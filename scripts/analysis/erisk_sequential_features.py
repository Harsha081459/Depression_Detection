"""Sequential plan, Stages 0-2: one numeric vector per eRisk Task 2 writing.

Text (title + body) is read here ONCE, turned into NRC emotion counts, and then
discarded. Everything else comes from the timestamp and the writing kind.

Every ``*_so_far`` column on writing t uses writings 1..t of the same user and
nothing else (no future writings, no full-timeline span). A random self-check
recomputes a sample of rows from the raw prefix to prove that.

Outputs
    data/processed/erisk/task2_writing_vectors.csv   (no title / body columns)
    results/erisk_sequential/features_report.json

Usage (from project root):
    python scripts/analysis/erisk_sequential_features.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "erisk"
LEX = ROOT / "data" / "raw" / "lexicons" / "nrc" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
OUT_DIR = ROOT / "results" / "erisk_sequential"
OUT_CSV = PROC / "task2_writing_vectors.csv"
OUT_REPORT = OUT_DIR / "features_report.json"

TZ = "Asia/Kolkata"  # same convention as the existing L2 parser; night = 0-5 IST
NIGHT_HOURS = set(range(0, 6))
WEEKEND_DOW = {5, 6}  # pandas dayofweek: Mon=0 ... Sun=6
RNG = 42

EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
    "positive",
    "negative",
]

# Words that name the condition / its treatment rather than express an emotion.
# Used only for the "emonc" ablation (NRC minus clinical vocabulary), so we can
# ask: is emotion helping, or is the word "depressed" helping?
CLINICAL_CANDIDATES = [
    "depression", "depressed", "depressing", "depress", "depressive",
    "antidepressant", "anxiety", "anxious", "suicide", "suicidal",
    "therapy", "therapist", "psychiatrist", "psychiatric", "psychiatry",
    "psychologist", "psychology", "medication", "medicine", "diagnosis",
    "diagnosed", "diagnose", "disorder", "mental", "illness", "bipolar",
    "insomnia", "counseling", "counselor", "prescription", "pill", "pills",
    "hospital", "doctor", "symptom", "symptoms", "disease", "sick", "sickness",
]

# Minimum prefix before rate-change is trusted (else neutral 1.0).
RATE_CHANGE_MIN_WRITINGS = 6
RATE_CHANGE_MIN_SPAN_DAYS = 1.0
# Rates are computed over max(span, 1 day) so a same-day burst does not explode.
RATE_SPAN_FLOOR_DAYS = 1.0

TOKEN_PATTERN = r"(?u)\b[a-z][a-z'-]*[a-z]\b|\b[a-z]\b"


# --------------------------------------------------------------------------- #
# Stage 0 - emotions from the NRC lexicon
# --------------------------------------------------------------------------- #
def load_nrc(path: Path) -> dict[str, set[str]]:
    """word -> set of NRC categories it is associated with (value == 1)."""
    if not path.exists():
        sys.exit(
            f"NRC lexicon not found at {path}.\n"
            "Download NRC-Emotion-Lexicon.zip from saifmohammad.com and place "
            "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt there (kept out of git)."
        )
    lex: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, emo, flag = parts
            if flag == "1" and emo in EMOTIONS:
                lex.setdefault(word.lower(), set()).add(emo)
    return lex


def emotion_counter(lex: dict[str, set[str]]) -> tuple[CountVectorizer, csr_matrix, list[str]]:
    """Fixed-vocabulary counter + (V x 10) indicator so counts = X @ E."""
    vocab = sorted(lex)
    vec = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern=TOKEN_PATTERN)
    rows, cols = [], []
    for j, word in enumerate(vocab):
        for emo in lex[word]:
            rows.append(j)
            cols.append(EMOTIONS.index(emo))
    ind = csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(vocab), len(EMOTIONS)),
    )
    return vec, ind, vocab


def count_emotions(texts: pd.Series, lex: dict[str, set[str]], prefix: str) -> tuple[pd.DataFrame, dict]:
    vec, ind, vocab = emotion_counter(lex)
    analyzer = vec.build_analyzer()
    unmatchable = [w for w in vocab if analyzer(w) != [w]]
    x = vec.transform(texts.tolist())
    counts = np.asarray((x @ ind).todense(), dtype=np.int32)
    df = pd.DataFrame(counts, columns=[f"{prefix}_{e}_cnt" for e in EMOTIONS], index=texts.index)
    info = {
        "lexicon_words": len(vocab),
        "lexicon_words_not_matchable_by_tokenizer": len(unmatchable),
        "example_unmatchable": unmatchable[:10],
    }
    return df, info


# --------------------------------------------------------------------------- #
# Stage 1 - prefix-only clocks for one user
# --------------------------------------------------------------------------- #
def _entropy_bits_rows(cum_counts: np.ndarray) -> np.ndarray:
    tot = cum_counts.sum(axis=1, keepdims=True).astype(float)
    p = np.divide(cum_counts, tot, out=np.zeros_like(cum_counts, dtype=float), where=tot > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * lp).sum(axis=1)


def user_prefix_clocks(g: pd.DataFrame) -> pd.DataFrame:
    """g: one user's writings sorted by time. Returns per-writing clock columns."""
    n = len(g)
    ts = g["ts_sec"].to_numpy(dtype=np.int64)
    hour = g["hour"].to_numpy(dtype=np.int64)
    night = g["night"].to_numpy(dtype=np.int64)
    weekend = g["weekend"].to_numpy(dtype=np.int64)
    is_comment = g["is_comment"].to_numpy(dtype=np.int64)

    t = np.arange(1, n + 1, dtype=np.int64)
    hours_since_first = (ts - ts[0]) / 3600.0
    days_since_first = hours_since_first / 24.0
    gap_hours = np.zeros(n, dtype=float)
    gap_hours[1:] = np.diff(ts) / 3600.0

    span_floor = np.maximum(days_since_first, RATE_SPAN_FLOOR_DAYS)
    posts_per_week_so_far = t / span_floor * 7.0

    night_frac_so_far = np.cumsum(night) / t
    weekend_frac_so_far = np.cumsum(weekend) / t
    comment_frac_so_far = np.cumsum(is_comment) / t

    # gap statistics use gaps 2..t (t-1 gaps); neutral 0 when there is no gap yet
    k_gaps = np.maximum(t - 1, 1)
    gaps = gap_hours[1:]
    median_gap_so_far = np.zeros(n, dtype=float)
    if n > 1:
        median_gap_so_far[1:] = pd.Series(gaps).expanding().median().to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(gaps)])
    css = np.concatenate([[0.0], np.cumsum(gaps**2)])
    mean_gap = cs / k_gaps
    var_gap = np.maximum(css / k_gaps - mean_gap**2, 0.0)
    gap_cv_so_far = np.where(t >= 3, np.sqrt(var_gap) / np.maximum(mean_gap, 1e-9), 0.0)
    burst = np.concatenate([[0], (gaps < 1.0).astype(np.int64)])
    burst_1h_frac_so_far = np.where(t >= 2, np.cumsum(burst) / k_gaps, 0.0)

    onehot = np.zeros((n, 24), dtype=np.int64)
    onehot[np.arange(n), hour] = 1
    hour_entropy_so_far = _entropy_bits_rows(np.cumsum(onehot, axis=0))

    # rate change inside the prefix: writings in the 2nd half of [ts_1, ts_t]
    # versus the 1st half; his "weekly -> every seven hours" clue, prefix-only
    mid = ts[0] + (ts - ts[0]) / 2.0
    n_first_half = np.searchsorted(ts, mid, side="right")
    n_second_half = t - n_first_half
    ratio = (n_second_half + 0.5) / (n_first_half + 0.5)
    trusted = (t >= RATE_CHANGE_MIN_WRITINGS) & (days_since_first >= RATE_CHANGE_MIN_SPAN_DAYS)
    rate_change_so_far = np.where(trusted, ratio, 1.0)

    # recent activity: writings in the trailing 7 / 28 days (inclusive of t)
    n_last7d = t - np.searchsorted(ts, ts - 7 * 86400, side="right")
    n_last28d = t - np.searchsorted(ts, ts - 28 * 86400, side="right")
    recent_rate_ratio = n_last7d / np.maximum(posts_per_week_so_far, 1e-9)

    return pd.DataFrame(
        {
            "t": t,
            "days_since_first": days_since_first,
            "gap_hours": gap_hours,
            "posts_per_week_so_far": posts_per_week_so_far,
            "night_frac_so_far": night_frac_so_far,
            "weekend_frac_so_far": weekend_frac_so_far,
            "comment_frac_so_far": comment_frac_so_far,
            "median_gap_so_far": median_gap_so_far,
            "gap_cv_so_far": gap_cv_so_far,
            "burst_1h_frac_so_far": burst_1h_frac_so_far,
            "hour_entropy_so_far": hour_entropy_so_far,
            "rate_change_so_far": rate_change_so_far,
            "n_last7d": n_last7d,
            "n_last28d": n_last28d,
            "recent_rate_ratio": recent_rate_ratio,
        },
        index=g.index,
    )


# --------------------------------------------------------------------------- #
# Stage 2 - cumulative emotion (token-weighted mean over writings 1..t)
# --------------------------------------------------------------------------- #
def cumulative_emotions(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    grp = df.groupby("user_id", sort=False)
    cum_tokens = grp["n_tokens"].cumsum().to_numpy(dtype=float)
    out = {}
    for e in EMOTIONS:
        cum_cnt = grp[f"{prefix}_{e}_cnt"].cumsum().to_numpy(dtype=float)
        out[f"cum_{prefix}_{e}"] = np.divide(
            cum_cnt, cum_tokens, out=np.zeros_like(cum_cnt), where=cum_tokens > 0
        )
    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------- #
# Self-check: recompute a few rows straight from the raw prefix
# --------------------------------------------------------------------------- #
def verify_prefix_only(df: pd.DataFrame, n_checks: int = 40, seed: int = RNG) -> int:
    rng = np.random.default_rng(seed)
    users = df["user_id"].unique()
    checked = 0
    for user in rng.choice(users, size=min(n_checks, len(users)), replace=False):
        g = df[df["user_id"] == user].sort_values("t")
        t = int(rng.integers(1, len(g) + 1))
        pre = g.iloc[:t]
        row = g.iloc[t - 1]
        ts = pre["ts_sec"].to_numpy(dtype=np.int64)
        span_days = (ts[-1] - ts[0]) / 86400.0
        exp_rate = t / max(span_days, RATE_SPAN_FLOOR_DAYS) * 7.0
        assert np.isclose(row["posts_per_week_so_far"], exp_rate), (user, t, "rate")
        assert np.isclose(row["night_frac_so_far"], pre["night"].mean()), (user, t, "night")
        assert np.isclose(row["comment_frac_so_far"], pre["is_comment"].mean()), (user, t, "comment")
        if t > 1:
            gaps = np.diff(ts) / 3600.0
            assert np.isclose(row["median_gap_so_far"], np.median(gaps)), (user, t, "median gap")
        exp_last7 = int(((ts > ts[-1] - 7 * 86400) & (ts <= ts[-1])).sum())
        assert int(row["n_last7d"]) == exp_last7, (user, t, "n_last7d")
        if t >= RATE_CHANGE_MIN_WRITINGS and span_days >= RATE_CHANGE_MIN_SPAN_DAYS:
            mid = ts[0] + (ts[-1] - ts[0]) / 2.0
            n1 = int((ts <= mid).sum())
            exp_rc = (t - n1 + 0.5) / (n1 + 0.5)
            assert np.isclose(row["rate_change_so_far"], exp_rc), (user, t, "rate change")
        else:
            assert row["rate_change_so_far"] == 1.0, (user, t, "rate change neutral")
        tok = pre["n_tokens"].sum()
        exp_sad = pre["emo_sadness_cnt"].sum() / tok if tok > 0 else 0.0
        assert np.isclose(row["cum_emo_sadness"], exp_sad), (user, t, "cum sadness")
        checked += 1
    return checked


# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    text = pd.read_csv(PROC / "task2_target_text.csv")
    labels = pd.read_csv(PROC / "task2_user_l2_features.csv", usecols=["user_id", "label"])
    print(f"writings: {len(text):,}  users: {text['user_id'].nunique()}")

    dt_utc = pd.to_datetime(text["date"], utc=True)
    text["dt"] = dt_utc.dt.tz_convert(TZ)
    text["ts_sec"] = (
        dt_utc.dt.tz_convert(None).to_numpy().astype("datetime64[s]").astype(np.int64)
    )
    text["hour"] = text["dt"].dt.hour.astype(np.int64)
    text["dow"] = text["dt"].dt.dayofweek.astype(np.int64)
    text["night"] = text["hour"].isin(NIGHT_HOURS).astype(np.int64)
    text["weekend"] = text["dow"].isin(WEEKEND_DOW).astype(np.int64)
    text["is_comment"] = (text["kind"] == "comment").astype(np.int64)

    text = text.sort_values(["user_id", "ts_sec"], kind="mergesort").reset_index(drop=True)

    # ---- Stage 0: emotions (the only place text is touched) ----------------
    lex_full = load_nrc(LEX)
    clinical_used = sorted(w for w in CLINICAL_CANDIDATES if w in lex_full)
    lex_nc = {w: s for w, s in lex_full.items() if w not in set(clinical_used)}

    body = text["body"].fillna("").astype(str)
    title = text["title"].fillna("").astype(str)
    docs = (title + " " + body).str.strip()

    emo_full, info_full = count_emotions(docs, lex_full, "emo")
    emo_nc, _ = count_emotions(docs, lex_nc, "emonc")
    analyzer = CountVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN).build_analyzer()
    n_tokens = np.fromiter((len(analyzer(d)) for d in docs.tolist()), dtype=np.int64, count=len(docs))

    text = text.drop(columns=["title", "body", "author"])  # text leaves the pipeline here
    del docs, body, title

    text["n_tokens"] = n_tokens
    text["is_empty_text"] = (n_tokens == 0).astype(np.int64)
    text = pd.concat([text, emo_full, emo_nc], axis=1)
    for e in EMOTIONS:
        text[f"emo_{e}_frac"] = np.divide(
            text[f"emo_{e}_cnt"].to_numpy(float), np.maximum(n_tokens, 1)
        )
    print(f"emotions done in {time.time() - t0:.1f}s")

    # ---- Stage 1: prefix-only clocks --------------------------------------
    clocks = (
        text.groupby("user_id", sort=False, group_keys=False)[
            ["ts_sec", "hour", "night", "weekend", "is_comment"]
        ]
        .apply(user_prefix_clocks)
    )
    text = pd.concat([text, clocks], axis=1)
    print(f"clocks done in {time.time() - t0:.1f}s")

    # ---- Stage 2: cumulative emotion --------------------------------------
    text = pd.concat(
        [text, cumulative_emotions(text, "emo"), cumulative_emotions(text, "emonc")], axis=1
    )
    grp = text.groupby("user_id", sort=False)
    text["mean_log_tokens_so_far"] = (
        grp["n_tokens"].transform(lambda s: np.log1p(s).cumsum()) / text["t"]
    )
    text = text.merge(labels, on="user_id", how="left", validate="many_to_one")
    assert text["label"].notna().all(), "every writing must have a user label"
    text["label"] = text["label"].astype(np.int64)

    checked = verify_prefix_only(text)
    print(f"prefix-only self-check passed on {checked} random (user, t) rows")

    forbidden = {"title", "body", "author"}
    assert not (forbidden & set(text.columns)), "text columns must not be written"

    id_cols = ["user_id", "label", "t", "date", "days_since_first", "hour", "dow", "kind"]
    inst_cols = (
        ["night", "weekend", "is_comment", "gap_hours", "n_tokens", "is_empty_text"]
        + [f"emo_{e}_frac" for e in EMOTIONS]
        + [f"emo_{e}_cnt" for e in EMOTIONS]
        + [f"emonc_{e}_cnt" for e in EMOTIONS]
    )
    cum_cols = [
        "posts_per_week_so_far",
        "night_frac_so_far",
        "weekend_frac_so_far",
        "comment_frac_so_far",
        "median_gap_so_far",
        "gap_cv_so_far",
        "burst_1h_frac_so_far",
        "hour_entropy_so_far",
        "rate_change_so_far",
        "n_last7d",
        "n_last28d",
        "recent_rate_ratio",
        "mean_log_tokens_so_far",
    ] + [f"cum_emo_{e}" for e in EMOTIONS] + [f"cum_emonc_{e}" for e in EMOTIONS]
    out = text[id_cols + inst_cols + cum_cols]
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  shape={out.shape}")

    # ---- report (numbers only; no text) -----------------------------------
    last = out.groupby("user_id").tail(1)
    by_label = last.groupby("label")
    report = {
        "n_writings": int(len(out)),
        "n_users": int(out["user_id"].nunique()),
        "n_positive_users": int(labels["label"].sum()),
        "timezone_for_hour": TZ,
        "night_hours": sorted(NIGHT_HOURS),
        "lexicon": {
            "file": LEX.name,
            **info_full,
            "clinical_words_removed_in_emonc": clinical_used,
        },
        "text_coverage": {
            "writings_with_zero_tokens": int((out["n_tokens"] == 0).sum()),
            "median_tokens_per_writing": float(out["n_tokens"].median()),
            "writings_with_any_emotion_hit": float(
                (out[[f"emo_{e}_cnt" for e in EMOTIONS]].sum(axis=1) > 0).mean()
            ),
        },
        "end_of_history_medians_by_label": {
            str(lbl): {
                c: float(g[c].median())
                for c in [f"cum_emo_{e}" for e in EMOTIONS]
                + ["posts_per_week_so_far", "night_frac_so_far", "median_gap_so_far",
                   "rate_change_so_far", "recent_rate_ratio", "hour_entropy_so_far"]
            }
            for lbl, g in by_label
        },
        "columns": {"id": id_cols, "instantaneous_2a": inst_cols, "cumulative_2b": cum_cols},
        "prefix_only_selfcheck_rows": checked,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT_REPORT}  ({report['runtime_sec']}s)")


if __name__ == "__main__":
    main()
