"""Sequential plan, Stages 3-4: prefix risk scores, decision policy, early metrics.

Input : data/processed/erisk/task2_writing_vectors.csv  (from erisk_sequential_features.py)
Output: results/erisk_sequential/metrics.json + figures
        data/processed/erisk/task2_sequential_oof.csv   (per-writing OOF scores; gitignored)

Protocol
  * 523 users, StratifiedKFold(5, seed 42) on users -> a user is never in train and test.
  * Every prefix row (user u, writing t) of a training user is a training example with
    the user's label. Row weight = 1/n_u x class balance, so each user counts once.
  * At test time the model scores every prefix row -> p_t. No row uses writings > t.
  * Feature sets: clocks-only (M0), clocks+NRC emotions (M1), clocks+NRC minus clinical
    words (M1nc), emotions-only (E). Models: logistic regression, XGBoost.
  * Windows: p_t at the last writing within 4 / 8 / 12 weeks of the first writing,
    at writing k = 5..200, and at the last writing. AUROC/AUPRC + user bootstrap CI.
  * Decision policy: flag at first t with p_t >= tau and t >= t_min. (tau, t_min) chosen
    by inner 3-fold CV on training users only, then frozen for the outer test fold.
    ERDE5/50 (Losada & Crestani 2016), F1, latency-weighted F1 (Sadeque 2018).

Usage (from project root):
    python scripts/analysis/erisk_sequential_models.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erisk_early_metrics import Decisions, apply_policy, summarize, sweep_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "erisk"
OUT = ROOT / "results" / "erisk_sequential"
RNG = 42
N_OUTER = 5
N_INNER = 3
N_BOOT = 1000

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "positive", "negative"]

# ---- feature groups (all prefix-only; no n_so_far, no span, see plan) --------
CLOCKS = [
    "log_posts_per_week_so_far",
    "night_frac_so_far",
    "weekend_frac_so_far",
    "comment_frac_so_far",
    "log_median_gap_so_far",
    "gap_cv_so_far",
    "burst_1h_frac_so_far",
    "hour_entropy_so_far",
    "log_rate_change_so_far",
    "log_n_last7d",
    "log_n_last28d",
    "log_recent_rate_ratio",
]
EMO = [f"cum_emo_{e}" for e in EMOTIONS] + ["mean_log_tokens_so_far"]
EMO_NC = [f"cum_emonc_{e}" for e in EMOTIONS] + ["mean_log_tokens_so_far"]

# Robustness sets: comment_frac could be a collection artifact (positives' timelines
# are ~98% comments in month 1) and text length is text-derived; drop both.
RHYTHM = [f for f in CLOCKS if f != "comment_frac_so_far"]
EMO_NOLEN = [f for f in EMO if f != "mean_log_tokens_so_far"]

FEATURE_SETS = {
    "M0_clocks": CLOCKS,
    "M1_clocks_emo": CLOCKS + EMO,
    "M1nc_clocks_emo_noclinical": CLOCKS + EMO_NC,
    "E_emo_only": EMO,
    "M0r_rhythm_only": RHYTHM,
    "M1r_rhythm_emo_nolen": RHYTHM + EMO_NOLEN,
    "Er_emo_only_nolen": EMO_NOLEN,
}
MAIN_SETS = ("M0_clocks", "M1_clocks_emo", "M1nc_clocks_emo_noclinical", "E_emo_only")
MODELS = ("logistic", "xgb")

CAL_WINDOWS_DAYS = {"4w": 28, "8w": 56, "12w": 84}
K_WINDOWS = (5, 10, 25, 50, 100, 200)
TAUS = np.round(np.arange(0.30, 0.951, 0.025), 3)
T_MINS = (1, 3, 5, 10)
POLICY_OBJECTIVES = ("erde5", "erde50", "f_latency")
FIXED_POLICY = {"tau": 0.5, "t_min": 5}
TALK_FEATURES = [
    "posts_per_week_so_far", "night_frac_so_far", "comment_frac_so_far", "median_gap_so_far",
    "rate_change_so_far", "recent_rate_ratio", "hour_entropy_so_far", "mean_log_tokens_so_far",
    "cum_emo_sadness", "cum_emo_negative", "cum_emo_positive", "cum_emo_anger", "cum_emo_fear",
]


# --------------------------------------------------------------------------- #
def add_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_posts_per_week_so_far"] = np.log1p(df["posts_per_week_so_far"])
    df["log_median_gap_so_far"] = np.log1p(df["median_gap_so_far"])
    df["log_rate_change_so_far"] = np.log(df["rate_change_so_far"])
    df["log_n_last7d"] = np.log1p(df["n_last7d"])
    df["log_n_last28d"] = np.log1p(df["n_last28d"])
    df["log_recent_rate_ratio"] = np.log(df["recent_rate_ratio"].clip(lower=1e-3))
    return df


def make_model(kind: str) -> Pipeline | XGBClassifier:
    if kind == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=2000)),
            ]
        )
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RNG,
    )


def fit(kind: str, x: np.ndarray, y: np.ndarray, w: np.ndarray):
    model = make_model(kind)
    if kind == "logistic":
        model.fit(x, y, clf__sample_weight=w)
    else:
        model.fit(x, y, sample_weight=w)
    return model


def row_weights(rows: pd.DataFrame) -> np.ndarray:
    """1/n_u per row (each user weighs 1) x class balance; rescaled to mean 1."""
    n_per_user = rows.groupby("user_id")["t"].transform("size").to_numpy(dtype=float)
    users = rows.drop_duplicates("user_id")
    n_users = len(users)
    n_pos = int(users["label"].sum())
    class_factor = np.where(rows["label"].to_numpy() == 1, n_users / (2 * n_pos), n_users / (2 * (n_users - n_pos)))
    w = class_factor / n_per_user
    return w * (len(w) / w.sum())


# --------------------------------------------------------------------------- #
def window_rows(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One row per user for each evaluation window (last writing inside the window)."""
    out: dict[str, pd.DataFrame] = {}
    for name, days in CAL_WINDOWS_DAYS.items():
        sub = df[df["days_since_first"] <= days]
        out[name] = sub.groupby("user_id", sort=True).tail(1).set_index("user_id")
    for k in K_WINDOWS:
        sub = df[df["t"] <= k]
        out[f"k{k}"] = sub.groupby("user_id", sort=True).tail(1).set_index("user_id")
    out["full"] = df.groupby("user_id", sort=True).tail(1).set_index("user_id")
    return out


def bootstrap_auc(y: np.ndarray, scores: dict[str, np.ndarray], n_boot: int, seed: int) -> tuple[dict, dict]:
    """AUROC + CI per score column, and paired deltas vs the clocks-only score (same resamples)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    boots = {k: np.empty(n_boot) for k in scores}
    for b in range(n_boot):
        # stratified resample keeps both classes present
        take = np.concatenate([rng.choice(idx_pos, len(idx_pos)), rng.choice(idx_neg, len(idx_neg))])
        yb = y[take]
        for k, s in scores.items():
            boots[k][b] = roc_auc_score(yb, s[take])
    res = {}
    for k, s in scores.items():
        res[k] = {
            "auroc": float(roc_auc_score(y, s)),
            "auroc_ci95": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))],
            "auprc": float(average_precision_score(y, s)),
        }
    return res, boots


# --------------------------------------------------------------------------- #
def run_cv(df: pd.DataFrame, users: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Outer CV -> OOF p_t per config; nested policy tuning; permutation importances."""
    y_users = users["label"].to_numpy()
    outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RNG)
    oof = df[["user_id", "label", "t", "days_since_first"]].copy()
    tuned_decisions: dict[str, dict[str, list[Decisions]]] = {}
    fold_policies: dict[str, list[dict]] = {}
    perm_acc: dict[str, list[np.ndarray]] = {}
    user_idx = {u: i for i, u in enumerate(users["user_id"])}
    row_user_pos = df["user_id"].map(user_idx).to_numpy()

    for fold, (tr_u, te_u) in enumerate(outer.split(np.zeros(len(y_users)), y_users), 1):
        t_fold = time.time()
        tr_mask = np.isin(row_user_pos, tr_u)
        te_mask = ~tr_mask
        train = df[tr_mask]
        test = df[te_mask]
        w_train = row_weights(train)
        y_train = train["label"].to_numpy()
        prev_train = float(users.iloc[tr_u]["label"].mean())

        # inner split for policy tuning (training users only)
        inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RNG + fold)
        inner_users = users.iloc[tr_u].reset_index(drop=True)
        inner_pos = train["user_id"].map({u: i for i, u in enumerate(inner_users["user_id"])}).to_numpy()

        for fs_name, feats in FEATURE_SETS.items():
            x_train = train[feats].to_numpy(dtype=float)
            x_test = test[feats].to_numpy(dtype=float)
            for kind in MODELS:
                cfg = f"{kind}__{fs_name}"
                model = fit(kind, x_train, y_train, w_train)
                p_test = model.predict_proba(x_test)[:, 1]
                oof.loc[te_mask, f"p__{cfg}"] = p_test

                # nested: inner-OOF scores on training rows -> choose (tau, t_min)
                p_inner = np.zeros(len(train))
                for itr, ite in inner.split(np.zeros(len(inner_users)), inner_users["label"].to_numpy()):
                    m_in = np.isin(inner_pos, itr)
                    sub = train[m_in]
                    mdl = fit(kind, sub[feats].to_numpy(dtype=float), sub["label"].to_numpy(), row_weights(sub))
                    p_inner[~m_in] = mdl.predict_proba(train[~m_in][feats].to_numpy(dtype=float))[:, 1]
                inner_stream = pd.DataFrame(
                    {"user_id": train["user_id"].to_numpy(), "t": train["t"].to_numpy(),
                     "label": y_train, "p": p_inner}
                )
                best, _ = sweep_policy(inner_stream, TAUS, T_MINS, POLICY_OBJECTIVES, c_fp=prev_train)
                test_stream = pd.DataFrame(
                    {"user_id": test["user_id"].to_numpy(), "t": test["t"].to_numpy(),
                     "label": test["label"].to_numpy(), "p": p_test}
                )
                for obj, (tau, t_min) in best.items():
                    tuned_decisions.setdefault(cfg, {}).setdefault(obj, []).append(
                        apply_policy(test_stream, tau, t_min)
                    )
                fold_policies.setdefault(cfg, []).append(
                    {"fold": fold, **{obj: {"tau": tau, "t_min": t_min} for obj, (tau, t_min) in best.items()}}
                )

                # permutation importance on the 4-week rows of the test fold (XGB, descriptive)
                if kind == "xgb" and fs_name in ("M0_clocks", "M1_clocks_emo"):
                    rows4 = test[test["days_since_first"] <= CAL_WINDOWS_DAYS["4w"]].groupby("user_id").tail(1)
                    if rows4["label"].nunique() == 2:
                        pi = permutation_importance(
                            model, rows4[feats].to_numpy(dtype=float), rows4["label"].to_numpy(),
                            scoring="roc_auc", n_repeats=10, random_state=RNG, n_jobs=-1,
                        )
                        perm_acc.setdefault(fs_name, []).append(pi.importances_mean)
        print(f"fold {fold}/{N_OUTER} done in {time.time() - t_fold:.0f}s")

    # pool tuned decisions across folds (one decision per user, made in its test fold)
    pooled: dict[str, dict[str, Decisions]] = {}
    for cfg, by_obj in tuned_decisions.items():
        for obj, decs in by_obj.items():
            pooled.setdefault(cfg, {})[obj] = Decisions(
                user_id=np.concatenate([d.user_id for d in decs]),
                y_true=np.concatenate([d.y_true for d in decs]),
                flagged=np.concatenate([d.flagged for d in decs]),
                delay=np.concatenate([d.delay for d in decs]),
                n_writings=np.concatenate([d.n_writings for d in decs]),
            )
    perm = {k: np.mean(np.vstack(v), axis=0) for k, v in perm_acc.items()}
    return oof, {"decisions": pooled, "policies": fold_policies}, perm


# --------------------------------------------------------------------------- #
def plot_score_over_time(oof: pd.DataFrame, configs: list[str], path: Path, max_weeks: int = 26) -> None:
    """Median OOF p_t per week since first writing, carrying the last score forward."""
    fig, axes = plt.subplots(1, len(configs), figsize=(5.2 * len(configs), 4), sharey=True)
    axes = np.atleast_1d(axes)
    oof = oof.copy()
    oof["week"] = np.floor(oof["days_since_first"] / 7).astype(int)
    for ax, cfg in zip(axes, configs):
        col = f"p__{cfg}"
        last_per_week = oof.groupby(["user_id", "week"], sort=True)[col].last().unstack("week")
        last_per_week = last_per_week.reindex(columns=range(0, max_weeks + 1))
        # forward-fill only up to each user's last observed week (a monitor keeps the last score)
        last_week = oof.groupby("user_id")["week"].max().reindex(last_per_week.index).to_numpy()
        filled = last_per_week.ffill(axis=1)
        beyond = np.asarray(filled.columns)[None, :] > last_week[:, None]
        filled = filled.mask(beyond)
        labels = oof.groupby("user_id")["label"].first().reindex(filled.index)
        for lbl, colour, name in [(1, "#c0392b", "positive"), (0, "#2c7fb8", "control")]:
            block = filled[labels == lbl]
            med = block.median(axis=0)
            q1 = block.quantile(0.25, axis=0)
            q3 = block.quantile(0.75, axis=0)
            ax.plot(med.index, med.values, color=colour, lw=2, label=f"{name} (n={int((labels == lbl).sum())})")
            ax.fill_between(med.index, q1.values, q3.values, color=colour, alpha=0.15)
        ax.set_title(cfg.replace("__", "  |  "), fontsize=10)
        ax.set_xlabel("weeks since first writing")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("OOF risk score p_t (median, IQR)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("eRisk Task 2 - prefix-only score over time (no text in the model)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_windows_and_erde(win_metrics: dict, early: dict, baselines: dict, path: Path) -> None:
    order = ["4w", "8w", "12w", "full"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    colours = {"M0_clocks": "#1b9e77", "M1_clocks_emo": "#d95f02", "M1nc_clocks_emo_noclinical": "#7570b3", "E_emo_only": "#666666"}
    for kind, ls in (("xgb", "-"), ("logistic", "--")):
        for fs, c in colours.items():
            cfg = f"{kind}__{fs}"
            vals = [win_metrics[w][cfg]["auroc"] for w in order]
            lo = [win_metrics[w][cfg]["auroc_ci95"][0] for w in order]
            hi = [win_metrics[w][cfg]["auroc_ci95"][1] for w in order]
            x = np.arange(len(order)) + (0.05 if kind == "xgb" else -0.05)
            ax1.errorbar(x, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="o" + ls, color=c,
                         capsize=3, lw=1.5, ms=5, label=f"{kind} {fs}")
    ax1.axhline(0.5, color="k", lw=0.8, ls=":")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(["first 4 wk", "first 8 wk", "first 12 wk", "all history"])
    ax1.set_ylabel("AUROC (user bootstrap 95% CI)")
    ax1.set_title("How early does the score separate the groups?", fontsize=10)
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=6.5, ncol=2, loc="lower right")

    cfgs = [f"xgb__{fs}" for fs in MAIN_SETS] + [f"logistic__{fs}" for fs in MAIN_SETS]
    e5 = [early[c]["tuned"]["erde5"] for c in cfgs]
    e50 = [early[c]["tuned"]["erde50"] for c in cfgs]
    x = np.arange(len(cfgs))
    ax2.bar(x - 0.2, e5, 0.4, label="ERDE5", color="#fdae61")
    ax2.bar(x + 0.2, e50, 0.4, label="ERDE50", color="#abd9e9")
    ax2.axhline(baselines["always_negative"]["erde50"], color="k", ls=":", lw=1, label="never flag")
    ax2.axhline(baselines["flag_everyone_at_k1"]["erde50"], color="r", ls=":", lw=1, label="flag everyone at k=1 (ERDE50)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("__", "\n") for c in cfgs], fontsize=6.5)
    ax2.set_ylabel("ERDE (lower is better)")
    ax2.set_title("Early decision error, policy tuned on train folds only", fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_windows(wins: dict[str, pd.DataFrame], configs: list[str], path: Path) -> None:
    order = ["4w", "8w", "12w", "full"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, w in zip(axes, order):
        rows = wins[w]
        y = rows["label"].to_numpy()
        for cfg in configs:
            s = rows[f"p__{cfg}"].to_numpy()
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, lw=1.6, label=f"{cfg.split('__')[1]} (AUC {roc_auc_score(y, s):.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(f"{w}  (n={len(rows)})", fontsize=10)
        ax.set_xlabel("FPR")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("TPR")
    fig.suptitle("ROC by observation window - XGBoost, prefix-only features", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(perm: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (fs, name) in zip(axes, [("M0_clocks", "M0 clocks only"), ("M1_clocks_emo", "M1 clocks + emotions")]):
        feats = FEATURE_SETS[fs]
        vals = perm[fs]
        order = np.argsort(vals)
        ax.barh([feats[i] for i in order], vals[order], color=["#1b9e77" if feats[i] in CLOCKS else "#d95f02" for i in order])
        ax.set_title(f"{name} - permutation drop in AUROC, 4-week rows (mean of 5 test folds)", fontsize=9)
        ax.grid(alpha=0.3, axis="x")
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROC / "task2_writing_vectors.csv")
    assert not {"title", "body"} & set(df.columns)
    df = add_transforms(df)
    # same user order/seed as erisk_l1_l2.py so folds are comparable across experiments
    users = pd.read_csv(PROC / "task2_user_l2_features.csv", usecols=["user_id", "label"])
    assert set(users["user_id"]) == set(df["user_id"])
    n_users, n_pos = len(users), int(users["label"].sum())
    prevalence = n_pos / n_users
    print(f"rows {len(df):,}  users {n_users}  positives {n_pos}")

    oof, early_raw, perm = run_cv(df, users)
    oof.to_csv(PROC / "task2_sequential_oof.csv", index=False)
    configs = [f"{k}__{fs}" for fs in FEATURE_SETS for k in MODELS]

    # ---- windows -------------------------------------------------------------
    wins = window_rows(oof)
    win_metrics: dict[str, dict] = {}
    deltas: dict[str, dict] = {}
    for w, rows in wins.items():
        y = rows["label"].to_numpy()
        scores = {cfg: rows[f"p__{cfg}"].to_numpy() for cfg in configs}
        res, boots = bootstrap_auc(y, scores, N_BOOT, RNG)
        n_in_window = int((df.groupby("user_id")["days_since_first"].max() >= CAL_WINDOWS_DAYS[w]).sum()) if w in CAL_WINDOWS_DAYS else None
        for cfg in configs:
            res[cfg]["n_users"] = int(len(rows))
            if n_in_window is not None:
                res[cfg]["n_users_with_history_beyond_window"] = n_in_window
            res[cfg]["median_writings_seen"] = float(rows["t"].median())
        win_metrics[w] = res
        deltas[w] = {}
        pairs = [
            ("M1_clocks_emo", "M0_clocks"),
            ("M1nc_clocks_emo_noclinical", "M0_clocks"),
            ("E_emo_only", "M0_clocks"),
            ("M1_clocks_emo", "E_emo_only"),  # does behaviour add to emotion?
            ("M1nc_clocks_emo_noclinical", "E_emo_only"),
            ("M1r_rhythm_emo_nolen", "Er_emo_only_nolen"),  # same question, suspicious features removed
            ("M0_clocks", "M0r_rhythm_only"),  # how much does comment_frac carry?
        ]
        for kind in MODELS:
            for a, b in pairs:
                d = boots[f"{kind}__{a}"] - boots[f"{kind}__{b}"]
                deltas[w][f"{kind}__{a}_minus_{b}"] = {
                    "delta_auroc": res[f"{kind}__{a}"]["auroc"] - res[f"{kind}__{b}"]["auroc"],
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                }

    # ---- early decisions -------------------------------------------------------
    early: dict[str, dict] = {}
    for cfg in configs:
        stream = oof.rename(columns={f"p__{cfg}": "p"})[["user_id", "t", "label", "p"]]
        fixed = summarize(apply_policy(stream, FIXED_POLICY["tau"], FIXED_POLICY["t_min"]), c_fp=prevalence)
        early[cfg] = {
            **{f"tuned_for_{obj}": summarize(d, c_fp=prevalence) for obj, d in early_raw["decisions"][cfg].items()},
            "tuned_policy_per_fold": early_raw["policies"][cfg],
            "fixed_tau0.5_tmin5": fixed,
        }
        early[cfg]["tuned"] = early[cfg]["tuned_for_erde50"]  # headline policy, as in the plan

    # medians of the raw (untransformed) prefix features at the 4-week window, by label
    raw4 = df[df["days_since_first"] <= CAL_WINDOWS_DAYS["4w"]].groupby("user_id").tail(1)
    medians_4w = {
        str(lbl): {c: float(g[c].median()) for c in TALK_FEATURES} | {"n_users": int(len(g)), "median_writings_seen": float(g["t"].median())}
        for lbl, g in raw4.groupby("label")
    }
    n_w = df.groupby("user_id")["t"].max().reindex(users["user_id"]).to_numpy()
    y_u = users["label"].to_numpy()
    base_neg = Decisions(users["user_id"].to_numpy(), y_u, np.zeros(n_users, int), n_w, n_w)
    base_all1 = Decisions(users["user_id"].to_numpy(), y_u, np.ones(n_users, int), np.ones(n_users, int), n_w)
    base_all5 = Decisions(users["user_id"].to_numpy(), y_u, np.ones(n_users, int), np.full(n_users, 5), n_w)
    baselines = {
        "always_negative": summarize(base_neg, prevalence),
        "flag_everyone_at_k1": summarize(base_all1, prevalence),
        "flag_everyone_at_k5": summarize(base_all5, prevalence),
    }

    # ---- figures -------------------------------------------------------------------
    plot_score_over_time(oof, ["xgb__M0_clocks", "xgb__M1_clocks_emo"], OUT / "fig_score_over_time.png")
    plot_windows_and_erde(win_metrics, early, baselines, OUT / "fig_erde_vs_window.png")
    plot_roc_windows(wins, ["xgb__M0_clocks", "xgb__M1_clocks_emo", "xgb__E_emo_only"], OUT / "fig_roc_windows.png")
    if perm:
        plot_importance(perm, OUT / "fig_feature_importance_4w.png")

    # ---- metrics.json -------------------------------------------------------------
    metrics = {
        "dataset": {
            "name": "eRisk 2026 Task 2 (Reddit, self-mention protocol labels; not clinical)",
            "n_users": n_users,
            "n_positive": n_pos,
            "n_writings": int(len(df)),
            "chance_auprc": prevalence,
        },
        "protocol": {
            "outer_cv": f"StratifiedKFold({N_OUTER}) on users, seed {RNG}, same order as erisk_l1_l2.py",
            "training_rows": "all prefix rows of training users, weight 1/n_user x class balance",
            "scoring": "p_t from the cumulative prefix vector at each writing t (no future writings)",
            "policy": f"flag at first t with p_t>=tau and t>=t_min; (tau,t_min) from inner {N_INNER}-fold CV on train users, one policy per objective {POLICY_OBJECTIVES}; headline = ERDE50",
            "tau_grid": TAUS.tolist(),
            "t_min_grid": list(T_MINS),
            "erde_costs": {"c_fn": 1, "c_tp": 1, "c_fp": prevalence},
            "bootstrap": f"{N_BOOT} stratified user resamples for AUROC CI",
            "excluded_on_purpose": ["raw text / TF-IDF", "n_so_far", "span_days (full or so-far)", "any feature from writings > t"],
            "main_sets_in_figures": list(MAIN_SETS),
            "robustness_sets": ["M0r_rhythm_only", "M1r_rhythm_emo_nolen", "Er_emo_only_nolen"],
        },
        "features": FEATURE_SETS,
        "windows": win_metrics,
        "delta_auroc_paired_bootstrap": deltas,
        "feature_medians_at_4w_by_label": medians_4w,
        "early_decisions": early,
        "baselines": baselines,
        "permutation_importance_4w": {fs: dict(zip(FEATURE_SETS[fs], map(float, v))) for fs, v in perm.items()},
        "notes": [
            "Emotion columns come from the NRC Emotion Lexicon applied to title+body once; text never enters the model.",
            "M1nc removes explicit condition/treatment words (depression, anxiety, suicide, therapy, ...) from the lexicon before counting.",
            "Robustness sets (M0r/M1r/Er) drop comment_frac_so_far (positives' month-1 timelines are ~98% comments vs ~82% for controls; may reflect how eRisk assembles positives) and mean_log_tokens_so_far (text length).",
            "Calendar windows: score at the last writing within 4/8/12 weeks of the user's first writing; users whose whole history is shorter use their last writing.",
            "k windows: score at writing min(k, n_user).",
            "ERDE50 of 'flag everyone at k=1' equals c_fp*(1-prevalence): any useful policy must beat it and never-flag.",
            "hour/night use IST (UTC+5:30) as in the existing L2 tables; the entropy/fraction clocks do not depend on the offset choice.",
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ---- console summary --------------------------------------------------------
    print("\nAUROC by window (xgb / logistic):")
    order = ["4w", "8w", "12w", "k10", "k50", "full"]
    hdr = f"{'feature set':32s}" + "".join(f"{w:>15s}" for w in order)
    print(hdr)
    for fs in FEATURE_SETS:
        for kind in MODELS:
            cfg = f"{kind}__{fs}"
            print(f"{cfg:32s}" + "".join(f"{win_metrics[w][cfg]['auroc']:>15.3f}" for w in order))
    print("\nEarly decisions (policy tuned on train folds for each objective; pooled test folds):")
    for cfg in configs:
        for obj in POLICY_OBJECTIVES:
            m = early[cfg][f"tuned_for_{obj}"]
            print(f"{cfg:32s} [{obj:9s}] ERDE5 {m['erde5']:.3f}  ERDE50 {m['erde50']:.3f}  F1 {m['f1']:.3f}  "
                  f"P {m['precision']:.2f} R {m['recall']:.2f}  Flat {m['f_latency']:.3f}  med delay {m['median_tp_delay']}")
    for k, m in baselines.items():
        print(f"{k:32s} ERDE5 {m['erde5']:.3f}  ERDE50 {m['erde50']:.3f}  F1 {m['f1']:.3f}")
    print(f"\nwrote {OUT / 'metrics.json'}  ({metrics['runtime_sec']}s)")


if __name__ == "__main__":
    main()
