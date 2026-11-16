"""Early-risk metrics for a stream of per-writing scores.

ERDE_o  - Losada & Crestani (CLEF 2016), Sect. 3.1:
            FP -> c_fp ; FN -> c_fn ; TP -> lc_o(k) * c_tp ; TN -> 0
            lc_o(k) = 1 - 1 / (1 + exp(k - o)),  k = writings seen at decision
            c_fn = c_tp = 1 ; c_fp = proportion of positive users (their default)
          ERDE is the mean over users.
F_latency - Sadeque, Xu & Bethard (WSDM 2018): F1 x (1 - median penalty over TPs),
            penalty(k) = -1 + 2 / (1 + exp(-p (k - 1))), eRisk uses p = 0.0078.

A "decision" is one per user: positive at delay k (first writing where the
policy fires) or negative at the end of the stream (k = number of writings).

Run this file directly for a self-test on hand-computed streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass(frozen=True)
class Decisions:
    """One row per user. flagged=1 means the policy emitted 'positive' at delay k."""

    user_id: np.ndarray
    y_true: np.ndarray
    flagged: np.ndarray
    delay: np.ndarray  # writings seen when the decision was emitted
    n_writings: np.ndarray


def latency_cost(k: np.ndarray, o: float) -> np.ndarray:
    # 1 - 1/(1+exp(k-o)) == sigmoid(k-o); expit avoids overflow for long streams
    return expit(np.asarray(k, dtype=float) - o)


def erde(d: Decisions, o: float, c_fp: float | None = None, c_fn: float = 1.0, c_tp: float = 1.0) -> float:
    y = d.y_true.astype(bool)
    f = d.flagged.astype(bool)
    if c_fp is None:
        c_fp = float(y.mean())
    cost = np.zeros(len(y), dtype=float)
    cost[f & ~y] = c_fp
    cost[~f & y] = c_fn
    tp = f & y
    cost[tp] = latency_cost(d.delay[tp], o) * c_tp
    return float(cost.mean())


def latency_penalty(k: np.ndarray, p: float = 0.0078) -> np.ndarray:
    return -1.0 + 2.0 * expit(p * (np.asarray(k, dtype=float) - 1.0))


def precision_recall_f1(d: Decisions) -> tuple[float, float, float]:
    y = d.y_true.astype(bool)
    f = d.flagged.astype(bool)
    tp = int((f & y).sum())
    fp = int((f & ~y).sum())
    fn = int((~f & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def latency_weighted_f1(d: Decisions, p: float = 0.0078) -> tuple[float, float, float]:
    """Returns (F_latency, speed, median TP delay)."""
    y = d.y_true.astype(bool)
    f = d.flagged.astype(bool)
    _, _, f1 = precision_recall_f1(d)
    tp = f & y
    if tp.sum() == 0:
        return 0.0, 0.0, float("nan")
    med = float(np.median(d.delay[tp]))
    speed = 1.0 - float(latency_penalty(np.array([med]), p)[0])
    return f1 * speed, speed, med


def summarize(d: Decisions, c_fp: float | None = None) -> dict:
    prec, rec, f1 = precision_recall_f1(d)
    f_lat, speed, med = latency_weighted_f1(d)
    return {
        "erde5": erde(d, 5.0, c_fp),
        "erde50": erde(d, 50.0, c_fp),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f_latency": f_lat,
        "speed": speed,
        "median_tp_delay": med,
        "n_flagged": int(d.flagged.sum()),
        "n_users": int(len(d.y_true)),
    }


class _SortedStream:
    """Stream sorted by (user, t) with per-user segment bounds, so a policy sweep is cheap."""

    def __init__(self, stream: pd.DataFrame, score_col: str = "p"):
        s = stream.sort_values(["user_id", "t"], kind="mergesort")
        self.p = s[score_col].to_numpy(dtype=float)
        self.t = s["t"].to_numpy(dtype=np.int64)
        users, starts = np.unique(s["user_id"].to_numpy(), return_index=True)
        self.user_id = users
        self.starts = starts
        self.ends = np.append(starts[1:], len(s))
        self.y_true = s["label"].to_numpy()[starts].astype(int)
        self.n_writings = self.t[self.ends - 1]

    def decide(self, tau: float, t_min: int) -> Decisions:
        fire = (self.p >= tau) & (self.t >= t_min)
        fire_t = np.where(fire, self.t, np.iinfo(np.int64).max)
        first = np.minimum.reduceat(fire_t, self.starts)
        flagged = first < np.iinfo(np.int64).max
        delay = np.where(flagged, first, self.n_writings).astype(int)
        return Decisions(
            user_id=self.user_id,
            y_true=self.y_true,
            flagged=flagged.astype(int),
            delay=delay,
            n_writings=self.n_writings.astype(int),
        )


def apply_policy(stream: pd.DataFrame, tau: float, t_min: int, score_col: str = "p") -> Decisions:
    """First writing t with p >= tau and t >= t_min flags the user; else negative at end.

    ``stream`` needs columns user_id, t, label, <score_col>; any row order.
    """
    return _SortedStream(stream, score_col).decide(tau, t_min)


MAXIMISE = {"f_latency", "f1", "precision", "recall", "speed"}


def sweep_policy(
    stream: pd.DataFrame,
    taus: np.ndarray,
    t_mins: tuple[int, ...],
    objectives: tuple[str, ...] = ("erde50",),
    score_col: str = "p",
    c_fp: float | None = None,
) -> tuple[dict[str, tuple[float, int]], pd.DataFrame]:
    """Grid search (tau, t_min); returns the best pair per objective plus the full table.

    ERDE-type objectives are minimised, F-type objectives maximised. Ties prefer the
    higher tau (fewer flags) and then the smaller t_min.
    """
    rows = []
    sorted_stream = _SortedStream(stream, score_col)
    for t_min in t_mins:
        for tau in taus:
            d = sorted_stream.decide(float(tau), int(t_min))
            m = summarize(d, c_fp)
            rows.append({"tau": float(tau), "t_min": int(t_min), **m})
    table = pd.DataFrame(rows)
    best: dict[str, tuple[float, int]] = {}
    for obj in objectives:
        asc = obj not in MAXIMISE
        row = table.sort_values([obj, "tau", "t_min"], ascending=[asc, False, True]).iloc[0]
        best[obj] = (float(row["tau"]), int(row["t_min"]))
    return best, table


# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # Four users: A pos flagged at k=1, B ctrl flagged (FP), C pos never flagged, D ctrl never flagged
    d = Decisions(
        user_id=np.array(["A", "B", "C", "D"]),
        y_true=np.array([1, 0, 1, 0]),
        flagged=np.array([1, 1, 0, 0]),
        delay=np.array([1, 3, 10, 10]),
        n_writings=np.array([10, 10, 10, 10]),
    )
    c_fp = 0.5  # 2 of 4 users positive
    lc5_1 = 1 - 1 / (1 + np.exp(1 - 5))  # 0.01798...
    exp_erde5 = (lc5_1 + c_fp + 1.0 + 0.0) / 4
    exp_erde50 = ((1 - 1 / (1 + np.exp(1 - 50))) + c_fp + 1.0 + 0.0) / 4
    assert np.isclose(erde(d, 5), exp_erde5), erde(d, 5)
    assert np.isclose(erde(d, 50), exp_erde50), erde(d, 50)
    assert np.isclose(erde(d, 5), 0.37949, atol=1e-4)
    assert np.isclose(erde(d, 50), 0.375, atol=1e-6)
    prec, rec, f1 = precision_recall_f1(d)
    assert (prec, rec) == (0.5, 0.5) and np.isclose(f1, 0.5)

    # latency cost end points: very early ~0, at k=o exactly 0.5, very late ~1
    assert np.isclose(latency_cost(np.array([5.0]), 5.0)[0], 0.5)
    assert latency_cost(np.array([1.0]), 50.0)[0] < 1e-12
    assert latency_cost(np.array([200.0]), 50.0)[0] > 1 - 1e-12

    # Sadeque penalty: 0 at k=1; ~0.5 near the median-writings anchor used by eRisk (p=0.0078 -> k~142)
    assert latency_penalty(np.array([1.0]))[0] == 0.0
    assert abs(latency_penalty(np.array([142.0]))[0] - 0.5) < 0.01

    # apply_policy on a tiny stream: user X fires at t=3 (t_min=2 blocks t=1), Y never fires
    stream = pd.DataFrame(
        {
            "user_id": ["X"] * 4 + ["Y"] * 3,
            "t": [1, 2, 3, 4, 1, 2, 3],
            "label": [1] * 4 + [0] * 3,
            "p": [0.9, 0.2, 0.7, 0.1, 0.3, 0.4, 0.2],
        }
    )
    dec = apply_policy(stream, tau=0.6, t_min=2)
    assert dec.flagged.tolist() == [1, 0] and dec.delay.tolist() == [3, 3]
    dec = apply_policy(stream, tau=0.6, t_min=1)
    assert dec.flagged.tolist() == [1, 0] and dec.delay.tolist() == [1, 3]

    # baselines: always-negative ERDE == prevalence ; flag-everyone-at-k=1 ERDE ~ c_fp*(1-prev)
    y = np.array([1] * 3 + [0] * 7)
    neg = Decisions(np.arange(10), y, np.zeros(10, int), np.full(10, 20), np.full(10, 20))
    assert np.isclose(erde(neg, 5), 0.3)
    allpos = Decisions(np.arange(10), y, np.ones(10, int), np.ones(10, int), np.full(10, 20))
    assert np.isclose(erde(allpos, 50), 0.3 * 0.7, atol=1e-6)
    print("erisk_early_metrics self-test: OK")


if __name__ == "__main__":
    _selftest()
