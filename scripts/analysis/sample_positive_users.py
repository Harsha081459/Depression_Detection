"""
Stage A3: check the frozen 9-user hand-pick against duration strata.

Source of truth: results/erisk_understand/sample_positive_ids.json
(written by update_sample_selection.py). Do not random-sample.
"""

import json
from pathlib import Path

import pandas as pd


def repo_root():
    return Path(__file__).resolve().parent.parent.parent


def main():
    base = repo_root()
    sample_path = base / "results/erisk_understand/sample_positive_ids.json"
    duration = pd.read_csv(base / "results/erisk_understand/user_duration.csv")
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    ids = sample["all_ids"]

    if len(ids) != 9 or len(set(ids)) != 9:
        raise SystemExit(f"Need 9 unique ids, got {ids}")

    rows = duration[duration["user_id"].isin(ids)]
    if len(rows) != 9:
        missing = set(ids) - set(rows["user_id"])
        raise SystemExit(f"Ids missing from duration table: {missing}")
    if not (rows["label"] == 1).all():
        raise SystemExit("Every sampled user must be eRisk-positive")

    short = sample["short_span"]
    medium = sample["medium_span"]
    long = sample["long_span"]
    checks = [
        (short[0]["span_days"] < 14, "short[0] must be burst <14d"),
        (20 <= short[1]["span_days"] <= 40, "short[1] must be ~1 month"),
        (60 <= short[2]["span_days"] < 180, "short[2] must be 60-179d"),
        (medium[0]["n_writings"] >= 498, "medium[0] must be dense wall"),
        (100 <= medium[1]["n_writings"] <= 300, "medium[1] must be 100-300 writings"),
        (
            180 <= medium[2]["span_days"] < 365 and medium[2]["n_writings"] < 400,
            "medium[2] must be 180-365d and not a wall",
        ),
        (all(u["span_days"] >= 730 for u in long), "long users must be >=730d"),
        (any(u["user_id"] == "subject_rS5HgHd" for u in long), "keep sparse long user"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        raise SystemExit("Strata checks failed:\n  " + "\n  ".join(failed))

    print("Sample JSON matches duration labels and strata rules.")
    print("Ids:", ids)
    print("Readers:", sample.get("teammate_assignments"))


if __name__ == "__main__":
    main()
