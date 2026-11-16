"""
Stage A4: first-person depression self-disclosure audit (eRisk Task 2).

How many positives / controls ever write a first-person disclosure, and how early
it appears in their timeline. Aggregates go to results/erisk_understand/disclosure_summary.json;
the per-user table is written next to it and stays local (results/**/*.csv is gitignored).
"""

import json
import re
from pathlib import Path

import pandas as pd

MOD = r"(?:severe|clinical|chronic|major|mild|moderate|own|untreated|diagnosed|long[- ]term|recurring|bad)\s+"
MEDS = r"(?:antidepressants?|zoloft|prozac|lexapro|sertraline|fluoxetine|escitalopram|citalopram|paxil|paroxetine|wellbutrin|bupropion|venlafaxine|effexor|cymbalta|duloxetine|mirtazapine|remeron|trintellix)"

STRICT = re.compile("|".join([
    r"\bi(?:'?ve| have| had| was| am|'?m| got)?(?: been)?\s+(?:\w+\s+){0,2}?diagnosed\s+with\s+(?:\w+\s+){0,3}?(?:depression|mdd|major depressive|depressive disorder)",
    r"\bdiagnosed me with\s+(?:\w+\s+){0,3}?(?:depression|mdd|major depressive)",
    rf"\bmy\s+(?:{MOD})?depression\b",
    rf"\bi(?: have|'?ve had| have had| had| suffer from| suffered from| struggle with| struggled with| deal with| dealt with| battle| battled| live with| lived with)\s+(?:{MOD}){{0,2}}depression\b",
    r"\bmy\s+(?:\w+\s+)?depressive\s+episodes?\b",
]))
BROAD_EXTRA = re.compile("|".join([
    r"\bi(?:'?m| am| was| feel| felt|'?ve been| have been| get| got)\s+(?:so |really |very |clinically |severely |extremely )?depressed\b",
    rf"\bi(?:'?m| am)\s+(?:\w+\s+)?on\s+(?:\w+\s+)?{MEDS}",
    rf"\bmy\s+{MEDS}",
]))


def repo_root():
    return Path(__file__).resolve().parent.parent.parent


def load(base):
    text = pd.read_csv(base / "data/processed/erisk/task2_target_text.csv", usecols=["user_id", "date", "title", "body"])
    labels = pd.read_csv(base / "results/erisk_understand/user_duration.csv", usecols=["user_id", "label"])
    text["date"] = pd.to_datetime(text["date"], utc=True)
    t = (text["title"].fillna("") + " " + text["body"].fillna("")).str.lower().str.replace("\u2019", "'", regex=False)
    text["strict"] = t.str.contains(STRICT)
    text["broad"] = text["strict"] | t.str.contains(BROAD_EXTRA)
    return text.merge(labels, on="user_id")


def per_user(df):
    df = df.sort_values(["user_id", "date"])
    df["idx"] = df.groupby("user_id").cumcount()
    first = df.groupby("user_id")["date"].transform("min")
    df["days_in"] = (df["date"] - first).dt.total_seconds() / 86400
    rows = []
    for uid, g in df.groupby("user_id"):
        row = {"user_id": uid, "label": int(g["label"].iloc[0]), "n_writings": len(g)}
        for kind in ("strict", "broad"):
            hits = g[g[kind]]
            row[f"{kind}_n"] = len(hits)
            row[f"{kind}_first_day"] = float(hits["days_in"].iloc[0]) if len(hits) else None
            row[f"{kind}_writings_before"] = int(hits["idx"].iloc[0]) if len(hits) else None
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(users):
    out = {"definition": {
        "strict": "first-person diagnosis / 'my depression' / 'I have|suffer from|struggle with depression'",
        "broad": "strict + 'I am/feel/was depressed' + first-person antidepressant use",
        "caveat": "regex audit; not manually validated; counts are lower bounds for implicit disclosure and include some colloquial use (broad)",
    }}
    for kind in ("strict", "broad"):
        k = {}
        for lab, name in ((1, "positives"), (0, "controls")):
            u = users[users["label"] == lab]
            d = u[u[f"{kind}_n"] > 0]
            k[name] = {
                "n_users": len(u),
                "n_disclosing": len(d),
                "pct_disclosing": round(100 * len(d) / len(u), 1),
                "median_disclosing_writings_per_user": float(d[f"{kind}_n"].median()) if len(d) else None,
            }
            if lab == 1 and len(d):
                days = d[f"{kind}_first_day"]
                k[name].update({
                    "median_days_to_first_disclosure": round(float(days.median()), 1),
                    "median_writings_before_first_disclosure": float(d[f"{kind}_writings_before"].median()),
                    "disclose_within_4_weeks": int((days <= 28).sum()),
                    "disclose_within_12_weeks": int((days <= 84).sum()),
                    "first_writing_is_disclosure": int((d[f"{kind}_writings_before"] == 0).sum()),
                })
        out[kind] = k
    return out


def main():
    base = repo_root()
    users = per_user(load(base))
    summary = summarise(users)
    out_dir = base / "results/erisk_understand"
    users.to_csv(out_dir / "disclosure_by_user.csv", index=False)
    (out_dir / "disclosure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
