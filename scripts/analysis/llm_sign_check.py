"""
Stage D1: ask a small local LM to look for the agreed signs in ONE user's timeline.

The model is told what to look for (the sign list below) and must say which signs it
sees, with dates and paraphrased evidence. We then compare its call with the eRisk label.

Runs against a local Ollama server only (no cloud APIs: eRisk DUA). Output is written to
data/intake/llm_checks/ (gitignored) because it contains paraphrased evidence.

  python scripts/analysis/llm_sign_check.py --user subject_sOPw3Ku --dry-run
  python scripts/analysis/llm_sign_check.py --user subject_sOPw3Ku --model qwen2.5:7b-instruct
  python scripts/analysis/llm_sign_check.py --user subject_WuGhoOy --mask-disclosure
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from erisk_disclosure_audit import BROAD_EXTRA, STRICT

ROOT = Path(__file__).resolve().parents[2]

SIGNS = {
    "explicit_disclosure": "The writer states in the first person that they have or had depression, a depression diagnosis, or depression treatment.",
    "self_blame": "The writer blames themselves, calls themselves worthless, a failure or not good enough.",
    "social_withdrawal": "The writer describes isolation, having no one to talk to, avoiding people, or feeling alone even among others.",
    "family_strain": "The writer describes conflict, pressure or distance with family, partners or close friends as a source of distress.",
    "treatment_history": "The writer mentions their own therapy, medication or hospitalisation, including that it did not help.",
    "distress_episodes": "The writer describes crying, recurring bad periods, or looping negative thoughts.",
    "help_seeking": "The writer asks others whether they might be depressed or how to get help for their own feelings.",
    "hopelessness": "The writer expresses that the future is bleak or nothing will improve.",
    "self_harm_or_suicidal": "The writer describes their own self-harm, suicidal thoughts or wishes to die, including euphemisms.",
    "emptiness_numbness": "The writer describes feeling empty, numb, or losing interest or motivation.",
}

INSTRUCTIONS = """You are assisting a research project on early risk detection from social media timelines.
You will read the dated writings of ONE anonymous person, oldest first.

Look ONLY for the signs defined below. For each sign, decide whether it is present in this person's OWN
experience (not when they discuss other people or topics in general).

Rules:
- Evidence must cite the writing date and a SHORT paraphrase in your own words. Never quote the text.
- If a sign is absent, say so; do not guess.
- Most writings may be about ordinary topics; that is normal.
- This is not a diagnosis. Finally, say whether the timeline as a whole looks more like an at-risk person
  ("at_risk"), not at risk ("not_at_risk"), or you cannot tell ("unsure"), and which signs drove that.

Signs:
{signs}

Reply with JSON only, in this shape:
{{"signs": {{"<sign_name>": {{"present": true|false, "evidence": [{{"date": "YYYY-MM-DD", "paraphrase": "..."}}]}}}},
  "overall": "at_risk"|"not_at_risk"|"unsure",
  "driving_signs": ["<sign_name>", ...],
  "rationale": "two sentences"}}
"""


def load_user(user, max_writings, max_chars, mask_disclosure):
    text = pd.read_csv(ROOT / "data/processed/erisk/task2_target_text.csv", usecols=["user_id", "date", "kind", "title", "body"])
    g = text[text["user_id"] == user].copy()
    if g.empty:
        raise SystemExit(f"No writings for {user}")
    g["date"] = pd.to_datetime(g["date"], utc=True)
    g = g.sort_values("date")
    g["text"] = (g["title"].fillna("") + " " + g["body"].fillna("")).str.strip()
    masked = 0
    if mask_disclosure:
        low = g["text"].str.lower().str.replace("\u2019", "'", regex=False)
        hit = low.str.contains(STRICT) | low.str.contains(BROAD_EXTRA)
        masked, g = int(hit.sum()), g[~hit]
    g = g[g["text"] != ""].head(max_writings)
    lines = [f"[{d:%Y-%m-%d} {k}] {t[:max_chars]}" for d, k, t in zip(g["date"], g["kind"], g["text"])]
    return lines, masked


def label_of(user):
    lab = pd.read_csv(ROOT / "results/erisk_understand/user_duration.csv", usecols=["user_id", "label"])
    row = lab.loc[lab["user_id"] == user, "label"]
    return int(row.iloc[0]) if len(row) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--max-writings", type=int, default=60, help="first N writings (prefix), oldest first")
    ap.add_argument("--max-chars", type=int, default=600, help="truncate each writing")
    ap.add_argument("--mask-disclosure", action="store_true", help="drop writings with first-person disclosure")
    ap.add_argument("--dry-run", action="store_true", help="report prompt size only; send nothing")
    args = ap.parse_args()

    lines, masked = load_user(args.user, args.max_writings, args.max_chars, args.mask_disclosure)
    signs = "\n".join(f"- {k}: {v}" for k, v in SIGNS.items())
    system = INSTRUCTIONS.format(signs=signs)
    prompt = "Writings:\n" + "\n".join(lines)
    n_chars = len(system) + len(prompt)
    print(f"{args.user}: {len(lines)} writings in prompt, {masked} masked, ~{n_chars // 4:,} tokens")
    if args.dry_run:
        return

    r = requests.post(f"{args.host}/api/chat", timeout=900, json={
        "model": args.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "num_ctx": max(8192, n_chars // 3)},
    })
    r.raise_for_status()
    answer = json.loads(r.json()["message"]["content"])

    label = label_of(args.user)
    record = {"user_id": args.user, "model": args.model, "eRisk_label": label, "n_writings_in_prompt": len(lines),
              "masked_disclosure_writings": masked, "settings": vars(args), "answer": answer}
    out_dir = ROOT / "data/intake/llm_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_masked" if args.mask_disclosure else ""
    out = out_dir / f"{args.user}_{args.model.replace(':', '-').replace('/', '-')}{suffix}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    present = [k for k, v in answer.get("signs", {}).items() if v.get("present")]
    print(f"eRisk label: {label} | model says: {answer.get('overall')} | signs present: {present}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
