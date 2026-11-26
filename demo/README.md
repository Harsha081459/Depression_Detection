# Screening demo

## v11 (26 Nov 2026): final model = mpnet sentence embeddings
- **Decision model:** `MPN32`. all-mpnet-base-v2 embedding per writing → PCA 32 (fitted on dev writings) → 10-writing chunks (current, previous, history) → logistic regression with C 0.003. It refers at the first writing with score ≥ 0.942 after ≥ 5 writings. Chosen by the rule in `research_v11/PREREGISTRATION.md`.
- **Dev (418 users, nested CV 5×5):** F1 0.800 all / 0.740 NP, against 0.699 / 0.637 for the v10 signals model.
- **All 523 users:** F1 0.771 / 0.710, against 0.684 / 0.587.
- **Held-out second look:** AUROC 0.857, F1 0.556 (v10: 0.938, 0.588). The cause is short-history controls; see the report.
- **Explanations:** exact per-writing shares of the log-odds, plus readable signals (Qwen 5, go_emotions 28, 21 symptom statements via bge-large), plus the v10 scorecard.
- **Files:**
  - `model_v11.py`: scoring and exact attribution.
  - `build_cases_v11.py`: held-out case records.
  - `pick_walkthrough_v11.py`
  - `embed_live.py`: new writings. Runs in `.venv`: torch + transformers, offline, laptop GPU.
  - `tests/test_v11.py`
- **Research code:** `research_v11/` (`v11.py` harness, `compare.py`, `freeze_v11.py`, `runq.py`). Heavy runs went to the lab (`~/aixpe_v11`, a text-free bundle).

## v10 (18 Nov 2026): signals model, kept for comparison

A local Streamlit app that walks through one user step by step: Qwen signals → rule features → 14-day chunks → model score → flag → scorecard → reasons.
The output is **"refer for a human counselling conversation"** or **"no referral at this point"**. It is never a diagnosis.

## Run (in this order, from the repo root)
```bash
pip install -r demo/requirements.txt
python demo/freeze_model.py          # refits the held-out (lockbox) model, checks it reproduces, writes artifacts/frozen_model.json
python demo/calibrate_scorecard.py   # dev users only: percentiles, correlation merge, RAG rules, agreement report
python demo/build_cases.py           # 105 held-out users -> private/demo/cases/HO-xxx.json
python demo/pick_walkthrough.py      # fixed rule -> TP / FP / FN / TP-late cases + paraphrase template
pytest demo/tests -q
streamlit run demo/app.py --server.address localhost --client.toolbarMode minimal
```
Never deploy the app (no Streamlit Cloud, no ngrok). It reads private data.

## Frozen model
Fitted with the same procedure as the held-out run (`USERS=lockbox`, B encoding, decide after every writing, standard threshold policy):

| | |
|---|---|
| spec / C | d14 / 1e-4 |
| τ / t_min | 0.754 / 3 (prev_adj τ 0.801, shown as a second marker) |
| inputs | 277 |
| held-out (105 users, 10 positive) | AUROC 0.938, F1 0.588, TP 5 / FP 2 / FN 5. Reproduced exactly (max score diff 3e-16) |

## Scorecard (calibrated on the 418 dev users only)
- Window: history so far, at the decision writing (the flag point, or the last writing).
- Item → percentile among dev chunks → 0–10 points.
- **Red**: any item ≥ P97.5, or ≥ 2 items ≥ P90, or composite ≥ P90.
- **Amber**: the same rules at P90 / P75 / P75.
- **Correlation:** the five Qwen signals and the distressed share correlate at user-level ρ 0.75–0.90, so they are merged into one group. Mood swings and absolutist words are also merged (ρ ≥ 0.7). That leaves 5 groups.
- **Direction check:** low positivity was dropped (dev AUROC 0.40, i.e. the wrong direction).
- **Agreement with the model flag (κ, red vs flag):**
  - dev 0.46, dev non-pilot 0.48;
  - held-out 0.56 (second look, descriptive only).
- **Positive rate by colour on dev:** red 51%, green 5%.
- **Composite score at full history:** dev AUROC 0.91, against 0.90 for the model. Treat this as optimistic, because item directions and merges were chosen on dev.

## Explanations
The model's log-odds split exactly into one contribution per quantity: coef × standardised input, summed over that quantity's c_/p_/d_/level/up/down inputs. The app shows the top 5 for and the top 2 against, plus a template sentence. No LLM narration.

## Privacy
- The app never loads the `text` column. Held-out users appear only as `HO-xxx`; the ID map stays in `private/`.
- Paraphrases: team members run `python demo/show_key_writings.py TP` **in their own terminal** and write ≤ 25-word paraphrases (no names, places, dates or quotes) into `private/demo/paraphrases.json`, then re-run `build_cases.py`. A second member checks each one.
- Live input (upload / typed) is scored by the Ollama server on 127.0.0.1:11434 with the frozen v1 prompts. The cache keeps text hashes only. Use fictional writings (`demo/examples/`).

## Live Qwen on the lab GPU
Run `bash demo/lab_tunnel.sh` in a terminal, keep it open, and reload the app. This needs the `erisk` SSH host and the campus network or VPN.
- The tunnel forwards local port 11434 to the lab's isolated project runner (port 12466). That runner has the same model as the cohort run: Ollama 0.34.0, `qwen2.5:7b-instruct` digest `845dbda0…`. The app checks the digest and the prompt hash before scoring.
- Speed: about 50 calls in 6 s (10 writings).
- Checked on 18 Nov:
  - the rising example → refer at writing 4 (RED);
  - the ordinary example → no referral (AMBER).
- Agreement with the stored cohort scores on 30 random writings: 148/150 within 0.05, 99.3% same yes/no, 2 outliers (0.68 and 0.33).
- Reason for the outliers: the cohort run cleared Qwen's prompt cache before every call, and over the tunnel that isn't possible. Live values shift a little with the previous prompt.

## Files
| file | role |
|---|---|
| `common.py` | paths and loaders (no text) |
| `freeze_model.py` | refit + held-out reproduction check |
| `pipeline.py` | one user → per-writing, chunk rows, scores, first flag |
| `scorecard.py`, `calibrate_scorecard.py` | RAG scorecard |
| `explain.py` | exact LR contributions + template sentence |
| `build_cases.py`, `pick_walkthrough.py` | held-out case JSONs, walkthrough choice |
| `live.py` | new writings: rule features + NRC + Qwen via Ollama |
| `app.py` | Streamlit UI |
| `tests/test_reproduce.py` | dev rows = research tables; held-out scores and flags = lockbox; no text in exports |
