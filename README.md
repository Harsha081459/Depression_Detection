# Depression Detection: Early Risk Screening from Social-Media Writings

After every new post or comment, the system decides whether a person should be **referred for a human
counselling conversation**. It is a screening flag for a person to review, **never a diagnosis**.

Team 6, AIX Project Elective, IIIT Bangalore, 2026–27: **Doppalapudi Harsha Vardhan, Munjikesh N, Polepalli Venkata Sai Prajwal**
Supervisor: Dr. Anutosh Maitra · Task: [eRisk 2026 Task 2](https://erisk.irlab.org/) (early risk detection of depression)

> No user text, user identifiers or per-user outputs are in this repository. Labels are dataset task labels,
> not clinical diagnoses.

---

## Results at a glance

Each model is evaluated with **nested, user-disjoint cross-validation (5 folds × 5 repeats)**. It decides after every writing, and every setting is tuned inside training folds only.

| Model | F1 (all) | F1 (non-pilot) | AUROC (all) | Avg. precision |
|---|---|---|---|---|
| Length only (baseline) | 0.37 | 0.22 | 0.64 | 0.26 |
| v10 signals model (Qwen signals + word lists + timing) | 0.70 | 0.64 | 0.92 | 0.73 |
| **v11 final model (sentence embeddings, all-mpnet-base-v2)** | **0.80** | **0.74** | **0.94** | **0.82** |

*Development set: 418 users, 81 positive. "Non-pilot" means users never seen while the LLM prompts were designed (175 users, 16 positive).*

**Larger check, all 523 users (91 positive):** F1 0.77 against 0.68 for v10, a gain of **+0.09, 95% CI [0.04, 0.14]**.

**Robustness check:** the gain holds when only writings from before any disclosure are used, with all users cut to matched lengths (F1 0.66 vs 0.59).

**Held-out users (105, 10 positive), second look:**

| | AUROC | F1 |
|---|---|---|
| v11 | 0.86 | 0.56 |
| v10 | 0.94 | 0.59 |

v11 does *not* beat v10 here. It over-scores short-history controls (median 50 writings); see **Limitations**.

## How it works

```
writing ──► sentence embedding (all-mpnet-base-v2, frozen) ──► PCA 32 (fitted on training writings only)
        ──► 10-writing chunks: current / previous / history averages (96 inputs)
        ──► logistic regression ──► score after every writing
        ──► refer at the first writing with score ≥ 0.942 once ≥ 5 writings are seen
```

Readable signals run alongside, to explain each decision. They are:
- **Qwen2.5-7B** yes/no signals: self-blame, helplessness, isolation, family strain, emptiness. The score is P(Yes) read from the answer-token probabilities, with frozen prompts and 462,445 calls.
- **go_emotions** (RoBERTa): 28 emotions.
- **Similarity** to 21 plain-language symptom statements (bge-large), for example *"I feel hopeless about the future and expect things to only get worse"*.

**Explanations are exact.** The decision model is linear in averaged writing embeddings, so each score splits into **one share per writing**. The shares add up to the score, and the tests check this. For every referral the demo shows:
- the writings that pushed the score up;
- what each one reads like (symptoms, emotions, Qwen levels);
- the person's symptom profile as percentiles;
- a green/amber/red scorecard;
- what the model looks for overall.

**Why v11 beats v10:** the v10 score jumped from writing to writing. 22% of controls had a spike, against 4% for v11, and the "first writing above the threshold" rule turned those spikes into false alarms.

## Run the demo

```bash
pip install -r demo/requirements.txt
pip install torch transformers        # for scoring new writings (CPU works; a GPU is faster)
python -m streamlit run demo/app.py --server.address localhost
```

- **Without the private data** (a fresh clone): use **Upload JSON → fictional example**, or **Type writings**. The models download once (about 2 GB), and new writings are then scored and explained.
- **With the private eRisk data:** walkthrough cases and all 105 held-out users (pseudonymous IDs), step by step.
- **Qwen signals:** optional. They need a local Ollama with `qwen2.5:7b-instruct`, or the lab tunnel `demo/lab_tunnel.sh`. Without them the final score still works.
- **Another Python for the embedding step:** set `EMBED_PYTHON=/path/to/python` if torch and transformers live in another environment.

Do not deploy the app publicly: with the private data it reads user-level files.

**Tests:**

```bash
python -m pytest demo/tests -q
```

`test_public.py` runs anywhere. The other tests need the private data and skip without it.

## Repository layout

```
demo/                    Streamlit app, frozen models (artifacts/), scoring + exact explanations, tests
  app.py                 the demo
  model_v11.py           final model: scoring and per-writing attribution
  pipeline.py            v10 signals model, step by step
  scorecard.py           green / amber / red rules (dev percentiles)
  live.py, embed_live.py new writings: word features, Qwen (optional), transformer features
research_v11/            final study: pre-registered candidate comparison, freeze script
  PREREGISTRATION.md     candidates and selection rule, written before the runs
  v11.py, compare.py     nested-CV harness and comparison (results/ = aggregate numbers)
research_v10/            v10 full-history chunk features and sequential evaluation
scripts/analysis/        LLM scorer (frozen prompts), feature builders, earlier pipeline rounds, tests
results/                 aggregate metrics and figures from earlier rounds
docs/, output/reports/   decision log and the 10 Nov findings report
Progress_Report_Team6_eRisk.pdf   progress report (12 Nov)
```

## Reproducing the study

You need the eRisk 2026 Task 2 data (data-use agreement) placed under `private/`. You also need the per-writing features and transformer embeddings (`private/encoder/embed_v11.py`, GPU).

```bash
python research_v11/runq.py 3 "MPN32@0" "B@0"   # nested CV for one candidate and repeat
python research_v11/compare.py                  # all candidates + selection rule + bootstrap
python research_v11/freeze_v11.py MPN32         # fit on dev, score held-out once
```

Every run writes its configuration next to its metrics, so each reported number traces back to the spec that produced it.

## Method discipline

- **Pre-registered selection.** The candidate list and selection rule were fixed in writing before any candidate was run. Later additions are labelled exploratory and could not be selected.
- **Nothing tuned on test users.** Chunk size, regularisation, PCA, encoding cut points and thresholds are all fitted inside training folds.
- **Same protocol for every model.** The harness first reproduced the published v10 numbers exactly.
- **Frozen prompts and hashes** for the LLM signals.
- **Disclosure removed.** 610 writings that openly disclose a diagnosis are removed before any feature is built.
- **Collection artefacts never used as features:** writing length, comment share, elapsed time and number of writings.

## Limitations

- **Short histories.** v11 is overconfident with few writings. This explains its weaker held-out result. Scores before writing 5 are never used, but a stronger guard is untested.
- **Small samples.** The held-out set has only 10 positives and dev has only 16 non-pilot positives, so F1 moves by 0.05–0.1 per user.
- **Early detection is weak.** Positives write little in their first months.
- **Explanations describe the model.** They are not clinical reasoning.

## Data and privacy

The eRisk corpus is used under a data-use agreement and is never committed. Everything user-level stays under `private/`, which `.gitignore` excludes, including all `*_PRIVATE.*` files. Committed files are limited to:
- source code;
- protocols;
- frozen model coefficients;
- aggregate metrics and percentile tables;
- fictional examples.
