# Depression Detection — Early Risk Screening from Social-Media Writings

Person-level early screening on **eRisk Task 2**: separating eRisk-positive users from control
users using only their own writings and the timestamps of those writings.

> **This is not a diagnostic system.** The labels are dataset task labels, not clinical diagnoses.
> The output of the pipeline is a *flag for human review*, never a medical judgement.
> No user text and no user identifiers are committed to this repository.

Team 6 · AIX Project Elective · IIIT Bangalore · 2026–27

---

## 1. Problem

Given a user's history of posts and comments (text + timestamp), decide **as early as possible**
whether that user should be flagged for human follow-up.

Three research questions drive the work:

1. **Which signals carry the information** — hand-written word rules, LLM-judged semantic signals,
   posting behaviour, or some combination?
2. **Do LLM signals beat simple word rules** once both are measured under the same protocol?
3. **How early can a user be flagged** without an unacceptable rise in false alarms?

## 2. Data

| Item | Value |
|---|---|
| Users | 523 (91 eRisk-positive, 432 control) |
| Writings marked `target: true` | 155,284 |
| 12-week eligible writings (scored set) | 62,183 from 522 users |
| Disclosure writings (masked, never a feature) | 610 (571 positive, 39 control) |
| Locked split | 418 development / 105 held-out |

Windows are measured from each user's **first** writing, taken *before* disclosure writings are
removed, so the clock cannot leak the label. Chance AUPRC on this cohort is **0.174**.

The raw corpus is used under a data-use agreement and is **never** committed. `data/` holds only
placeholders; everything user-level lives locally under `private/` and is excluded by `.gitignore`.

## 3. Method

### 3.1 Scoring one writing

A local **Qwen2.5-7B-Instruct** (Ollama, `qwen2.5:7b-instruct`, temperature 0, seed 42,
one request at a time, prompt cache cleared between calls) is asked one frozen yes/no question
per signal, about one writing at a time. Five signals:

`self_blame` · `helplessness` · `isolation` · `family_strain` · `emptiness`

Nothing is classified at this step. The score is read from the **first generated token's**
log-probabilities:

```
score = exp(l_yes) / (exp(l_yes) + exp(l_no))
```

where every top-20 token that lower-cases to `yes` or `no` is folded into its side with
log-sum-exp. Rows where only one side appears are flagged (`missing_flag`); writings longer than
2,000 characters are scored on the prefix (`truncated_flag`). This produced **310,915 score rows**
(62,183 writings × 5 signals) under a single frozen prompt hash.

Each LLM signal has a matching **word-rule** version (0/1 per writing) so that the two can be
compared under exactly the same downstream protocol.

### 3.2 Code-computed signals

Computed by script, no model involved:

- **Polarity**, **polarity flip** (sign changes between successive writings), **polarity jump**
- **Distress** (share of negative-list tokens) and **distress burst** (top-3 distress above the
  window median)
- **Absolutist** language (absolutist word + negative word in the same writing)
- Posting behaviour: rate, median gap, night posting, **silence after distress**

### 3.3 From writing scores to user features

The classifier never sees raw text or per-writing scores — only one row per user per decision point.

- **v1 (window features):** for each signal, six candidate summaries (`mean`, `max`, `top3`,
  `fraction_0.25/0.5/0.75`) at 4, 8 and 12 weeks. The summary that represents a signal is chosen
  by 5-fold AUROC **inside training folds only**; direction is fixed in advance and never flipped
  after seeing test users.
- **v2–v6 (temporal features):** change *between* time chunks, ordinal level encodings instead of
  raw floats, post frequency and its change, silence after distress, mood swings.
- **v10 (full-history chunks):** each history is cut into chunks (time chunks of 7/14/28 days,
  post chunks of 10/25/50 writings — the spec is a hyperparameter chosen inside training folds).
  A decision row is emitted at the end of every non-empty chunk, using only writings at or before
  that moment. Per chunk, 27 base quantities are stored for the current chunk, the previous
  non-empty chunk, and the whole prefix; the lower/same/higher and ordinal-level encodings are
  derived **inside** the CV loop so the cut points come from training users only.

Known collection artefacts (comment share, writing length, elapsed time, writings seen) are
excluded on purpose; writings-seen survives only as metadata and as a count-only baseline.

### 3.4 Evaluation

User-disjoint repeated cross-validation (5×5), with feature-set and hyperparameter selection
nested inside the training folds. Reported: AUROC, AUPRC, sequential F1, latency-F1, ERDE₅/ERDE₅₀,
and median delay to a true-positive flag. Sensitivity analyses: drop-one-feature-group,
permutation, and an all-users vs. without-pilot-users split.

## 4. Results so far

### v1 — window features, all 523 users, 12 weeks (5×5 repeated user CV, logistic)

| Feature set | AUROC |
|---|---|
| Post count only | not separable |
| Code (style) features | **0.839** |
| LLM signal means | **0.822** |
| Both | **0.841** |

Removing the pilot-exposed users drops these into roughly the **high 0.6s–low 0.7s** with wide
fold scatter. v1's weakness: averaging a signal over a window throws away how it *changed*.

### v10 — full-history chunk features, non-pilot users

| Metric | Value (mean ± sd over 5 repeats) |
|---|---|
| AUROC, end of history (all users) | 0.901 ± 0.010 |
| AUROC, end of history (non-pilot) | 0.884 ± 0.010 |
| AUROC after 25 writings | 0.825 ± 0.037 |
| AUROC after 50 writings | 0.843 ± 0.029 |
| AUROC at day 28 (non-pilot) | 0.730 ± 0.029 |
| AUROC at day 84 (non-pilot) | 0.762 ± 0.031 |
| Sequential F1 (latency-F1 policy, non-pilot) | 0.549 ± 0.047 |
| ERDE₅ / ERDE₅₀ (non-pilot) | 0.091 / 0.043 |
| Median delay to true-positive flag | 13 writings |

Moving from window means to chunk-to-chunk change is what recovered the early-detection signal:
the paired comparison gives **+0.043 AUROC at full history** and **+0.079 at 50 writings** for the
non-pilot set, with bootstrap intervals excluding zero.

### LLM signals vs. word rules

Dropping the LLM signals and keeping only the word rules costs **0.110 AUROC** (non-pilot, full
history; 95% bootstrap CI `[0.038, 0.193]`). On an enriched held-out sample adjudicated by two
raters, all five LLM signals outrank their word lists, with `emptiness` the closest call.

### What cannot be claimed

- That the all-user 0.84 figure is free of pilot exposure — it is not.
- That any single signal is best; fold noise is as wide as the gaps between signals.
- That the 105-user holdout (only 10 positives) supports a precise estimate.
- That any number here is a clinical sensitivity or a screening recommendation.

## 5. Repository layout

```
scripts/analysis/        active pipeline: feature builders (v2–v6), runners, LLM scorer,
                         pilot/adjudication tooling, and the unit tests for all of it
scripts/analysis/set3_v6protocol/   locked v6 protocol: feature build, eval, early-flag, topic check
research_v10/            full-history chunk features, sequential evaluation, horizons,
                         robustness, ablation, student-model fidelity  ← current line of work
docs/handoff.md          decisions, paths, open threads
output/reports/          findings report (10 Nov)
results/                 aggregate metrics and figures only — no user-level data
Progress_Report_Team6_eRisk.pdf   latest written report
report_updated_1Nov.tex           paper draft
```

### Where to start reading

1. `output/reports/findings_10Nov.md` — the fullest account of the method and the evidence
2. `docs/handoff.md` — decisions, paths, and open threads
3. `research_v10/chunks.py` and `research_v10/seq_eval.py` — the current feature and evaluation code
4. `scripts/analysis/README_LLM_SIGNALS.md` — the LLM scorer, its frozen prompts and the human gate

## 6. Reproducing

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # fill in local paths; .env is never committed
```

LLM scoring additionally needs a local Ollama with `qwen2.5:7b-instruct` pulled:

```bash
bash scripts/analysis/start_isolated_ollama.sh      # one-slot server, prompt cache cleared
python scripts/analysis/llm_signal_scorer.py --help
```

Pipeline entry points:

```bash
python scripts/analysis/run_erisk_full_pipeline.py   # v1 window pipeline
python scripts/analysis/run_v2.py                    # temporal features
python -m research_v10.chunks                        # full-history chunks
python -m research_v10.seq_eval                      # sequential evaluation
bash research_v10/run_reps.sh                        # repeated runs → results/agg.json
```

Tests:

```bash
python -m unittest discover -s scripts/analysis -p 'test_*.py' -v
```

Every run writes a JSON record of its configuration alongside its metrics, so a reported number can
always be traced back to the spec that produced it.

## 7. Method discipline

- **Pre-registration.** Each round's rules are written and frozen as a JSON prespec before any
  metric from that round is computed.
- **Nothing selected on test users.** Summary choice, feature-set choice, encoding cut points and
  hyperparameters are all decided inside training folds.
- **Frozen prompts.** Prompt text is hashed; the same hash appears in the prompt file, the preflight
  report, the pilot gate and every stored score row.
- **Human gate.** LLM signals were validated against two independent raters on an enriched sample
  before being used as cohort features.
- **Artefacts excluded on purpose**, and said so in writing, rather than quietly dropped.

## 8. Privacy and data policy

No raw text, per-user CSV, score row, model prediction or annotation page is committed. Everything
of that kind is produced into `private/` or `data/` and excluded by `.gitignore`
(including the `*_PRIVATE.*` and `*.sqlite` patterns). What is committed: source code, protocols,
aggregate metrics, figures and share-safe documentation.

## 9. Status and next steps

v10 is the current line of work. Open threads:

- Finish the sensitivity analysis and the compact feature table (dataset · feature · model · output).
- Short-window early detection, below the 28-day mark.
- External-dataset evaluation on a second corpus.
- Longer term: extending the signal set to multimedia.

Intermediate research rounds (v7–v9), the per-round prespecs and the dated progress reports are
kept locally; this repository carries the active pipeline, the current v10 line of work, and the
reports that stand on their own.
