# Handoff, 10 November 2026

Team 6, IIIT Bangalore, AIX Project Elective. Faculty: the course faculty.
eRisk 2026 Task 2. The labels are dataset task labels. They are not a clinical diagnosis.
Say eRisk-positive and control.

This file is first on the Read first list in `README.md`.

## Current position

v1 is finished and closed. The Ollama scores exist. The prespecified classifier
has been run once. The professor’s 2 November meeting asked for a different
feature shape: time chunks, change between chunks, frequency, and ordinal
LLM outputs. v2 was specified and locked. The v2 code has not been written
and no v2 metric exists.

Do not rescore. Do not edit the frozen prompts. Do not open the closed
evaluation to choose anything.

## Decisions already made

1. Data is eRisk only. Writings with `target: true`. 523 users: 91 eRisk-positive, 432 control, 155,284 writings.
2. Disclosure text is a mask, not a feature. 610 disclosure writings (571 positive, 39 control). Windows start at each user’s first writing, before that mask.
3. The cohort scorer is Ollama `qwen2.5:7b-instruct`, digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`, Ollama 0.34.0, one request at a time. Prompt version 1. Canonical hash `238ac1a34c03c868fcb093af92b30b7bbebae9f394e4d66e10bc27bf006a95f1`.
4. A score is P(Yes) on the first answer token. `Yes` / ` yes` / `YES` are combined. If one side is missing from the top 20, it is filled with the lowest returned log probability minus one. The writing is cut at 2,000 characters. 32,535 of 62,183 writings were one-sided. 1,151 were truncated.
5. The BF16 Hugging Face scorer failed Gate 1. Same prompt, same Yes/No ratio, different weights. Game self-blame 0.953 against Ollama 0.220. Game helplessness 0.905 against 0.050. Left-padding moves the first of those by 0.028, which already fails Gate 2’s 0.01 limit. No prompt and no threshold was changed. BF16 cohort scoring was not run.
6. `meeting_screener.py train` does not read `split.json`. It trains on every user, so the 105 test users would be fitting data in four of five folds. It was not used.
7. Steps 3–7 of `erisk_full_pipeline.py` ran once on the Ollama scores. The split was frozen first, seed 42. All 243 pilot-exposed users are in training. Test is 105 users, 10 of them positive (prevalence 9.5% against 19.4% in training). The directory `private/cohort_v1/evaluation/` must not be rerun or deleted.
8. v2 is exploratory. Chunk rules, ordinal cuts, and model grids were locked in `diagnostics_v2/prespec_v2.json` at 2026-11-10T04:41:27Z, before any v2 metric. Headline ordinal is the share of writings in a chunk with P(Yes) > 0.5, cut at 0, 0.10, and 0.30 into levels 0–3. The max-P(Yes) variant is unit-tested only and is not fit to labels.
9. The professor asked for post frequency and its change. v2 therefore includes them, as an integer count plus an ordinal and a lower/same/higher change. v1’s count-only model stays the baseline. It was about 0.53 AUROC on all 523 users.
10. Night is 00:00–05:59 UTC. There is no parent-post id. `kind == comment` is the reply proxy and `kind == submission` is the own-post proxy.
11. The model output for the course is a flag for a person to review, not a statement that someone is depressed.
12. Nothing is committed unless someone asks. `private/` is gitignored and contains scores plus some writing text.

## What the numbers say

Human pilot, enriched held-out sample, 60 judgements per signal, not a population rate. Every signal cleared AUROC ≥ 0.80 and beat its word list.

| Signal | Kappa | Model AUROC | Word-list AUROC |
|---|---:|---:|---:|
| Self-blame | 0.963 | 0.945 | 0.773 |
| Helplessness | 0.955 | 0.939 | 0.675 |
| Isolation | 0.896 | 0.951 | 0.750 |
| Family strain | 0.914 | 0.938 | 0.491 |
| Emptiness | 0.964 | 0.888 | 0.837 |

Word-rule nested XGBoost, masked arm: 4 weeks 0.786, 8 weeks 0.791, 12 weeks 0.839, full history 0.867 AUROC.

v1 headline, 5×5 repeated user CV, logistic, all 523 users, 12 weeks: code features 0.839, LLM means 0.822, both 0.841. The five LLM scores correlate 0.64–0.76, and the mean was the selected summary every time, so change over time was thrown away. That is the gap the professor pointed at.

Same CV with the 243 pilot users removed (280 users, 26 positive): 12-week logistic falls to about 0.68 (code), 0.71 (LLM), and 0.65 (both).

Holdout point estimate, 12-week LLM logistic: 0.784, interval 0.639–0.899. Ten positive users. Emptiness against its word rule was exploratory at 4, 8, and 12 weeks. Do not quote the all-user 0.84 without the without-pilot table.

Posting, measured 10 November, not what was said from memory in the meeting:

| | Control | eRisk-positive |
|---|---:|---:|
| Median span | 282 days | 726 days |
| Median writings | 277 | 383 |
| Median writings per week | 5.82 | 2.93 |
| Median per week in the first 84 days | 3.80 | 3.50 |
| Median longest gap | 50 days | 137 days |
| Median share that are comments | 0.80 | 0.94 |
| Share with span under 9 days | 6.5% | 2.2% |

Positive users post less often and go quiet longer. They do not have fewer writings in total, because they are observed for longer. Inside 84 days the weekly rates are close. The “about 9 days” user exists and is uncommon.

## Where the work lives

Laptop workspace:

`C:\Users\itsme\OneDrive - iiit-b\Cursor_folder\sem-5\AIX PE`

Lab, the only machine verified on 9–10 November:

- `user5@172.16.121.45`, hostname `redhatacademy5`
- GPU: NVIDIA GeForce RTX 4060 Ti, 16 GB. About 15 GB was free when the diagnosis ran.
- Project: `/home/user5/erisk_llm_signals`
- On the lab, scripts sit flat in `scripts/`. On the laptop they sit in `scripts/analysis/`.
- Python with torch: `/home/user5/bio/bin/python` (torch 2.11.0+cu128, transformers 5.18.0). `~/miniconda3/bin/python` has no torch.
- HF weights: `/home/user5/erisk_llm_signals/models/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`
- Ollama 0.34.0, isolated, `127.0.0.1:12466`, started 30 October with `scripts/start_isolated_ollama.sh`. Do not use the shared Ollama on port 11434. Do not kill other users’ jobs (`papertrader.py`, the SIH uvicorn).
- Laptop SSH key for this account: `%USERPROFILE%\.erisk_lab_key`. Do not write the lab password into the repo.
- The laptop must be on the campus network. Off campus, port 22 times out.

Other lab hosts were listed by the team and were not rechecked this week. Use `user5` unless that GPU is busy. The list, if a move is required: 172.16.121.40 user22, .27 user2, .45 user5 and user23, .31 user6 and user24, .24 user7 and user25, .19 user8 and user26, .34 user10 and user27, .5 user19 and user28, .78 user19a and user29, .72 user20 and user21, .60 user30 and user31, .63 user32 and user33, .68 User35, .66 sem11, .38 sem10, .65 sem9, .36 sem8, .74 sem6, .18 sem5, .79 sem4, .80 sem3, .82 sem2, .84 sem1.

v2 does not need a GPU. The scores are cached. Run it on the laptop.

## Files to open

Read:

- `docs/handoff.md` (this file)
- `output/reports/findings_10Nov.md`
- `diagnostics_v2/prespec_v2.json`
- `diagnostics_v2/00_state.md`
- `diagnostics_v2/01_plan.md`
- `output/reports/pipeline_progress_9Nov.md`
- `scripts/analysis/erisk_full_pipeline.py` (`window_features`, `LocalScorer`, `evaluate_all`)
- `scripts/analysis/llm_signal_scorer.py` (`payload_for`, `parse_logprobs`)
- `scripts/analysis/final_indirect_pipeline.py` (`flags`, `disclosure`)
- `scripts/analysis/erisk_repeated_cv.py`

Use, do not rewrite:

- `private/cohort_v1/prompts.v1.json` — frozen prompts. The copy in `scripts/analysis/prompts.json` is still version 0. Do not overwrite it.
- `private/cohort_v1/cohort_scores.sqlite` — 310,915 rows
- `private/cohort_v1/scores/part-00000000.parquet` — same scores, joined, 62,183 rows, columns `user_id`, `post_id`, `date`, five signals
- `private/cohort_v1/eligible_posts.parquet` — those writings, including text. Do not print text.
- `private/cohort_v1/labels.parquet`, `origins.parquet`, `pilot_users.parquet`, `split.json`
- `private/cohort_v1/ollama_score_notes.json`
- `private/cohort_v1/phase1_v1_heldout.json`
- `results/llm_signals/pilot_heldout_v1.json`, `pilot_gate_v1.json`
- `results/final_indirect/metrics.json`
- `data/processed/erisk/task2_target_text.csv` — columns `user_id`, `author`, `date`, `hour`, `kind`, `title`, `body`

Closed. Do not rerun, delete, or mine for a new idea:

- `private/cohort_v1/evaluation/`
- `private/cohort_v1/private_sanity/` (train-user text)

Diagnosis, already done:

- `scripts/analysis/diagnose_backend.py`
- `scripts/analysis/test_diagnose_backend.py`
- `private/cohort_v1/diagnostics/02_verdict.md` and `lab_v1/`

Not written yet, and this is the job:

- `scripts/analysis/v2_temporal_features.py`
- `scripts/analysis/test_v2_temporal_features.py`
- `scripts/analysis/run_v2.py`
- `output/reports/v2_temporal_progress.md`
- aggregates under a new `private/cohort_v2/`. Do not put them in `private/cohort_v1/evaluation/`.

## What v2 must do

Follow `diagnostics_v2/prespec_v2.json`. Do not retune its cuts after seeing a metric.

- Chunks of 7, 14, and 28 days, plus every 10 and every 20 writings. Horizon 84 days. Empty day-chunks stay, so a user going quiet is visible.
- Each LLM signal becomes level 0–3 from the share of writings with P(Yes) > 0.5. Change against the previous chunk is lower, same, or higher. The first chunk’s change is same. Trend over three chunks is falling, flat, or rising.
- One-hot the ordinal and categorical columns. Drop the reference level (0, same, flat). Keep `n_posts` and max silence days as integers.
- Frequency ordinal: 0, 1–2, 3–7, 8 or more posts in the chunk, plus its change.
- Silence after distress: a gap that starts on a writing at or above the user’s 75th distress percentile, compared with 1.5 times the median earlier gap. Classes shorter, similar, longer. Also the longest gap in the chunk, in whole days.
- Mood: count of polarity sign flips in the chunk, binned 0, 1, 2–3, 4 or more, plus its change.
- Candidates that do exist: night share (UTC hours 0–5), reply share (`kind == comment`).
- Feature sets: v1 static prefix, v2 LLM, v2 code, v2 all, frequency only, post count only.
- Chunk length is chosen inside each outer training fold, on v2-all logistic with C = 1, at the day-28 decision. Post-count chunks use the complete chunk whose median end, on training users, is closest to day 28.
- Headline: repeated stratified 5×5, seed 42, all 523 users and again without the 243 pilot users. Logistic C in {0.01, 0.1, 1, 10}. Boosting only for v2-all and v1-static, at day 28 and day 84, grid `n_estimators` {50, 100}, `max_depth` {1, 2}, `learning_rate` {0.1}.
- Flag rate is fixed on each training fold at that fold’s positive prevalence. Report recall, precision, flags per 100 users, median days to first flag among positives who are flagged, and the share never flagged. Repeat 0 only for the time-to-flag, so each user is tested once.
- Paired bootstrap of v2-all minus v1-static at day 28, repeat 0, 2,000 resamples, seed 42.
- Sensitivity: drop-one group at the fixed 28-day spec, and permutation importance on a depth-1 boosting model inside repeat 0. A gap under about 0.02 is noise.
- A unit test must show that chunk k does not use a writing from a later chunk.
- The report is a table: feature name, type, how it is computed, window, LLM or rule. State that the output is a flag for human review. Count every variant. Say the run is exploratory.

## Open items

1. Write and test `v2_temporal_features.py`. The prespec is the spec. The plan is `diagnostics_v2/01_plan.md`.
2. Run `run_v2.py` on the laptop. It does not need the GPU. Save user-level rows only under `private/cohort_v2/`.
3. Write `output/reports/v2_temporal_progress.md` from the saved aggregates. Until that file cites a saved metric file, the v2 run is not done.
4. The professor asked for a short feature table on the Teams channel: dataset, feature, model, output. That note is not written.
5. Multimedia was deferred to a possible next semester. Do not start it.
6. `bitsandbytes` is not on the lab, and no second GGUF precision was pulled, so 4-bit weights and llama.cpp were not separated. That does not block v2.
7. Full `unittest discover` once failed while importing `test_meeting_screener` from outside `scripts/analysis`. Running that file directly passed 11 tests. Do not treat that as a v2 failure.

## Stop conditions

Stop and ask before:

- editing `prompts.v1.json` or rescoring with a new prompt
- reading or rerunning `private/cohort_v1/evaluation/`
- using the 105-user holdout to pick a chunk length, a cut, or a model
- changing a threshold after a v2 metric has been seen
- printing writing text, user ids, or the lab password
- committing or pushing
