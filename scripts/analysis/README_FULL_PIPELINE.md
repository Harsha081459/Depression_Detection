# Frozen-prompt user-level pipeline

**Implementation complete, untested on real data.** Synthetic tests validate
selected software behaviours only, not the cohort pipeline or model validity.

## Ordered gates (9 November update)

Run `run_backend_pilot_gates.py` first; it has no cohort scoring path.
Gate 1 starts at batch 1: synthetic `check_backend`, followed by Spearman
comparison on the locked 300-question version-1 held-out pilot. All five
signals require rho >= 0.95, with undefined/constant scores treated as failures.
Do not mix version-0 development cache rows into this comparison.

Gate 2 compares 4 and 8 against batch 1 on 100 distinct pilot posts, with
all five questions. The prespecified rule is maximum absolute difference
<= 0.01 AND Spearman >= 0.99 separately per signal, against batch 1.
Only passing sizes can be selected; batch 1 is the fallback if larger sizes
fail or exhaust GPU memory. Gate 3 generates top-ten-per-signal and game
candidate pages privately. An actual recorded read is mandatory. Domain
tags are not ground truth for non-emotional content. No model adjudication
of held-out human labels is performed.

`assemble_gate_report` requires actual review attestations tied to the
page hashes, after gates 1 and 2 pass. The cohort runner requires this
`--gate-report`; it rejects synthetic-only passes, failed correlations,
unreviewed pages, and mismatched backend/batch configurations.

The worker stops after Gate 1, including on a pass, pending interpretation.
It verifies every checkpoint shard SHA256 against the pinned HF revision
before loading. Partial/missing shards fail verification. Gate 1 reports
exact tie rates, near-zero rates, unique scores and maximum absolute gaps.

The headline is prespecified 5x5 repeated stratified CV on all 523 users.
Each outer training partition contains the entire inner selection/tuning
procedure. Mean outer-fold AUROC and descriptive standard deviation are
reported; 25 overlapping folds are not 25 independent observations.
Repeat the full nested procedure excluding ALL pilot users as a sensitivity
analysis. Pilot users shaped the frozen questions; their inclusion in
all-user validation folds can be optimistic. A gap can reflect pilot tuning
or changed population/sample size; it is not proof of either explanation.
The fixed 105-user holdout is secondary and is not independent either.
Neither analysis erases prior
full-cohort exploration. Report every prespecified window/model and all
post-count-only controls; do not choose a winner from small test gaps.

Steps 3–7 run once. Existing evaluation output blocks a rerun. Following a
crash, a person may manually remove partial output ONLY if no test result
has ever been read. No automatic delete or reset is implemented. Once test
numbers have been read, do not rerun after changes. CV output receives the
same protection. Truncation-by-label/window audit is also written.

There are multiple comparisons across five signals, three windows,
aggregation candidates and models. CIs are not multiplicity-adjusted.
Small differences may be noise; do not name a best signal from small gaps.

Run on lab Linux, Python 3.10+, with local Hugging Face BF16 safetensors
for Qwen/Qwen2.5-7B-Instruct. The existing Ollama GGUF is not a HF checkpoint.
Install requirements_full_pipeline.txt in an isolated environment. Record
`python -m pip freeze` alongside the run. The scorer loads locally only.

Files: erisk_full_pipeline.py (all seven steps), run_erisk_full_pipeline.py
(ordered runner), test_erisk_full_pipeline.py (synthetic CPU checks).
Existing dependencies: final_indirect_pipeline.py, erisk_disclosure_audit.py,
llm_signal_scorer.py. Copy the new files alongside those on the lab.

Input parquet schemas:
- posts: user_id, post_id, date, text (all original writings before masking).
- labels: user_id, label (one row per user; 0/1 dataset task labels).
- pilot_users: user_id (all users exposed in either pilot half).

`export_inputs(text_csv, labels_csv, pilot_manifest, directory)` privately
converts the existing title/body CSV, labels CSV, and locked pilot manifest.
It refuses to overwrite an existing input snapshot. No dataset export or
cohort scoring has been run as part of implementing this code.

Run from the lab project root:

```sh
python3 scripts/run_erisk_full_pipeline.py \
  --posts private/cohort_input/posts.parquet \
  --labels private/cohort_input/labels.parquet \
  --pilot-users private/cohort_input/pilot_users.parquet \
  --prompts scripts/prompts.json \
  --model-dir /path/to/local/Qwen2.5-7B-Instruct \
  --phase1-report results/phase1_v1_heldout.json \
  --gate-report private/hf_backend_validation/all_gates.json \
  --work private/cohort_v1 --batch-size 1
```

Important interpretation and leakage controls:

1. The requested two-token BF16 probability differs from the validated
   Ollama scorer's token-variant/top-logprob method. Prompts are identical,
   but the scores need not be. The synthetic gate checks ten probe answers,
   repetition and batch consistency (0.001 tolerance). Absolute cross-backend
   probe differences are diagnostic, not a gate. The user-prescribed
   cross-backend gate is pilot Spearman >=0.95 separately for all five signals.
   Passing is an operational check, not a new independent human validation.
   If it fails, investigate/revalidate the new backend without changing the
   frozen prompts or retuning on held-out answers. Do not bypass the gate.
2. BF16 7B weights use most of a 16 GB GPU. Batch size 1 is the default;
   last-token two-row projection and disabled KV caching reduce memory.
   An OOM stops the run; completed atomic parquet shards are resumable.
   Changing batching/model/input creates a different cache configuration.
3. The 105 test users are stratified among non-pilot users; pilot users are
   forced into training. The resulting full-training and test prevalence
   can differ. Report it. This cannot erase earlier full-cohort exploration;
   genuinely prospective confirmation needs new users/data.
4. Original first-post timestamps define windows BEFORE disclosure masking.
   Only retained writings inside each window enter features. Empty windows
   are zero with has_posts=0; missing model scores are a hard error.
5. Fixed candidate cuts are 0.25, 0.5, 0.75. Each signal's aggregation/cut is
   chosen by five-fold training CV, inside each classifier CV training fold.
   Five selected LLM features plus count/availability controls are used for
   LLM-only; four existing code features plus controls for code-only.
   Both combines them. n_posts-only is fitted separately for both models.
6. Report ALL prespecified windows and models; never select a headline model
   using test performance. Candidate aggregation AUROCs are training-CV only;
   the tidy test table contains only training-selected aggregations.
7. Emptiness compares separately train-selected LLM and word-rule summaries
   on the SAME test users with PAIRED bootstrap indices. Positive difference
   CI: keep; crosses zero: exploratory; negative CI: word_rule_better.
   A negative CI excludes zero but is not evidence to keep the LLM. This
   test conclusion never feeds back into the reported classifier selection.
8. CIs use 2000 class-stratified user resamples with fixed seeds. They do not
   incorporate training/tuning uncertainty or adjust for multiple comparisons.
   AUPRC is average precision, and depends on test prevalence. Small AUROC
   differences with 523 users may be noise; test has only 105 users.
9. Raw-text top-ten lists stay in private_sanity parquet files, on TRAIN
   users only. Game/news keyword candidates are not human non-emotional
   ground truth. Review them privately; do not upload or print them. The
   synthetic game probes provide a separate known-negative check.

Outputs under the private work directory:
- scores/metadata.json and part-*.parquet: identity-checked atomic cache.
- split.json: immutable private split.
- backend_check.json: operational scorer comparison.
- private_sanity/: histogram data, Spearman correlation, private top posts,
  candidate domain checks, training post-count audit.
- evaluation/signal_summary.csv: signal, weeks, aggregation, estimate,
  ci_low, ci_high (AUROC; 15 rows).
- evaluation/train_candidates.csv: every candidate's CV AUROC.
- evaluation/emptiness_check.csv: paired differences and decisions.
- evaluation/classifiers.csv: ablation/model/window/metric and CIs.
- evaluation/empty_windows.csv and post_count_audit.csv.
- evaluation/run.json: seed, software versions, source hashes, limitations.

The runner refuses to overwrite an existing test evaluation directory.
Do not repeatedly evaluate against the test set after making choices.
