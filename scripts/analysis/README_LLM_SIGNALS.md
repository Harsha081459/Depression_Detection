# Local-model signal pipeline

The held-out human signal-validity pilot passed on 6 November 2026 for all
five semantic signals. A full-cohort Qwen scoring run is now active on the
lab; the Qwen-backed classifier waits for that private cache to complete.
See `README_MEETING_WINDOWS.md` for the current cohort workflow and gate.

## Implemented

- `llm_signal_scorer.py`: one post/question, first-token log probabilities,
  log-sum-exp over Yes/No variants, missing-alternative flags, SQLite
  resume, exact model-digest checks, and text-free progress/errors.
- `prompts.json`: five development questions. Version **0** is intentional;
  freeze version 1 after development annotation and before held-out evaluation.
- `prepare_llm_pilot.py`: exact cohort/disclosure audit, 12-week candidate
  pool, user-disjoint pilot halves, lexical/domain enrichment. This script
  reads data programmatically and prints counts only.
- `llm_pilot_review.py`: 60 items per signal per half, equal numbers above
  and below model score 0.5, at most three annotated posts per user globally,
  60 distinct posts per signal/half, and game/sport/news examples. A post
  may receive two different signal questions; the user cap counts distinct
  posts. Offline pages hide user
  identity, cohort label, model score, and split metadata beyond the page's
  development/held-out designation. Held-out pages require version 1.
- `evaluate_llm_signals.py`: two independent rater exports, Cohen's kappa,
  adjudication enforcement, model and word-list AUROC against agreed human
  labels. This is pilot evaluation, not cohort classification.
- `llm_signal_silence.py`: three graded, closed-gap columns and a separate
  open-silence diagnostic. Callers must retain masked disclosure timestamps
  with zero expression/tone, and supply the correct prefix/cut.
- `verify_llm_signal_cache.py`: privately repeats 100 questions and compares
  both repeats against the original cache; exports differences only.
- `run_llm_pilot.py`: lab-only, single-runner lock, crash recovery, limited
  extra questions about the same approved pool, and an automatic stop for
  human annotation. It never starts full-cohort scoring.

## Validated runtime (30 October 2026)

Working machine: `user5@172.16.121.45`, RTX 4060 Ti. Ollama 0.34.0,
`qwen2.5:7b-instruct`, digest
`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`.

The final synthetic checks passed 10/10 probes and two independent 100-call
runs matched exactly. Throughput was 15.899 calls/second over 200 short
synthetic calls with eight queued client workers. Actual inference is
serialized around cache erasure, with one server slot. All tested client
worker counts (1/4/8) gave identical scores. Real, longer posts can be slower.
Another 100 real pilot questions were repeated privately; both repeats and
the original cache matched exactly. These checks establish scorer operation,
not human validity or prediction accuracy. Reports are
`results/llm_signals/phase1_user5.json` and `private_repeatability.json`.

The tested private service uses port 12466, flash attention disabled,
`OLLAMA_NUM_PARALLEL=1`, request `num_batch=512`, and cache erasure before
every call. All network inference is
to `127.0.0.1` on the lab machine, with inherited HTTP proxies disabled.
The rejected prefix-cache reports are retained separately. An earlier
num_batch=8 run repeated consistently but differed from its original cache
by up to 0.01374; those results are excluded by their configuration hash.
There were two
synthetic self-blame wording revisions and zero human development rounds.

On `sem9@172.16.121.65`, the GPU is visible to `nvidia-smi`, but CUDA
discovery returns error 100 and Ollama falls back to CPU. This machine is
not eligible for cohort scoring until GPU loading and the same checks pass.
The temporary diagnostic service was stopped; existing services were kept.

Prompt-prefix caching can alter the numerical path. Ollama 0.34.2's
[runner implementation](https://github.com/ollama/ollama/blob/v0.34.2/llm/llama_server.go)
enables that cache in its request builder. Do not assume temperature zero
alone guarantees identical probabilities; use measured repeatability.
The isolated runtime uses the documented
[slot-erasure endpoint](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
and disables the RAM prompt cache. `start_isolated_ollama.sh` copies the
public executable into a private runtime bundle and wraps its runner to
enable slot actions. Shared binaries and services are not changed. Prefix
reuse is intentionally sacrificed to meet the repeatability gate.

## Data audit

The exact checks passed: 523 users (91 eRisk-positive, 432 control), 155,284
target writings, and 610 disclosure writings (571 positive, 39 control).
The 12-week prefixes contain 62,378 writings; two combined posts are empty
or removed/deleted placeholders. The initial approved pool contained 1,126
posts from 273 users. It could not fill all 600 annotation quotas under the
three-post-per-user cap. The user explicitly approved 637 more posts; the
expanded pool has 1,763 posts from 355 users and 2,313 initial questions.
Further questions may reuse those approved posts; no additional posts are
added automatically. A capacity-matching sampler enforces all quotas/caps.

No source text is in these reports. Source text, pilot manifests/pages,
annotation exports, and caches stay private. The local exports are under
`data/intake/llm_signals/` and `data/intake/llm_signals_expanded/` (gitignored); the lab copy is under
`~/erisk_llm_signals/private/`, with directory mode 700 and files mode 600.
The cache stores IDs, hashes, scores, flags, runtime version, and latency;
it stores neither post text nor generated response text.

## Continue on the lab machine

**31 October: both development rater exports have been located, validated,
and copied to `private/review/rater_A.csv` and `rater_B.csv`.** Each contains
300 complete answers matching the development items. There are 25
disagreements. `private/review/adjudication.html` is ready for joint human
resolution. Per-signal agreement counts and kappa are recorded in the
aggregate `results/pilot_development_agreement.json`.

A one-time worker is waiting for up to 120 minutes for all 25 agreed answers
in `private/review/adjudication.csv` (or the complete browser-exported
adjudication CSV in the usual download folders). It then invokes the
development evaluator and saves `results/pilot_development.json`. It stops
if prompts, the locked manifest, or either original rater file changes.
Worker state is in `results/development_completion_status.json`.
This worker makes no model calls and never releases held-out pages.

The pilot is ready: **600 judgments from 584 distinct posts and 243 users**.
All twenty score-stratum quotas contain thirty items; both cohort groups
and all three topic controls are represented. At most three distinct posts
come from a user, and at most two signals are assessed on one post.
There are 4,235 cached judgments under the validated configuration, all in
[0,1]; 11.57% use the specified missing-Yes/No-alternative fallback.
The model was unloaded after final checks to free the shared GPU.

The active working directory is `/home/user5/erisk_llm_signals`.
Read only `results/pilot_status.json` for status; do not send private files
to a coding assistant. The runner may request extra signals on the same
approved posts if a model-score stratum is too small. It adds no new posts.
If quotas remain short, it reports `pilot_sample_shortfall` instead of
silently reducing sample size or weakening the user cap.

Scoring recovery only, before human review: the following command is not
the next step while awaiting raters or adjudication.

```sh
cd ~/erisk_llm_signals
nohup python3 -u scripts/run_llm_pilot.py --port 12466 \
  --pool private/pilot_pool_expanded.jsonl --require-private-check \
  > results/pilot_runner.log 2>&1 < /dev/null &
```

When status is `awaiting_two_independent_human_raters`, two people open
`private/review/review_development_A.html` and
`private/review/review_development_B.html` on the lab machine. Each labels
300 items independently. Save answers locally as `rater_A.csv` and
`rater_B.csv` in `private/review/`. Discuss disagreements using the blind
item reference, and create `adjudication.csv` with only `item_id,answer`
for those disagreements. Answers are `Yes` or `No`.
Humans see the full posts. The model sees at most 2,000 characters; 49 of
the 600 judgments have that truncation flag. The word-list comparison uses
the full post, so the human pilot also measures this model-input limitation.
Only development pages have been released. Held-out users are disjoint,
and their pages stay withheld until the prompts are frozen.

To create an offline page containing just the disagreements:

```sh
python3 scripts/llm_pilot_review.py \
  --adjudication-manifest private/review/private_manifest.json \
  --rater-a private/review/rater_A.csv --rater-b private/review/rater_B.csv \
  --output private/review/adjudication.html
```

Open that page together, settle the answers, and save its export as
`private/review/adjudication.csv`. Then compute the development metrics:

```sh
python3 scripts/evaluate_llm_signals.py \
  --manifest private/review/private_manifest.json \
  --cache private/pilot_scores.sqlite \
  --rater-a private/review/rater_A.csv \
  --rater-b private/review/rater_B.csv \
  --adjudication private/review/adjudication.csv \
  --phase1-report results/phase1_cache_control.json \
  --split development --output results/pilot_development.json
```

Omit `--adjudication` if there are no disagreements. At most three human
development prompt rounds are allowed. If a prompt changes, rerun the
matching preflight and rescore the pilot; its hash invalidates old scores.
Do not read held-out answers during that process.

After development, freeze the questions as version 1. Run a matching
preflight and score those frozen questions on the same private pilot posts;
then release held-out pages with `llm_pilot_review.py --release-heldout`.
The first complete manifest locks both halves: subsequent prompt rounds
reuse those items and retain their original sampling strata. They never
resample held-out posts in response to changed model scores.
Evaluate held-out annotations once. The numeric gate is AUROC >= 0.80 plus
all probes passed, and researchers must also judge whether the model
improves on the word list. Do not invent human labels or infer them from
the user's eRisk label. Human labels are about the specific signal.

Only then implement/run Phases 3–6. In particular, do not use the
full-cohort signal screen to choose the headline classifier's features.

## Tests

```powershell
python -m unittest discover -s scripts/analysis -p 'test_*.py'
```

All 38 tests passed, including all nine existing indirect-pipeline tests.
The lab scorer, runner, review builder, evaluator, and silence module use
the Python standard library and are compatible with the lab's Python 3.6.
The local data-preparation adapter uses the existing project dependencies.
