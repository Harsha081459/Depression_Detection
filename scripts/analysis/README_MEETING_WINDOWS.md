# November meeting experiment

The 2 November meeting asks whether the groups differ in posting frequency
(05:32–07:15), how to compare each feature with its previous window
(14:20–18:50), and how to represent categorical evidence in a screening flag
(19:00–19:26 and 20:46–26:00). The professor also asked for a compact feature
table and later sensitivity analysis (29:06–30:24). Multimedia is a possible
extension for next semester (01:42–01:58).

## Run

From the repository root, in the project Python environment:

```text
python scripts/analysis/meeting_windows.py
python scripts/analysis/meeting_screener.py train
python scripts/analysis/meeting_screener.py check
python scripts/analysis/meeting_screener.py replay --day 84 --width 4 --arm before_disclosure --policy stable
python -m unittest discover -s scripts/analysis -p "test_*.py" -q
```

The first script reproduces the aggregate comparison from the meeting.
The screener saves 20 fitted classifiers in the ignored private
`data/intake/meeting_screener.joblib`. Each combination of 4 or 28 day
windows and masked or before-disclosure handling gets five models. For each
model, three user folds fit the classifier, a fourth user fold sets the
cutoff, and the fifth supplies an independent replay. The calibration
cutoff is at least 0.5 and admits no more than 10% of calibration controls
by day 84. Ties at the cutoff are excluded. The `stable` policy also
requires a candidate flag to survive every available one-step change to
one feature at a time. These policies are research operating points.

The `check` command prints aggregate results, including how many first
flags a single feature change would reverse. It also compares 4-day and
28-day schedules on people assessed by both, separately for positive and
control labels. This paired comparison distinguishes differences in flags
from differences in who had enough writing to receive a decision. The `replay` command writes
one private JSON decision per original user under
`data/intake/meeting_decisions/`. The JSON contains each completed
window's status and score, and the first review flag with its features.
The script checks the saved evaluation digest of every user's first flag
day and whether they had a decision at every completed window, including
cutoffs between windows. An individual original
`subject_*.json` can be screened at any cutoff day from 0 to 84 with:

```text
python scripts/analysis/meeting_screener.py screen data/raw/erisk26/PATH_TO_ORIGINAL/subject_NNN.json --day 28 --width 4 --arm before_disclosure --policy stable
```

Use the actual file path. The screener reads the target writer's
submissions and comments, omits the surrounding conversation and ignores
all writings after the requested day. Each score is trained in a fold that
excludes that original user. The screening output is a flag for human
review and has no clinical interpretation.

The experiment reads the existing private target-text CSV and user labels at
the paths defined in `final_indirect_pipeline.py`. It checks the cohort and
disclosure counts with the pilot audit. The only output is
`results/meeting_windows/summary.json`, which contains aggregate numbers.
`lab_pilot_check.json` in the same folder is the separate 3 November lab
verification. It also contains development-only Yes/No confusion counts from
the locked pilot manifest. The cohort script makes no LLM calls.

When the original 523 private user JSON files are present, it also checks
the discussion structure raised at 01:59–04:15. Each target comment is
matched to its submission ID and the submission author's target flag.
27,229 comments are in the target writer's own threads and 83,121 are in
other writers' threads. All 110,350 match. Thread ownership is descriptive
only and does not identify the author of a comment's immediate parent.

## What a window contains

Each user's first target writing starts both fixed schedules. The first
window covers days [0, D], and later windows cover (D, 2D] and so on through
day 84. Choosing the width from a person's eventual history length would
reveal future information. Instead, run D = 4 and D = 28 separately for
everyone. To illustrate, in a 4-day replay, writings on days 0 and 4 form
the baseline. A writing on day 5 can first change a flag at day 8. A writing
on day 9 cannot change that decision.

The first window always has no flag. Subsequent windows need at least one
eligible writing to issue a new score. A zero writing count in the preceding
scheduled interval means zero **recorded** writings. Content features with
insufficient evidence carry an explicit unknown category. No event is
invented after the final writing. A short or sparsely sampled history can
therefore receive no decision.

The masked arm retains all observed timestamps and counts and removes
disclosure, empty and removed/deleted writings from content scoring. The
before-disclosure arm additionally stops before the first window end at or
after a matched disclosure. Both arms keep the original clock origin. A
positive user's first disclosure may remove every opportunity for a flag.

## Features

The existing five first-person lexical flags measure self-blame, helplessness,
isolation, family strain and emptiness. Negative absolutist wording uses
the same first-person and game-context checks. Per window, each of these six
features is absent, present in fewer than half of eligible writings, or
present in at least half. The half-share boundary is a fixed design choice.
For the Qwen pilot comparison only, relative Yes support at or above 0.5 is
Yes. No decimal LLM scores enter this classifier.

- **Distress burst.** Present when the mean of the three largest lexical
  distress densities exceeds the window median, with at least four eligible
  writings. Distress density is indirect negative-word count per token
  after the existing context checks.
- **Silence after distress.** A gap closes in the current window after an
  instant with personal distress. It must last at least one day and at
  least three times the median of four or more earlier positive-time gaps.
  Masked writings still close gaps. An unevaluable gap supplies no evidence.
- **Polarity swing.** Opposite nonzero signs at adjacent observed timestamps.
  At tied timestamps, sum eligible post polarities before comparing instants.
  A timestamp with no eligible content breaks the comparison.
- **Frequency.** Number of all target submissions and comments in the window.
  The count is comparable across adjacent windows of the same width.

For each of the ten features, compare the current level with the immediately
preceding level. Encode lower, unchanged, higher or unknown. Binary events
are unknown if no eligible comparison exists. Categorical levels and changes
are one-hot encoded. Only the frequency count is standardized using the
training rows. Rising or falling alone does not establish deterioration.

## Training and reading the results

The five stratified folds use sorted user IDs and seed 42. All windows of
a person remain on one side of a split. Each fold fits a logistic regression
on up to 84 days of historical training users, then replays held-out people
as writings arrive. Training weights give users equal total weight within
each class, and the classes equal total weight. All windows inherit only
their person's dataset label. The dataset has no ground truth for onset or
for a particular window.

At a reporting day, rank each person by their highest score up to that day.
Break no ties among people who never had a decision and rank them below
scored people. A score at or above 0.5 at any eligible close records a
flag for human review. This fixed research threshold is uncalibrated.
Shorter windows give more opportunities to cross it. Coverage by class,
eligible-only metrics, confusion counts and conditional first-flag days
accompany all-cohort AUROC and average precision. All-cohort average
precision has a prevalence baseline of 91/523. Eligible-only prevalence
varies by arm and day.

The matched comparators use frequency and its change alone, all current
levels alone, and all levels plus changes. For each masked schedule, ten
further models omit one feature and its change at a time and refit on the
same folds. These are exploratory comparisons on a previously studied cohort.
The earlier prefix models use a different decision schedule and scoring rule.

## Independent Qwen review

The checked lab account is `user5@172.16.121.45`, project
`/home/user5/erisk_llm_signals`. Both complete rater exports match the
300 development IDs. The 25 development disagreements have AI tie-breaks.
The separate version 1 release scored 300 held-out items on 287 posts.
Its review pages are `private/review/review_heldout_A.html` and
`private/review/review_heldout_B.html`. On 6 November, both independent
held-out answer exports were checked against all 300 locked IDs. They had
six disagreements. Per the user's explicit choice, reviewer A's answers
settled those six; the lab records this in
`results/heldout_adjudication_provenance.json`. The frozen evaluation was
run once to `results/pilot_heldout.json`. All five signals passed the
predefined AUROC threshold of 0.80, passed probes, and exceeded their
word-rule comparators. This is an enriched signal pilot, not an estimate
of population prevalence.

The local gate is `results/llm_signals/pilot_gate_v1.json`. On 6 November,
`prepare-qwen` produced 62,183 eligible writings under the private
`data/intake/meeting_qwen_input.jsonl`. A resumable five-signal lab scorer
run started with 310,915 requested calls and private cache
`private/cohort_scores.sqlite`. Its progress is in
`results/cohort_scoring.log` on the lab. The Qwen-backed classifier can be
trained only after this cache is complete and copied back.

The Qwen integration is implemented behind that independent check.
`meeting_screener.py prepare-qwen` accepts `--prompts`,
`--pilot-gate`, `--heldout-report` and `--phase1-report` after the
held-out evaluation. It checks all admitted signals and writes the private
cohort scorer input to `data/intake/meeting_qwen_input.jsonl`. The lab
`llm_signal_scorer.py score --purpose cohort` reads that input and scores
only the admitted signals. It requires the same `--pilot-gate` and
`--heldout-report`, in addition to the scorer's existing
`--prompts`, `--phase1-report`, `--input`, `--cache`,
`--output`, `--port` and `--workers` options. Copy the resulting
private SQLite cache back through the approved project workflow.
`meeting_screener.py train`
then accepts the cache and the same four validation files. It converts
the approved Qwen scores at or above 0.5 to Yes and uses the existing
word rules for unapproved semantic signs and the other features. It
checks the prompt hash, model digest, runtime version and text hash.
Missing, stale or conflicting cached answers stop training. For a cached
answer with an incomplete Yes/No alternative or truncated text, the bridge
uses the existing word rule for that signal and records fallback counts in
the saved model. The frozen prompt permits at most 2,000 characters, so this
fallback is necessary for long writings. The Qwen
classifier is saved separately as
`data/intake/meeting_screener_qwen.joblib`; `check --qwen` and
`screen` or `replay` with `--qwen-cache` use it.
