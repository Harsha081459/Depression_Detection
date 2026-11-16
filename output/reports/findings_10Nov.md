# Findings report, 10 November 2026

Team 6, AIX Project Elective. eRisk 2026 Task 2, person-level separation of eRisk-positive and control users from their own writings. This report covers how a writing is scored, how those scores become user-level features, the word-rule baseline, the human pilot of five signals, the Hugging Face versus Ollama backend diagnosis, and the one prespecified evaluation of the frozen Ollama scores. It contains no post text and no user identifiers.

Labels below mean the dataset task labels. They are not a clinical diagnosis.

## 1. Data and locked rules

The cohort is 523 users: 91 eRisk-positive and 432 control, with 155,284 writings marked `target: true`. Disclosure writings are a mask, not a feature. There are 610 of them (571 in the positive group, 39 in the control group). Windows are measured from each user's first writing, before any disclosure is removed. Scores exist for the 12-week eligible set: inside 84 days of the first writing, not a disclosure, and not empty, `[removed]`, `[deleted]`, or `none`. That set is 62,183 writings from 522 users. One user has none.

The model is local Qwen2.5-7B-Instruct. The cohort scores come from Ollama `qwen2.5:7b-instruct`, digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`, Ollama 0.34.0, running one request at a time. Prompts are frozen version 1. The canonical prompt hash is `238ac1a34c03c868fcb093af92b30b7bbebae9f394e4d66e10bc27bf006a95f1`. That hash is the same in the prompt file, the preflight report, the pilot gate, and all 310,915 score rows (62,183 writings times five signals).

## 2. How a writing is scored

Scoring is one frozen question per signal, asked of one writing at a time. The five signals are self-blame, helplessness, isolation, family strain, and emptiness. Nothing is classified at this step. The model only answers Yes or No about that one sign in that one writing.

The call is Ollama `/api/chat` with two messages. The system message is the frozen instruction. The user message is:

```text
Post:
"""
<the writing, first 2000 characters>
"""

Question: Does the writer <yes-clause for this signal>?
Answer No if <no-clause for this signal>.
Answer Yes or No.
```

The yes-clause and no-clause are the frozen wording for that signal. A writing about a game loss is named in the self-blame no-clause, which is why the game probes exist. Options are temperature 0, one new token, context 2048, seed 42. The private one-slot server erases its prompt cache before every call, so one writing cannot leak into the next through the cache.

The score uses the first generated token only. Ollama returns the top 20 log probabilities for that token. Every token whose text, after stripping and lower-casing, is `yes` or `no` is kept. Variants such as `Yes`, ` yes`, and `YES` therefore count as the same side, and their log probabilities are combined with log-sum-exp. The score is

```text
score = exp(l_yes) / (exp(l_yes) + exp(l_no))
```

which is the share of the Yes-versus-No mass sitting on Yes. It is always between 0 and 1. If the top 20 contains only one of the two sides, the missing side is set to the lowest returned log probability minus one, and the row is marked `missing_flag`. If it contains neither, the call fails and is not stored. 32,535 writings hit the one-sided case. 1,151 were longer than 2,000 characters and were scored on the prefix, marked `truncated_flag`.

Each writing is asked five times, once per signal. That produced 310,915 rows. The same writing can be high on one signal and low on another. There is no single "depression score" at the writing level.

The BF16 scorer asks the same question and reads the same next-token probability, but it uses only the two exact tokens `Yes` and `No`. Section 6 shows those two implementations agree on the easy probes and disagree on the borderline ones. The cohort features below are built from the Ollama scores.

## 3. How scores become features

The classifier never sees the raw writings or the 62,183 scores. It sees one row per user per window.

A window is the writings from a user's first post up through 4, 8, or 12 weeks. The clock is the timestamp of the first writing, taken before disclosure writings are removed. A disclosure writing does not enter the features. Writings after the window do not enter that window's row. A user with nothing left in the window still gets a row, filled with zeros, and `has_posts = 0`. Two training users are in that state at 4 weeks, and one at 12 weeks.

For each of the five signals, the writings inside the window produce six candidate summaries. All of them are computed without looking at the label:

| Summary | Definition |
|---|---|
| `mean` | Average score in the window |
| `max` | Highest score in the window |
| `top3` | Average of the three highest scores, or of all of them if there are fewer than three |
| `fraction_0.25` | Share of writings scored above 0.25 |
| `fraction_0.5` | Share of writings scored above 0.50 |
| `fraction_0.75` | Share of writings scored above 0.75 |

That is 5 signals times 6 summaries = 30 LLM candidates. The same six summaries are also computed for the word-list version of each signal (0/1 per writing: the word list hit or it did not). Family strain on the word list requires a family word and a negative word. A writing that is only about a game, or that has no first-person wording, is set to 0 on the word-list signs.

Four style features are computed from the same writings, again without a label:

- **Polarity** of a writing is (positive words − negative words) / (positive + negative + 1). **Polarity flip** is the share of successive writings whose polarity changes sign. **Polarity jump** is the average absolute change in polarity, divided by 2.
- **Distress** is the share of tokens that are on the negative list. **Distress burst** is how far the three most distressed writings sit above the window median, floored at 0.
- **Absolutist** is the share of writings that contain an absolutist word and a negative word.

Two controls are attached to every row: `n_posts`, the number of eligible writings in the window, and `has_posts`.

Which of the six summaries represents a signal is chosen inside training data only. For each signal, each of the six candidates is scored by 5-fold user AUROC on the training users, and the one with the highest mean is kept. The direction is fixed: a higher score is always the positive direction, and the sign is never flipped after seeing the test users. In this run the winner was `mean` for every signal at 4, 8, and 12 weeks. The fraction and max candidates were available and lost on the training folds.

The model then receives a small set, chosen again inside each training fold:

| Feature set | What goes in |
|---|---|
| Count | `n_posts` only |
| Code | `n_posts`, `has_posts`, and the four style features |
| LLM | `n_posts`, `has_posts`, and the one selected summary for each of the five signals |
| Both | The code set plus the five selected LLM summaries |

Scaling and the choice of logistic `C`, or of the boosting depth, learning rate, and tree count, are fit inside that same training fold. The test users, and the outer-fold users in the 5×5 cross-validation, are scored with a model that has already made those choices.

So the chain is: writing, then five Yes/No probabilities, then six summaries per signal inside a time window, then one summary per signal chosen on training users, then a logistic or boosting model on those five numbers plus the style features. Sections 9 to 11 report that last step.

## 4. Word-rule baseline

Before any LLM scores, the same five signs plus writing-style features were built from word lists. Nested XGBoost, with feature selection inside each training fold, on the masked arm (disclosure text removed, timestamps kept):

| Window | AUROC | Average precision |
|---|---:|---:|
| 4 weeks | 0.786 | 0.552 |
| 8 weeks | 0.791 | 0.603 |
| 12 weeks | 0.839 | 0.623 |
| Full history | 0.867 | 0.575 |

The before-disclosure arm is lower: 0.731, 0.740, 0.778, and 0.784. The file that records this, `results/final_indirect/metrics.json`, says the fixed-feature screen used the full cohort, so that earlier cross-validation is optimistic. These numbers are the baseline the LLM features have to be read against. They are not a clinical result.

## 5. Human pilot of the five signals

Two raters labelled an enriched sample, not a random population sample. Version 1 prompts, held-out half, 60 judgements per signal. Every signal passed the pre-registered rule: held-out AUROC at least 0.80, probes passed, and a higher AUROC than the word list.

| Signal | Kappa | Model AUROC | Word-list AUROC | Human Yes out of 60 |
|---|---:|---:|---:|---:|
| Self-blame | 0.963 | 0.945 | 0.773 | 21 |
| Helplessness | 0.955 | 0.939 | 0.675 | 14 |
| Isolation | 0.896 | 0.951 | 0.750 | 12 |
| Family strain | 0.914 | 0.938 | 0.491 | 6 |
| Emptiness | 0.964 | 0.888 | 0.837 | 21 |

Agreement was 0.97 to 0.98. One or two items per signal needed adjudication. Family strain had only six human-Yes items, so its AUROC is the least stable of the five even though the point estimate is high. Emptiness is the signal where the model beats the word list by the smallest margin (0.051). This pilot is evidence that the Ollama scores track the raters on the sampled posts. It is not an estimate of how common the signs are in the 523 users.

The pilot gate file admits all five signals for the Ollama backend only.

## 6. Why the BF16 Hugging Face backend was not used

A second scorer was built on the full-precision BF16 weights of the same Qwen2.5-7B-Instruct revision (`a09a35458c702b33eeacc393d103063234e8bc28`), on the lab RTX 4060 Ti. Gate 1 required the ten synthetic probes to pass and, after that, a Spearman correlation of at least 0.95 against the Ollama pilot scores on each signal. The workflow stopped at the probes. Eight of ten passed. The two game-negative probes did not:

| Probe | What a pass requires | Ollama | BF16 |
|---|---|---:|---:|
| Game, self-blame | Low | 0.220 | 0.953 |
| Game, helplessness | Low | 0.050 | 0.905 |
| Numb, emptiness | High, and it passed both | 0.902 | 0.679 |

The other seven probes agree to many decimal places, at 0 or at 1. Repeated BF16 scores matched exactly. Batch size 1 was stable. Pilot Spearman, Gates 2 and 3, and BF16 cohort scoring were not run. Gate 1 has not passed for BF16.

The diagnosis used only the ten probes. It compared the two backends on the frozen version-1 wording.

| Check | Result |
|---|---|
| Prompt text | Character-identical on 10 of 10 probes. Sending the Hugging Face string to Ollama as a raw prompt returns the chat score. |
| Chat template | The prompt ends by opening the assistant turn. The game self-blame prompt is 186 tokens in both backends. |
| Token ids | `Yes` is 9454 and `No` is 2753. Both are single tokens. The BF16 scorer uses those ids. |
| Where the BF16 probability sits | All of it is on Yes or No (mass 1.000 on 10 of 10). Applying Ollama's variant rule to the BF16 distribution leaves the scores at 0.953 and 0.905. |
| Last-token path, one item | A manual slice of the output weights matches the plain logits on 10 of 10 probes, within 0.0001. |
| Ollama | Log probabilities are returned, Yes plus No has mass 1.000, and setting the repeat penalty to 1.0 changes nothing. |
| Left padding | The game self-blame score moves from 0.953 alone to 0.924 when the same text is left-padded in a batch of two. The gap is 0.028. |

So the two game failures are a difference between the weight files. The BF16 model and the 4-bit Ollama model answer the borderline probes differently. A 4-bit Hugging Face reload was not available (`bitsandbytes` is not installed, and no second Ollama quantisation was already pulled), so quantisation and the llama.cpp kernels were not separated. The decision does not depend on that split.

A second, separate finding: the 0.028 padding shift is already above Gate 2's limit of 0.01. Batched BF16 scoring would fail that gate even if the probes had passed. Ollama scores one request at a time, so the Ollama cache is not affected.

No prompt was edited. No threshold was loosened. The cohort evaluation below uses the Ollama scores, which are the scores the raters were compared with.

## 7. Which evaluation was run, and why

`meeting_screener.py train` does not read `split.json`. It assigns every one of the 523 users to a 5-fold split, so a user who is held out in one fold is training data in the other four. That would have shown the 105 test users to the fitter. It was not run.

There was no `split.json` on the lab. The prespecified split was created before any test metric was computed, with seed 42. Pilot-exposed users stay in training. The test users are a stratified 20 percent of the remaining users.

| | Users | eRisk-positive | Prevalence |
|---|---:|---:|---:|
| Train | 418 | 81 | 19.4% |
| Test | 105 | 10 | 9.5% |
| Pilot-exposed, all inside train | 243 | 65 |  |
| Everyone except the pilot | 280 | 26 |  |

The test set is small on the positive side because the pilot already used 65 of the 91 positive users. Prevalence also differs, so average precision on the test set is not comparable to average precision on the full cohort. This split cannot undo earlier full-cohort exploration. It is a locked secondary check, not a pristine prospective test.

Steps 3 to 7 then ran once: window features at 4, 8, and 12 weeks, signal summaries, the paired emptiness check, the classifiers, and the train-only sanity audit. The headline is a 5 by 5 repeated stratified nested user cross-validation. Aggregation and cut are chosen inside training folds only. The chosen summary was the mean for every signal at every window. Cuts of 0.25, 0.50, and 0.75 were available and were not selected. The evaluation directory is `private/cohort_v1/evaluation/`. It must not be rerun.

Software for that run: numpy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0, pyarrow 25.0.0. Bootstrap intervals use 2,000 class-stratified resamples and are conditional on the fitted model and this split. They do not include training uncertainty, and they are not adjusted for the number of comparisons.

## 8. Score quality on the 62,183 writings

Of those writings, 32,535 had only Yes or only No inside the top 20, so one side was filled in by the rule above. 1,151 writings were longer than 2,000 characters and were scored on the prefix. Both behaviours are the same rule the pilot used.

On the training users, the five scores move together. Spearman correlations are between 0.64 and 0.76. The highest is family strain with emptiness (0.76). The lowest is self-blame with isolation (0.64). A classifier that uses all five is therefore using overlapping information.

Train-only keyword audit, which is not ground truth: 2,267 training writings contain game, boss, quest, sport, or news. The share scored above 0.5 is 1.1% for self-blame, 2.9% for helplessness, 0.8% for isolation, 2.2% for family strain, and 0.6% for emptiness. On this keyword slice the cohort scores are mostly low, which is the direction the game probes were written to protect.

Truncation falls unevenly. Inside 12 weeks, 2.0% of control writings and 0.6% of positive writings exceed 2,000 characters. Median length is 76 characters for control writings and 131 for positive writings. The long tail, and so most of the truncated mass, is in the control group. Mean fraction of characters retained is 0.73 for control writings and 0.98 for positive writings.

Eligible writings per user at 12 weeks, training users: control mean 143 and median 48; positive mean 93 and median 46. The means differ because a minority of control users write a great deal. The medians are close. In the test set the positive users have little text: mean 20 and median 16.5 writings at 12 weeks, against control median 12. Two training users have no eligible writing in the first 4 weeks, and one has none in the first 12 weeks. Empty windows are stored as zeros with `has_posts = 0`. Missing scores were not imputed; a writing inside the scored set always had all five scores.

## 9. Headline: 5 by 5 cross-validation, all 523 users

Mean AUROC over 25 folds. The figure in parentheses is the fold standard deviation. Average precision is in the second table. Positive users: 91. Prevalence inside a fold is about 17%.

Logistic regression and gradient boosting, four feature sets. Code features are polarity jump, polarity flip, distress burst, and absolutist language, plus post-count controls. LLM features are the five selected signal means plus the same controls. Both uses all of them. Count uses the post count only.

| Weeks | Model | Code | LLM | Both | Count |
|---|---|---:|---:|---:|---:|
| 4 | Logistic | 0.840 (0.041) | 0.793 (0.064) | 0.839 (0.055) | 0.532 (0.064) |
| 4 | Boosting | 0.805 (0.055) | 0.770 (0.070) | 0.786 (0.064) | 0.578 (0.070) |
| 8 | Logistic | 0.833 (0.044) | 0.794 (0.060) | 0.837 (0.050) | 0.529 (0.076) |
| 8 | Boosting | 0.798 (0.060) | 0.736 (0.072) | 0.797 (0.063) | 0.509 (0.064) |
| 12 | Logistic | 0.839 (0.043) | 0.822 (0.055) | 0.841 (0.048) | 0.531 (0.082) |
| 12 | Boosting | 0.812 (0.054) | 0.793 (0.073) | 0.820 (0.067) | 0.515 (0.063) |

Average precision, same models:

| Weeks | Model | Code | LLM | Both | Count |
|---|---|---:|---:|---:|---:|
| 4 | Logistic | 0.605 | 0.527 | 0.621 | 0.208 |
| 8 | Logistic | 0.576 | 0.528 | 0.628 | 0.204 |
| 12 | Logistic | 0.596 | 0.557 | 0.649 | 0.211 |
| 12 | Boosting | 0.587 | 0.587 | 0.638 | 0.184 |

Single signals, mean over the same 25 folds:

| Signal | 4 weeks | 8 weeks | 12 weeks |
|---|---:|---:|---:|
| Self-blame | 0.745 (0.063) | 0.749 (0.067) | 0.774 (0.071) |
| Helplessness | 0.745 (0.076) | 0.758 (0.079) | 0.769 (0.079) |
| Family strain | 0.732 (0.058) | 0.745 (0.071) | 0.763 (0.069) |
| Emptiness | 0.730 (0.046) | 0.748 (0.054) | 0.758 (0.063) |
| Isolation | 0.678 (0.063) | 0.691 (0.073) | 0.694 (0.078) |

What these tables support:

- The post count alone carries almost no ranking signal on the full cohort. Logistic AUROC stays near 0.53.
- Each LLM signal ranks users above chance. Isolation is the weakest, near 0.69 at 12 weeks. The other four sit between 0.76 and 0.77 at 12 weeks. Fold noise is about 0.05 to 0.08, so the gaps between those four are small. The pipeline forbids picking a best signal from gaps of that size.
- The LLM logistic reaches 0.822 at 12 weeks. The code-feature logistic is 0.839, 0.833, and 0.839 at 4, 8, and 12 weeks. The combined logistic is 0.839, 0.837, and 0.841. On this cross-validation the writing-style features already reach the level of the combined model. The combined model does raise average precision: 0.649 at 12 weeks, against 0.596 for code features and 0.557 for the LLM signals.
- These 25 folds include the 243 pilot-exposed users. Section 10 is the prespecified check on that fact.

The 12-week LLM logistic (0.822) sits next to the earlier nested word-rule XGBoost at 12 weeks (0.839). They are different models, different feature sets, and different validation schemes. The useful statement is that the LLM signals, on their own, reach a similar range to the nested word-rule model inside 12 weeks, and the style features are in that same range.

## 10. The same cross-validation with pilot users removed

280 users remain, 26 of them eRisk-positive. Fold standard deviations roughly double, because each positive fold is now a handful of people.

| Weeks | Model | Code | LLM | Both |
|---|---|---:|---:|---:|
| 4 | Logistic | 0.719 (0.092) | 0.716 (0.127) | 0.662 (0.108) |
| 8 | Logistic | 0.701 (0.101) | 0.674 (0.121) | 0.658 (0.131) |
| 12 | Logistic | 0.683 (0.078) | 0.712 (0.121) | 0.652 (0.123) |
| 12 | Boosting | 0.636 (0.122) | 0.596 (0.133) | 0.638 (0.131) |

Single signals at 12 weeks fall to self-blame 0.632, helplessness 0.601, family strain 0.608, emptiness 0.588, and isolation 0.554.

The drop from the all-user logistic to this one is 0.11 to 0.19 AUROC depending on the feature set and window. For the 12-week combined logistic the gap is 0.189 (0.841 against 0.652). Part of that drop is the smaller positive count, which makes every fold noisy. Part of it is that the pilot users, who were chosen and read while the signals were being developed, are no longer in the average. Both parts belong in any number quoted from section 9.

On these 280 users the count-only logistic is higher than it was on the full cohort (0.638, 0.639, 0.643). With only 26 positive users, post count is no longer a null baseline. That is another reason the two populations should be quoted separately.

## 11. Secondary holdout, 105 users never used for fitting

Point estimate and 95 percent interval. Ten positive users, so several intervals run from the mid 0.4s to the high 0.8s. A point estimate here can move a long way if two users change.

Single-signal means, the aggregation chosen on the training users:

| Signal | 4 weeks | 8 weeks | 12 weeks |
|---|---|---|---|
| Self-blame | 0.580 (0.389–0.763) | 0.558 (0.392–0.727) | 0.622 (0.443–0.792) |
| Helplessness | 0.640 (0.447–0.818) | 0.659 (0.489–0.808) | 0.666 (0.488–0.823) |
| Isolation | 0.653 (0.468–0.818) | 0.651 (0.501–0.789) | 0.640 (0.489–0.779) |
| Family strain | 0.644 (0.466–0.791) | 0.661 (0.541–0.769) | 0.671 (0.544–0.789) |
| Emptiness | 0.566 (0.380–0.739) | 0.605 (0.441–0.757) | 0.625 (0.442–0.796) |

Every one of these intervals is wide. Family strain at 8 and 12 weeks is the only signal whose interval starts above 0.54. None of them support a precise AUROC claim.

Classifiers, AUROC:

| Weeks | Features | Logistic | Boosting |
|---|---|---|---|
| 4 | Code | 0.655 (0.444–0.851) | 0.678 (0.486–0.859) |
| 4 | LLM | 0.733 (0.581–0.864) | 0.687 (0.528–0.836) |
| 4 | Both | 0.631 (0.414–0.842) | 0.573 (0.349–0.790) |
| 8 | Code | 0.682 (0.471–0.872) | 0.633 (0.443–0.813) |
| 8 | LLM | 0.664 (0.484–0.825) | 0.484 (0.278–0.694) |
| 8 | Both | 0.691 (0.484–0.876) | 0.601 (0.391–0.792) |
| 12 | Code | 0.687 (0.467–0.885) | 0.683 (0.467–0.883) |
| 12 | LLM | 0.784 (0.639–0.899) | 0.622 (0.417–0.813) |
| 12 | Both | 0.735 (0.554–0.897) | 0.694 (0.473–0.883) |
| 12 | Count | 0.539 (0.363–0.715) | 0.526 (0.505–0.553) |

The highest holdout point estimate is the 12-week LLM logistic at 0.784, with interval 0.639 to 0.899. The 8-week LLM boosting estimate is 0.484, and its interval includes both poor and useful values. Holdout average precision for the 12-week LLM logistic is 0.294 (0.162–0.578). The test prevalence is 9.5 percent, so a useless ranker would sit near 0.10; the estimates are above that, and the intervals are still wide.

Selected hyperparameters on this split were recorded and not reused to pick a winner. The LLM logistic chose `C = 0.01` at every window. The code logistic chose `C = 10` at 4 weeks and `C = 1` at 8 and 12 weeks.

### Emptiness against its word rule

The rule was fixed beforehand. The LLM summary and the word-rule summary are chosen on training users, then compared on the same test users with a paired bootstrap. Keep only if the interval lies entirely above zero. Word-rule-better only if it lies entirely below zero. Otherwise exploratory. The decision is not fed back into the classifiers.

| Weeks | LLM AUROC | Word-rule AUROC | Difference | Interval | Decision |
|---|---:|---:|---:|---|---|
| 4 | 0.566 | 0.479 | +0.087 | −0.097 to +0.257 | Exploratory |
| 8 | 0.605 | 0.479 | +0.126 | −0.032 to +0.278 | Exploratory |
| 12 | 0.625 | 0.577 | +0.048 | −0.094 to +0.202 | Exploratory |

On the holdout, the emptiness LLM score is not shown to beat its word rule. This is the same signal that had the smallest gain in the pilot (0.051 AUROC on 60 enriched items).

## 12. What can be said, and what cannot

Can be said:

- On an enriched held-out sample, all five Ollama signals agree with two raters and outrank their word lists, with emptiness the closest call.
- The BF16 scorer sees the same prompt and scores it by the same Yes-versus-No ratio, and it still answers two borderline probes differently. That backend did not pass Gate 1. The cohort numbers in this report are Ollama numbers.
- On all 523 users, a logistic model on the style features reaches about 0.84 AUROC inside 4 to 12 weeks. A logistic model on the five LLM signals reaches 0.79 to 0.82. Combining them reaches 0.84 and improves average precision. Post count alone does not.
- Removing the pilot users lowers those AUROCs into roughly the high 0.6s and low 0.7s, with wide fold scatter.
- The locked 105-user holdout has only 10 positive users. Its intervals are too wide for a precise claim. The 12-week LLM logistic point estimate is 0.784.

Cannot be said:

- That a gate passed for the BF16 backend.
- That the 0.84 all-user figure is free of pilot exposure. Section 10 exists because it is not.
- That one signal is the best. The fold noise is as large as the gaps.
- That emptiness beats its word rule on unseen users. The holdout decision is exploratory.
- That any of these numbers is a clinical sensitivity or a screening recommendation.
- That the test set is untouched history. Full-cohort word-rule work happened before this split existed.

## 13. What was not done

- Gate 1 Spearman of BF16 against Ollama on the pilot posts.
- Gates 2 and 3 for BF16, and any BF16 cohort scoring.
- A second quantisation that would separate 4-bit weights from llama.cpp.
- `meeting_screener.py train`, on purpose.
- A second run of steps 3 to 7. The evaluation directory is closed.
- Any edit to the frozen prompts, the thresholds, the test labels, or the 105-user list.

## 14. Where the files are

| File | What it is |
|---|---|
| `private/cohort_v1/prompts.v1.json` | Frozen prompts, hash `238ac1a3…` |
| `private/cohort_v1/cohort_scores.sqlite` | 310,915 Ollama score rows |
| `private/cohort_v1/scores/part-00000000.parquet` | The same scores joined to user, post, and date |
| `private/cohort_v1/split.json` | The locked 418 / 105 split |
| `private/cohort_v1/evaluation/` | Steps 3 to 7, written once |
| `private/cohort_v1/diagnostics/02_verdict.md` | Backend diagnosis |
| `private/cohort_v1/private_sanity/` | Train-only histograms, correlations, and top-scoring posts |
| `results/llm_signals/pilot_heldout_v1.json` | The pilot table in section 5 |
| `results/final_indirect/metrics.json` | The word-rule table in section 4 |

The top-scoring-post files contain writing text and stay in `private/`. They were not read for this report.
