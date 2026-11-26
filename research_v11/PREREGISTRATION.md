# v11 candidate study: selection rule fixed before any candidate result (25 Nov 2026)

**Why:** the score itself limits F1, not the threshold. On dev repeat 0 the best threshold chosen in hindsight gives F1 0.70 (all) / 0.60 (NP), against 0.67 / 0.55 tuned. So better features are needed: stronger text representations and transformer-based, interpretable signals.

**Protocol (unchanged from research_v10):**
- 418 dev users; outer 5-fold × 5 repeats (seed 42 + 1000·rep).
- 3-fold inner CV chooses the chunk spec (6), C (7), τ and t_min.
- prev_adj policy, latency-F1 objective, decision after every writing.
- PCA for the embedding blocks is fitted on the outer-training users' writings only.
- The 105 held-out users are not read.

Harness check: base B repeat 0 reproduces the published run (F1 0.705 all / 0.688 NP).

**Candidates (fixed list):**

| id | blocks |
|---|---|
| B | research_v10 base (reference) |
| B+EMO | + 28 go_emotions probabilities (RoBERTa) |
| B+BDI | + 21 symptom-statement similarities (bge-large) |
| B+EMO+BDI | both interpretable transformer blocks |
| TXT32 | bge-large embedding, in-fold PCA 32 |
| MPN32 | all-mpnet-base-v2, in-fold PCA 32 (earlier encoder, re-run here) |
| B+EMO+BDI+TXT32 | everything in one model |
| ens:B\|TXT32 | mean logit of base and text-only |
| ens:B+EMO+BDI\|TXT32 | mean logit of interpretable model and text-only |

**Amendment (25 Nov, written before these runs):** the GPU download for bge-large is stalled, so the same two combinations are also run with mpnet: `B+MPN32` and `ens:B|MPN32`. Base B and MPN32 results were already known when this was added. They enter the same selection rule.

**Exploratory additions (25 Nov, after the results above and after seeing single-signal dev AUROCs):** `BDI` and `EMO+BDI` are run without the base block. They are **not** eligible for selection. They are reported as exploratory only.

**Selection score:** mean over the 5 repeats of (F1_all + F1_NP) / 2.

**Guard:** NP AUROC (repeat-averaged rank scores) may not fall more than 0.01 below B.

**Tie rule:** within 0.01 of the best score, take the more interpretable candidate (fewer embedding blocks).

**Reporting:**
- Every candidate is reported, not only the winner.
- Paired user-level bootstrap (2,000, class-stratified) of the winner against B for AUROC, AP and F1.

**Final model:**
- Refit on all 418 dev users with the same inner procedure (spec, C, τ, t_min from 3-fold inner CV on dev).
- Score the 105 held-out users once. The held-out set already scored the v10 model, so this is labelled a **second look**, not a fresh test.
