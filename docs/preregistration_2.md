# Preregistration 2: what the reader does with the evidence

Second set of confirmation tests, after `docs/preregistration.md`. It covers
three things measured on the exploration folds between 28 Oct and 3 Nov:
the intervention arms (`illusion-analysis`), deepening inside the ircot loop
and cutting the reader's context (`experience-memory`, docs/plan.md Stages
5b and 5c).

Status: written 3 Nov 2026, before any of the runs below. What I had seen of
the confirmation folds when writing it: the runs listed in the first
preregistration and their analyses, which use the same ids. None of the
interventions, the k=20 ircot run or the context cuts had been run on a
confirmation fold; the hotpot confirmation experience log
(`hybrid_ircot_hotpot_confirm_experience.jsonl`, backfilled on 4 Oct) had
been opened for its header only. Its outcomes are the
`hybrid_ircot_hotpot_confirm` run's, whose EM and retrieval_correct are in
the first preregistration's tables, so D3's k=10 baseline was visible; the
2k ranks and the policy's outcome were not.

## Scope

- Folds, reader, prompt, device, tests and definitions as in the first
  preregistration. Confirmation folds: 1000 ids per dataset.
- Runs (queue11 onward, `.local_notes/`):
  - `interventions_{hotpot,2wiki,musique}_confirm` (`src/eval/interventions.py`,
    distractor k from `configs/bm25.yaml`, `bm25_2wiki.yaml`,
    `bm25_musique.yaml`), summarized against `closed_book_*_confirm`.
  - `hybrid_ircot_k20_hotpot_confirm` (`configs/hybrid_ircot_k20.yaml`),
    paired against `hybrid_ircot_hotpot_confirm`.
  - `reader_ctx_hotpot_confirm` (`src/memory/reader_context.py`) on the k=20
    confirmation contexts, the chunk filter fit on the exploration fold's
    k=20 contexts (`--train-trace`), nothing fit on the confirmation fold.
  - D3 on the CPU: `scripts/inloop_ceiling.py --records
    results/memory/hybrid_ircot_hotpot_confirm_experience.jsonl
    --train-records results/memory/hybrid_ircot_hotpot_dev_experience.jsonl`.
- Analysis set = ids common to the arm and its comparison run. Arms defined
  on fewer questions (answer_only, bridge_only) are compared on the ids where
  both are defined, as on exploration.

## Hypotheses

Exploration numbers (n=500) motivate; they are not thresholds. Equivalence
rules use the three-way reading of the first preregistration: 95% CI inside
the margin is a pass; crossing it is equivalence not shown; wholly outside
is a difference. Both failure modes are failed replications.

Interventions:

- I1 (primary, equivalence), the bridge is not needed for EM. Gold vs
  answer_only, EM, hotpot and musique. Exploration -0.012 [-0.055, +0.030]
  and -0.013 [-0.054, +0.029]. Margin +-0.05, as before. Pass: CI inside on
  each of the two datasets, each its own test.
- I1w (descriptive), the same pair on 2wiki. Exploration -0.048 [-0.096,
  0.000]: I expect a drop of a few points and do not hold it to the margin.
- I2 (primary), the bridge alone adds nothing beyond closed-book.
  bridge_only minus closed-book, EM, on all three. Exploration +0.015
  [-0.020, +0.050], -0.082 [-0.122, -0.043], -0.004 [-0.019, +0.011]. Pass:
  upper CI bound below +0.05 on each dataset (one-sided non-superiority).
- I3 (primary), on 2wiki context without the answer lowers EM below
  closed-book. bridge_only vs closed-book and distractors vs closed-book,
  2wiki, McNemar. Exploration -0.082 (19 vs 53) and -0.080 (33 vs 73).
  Pass: lower, Holm-adjusted p < 0.05, CI excludes 0, each of the two.
- I4 (descriptive), distractors vs closed-book on hotpot and musique
  (exploration -0.026, p 0.35; -0.016, p 0.12): no direction fixed.
- Manipulation check, not a hypothesis: gold above closed-book on all three.
  If it fails on any dataset the intervention table for that dataset is not
  interpreted.

Deepening (Stage 5b):

- D1 (primary), deeper retrieval raises evidence coverage.
  `hybrid_ircot_hotpot_confirm` vs `hybrid_ircot_k20_hotpot_confirm`,
  retrieval_correct. Exploration +0.134 (70 vs 3). Pass: higher at k=20,
  Holm-adjusted p < 0.05, CI excludes 0.
- D2 (primary, equivalence), and EM does not follow. Same pair, EM.
  Exploration +0.016 [-0.010, +0.042]. Margin +-0.05. Pass: CI inside.
  Read together with D1: if D1 passes and D2 passes, deepening is a
  retrieval-level change only; if D2's CI lies above the margin, EM does
  follow and Stage 5b's conclusion is withdrawn.
- D3 (primary, non-superiority), a policy on the frozen features recovers at
  most half the matched-cost gap. Fit on the exploration log, applied once
  to the confirmation log, deepening 30% of hops; learned minus uniform
  k=13, all gold in context. On exploration the matched-cost oracle gap was
  0.908 - 0.816 = 0.092 and the in-sample learned policy +0.024 [+0.002,
  +0.046]. Pass: upper CI bound below +0.046. Failing it means the
  pre-retrieval features carry more about when to deepen than the
  exploration fold showed, and the "same shape as routing" reading goes.
- D4 (descriptive): the oracle, uniform k=13 and random rows of the same
  run; the musique confirmation log through the same script with no
  hypothesis (k=5 over 20-paragraph pools makes 2k half the pool).

Reader context (Stage 5c), on `reader_ctx_hotpot_confirm`:

- C0 (check, not a hypothesis): the full arm gives the k=20 run's answer on
  every question. If it does not, the arm comparisons are not interpreted
  until the difference is explained.
- C1 (primary), gold-only beats the full context. EM, all questions.
  Exploration +0.102 (79 vs 28). Pass: higher, Holm-adjusted p < 0.05, CI
  excludes 0. Also reported on the all-gold-in-context subset, unadjusted.
- C2 (primary), no cut short of the oracle beats the full context. rerank3,
  rerank5 and learned3 minus full, EM, all questions. Exploration -0.090,
  -0.062, -0.084. Pass for each arm: upper CI bound below 0 ("below the full
  context"). An upper bound in [0, +0.03) is reported as "not above", a
  weaker pass. An upper bound at or above +0.03 is a failed replication for
  that arm, and the gate reading in docs/plan.md is withdrawn for it.
- C3 (descriptive): random3; the split by whether a cut kept every gold
  chunk (selected subsets, not tested); kept-gold rates before generation.

## Multiplicity

- Holm at alpha 0.05 across the McNemar tests of I3 (two), D1 and C1: four
  tests.
- I1 (two), I2 (three), D2, D3 and C2 (three) are CI rules, held as their own
  family at the 95% level each; with ten of them I expect about one to fail
  by chance if every claim were exactly at its bound, so a single failure is
  reported as that and not explained away.
- Everything marked descriptive: no pass/fail.

## Failed replication

As in the first preregistration: a failed rule is reported where a pass would
be, exploration numbers are never substituted, no re-running with another
seed, prompt, k, cut size or id list. A crashed run is re-run from scratch
with the same config and listed below.

## Not claimed

- Everything in the first preregistration's list.
- The intervention arms build the reader's input from gold labels on
  purpose; they say what the reader does with given evidence, not what any
  retriever achieves.
- The context cuts are three sizes and two scorers. "No cut short of the
  oracle helps" is about these cuts, on hotpot, with this reader. A cut
  between 5 and the full context was not tried.
- The learned deepening policy and chunk filter are one logistic model each
  over frozen features. A failure to beat a baseline is about that model and
  those features, not about learning in general.

## Deviations

Anything that departs from the above, with date and reason.
