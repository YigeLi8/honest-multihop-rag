# Preregistration: measurement study, confirmation folds

Working notes for `illusion-analysis`. What I will test on the confirmation
folds, how, and what counts as passing. Everything here is fixed from the
exploration fold (dev positions 0-499) only.

Status: written 1 Oct 2026, after the confirmation runs of 30 Sep had
finished and before any analysis of them. The order should have been the other
way round. What I had seen of the confirmation folds when writing this: the
one-line summary each run prints (EM, F1, the 2x2 counts). No partition, no
paired test and no per-question look at a confirmation fold had been done.
The hypotheses, margins and pass rules below come from the exploration tables
only. `hybrid_ircot_musique_confirm` was still running.

## Scope

- Folds (`data/folds.py`, ids in `results/folds/`): exploration = positions
  0-499 of each dev file. Confirmation = 1000 ids per dataset: hotpot and
  2wiki positions 500-1499; musique 1000 ids stratified by composition shape
  (seed 13) from positions 500 onward of the shuffled dev file.
- One reader: `mlx-community/Qwen2.5-7B-Instruct-4bit`, temperature 0,
  seed 13, one answer prompt for every config, one M5 Pro.
- Retrieval is over each question's own distractor pool, not a corpus: hotpot
  ~10 paragraphs (~42 sentences), 2wiki 10 paragraphs (~32 sentences), musique
  20 paragraphs (the chunk is the paragraph).
- Runs, each on the full fold of its dataset, n=1000:
  - hotpot: `closed_book_hotpot_confirm`, `bm25_hotpot_confirm` (k=10),
    `bm25_k13_hotpot_confirm`, `hybrid_ircot_hotpot_confirm` (bm25 +
    bge-small, alpha 0.5, k=10 per hop, max_hops 3),
    `hybrid_ircot_rerank_hotpot_confirm` (k=20, bge-reranker-base to 5).
  - 2wiki: `closed_book_2wiki_confirm`, `bm25_2wiki_confirm` (k=10),
    `bm25_k13_2wiki_confirm`.
  - musique: `closed_book_musique_confirm`, `bm25_musique_confirm` (k=5),
    `bm25_k7_musique_confirm`, `hybrid_musique_confirm` (k=5),
    `hybrid_ircot_musique_confirm` (k=5, max_hops 4).
- Analysis set per dataset = ids common to all its runs and the closed-book
  run. I expect 1000; anything less goes under Deviations.

## Definitions

- Answer correct: EM after the hotpot normalisation, best over aliases
  (primary). Secondary, in `src/eval/answer_metrics.py`: F1 >= 0.5, and capped
  containment (gold span contiguous in the prediction, at most four extra
  tokens; yes/no golds by EM only).
- `retrieval_correct`: every gold sentence (musique: every gold paragraph) is
  in the context handed to the reader, unioned over hops.
- The cell: answer correct and `retrieval_correct` = 0, as a share of all
  questions, Wilson 95% interval.
- Buckets, first match wins, order as in `src/eval/analyze.py`: parametric
  (closed-book EM also right), yesno, shortcut (a gold sentence containing the
  answer string is in context), paragraph (every gold paragraph reached, no
  answer-bearing gold sentence), partial (some gold), none. Residual =
  paragraph + partial + none.
- Alternative order, reported beside it: evidence-first, i.e. shortcut, yesno,
  parametric, then the same tail. Only parametric's position matters (shortcut
  never fires on a yes/no answer). Parametric becomes "closed-book right, span
  answer, answer sentence not in context": a lower bound on what memory must
  explain, where the primary order gives the upper bound. The residual is the
  same under both. On exploration bm25, parametric goes 19 -> 4 of 74 on
  hotpot, 36 -> 9 of 79 on 2wiki, 2 -> 0 of 32 on musique.
- Evidence-criterion ladder, the cell recomputed with each rung in place of
  `retrieval_correct`: (1) all gold sentences; (2) answer-bearing gold
  sentence in context, yes/no questions fall back to (1); (3) all gold
  paragraphs; (4) any gold sentence; (5) answer string in any retrieved
  sentence, yes/no fall back to (4). Rungs 2 and 3 are not nested. On musique
  1 and 3 coincide.
- `analyze.py` has only the primary order today. The alternative order and the
  ladder must implement exactly the above.

## Tests

Paired on identical ids. Binary outcomes: McNemar exact, two-sided
(`stats.mcnemar`). Differences: paired bootstrap, 10,000 resamples, seed 13,
95% percentile interval (`stats.paired_bootstrap`). Differences are b minus a.

## Hypotheses

Exploration numbers are n=500 under the unified answer prompt: bm25 k=10 as
committed; k=13, ircot and ircot + rerank from the re-runs on the same ids.
The committed `hotpot_decomposition.csv` and `hotpot_paired.csv` still carry
ircot rows from the old loop prompt (EM 0.432, flat against bm25 k=10). With
one prompt it is not flat (0.428 -> 0.486), so I do not preregister "EM is
flat between bm25 k=10 and ircot". The numbers motivate; they are not
thresholds.

- H1 (primary), cell shrinks. `bm25_hotpot_confirm` vs
  `hybrid_ircot_hotpot_confirm`, metric the cell. Exploration 0.148 -> 0.068,
  diff -0.080 [-0.110, -0.052], discordant 50 vs 10. Pass: lower under ircot,
  Holm-adjusted McNemar p < 0.05, bootstrap CI excludes 0.
- H2 (primary, equivalence), EM at matched context. `bm25_k13_hotpot_confirm`
  vs `hybrid_ircot_hotpot_confirm`, metric EM. Exploration 0.462 vs 0.486,
  +0.024 [-0.008, +0.056], p = 0.17. Margin +-0.05: exploration half-widths
  were 0.032-0.036, so about 0.025 at n=1000; the margin is twice that and
  under a third of the 17-point `retrieval_correct` gap beside it. Pass: 95%
  CI inside (-0.05, +0.05). CI crossing a margin = equivalence not shown; CI
  wholly outside = EM differs. Either is a failed replication, and I drop
  "EM unchanged at matched context". Inside the margin but excluding 0 is a
  pass reported as a small difference. This one is close: it passes only if
  the point estimate stays under about 0.027.
- H3a (primary), reranker loses evidence. `hybrid_ircot_hotpot_confirm` vs
  `hybrid_ircot_rerank_hotpot_confirm`, metric `retrieval_correct`.
  Exploration 0.772 -> 0.698, -0.074 [-0.112, -0.036], 68 vs 31. Pass: lower
  under rerank, Holm-adjusted p < 0.05, CI excludes 0.
- H3b (primary, equivalence), and EM does not move. Same pair, metric EM.
  Exploration 0.486 vs 0.488, +0.002 [-0.034, +0.038]. Same margin and
  three-way rule as H2.
- H4 (primary), matched-context control. `bm25_k13_hotpot_confirm` vs
  `hybrid_ircot_hotpot_confirm`, metric the cell. The ircot union context
  averages 13.1 sentences. Exploration 0.118 -> 0.068, -0.050 [-0.076,
  -0.024], 36 vs 11. Pass: as H1. If H1 passes and H4 fails, the reduction is
  context size, and I say that.
- H5 (descriptive), composition of the bm25 cell.
  - hotpot k=10, primary order: shortcut largest, parametric second
    (exploration 46 and 19 of 74). Pass: that ranking, and the bootstrap CI
    for (shortcut - parametric)/n excludes 0. Under the alternative order
    shortcut stays largest (53).
  - 2wiki: no ranking of the top two under the primary order (parametric 36,
    shortcut 33, 20 cases satisfy both). Pass: together >= 75% of the cell
    (69 of 79), and shortcut largest under the alternative order (53).
  - musique: shortcut largest under both orders (26 of 32), and the lowest
    parametric share of the three datasets (2 of 32; closed-book EM 0.028
    against 0.170 hotpot and 0.226 2wiki).
  - Residual <= 0.03 of questions in each of the three bm25 runs (exploration
    0.014, 0.018, 0.008).
- H6 (descriptive), not a string-matching artefact. The cell's share of
  accepted answers under F1 >= 0.5 and under containment, against the same
  share under EM, for the four hotpot retrieval runs and `bm25_2wiki_confirm`.
  Exploration: bm25 hotpot 0.346 / 0.362 / 0.340, ircot 0.140 / 0.141 / 0.135,
  2wiki 0.457 / 0.459 / 0.433. Largest gap 2.3 points (2wiki containment,
  just over the "two points" in plan.md). Pass: every share within +-0.04 of
  its EM value, and H1's direction holds under both criteria (unadjusted
  McNemar p < 0.05). A run outside the band is reported as
  criterion-sensitive. musique is the known exception (0.478 -> 0.580 under
  F1 >= 0.5): reported with aliases, not held to the band.

## Multiplicity

- Holm at alpha 0.05 across the three McNemar tests: H1, H3a, H4.
- H2 and H3b are CI-in-margin rules with no p-value. A 95% interval is two
  one-sided tests at 0.025, so the pair is held at 0.05 by Bonferroni, as its
  own family.
- Descriptive, unadjusted, no pass/fail beyond the rules above: H5, H6, and
  - EM bm25 k=10 vs ircot on hotpot (exploration +0.058 [+0.024, +0.092]);
  - k=10 vs k=13 on hotpot (EM +0.034, cell 0.148 -> 0.118);
  - the cell, ircot vs rerank (0.068 -> 0.090, p = 0.15 at n=500);
  - F1, the residual, the ladder, per-hop recall;
  - everything on 2wiki k=13, musique k=7 and the two musique hybrid runs.
    These have no exploration counterpart, so no direction is fixed.

## Failed replication

A primary hypothesis whose rule is not met is a failed replication. It goes in
the README confirmation table and the write-up as that, in the same place a
pass would. Exploration numbers are never substituted for confirmation
numbers. No re-running with another seed, prompt, k or id list, no dropping
questions, and EM stays the primary criterion. A crashed run is re-run from
scratch with the same config and listed under Deviations.

## Not claimed

- Nothing beyond distractor pools: retrieval over 10 or 20 given paragraphs
  says nothing about open-corpus retrieval.
- One reader model, 4-bit, greedy, one device. No claim about other models or
  sizes, and I cannot rule out that Qwen2.5 saw these datasets.
- Gold is the datasets' supporting facts mapped to chunks heuristically
  (unmapped facts on full dev: 0.006% hotpot, 0.23% 2wiki). The cell is an
  upper bound on unsupported answers.
- Buckets are measured conditions, not causes. Parametric means the
  closed-book run was also right; shortcut is a string match in a gold
  sentence.
- That right answers survive incomplete evidence is known (DiRe, KILT-EM,
  Sufficient Context). No new metric, no method, no ranking of retrievers.

## Deviations

Anything that departs from the above, with date and reason.

- 2026-10-01, order. The plan says criteria first, runs second. The runs
  were queued on 30 Sep before this file existed; see Status.
- 2026-10-01, re-run. `hybrid_ircot_musique_confirm` was killed on 30 Sep at
  239 of 1000 with nothing scored. Re-run from scratch on 1 Oct, same config,
  same ids.
- 2026-10-01, code. `analyze.py` has the primary bucket order only. The
  alternative order and the evidence ladder get reported once they are
  implemented as defined above; until then the confirmation tables carry the
  primary order.
