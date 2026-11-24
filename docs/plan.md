# Plan

Working notes. Two branches:

- `illusion-analysis`: the measurement study (what the "answer right, gold
  evidence incomplete" cell is made of). Being finished, then frozen.
- `experience-memory`: forks from it. Uses the harness's gold per-hop labels
  to study experience memory for retrieval: when does a past retrieval
  success or failure actually apply to the question in front of me.

Items get ticked as they land; anything cut goes to FUTURE_WORK.md with a reason.

## Ground rules (both branches)

- No new-algorithm or SOTA language. Every claimed difference is a paired
  comparison on identical question ids (McNemar exact + paired bootstrap),
  Holm-corrected across the primary tests.
- Positions 0-499 of each dev file are the exploration fold. Claims are tested
  once on the confirmation folds in `results/folds/`, with the criteria in
  `docs/preregistration.md` written first.
- Nothing is tuned on a dev file. Anything learned (graph knobs, memories,
  boundaries) is built from train splits or from a held-out half.
- Gold-derived fields (`gold_chunk_ids`, `reasoning_path`, `evidence_triples`,
  musique `is_supporting`) are outcomes, never inputs: they do not enter a
  retriever, a graph or a memory feature. A leakage test greps for them.
- Generation only runs locally (Metal). Retrieval-only experiments, code,
  tests and notes can be done anywhere.

## Part A: measurement study (`illusion-analysis`), finish and freeze

- [x] closed-book control, per-run traces, gold recall / paragraph recall
- [x] partition of the illusion cell: parametric / yes-no / shortcut /
      paragraph / partial / none
- [x] paired statistics; cumulative and hop-aligned per-hop recall
- [x] one answer prompt for every config; loop stop reason logged
- [x] folds; `--ids` in the evals; musique shuffled at prep
- [x] runner on `stream_generate`: prompt tokens, TTFT, decode throughput
- [x] `docs/preregistration.md`: hypotheses, assignment order and one
      alternative order, correctness criteria, n, tests, pass/fail rules.
      Written on 1 Oct, after the confirmation runs and before their
      analysis; the file says so
- [x] confirmation-fold baselines: hotpot_confirm (closed-book, bm25 k=10,
      bm25 k=13 matched context, hybrid ircot, + rerank), 2wiki_confirm and
      musique_confirm (closed-book, bm25, matched context); musique hybrid
      single-hop and ircot max_hops 4. All in, n=1000 each. The preregistered
      rules are computed by `src/eval/prereg.py` into
      `results/confirm_prereg.csv` and all of them hold. Hotpot: the cell
      goes 0.131 -> 0.060 from bm25 k=10 to hybrid ircot (McNemar 88 vs 17,
      Holm p < 1e-11) and 0.106 -> 0.060 against the matched-context k=13
      (63 vs 17, p < 1e-6). The reranker cuts retrieval_correct 0.774 ->
      0.657 (178 vs 61) with EM inside the +-0.05 margin (+0.018, CI -0.005
      to 0.041). EM at matched context is inside the margin too, but it is
      not zero and the pass is thin: +0.025, CI 0.002 to 0.048. So I can say
      "within five points", not "unchanged".
      Not preregistered, and against what the old ircot prompt showed: EM is
      not flat. With one answer prompt, bm25 k=10 -> ircot is +0.051 EM
      (CI 0.027 to 0.075) on hotpot (+0.058 on the exploration fold) and
      +0.085 on musique, beside +0.27 and +0.38 retrieval_correct. With the
      reranker the cell is back at the k=13 level (0.106; 24 vs 70 against
      plain ircot), which the exploration fold only hinted at (p = 0.15).
      Musique (no hypotheses fixed): the cell is 0.078 bm25, 0.073 k=7,
      0.074 hybrid, 0.051 hybrid ircot (54 vs 27 against bm25, p = 0.004),
      and almost all of it is shortcut (67 of 78).
- [x] partition tables and bars per dataset on the confirmation folds, with
      the alternative assignment order and the evidence-criterion ladder (all
      gold sentences / answer-bearing gold / all gold paragraphs / any gold /
      answer string in context). Done: primary-order tables and bars
      (`results/*_confirm_decomposition.*`) and the alternative order for the
      bm25 cells (`bucket_alt`, counts in `confirm_prereg.csv`). bm25 cell,
      primary -> alternative: hotpot parametric 42 -> 7, shortcut 81 -> 105,
      yesno 2 -> 13 of 131; 2wiki parametric 56 -> 11, shortcut 57 -> 94 of
      133; musique parametric 9 -> 5, shortcut 67 -> 71 of 78. The primary
      parametric bucket includes closed-book-right yes/no questions (11 on
      hotpot, 8 on 2wiki), so it is not all memory. Musique has the lowest
      parametric share of the cell under the primary order only; under the
      alternative order hotpot's is lower (0.053 against 0.064). 4 Nov: the
      alternative order (alt_* columns, right-hand panel of every
      decomposition png) and the ladder (cell_* columns with Wilson
      intervals) for every run in the six decomposition tables, in
      `analyze.py`; the primary columns are unchanged. Confirmation folds,
      the cell under rungs 1 / 2 / 3 / 4 / 5 (all gold sentences /
      answer-bearing gold / all gold paragraphs / any gold / answer anywhere):
      hotpot bm25 0.131 / 0.027 / 0.032 / 0.002 / 0.004, hybrid ircot 0.060 /
      0.008 / 0.004 / 0 / 0.002; 2wiki bm25 0.133 / 0.050 / 0.086 / 0.007 /
      0.017; musique bm25 0.078 / 0.007 / 0.078 / 0.002 / 0.003. So the cell
      is mostly a property of rung 1: asking only for the answer-bearing gold
      sentence leaves 1-5% of questions, and an answer right with no gold in
      context at all is under 1% in every run. Under the alternative order
      parametric falls to 0-11 per run (of 1000), which is the lower bound on
      answers from memory.
- [ ] correctness beyond EM: F1 >= 0.5 and capped containment with aliases,
      reported beside EM in every 2x2. The criteria are in
      `src/eval/answer_metrics.py` (containment allows at most four tokens
      beyond the gold span: the ircot readers answer in sentences a third of
      the time, and uncapped containment would credit those) and
      `src/eval/criteria.py` reports them beside retrieval_correct for runs
      already scored (`results/<dataset>_criteria.csv`, exploration fold). On
      hotpot the cell moves in step with the answer rate: bm25 0.148 EM /
      0.162 contain / 0.214 F1>=0.5, hybrid ircot 0.068 / 0.072 / 0.092
      (re-run with the one answer prompt), and
      the cell's share of accepted answers stays within two points of its EM
      value on hotpot and 2wiki, so the cell is not a string-matching
      artefact. musique is the exception: F1>=0.5 adds 45 answers to bm25 and
      the share goes from 0.48 to 0.58. Confirmation folds
      (`results/*_confirm_criteria.csv`, musique with its aliases): every
      hotpot run and bm25 2wiki stay within the preregistered 0.04 band, the
      largest gap being 3.2 points (bm25 hotpot under F1>=0.5, above the 2.3
      seen on exploration); musique bm25 goes 0.565 -> 0.632 under F1>=0.5.
      Still open: the columns in run_qa's own 2x2.
- [x] stop audit from the ircot traces: hop at which each gold sentence first
      entered context, stop reason, correctness. `src/eval/stop_audit.py`
      over the committed traces, nothing re-run
      (`results/<dataset>_stop_audit.csv` per question,
      `_stop_audit_summary.csv` per run and stop reason). The stop is
      uninformed: on hotpot confirm (hybrid ircot) the gold is complete on
      0.74 of the questions where the model wrote done, 0.74 where it
      repeated a query and 0.83 where it ran out of hops, against 0.77
      overall, so the done marker says nothing about whether the gold is in
      context; 0.26 of the done stops are premature (gold still missing),
      and EM there is 0.25 against 0.57 where the gold was complete at the
      stop. Two thirds of the illusion cell is made by those premature
      stops: 41 of the 60 cell questions on confirm (25 of 34 on
      exploration; 62 of 106 with the reranker; 43 of 51 on musique, where
      0.49 of the repeated-query stops, two thirds of all stops, are
      premature). The other direction is as common: 0.22 of all questions
      hit max_hops with the gold complete since hop 0, and a max_hops
      question with complete gold spent 1.6 hops after it was complete.
      The hops the loop does take buy something: of the questions that
      went past hop 0, 0.20 (hotpot) and 0.45 (musique) gained a gold
      sentence later, and 0.17 / 0.31 were completed by it; EM when the
      gold was completed later is 0.46 against 0.54 when it was there at
      hop 0 (musique 0.29 / 0.34), so a late gold sentence is read a
      little worse than an early one. The answer-bearing gold sentence
      never arrives on 0.11 of the span questions (0.27 on musique). Not a
      test of anything, an audit; the number to carry into the write-up is
      that the cell is mostly a stopping failure under one prompt, which a
      sufficiency check at the stop could address and the stop marker does
      not.
- [ ] README rewrite: scope in paragraph one (distractor pools, one model, one
      device), confirmation-fold tables, limitations, reproduction steps
- [x] intervention arms on the exploration fold (gold-only, gold minus bridge,
      gold minus answer sentence, distractors only), if time.
      `src/eval/interventions.py`, positions 0-499, 7B 4-bit, the one answer
      prompt, paired against the closed-book run on the same ids, Holm over
      the six pairs within each dataset (`results/interventions_*_explore_*`).
      Exploratory, not preregistered. EM, hotpot / 2wiki / musique:
      gold 0.614 / 0.578 / 0.424 against closed-book 0.170 / 0.226 / 0.028.
      Gold minus the bridge (answer-bearing gold only), against gold on the
      same questions: -0.012 [-0.055, +0.030] Holm p 0.96 / -0.048 [-0.096,
      0.000] p 0.064 / -0.013 [-0.054, +0.029] p 1.0. So taking the bridge
      sentence away does not measurably lower EM on any of the three; on
      2wiki the drop is at the edge. Gold minus the answer sentence (bridge
      only) is at or below closed-book: 0.119 vs 0.104 (p 0.96) / 0.089 vs
      0.170 (-0.082, 19 vs 53, p 0.0002) / 0.021 vs 0.025 (p 1.0).
      Distractors only (bm25 top-k from paragraphs holding no gold): 0.143
      vs 0.169 (p 0.35) / 0.146 vs 0.226 (-0.080, 33 vs 73, p 0.0003) /
      0.012 vs 0.028 (p 0.12). Reading: the model answers from the
      answer-bearing sentence whether or not the bridge is there, which is
      what the shortcut bucket assumed from the decomposition and is now seen
      by intervention; and on 2wiki any context without the answer, bridge
      or distractor, suppresses answers it gives closed-book. Per bm25
      bucket the cells are small (shortcut 46 / 33 / 26) and are in the
      by_bucket files without tests. Even with only the gold evidence EM
      stays at 0.42-0.61, so the EM side of the 2x2 has a ceiling well below
      1 in this harness
- [x] intervention arms on the three confirmation folds, under
      `docs/preregistration_2.md` (I1-I4, outcomes there). I2 and I3 pass on
      all three; I1 passes on musique and fails on hotpot: gold minus the
      bridge -0.042 [-0.068, -0.017], so on this fold the bridge is worth
      about four EM points on hotpot and 2wiki, and the "bridge not needed"
      reading is kept for musique only. Distractors alone are below
      closed-book on all three (-0.032 / -0.070 / -0.017).

Cut from this branch: the selector arm, the model-size sweep, the concurrency
ladder, energy, the CUDA leg. The serving sweep stays as it is in the README,
relabelled: the two ends of its latency range (8-bit k=20 and 4-bit k=5,
rerank off) are re-run with the streaming runner in `results/serving_corner/`
and summarised by `scripts/serving_corner.py`. EM and retrieval_correct match
the old rows (0.53 / 0.85 and 0.46 / 0.45); the old tok/s column rebuilds to
4.4 and 9.5 against 4.7 and 10.1, i.e. it was answer tokens over prefill plus
decode and mostly prefill; per-step decode is 32.8 and 60.9 tok/s. What is
left for the README rewrite: the bullets still quote the old column, the
pareto x axis still says throughput, and the `tokens_per_s` that run_qa logs
counts n tokens over n-1 steps (43-45% high on five-token answers), which the
runner's field comment does not say. A small graph arm moves to Part B as
one of the retrieval strategies.

## Part B: experience memory (`experience-memory`)

The question: can a system learn from previous retrieval successes and
failures, and decide when those experiences should influence a new retrieval?
Not "bm25 failed before, avoid bm25", and not nearest-neighbour recall of past
failures. An experience generates a hypothesis; later experiences support or
contradict it; the thing to learn is where the hypothesis applies.

What is already published, so I do not claim it: online case-utility learning
on HotpotQA/2Wiki (Memento, CASCADE), per-query retriever routing with bandits
(MBA-RAG), failure-aware revisable graph memory (EvoGraph-Mem), boundary fields
and reliability lifecycles (MSCE, GSEM), static applicability text (BASM,
AutoGuide), and the classical machinery (version spaces, ripple-down rules,
drift-aware trees). Logging, similarity memory, failure memory and static
conditions are baselines here, not contributions.

What is left to test, stated narrowly: with gold per-hop hit/miss as the
supervision (no model judging itself), an explicit, interpretable applicability
model per retrieval lesson over retrieval-process features, maintained online
with logged narrow / expand / exception / split / retire steps, gives lower
false-application and repeated-failure rates at matched retrieval cost on
look-alike question pairs than: similarity memory, static conditions, a scalar
per-memory utility, one shared online router over the same features, and
always running both retrievers. If the shared router matches it, the memory
structure is not doing any work and I say so.

Outcomes are retrieval-level (all gold in top-k, paragraph recall, hop-aligned
hit), so every arm can be run on every question in seconds, the best arm per
question is known, and regret is exact. EM is secondary: it moves far less
than retrieval in this harness (hotpot confirmation fold, bm25 to ircot: +5
EM beside +27 retrieval_correct).

### Stage 1: experience logging (no behaviour change)

- [x] `memory:` block in `configs/base.yaml`, off by default
- [x] `src/memory/experience.py`: one record per (question, hop): strategy,
      retrieved ids/scores/ranks, gold hit/miss, hop / cumulative / aligned
      recall, the rank of every gold chunk in the full pool, features, final
      outcome row; shadow arms scored the same way; run header with git sha,
      config, data file hash and schema versions
- [x] `src/memory/features.py`: model-free, gold-free features, schema
      versioned and frozen (19 of them, each tagged by what it costs: query /
      pool / history / primary hits). Arm disagreement (top-k overlap with
      another retriever) is kept apart as a probe feature, because knowing it
      already costs the second retrieval
- [x] `src/memory/backfill.py`: rebuild the same records from committed traces
      by re-indexing each pool; the recomputed top-k must equal the trace
      (bm25_hotpot_dev: 500/500 hops identical)
- [x] hooks in the pipeline and `run_qa`, guarded by `memory.enabled`; time
      spent logging is kept out of latency
- [x] `scripts/check_baseline.py`: memory off vs logging on gives identical
      traces, csvs (timing columns aside) and per-hop stats; passes without
      generation on hotpot and musique bm25
- [x] the same identity check with real generation on an ircot config, and
      with hybrid and reranked primaries. On the Mac, Qwen2.5-7B 4-bit at
      temperature 0, first 50 hotpot dev questions per config, one pair of
      runs each (4 Oct): hybrid ircot (103 traced hops), hybrid ircot with
      the reranker (100) and bm25 single hop (50) give byte-identical trace
      and per-hop files and identical csvs on the 21 non-timing columns
      between memory off and logging on. The shadow arm in the check is
      bm25; a dense or hybrid shadow arm has not been run beside generation
- [x] backfill the hybrid and ircot traces with bm25 and dense shadow arms.
      Six committed traces, replayed on the Mac (bm25s 0.3.10, bge-small and
      the reranker on mps), each returning the trace's ordered chunk ids at
      every hop: hybrid_ircot_hotpot_dev 500 questions / 1057 hops,
      hybrid_ircot_hotpot_confirm 1000 / 2177, hybrid_ircot_rerank_hotpot_dev
      500 / 1082, hybrid_ircot_rerank_hotpot_confirm 1000 / 2201,
      hybrid_musique_confirm 1000 / 1000, hybrid_ircot_musique_confirm 1000 /
      2625; 0 hops differ. The shadow configs are the committed config plus
      `memory: {enabled: true, log_arms: [bm25, dense]}`, kept under the
      gitignored results/memory/_cfg. In the reranked runs the shadow arms
      are first-stage lists at top_k 20 beside a reranked primary, and past
      hop 0 the shadow arms run on the primary's queries, so regret is exact
      at hop 0 only.
      Backfill the Mac's traces on the Mac: on linux with bm25s 0.3.11 the
      replay of bm25_hotpot_dev differs from the trace on 89 of 500 hops and
      bm25_musique_dev on 11, every one a different order inside a run of
      equal scores (16 of them at the top-k boundary, so the id set differs
      too). bm25s is now pinned to the Mac's 0.3.10 in requirements.txt. The
      linux runs differ from the Mac's in bm25s version and machine at once,
      and the top-k is numpy's argpartition (2.5.2 on the Mac, jax not
      installed), so I do not know that the pin removes the difference.
      The earlier sentence here, that a fresh linux bm25_arms_hotpot_dev run
      gives the Mac's four Stage 1b tables to the last digit, does not hold
      against a Mac run of the same 1500 questions
      (`results/bm25_arms_hotpot_mac_*`, tables
      `results/hotpot_mac_arms_*.csv`, bge-small on mps). The three arm
      rates are the same (0.5053 / 0.6173 / 0.634) only because two bm25
      outcomes flip in opposite directions, both in the confirmation fold
      and each a swap at rank 10; the oracle is 0.734 against 0.7347, the
      bm25/hybrid discordance cells move by one, no routed rate or gain
      moves by more than 0.0014, and the pairwise AUCs on the bm25-vs-hybrid
      questions (301 on the Mac, 303 on linux) move by up to 0.03, inside
      their split sd except at primary hits (0.028 against an sd of 0.025 on
      the Mac, 0.051 on linux).
      Against the Mac's own committed bm25 runs of those questions nothing
      differs. So bm25's all-gold-in-top-k is stable to one or two questions
      in 1500 across the two machines, not identical; dense and hybrid were
      compared through the table cells only, not per question. The hotpot
      numbers quoted below stay the linux ones.

### Stage 1b: what is there to learn (no model calls)

- [x] bm25 / dense / hybrid first-stage outcomes per question; discordance
      table; best-arm-per-question ceiling. `configs/bm25_arms_*.yaml` log
      dense and hybrid in the shadow of a bm25 single-hop run with no
      generation; `src/memory/arms.py` turns the log into the pairwise
      discordance table (McNemar), each arm's exact regret against the
      oracle (paired bootstrap) and the all-arms floor
      (`results/<dataset>_arms_*.csv`). Hotpot, positions 0-1499 (both
      folds), all gold in top-10: bm25 0.505, dense 0.617, hybrid 0.634,
      oracle 0.735, every arm 0.407. The oracle is 10.1 points above hybrid
      (paired bootstrap 8.5-11.6); bm25 and hybrid disagree on 20% of
      questions, dense and hybrid on 15% (McNemar p = 0.10, so dense and
      hybrid are close to interchangeable). At paragraph level there is
      almost nothing left: hybrid 0.967 mean paragraph recall, oracle 0.986.
      Musique, full dev (2417), all gold in top-5 of 20 paragraphs: bm25
      0.177, dense 0.326, hybrid 0.281, oracle 0.411, every arm 0.117; the
      oracle is 8.5 points above dense (7.5-9.6) and every pair disagrees on
      15-27% of questions. Hop-aligned (the first gold paragraph in top-5):
      hybrid 0.825, oracle 0.891.
      2wiki, positions 0-1499, all gold in top-10 (Mac run, 4 Oct; the
      processed dev file is on the Mac, so it is run there;
      `results/2wiki_arms_*.csv`): bm25 0.361, dense 0.573, hybrid 0.470,
      oracle 0.626, every arm 0.312. Dense is the best single arm and the
      oracle is 5.3 points above it (4.1-6.4), the smallest of the three
      ceilings, though the three are separate samples at different k and
      pool unit. Of the 79 questions behind it bm25 hits 68 and hybrid 43,
      so the room over dense is mostly bm25's. Pairs disagree on 30%
      (bm25/dense), 16% (bm25/hybrid) and 16% (dense/hybrid) of questions,
      and here dense and hybrid are not interchangeable (198 dense-only
      against 43 hybrid-only; hybrid is the untuned alpha 0.5 fusion). The
      Mac's bm25 lists equal the committed bm25_2wiki runs on all 1500.
      The musique log was rebuilt on the Mac as well
      (`results/bm25_arms_musique_mac_*`, tables
      `results/musique_mac_arms_*.csv`): discordance and ceiling are
      byte-identical to the linux tables, bm25 all-gold agrees on every
      question (the top-5 id set differs on 19), and the probe-stage rows
      differ in the fourth decimal. The two tf-idf neighbour rows differ in
      the fourth decimal too, but the committed musique table predates the
      stable neighbour sort, so that is the code and not the machine.
- [x] how predictable is the best arm from the frozen features (held-out
      AUC), and how much is just the question template. Same module: a
      standardised logistic regression per target (each arm's hit; which
      arm hits where two disagree) and per feature stage (size control,
      qtype, query, pool, primary hits, probe), fit on one seeded half and
      scored on the other, both ways over five seeds, or fit on the
      exploration fold and scored on the confirmation fold. Hotpot: barely.
      Each arm's hit is predicted at AUC 0.50-0.63 (the top value needs the
      probe features, i.e. both retrievals), and which of two arms hits
      where they disagree at 0.56-0.66; the size control sits at 0.50 and
      qtype at 0.49-0.57; the fold split gives the same picture. The
      `_routed.csv` table says what that buys: a shared router over the
      same features, one hit model per arm and argmax on the held-out half,
      lands at 0.632-0.638 against always-hybrid 0.634 and the oracle
      0.735, a gain of 0.004 +- 0.007 at the best stage. So the ceiling is
      real and the frozen process features do not reach it: a fixed rule or
      a shared router over them captures none of it. Whatever Stages 2-5
      find must come from somewhere else than these features; the first
      thing tried, past questions that look like it (Stage 2 below), gets
      nothing on hotpot either. Musique says the same: the router over the
      frozen
      features gains at most 0.004 over always-dense (0.326 vs oracle
      0.411). One caution the size control was for: on musique each arm's
      hit is predicted at AUC 0.69-0.71 from qtype alone, which is the
      composition shape, i.e. how many gold paragraphs must fit in top-5;
      the pairwise routing AUC stays at 0.46-0.62, so that is difficulty,
      not a routing signal.
      2wiki says it a third way (`results/2wiki_arms_predictability.csv`,
      `_routed.csv`). Each arm's hit is predicted at AUC 0.68-0.73 with
      every feature, but the size control alone already gives 0.64-0.66
      (pool size; on hotpot it gave 0.54-0.56) and qtype alone 0.63-0.73, which
      is the template deciding difficulty (bridge_comparison needs four gold
      sentences in the top-10). Which of two arms hits where they disagree
      stays at 0.46-0.61, and at 0.46-0.48 from qtype, so the worry that the
      2wiki template predicts the best arm does not hold at this outcome:
      the qtype router lands on always-dense exactly. No router stage gains
      over always-dense (0.567-0.573 against 0.573, oracle 0.626), and the
      explore-to-confirm split in `results/arms_foldsplit/` says the same
      (router gains -0.006 to 0.000). The hop-aligned outcome (first gold
      chunk of the path in the top-10; `results/2wiki_aligned_arms_*.csv`)
      has the one row that points the other way: which of bm25 / dense hits
      where they disagree is at AUC 0.70 from the query stage and 0.74 +-
      0.04 from the pool stage (174 questions, 32 bm25-only), with qtype
      alone at 0.56. It buys nothing: router gains -0.002 to +0.001 against
      always-dense 0.879, oracle 0.903.
      The zero-utility share (14 Nov, after MemCredit, paper log
      2026-11-13): the fraction of (lesson, question) pairs on which
      applying the lesson changes the retrieval outcome at all. A lesson
      here is "run arm A instead of the default", so at the arm level it
      is read off the discordance tables: switching bm25 to hybrid changes
      all-gold-in-top-k on 20% of hotpot questions, 15% on musique and 16%
      on 2wiki, bm25 to dense on 31%, 27% and 30%, so 69-85% of (arm,
      question) pairs carry zero credit. At the firing level it is
      false_application minus harm in the stream tables (fired, and both
      arms hit or both missed): 0.72-0.77 over the hotpot utility /
      boundary rows, 0.67-0.86 on musique, 0.62-0.76 on 2wiki, the static
      rules at 0.77, 0.79 and 0.85. MemCredit reports 86.9% on ALFWorld,
      so the label is as sparse here as there, and the offline fit to the
      exact credits that their estimator reaches AUROC 0.744 with is the
      pairwise row above ("which of two arms hits where they disagree",
      fit and scored on the non-zero-credit questions only): 0.56-0.66 on
      hotpot, 0.46-0.62 on musique, 0.46-0.61 on 2wiki, against their
      0.483 from relevance features alone. So the sparse-label reading
      does not change the result: with the credits given exactly and the
      zero-credit pairs taken out, the frozen features still do not say
      which arm.
- [ ] look-alike pairs: within-template pairs on 2wiki with discordant
      outcomes; natural pairs on hotpot by question similarity (expected
      sparse, reported as such); constructed twins (`data/make_twins.py`):
      distractor sentences sharing bridge-entity tokens, alias substitution.
      Natural pairs, hotpot positions 0-1499 (`src/memory/lookalike.py`,
      `results/hotpot_lookalike*_summary.csv`): each question's nearest
      other question, by tf-idf and by the bge-small question embedding,
      against random pairs. Sparse, as expected: by tf-idf only 28 of 1225
      pairs sit above 0.4 cosine. Agreement on an arm's outcome (both hit
      or both miss) is 0.54 / 0.53 / 0.57 (bm25 / dense / hybrid) over all
      nearest pairs against 0.51 / 0.52 / 0.53 over random pairs, and the
      best-arm label agrees on 0.29 against 0.26; the dense neighbours are
      inside the random intervals everywhere (0.52-0.54 vs 0.51-0.53).
      Look-alike questions on hotpot do share outcomes a little more than
      random ones, by two to four points, and nothing about which arm.
      Constructed twins (`results/hotpot_twins.csv`, first 800 twins, 641
      distractor and 159 alias, arms run on each): the alias twin changes
      some arm's outcome on 28% of pairs (bm25 18%, dense 10%, hybrid
      11%), flips both ways and symmetric (McNemar p 0.33-0.80), so the
      best arm moves on 28% of pairs: a surface change a text-keyed memory
      would treat as a new question is also a change of retrieval
      situation. The distractor twin as built (five sentences sharing a
      bridge-title token) changes an outcome on only 4.7% of pairs (bm25
      loses 14 of 641, p = 0.0001; hybrid 12 vs 3, p = 0.035; dense 5 vs
      0), so it is too weak to make the "same surface, different
      situation" pairs the boundary study needs. The harder twin
      (`distractor_para`: ten sentences from other pools that share a
      content token with the question and one with a bridge paragraph's
      text, function words out, so each scores on the query and looks like
      the bridge; pool 41.6 -> 52.7 chunks) does more
      (`results/hotpot_twins_para.csv`, twins of the first 500 questions,
      365 distractor_para and 78 alias, arms run on each): an outcome
      changes on 14.5% of pairs, three times the bridge-token twin, and
      every arm loses (bm25 19 hit-to-miss against 2, dense 22 against 0,
      hybrid 22 against 3; p <= 0.0002). The change is "harder for every
      arm" rather than "a different arm wins": of the 53 changed pairs, 34
      still have an arm that hits and 21 flip bm25 itself. The alias twins
      of the same 500 move an outcome on 29% of pairs (23 of 78), as in the
      first 800. That is enough changed pairs for the stream (next item).
      Natural pairs on the other two (Mac logs, 4 Oct;
      `results/{musique,2wiki}_lookalike*_summary.csv`). Musique is the
      opposite of hotpot: 1356 of 1813 nearest tf-idf pairs sit at or above
      0.4 cosine (873 above 0.6), and nearest pairs agree on an arm's
      outcome far more than random ones (bm25 0.82 against 0.72, dense 0.76
      against 0.55, hybrid 0.76 against 0.60; best-arm label 0.58 against
      0.38). I do not read that as a routing signal. Agreement counts
      both-miss, and the label lumps "no arm hits" (most musique questions)
      with the arm-specific labels; musique builds its questions from shared
      single-hop questions, so near neighbours share gold paragraphs and
      difficulty; and the random baseline is not matched on question type.
      Checked by hand on the pairs file and the data, not in a table yet:
      79% of nearest pairs share a gold paragraph, pairs that share none
      agree at the random level, and where the arms disagree on both
      questions (242 pairs) the label matches on 0.51 against 0.30 for
      random pairs. So there is some arm signal on about an eighth of the
      pairs, which fits a vote that gains about a point at best (Stage 2).
      2wiki: small. Nearest tf-idf pairs agree on bm25 0.66 against 0.55,
      dense 0.55 against 0.51, hybrid 0.60 against 0.51, best-arm 0.36
      against 0.28, and only bm25's agreement rises with similarity (0.50 to
      0.73 over the bands, 60 and 45 pairs at the ends); nearest
      pairs mostly share the question type, which a type-matched baseline
      has to take out before any of it is called look-alike.
      2wiki twins of the first 500 questions (`results/2wiki_twins.csv`; 435
      twins, 381 distractor_para and 54 alias, built with PYTHONHASHSEED=0
      because `make_twins` iterates a set of tokens and the seed alone does
      not fix the file; arms run with `configs/bm25_arms_2wiki.yaml` and
      `dataset.name: 2wiki_twins`): the distractor_para twin changes some
      arm's outcome on 32% of pairs against 14.5% on hotpot, again harder
      for every arm rather than a different arm winning (hit-to-miss bm25 48
      against 0, dense 65 against 0, hybrid 55 against 1). The alias twin
      changes 15% of 54 pairs, too few to read. The twin set is a harder
      subset: comparison questions get no distractor twin. Rebuilt on 5 Oct
      with the sorted builder: 31.8% of pairs (164 against 1; bm25 47 / 0,
      dense 65 / 0, hybrid 52 / 1), alias unchanged. Musique twins exist
      since 5 Oct (`results/musique_twins.csv`, 484 distractor_para of the
      first 500, no alias kind): an outcome changes on 13.4% of pairs (74
      against 9; bm25 24 / 3, dense 17 / 0, hybrid 33 / 6, p < 1e-4 each),
      between hotpot and 2wiki; details under Stage 5. Within-template pairs
      on 2wiki are not written yet.

### Stages 2-4: the baselines

- [x] 2. similarity memory over past experiences (top-k neighbours vote).
      At retrieval level on hotpot it is already in `src/memory/arms.py`
      (`question_knn` rows of `results/hotpot_arms_routed.csv`): the k
      nearest fit-half questions by tf-idf cosine vote with their own
      outcomes, and the arm with the most neighbour hits is chosen on the
      held-out half. k=5 lands at 0.625 and k=20 at 0.628 against
      always-hybrid 0.634 (gain -0.009 +- 0.007 and -0.006 +- 0.010), the
      same nothing as the feature router. On hotpot, look-alike questions
      by surface text do not say which retriever to use. Musique: k=5 gets
      0.334 against always-dense 0.326 (gain 0.008 +- 0.005 over ten
      halves, oracle 0.411), the only positive neighbour vote in either
      table at that point and too small to claim without the confirmation
      fold. Left to try before
      calling the stage: dense question neighbours, and the vote restricted
      to neighbours with a discordant outcome. Both tried on hotpot
      (`question_dense_knn*` and `*_discordant` rows of
      `results/hotpot_arms_routed.csv`): dense neighbours k=5 / 20 land at
      0.628 / 0.634, the discordant-only vote at 0.628 / 0.632 by tf-idf and
      0.633 / 0.637 by embedding, against always-hybrid 0.634; the best of
      the eight is +0.003 +- 0.006. The same nothing. Stage 2 on hotpot is
      closed: nearest past questions, by any key and with any vote, do not
      say which retriever to use.
      Musique with the two new votes (Mac log, ten halves,
      `results/musique_mac_arms_routed.csv`): against always-dense 0.3256
      all eight votes are positive: tf-idf k=5 / 20 +0.008 / +0.008,
      discordant-only +0.005 / +0.006, dense neighbours +0.011 / +0.009,
      dense discordant-only +0.007 / +0.007, with a spread over the halves
      of 0.004-0.007. That spread is the sd of ten half-gains over resplits
      of the same questions, not a standard error, and the table has no
      paired test, so +0.011 (the best of eight rows, a point of the 8.5
      available) is not a claimed difference. Fit on the exploration fold
      and scored on the confirmation fold
      (`results/arms_foldsplit/musique_mac_arms_routed.csv`, one split) the
      dense neighbours give +0.006 at k=5 and -0.002 at k=20, tf-idf -0.001
      / +0.001, and the largest row is the tf-idf discordant-only vote at
      k=5, +0.009 (35 wins against 26 by hand, p = 0.31): not confirmed,
      with the caveats that the memory is 500
      questions there against about 1200 in the halves, and that the folds
      are random splits of a set whose questions share gold paragraphs.
      2wiki (`results/2wiki_arms_routed.csv`): every vote is at or below
      always-dense, the k=5 votes over all past questions by a point or more
      (-0.012 +- 0.004 tf-idf, -0.014 +- 0.006 dense). So the stage closes
      as: a similarity
      vote routes nothing on hotpot, sits up to a point and a half under
      always-dense on 2wiki (no paired test there either), and on musique
      shows up to a point that I cannot separate from the dataset's own
      reuse of sub-questions. Claiming it would need a paired test per vote
      in `arms.py`, a memory of the size the gain was seen at, and a split
      by shared gold paragraphs.
      The paired test is in the table now (14 Nov, the replayed musique
      log of the Stage 4 item, 2359 questions, pooled bootstrap over
      questions beside the per-split McNemar): against always-dense 0.3247
      the dense-neighbour votes are +0.010 [+0.003, +0.017] at k=5 (0.3348),
      +0.009 [+0.002, +0.016] at k=20, +0.010 [+0.003, +0.017] discordant-
      only at k=5, and the tf-idf vote at k=5 +0.008 [+0.001, +0.015]; the
      other four hold zero; the largest per-split McNemar p is 0.67-1.0 on
      every row, so no single half separates. A point, then, with its
      interval, and the other two caveats stand: the memory is about 1180
      questions, the split by shared gold paragraphs is not done, and the
      fold-split table gave -0.002 to +0.009 for the same votes. Not a
      claim; it is the row to re-test on the confirmation fold if a
      similarity memory is ever argued for on musique.
- [ ] 3. failure memory: failure type + what recovered it; rule-based failure
      typing with a 50-case hand audit. The typing half is in
      `src/memory/failures.py`: five ordered rules on the primary arm's
      hop-0 top-k against gold (none / near_miss, every missed gold within
      2k of the full-pool ranking / named_miss, a missed gold paragraph is
      named in the query / bridge_miss, something hit and every missed
      paragraph is unnamed / total_miss), and the shadow arms that hit as
      what recovered it. Hotpot bm25 k=10, positions 0-1499
      (`results/hotpot_failure_types.csv`): 742 failures, of which 379 near
      misses, 171 named, 187 bridge, 5 total. What recovers them differs by
      type (chi-square over type x recovering arm, p < 1e-29): near misses
      are recovered by dense or hybrid 62% of the time (hybrid alone 54%),
      named misses 24% (dense 24%, hybrid 9%), bridge misses 36% (dense
      33%, hybrid 15%). So "bm25 missed" is not one lesson: a near miss
      wants the fused list, a named or bridge miss wants dense if anything,
      and hybrid is close to useless there. What is not yet known is
      whether any of this is visible before the outcome (the near / named /
      bridge split uses gold); that is the memory's applicability question
      for Stage 5. The audit sample is `results/hotpot_failure_audit.csv`
      (50 failures, seed 13: 29 near, 13 bridge, 8 named) with empty
      audit_ok / audit_note columns; the tick waits for it and its
      disagreement rate.
      2wiki, bm25 k=10, positions 0-1499 (`results/2wiki_failure_types.csv`,
      Mac log): 959 failures, 537 near misses, 212 named, 210 bridge.
      Recovery differs by type as on hotpot (near 54%, named 17%, bridge
      34%) but not in which arm: dense does the recovering for every type
      and hybrid adds almost nothing to it (dense alone 52%, 17%, 34%), so
      on 2wiki the typed table says "use dense" and no more. The types also
      mean less here than on hotpot (by hand from the log): 129 of the 212
      named misses have no gold chunk in the top-10 at all, which is why
      total_miss is empty, and 120 of the 537 near misses sit in pools of 20
      chunks or fewer, where the 2k window is the whole pool. No audit
      sample for 2wiki. Until 5 Oct the typing was not usable on musique:
      `paragraph_of` took the part of the chunk id before `::`, which on
      musique is `p{idx}`, so the named-miss rule could never fire there.
      Musique, bm25 k=5 over 20-paragraph pools, full dev
      (`results/musique_failure_types.csv`, Mac, 5 Oct): 1990 failures of
      2417, of which 466 near misses, 369 named, 1018 bridge, 137 total.
      Recovery differs by type as on the other two (near 52%, named 18%,
      bridge 23%, total 18%; chi-square over type x recovering arm 339 on
      9 dof, p ~ 1e-67) and dense does the recovering for every type (25%
      of all failures against hybrid's 16%); hybrid alone adds 41 near
      misses, 7 named, 16 bridge and 1 total. The shares mean something
      different here: bridge misses are half the failures (against a quarter
      on hotpot and 2wiki) because compositional musique questions rarely
      name a gold paragraph, and total_miss is populated (137 against 5 and
      0) because the 2k = 10 window is half the pool and a question whose
      gold is all outside the top-5 and unnamed falls through to it; 72% of
      failures are recovered by no arm. No audit sample for musique either.
- [x] 4. static conditions: a fixed predicate per lesson, and the same as
      prose read by the generator; if a fixed rule captures most of the
      best-arm ceiling, stop here and report that. CAVE-Mem (paper log,
      2026-10-02) is this baseline in print with a cross-fitted utility veto
      and abstention, so the arm gets both, and any per-lesson utility
      reported anywhere here is cross-fitted (estimated from blocks that
      exclude the question being scored).
      The predicate half, 13 Nov, at retrieval level (`RuleTree` and
      `rule_tree_rate` in `src/memory/arms.py`; the `tree:` rows of
      `results/hotpot_arms_routed.csv`, the rules themselves in
      `results/hotpot_arms_rules.txt`). Until now the only static
      conditions were the three hand-fixed rules of the stream (`rule:`
      rows), which were read off the Stage 3 table of the same questions.
      Here the predicates are learned on one seeded half and frozen on the
      other, both ways over five seeds, so nothing is fit where it is
      scored: a regression tree of depth 1, 2 or 3 with at least 30
      questions per leaf, fit to every arm's hit at once, and each leaf
      sends its questions to the arm with the highest hit rate among its
      fit-half questions, or stays with the best single arm when none is
      strictly above it; with the veto a leaf may leave the best single arm
      only when its utility (mean gain of its arm over the best single arm
      on the fit-half questions in the leaf) clears one standard error,
      else it abstains. Hotpot, positions 0-1499, all gold in top-10, against
      always-hybrid 0.634 on the same held-out halves (the rebuilt linux log,
      which replays the committed arms trace on 1500 of 1500 hops and
      reproduces every old cell of the four tables): no row is above it.
      Without the veto the trees land at 0.625-0.632 (query d1 -0.0025
      [-0.0077, +0.0027], pool d2 -0.0093 [-0.0165, -0.0024], primary_hits
      d3 -0.0088 [-0.0163, -0.0013]; the pool and primary_hits trees at
      depth 2 and 3 are below hybrid with their interval), sending 8-30%
      of questions away from hybrid; with the veto they land at 0.629-0.634
      and leave hybrid on 0-7.6% of questions, the depth-1 pool and
      primary_hits trees never, so their rows are always-hybrid exactly.
      The widest upper bound of the 18 rows is +0.0031 (query d2 unvetoed),
      below the shared router's +0.0148 at the pool stage. The rules are
      legible and say why: the first split is query length (8.5 tokens),
      pool size (24.5 and 54.5 chunks) or bm25 score entropy (0.979), i.e.
      difficulty, and every leaf but two sends to hybrid; the two that want
      dense are "pool above 54 chunks" (+0.026 +- 0.049 on 77 fit
      questions) and "entropy at most 0.979 and pool above 34 chunks"
      (+0.025 +- 0.022 on 285), neither clearing one standard error on the
      first split; over all ten halves the vetoed depth-2 and depth-3 trees
      leave hybrid on 3-8% of questions and lose by doing so. So a
      fixed rule over these features captures none of the 10-point ceiling,
      which is what the shared router (Stage 1b) and the per-lesson
      boundary (Stage 5) said from their sides; the three baselines agree.
      The prose half (the same condition written as a sentence and read by
      the generator) is not run: it needs the Mac, and there is no rule
      worth writing down; it stays available if a feature set ever gives
      one. Musique and 2wiki wait for their arms logs on the Mac (the
      linux musique log is a 48k-paragraph dense encoding on cpu, not run
      today).
      Musique, 14 Nov, on linux (`results/musique_replay_arms_*.csv`,
      rules in `results/musique_replay_arms_rules.txt`). The log is the
      committed Mac arms trace replayed here with bm25s 0.3.10 and
      bge-small on cpu: 58 of 2417 hops differ from the trace, every one a
      different order inside a run of equal bm25 scores (the known machine
      difference), so with --allow-mismatch those 58 questions are left out
      and every table below is over 2359 questions. On them the three arm
      rates are 0.178 / 0.325 / 0.279 against 0.177 / 0.326 / 0.281 on the
      full set, the oracle 0.410 against 0.411, discordance 0.265 / 0.151 /
      0.168 against 0.266 / 0.153 / 0.169, and the predictability cells are
      within 0.05 of the Mac's (the replay took 71 minutes, most of it the
      two dense encoders over the 20-paragraph pools). Against always-dense
      0.3247 on the same held-out halves: no tree captures any of the 8.5
      points to the oracle. The 18 rows sit at 0.3223-0.3284, a gain of
      -0.0025 to +0.0037, and every per-split McNemar p is 1.0; the one
      pooled interval that does not contain zero is tree:query:d2+veto,
      +0.0031 [+0.0001, +0.0064], the lower bound at the edge over 18 rows,
      and the unvetoed tree of the same depth sits at +0.0037 [-0.0006,
      +0.0082]. The vetoed trees leave dense on 2.6-10% of questions, the
      unvetoed on 9-18%. The rules are legible and differ from hotpot's:
      the first split is query length at 21.5 tokens on every stage, and
      the one leaf that clears the veto is "a question longer than 25
      tokens" (210 fit questions of 1179 at the query stage: bm25 0.105,
      dense 0.110, hybrid 0.181, +0.071 +- 0.022 over dense on the fit
      half), i.e. the long compositional questions where dense is no better
      than bm25 and the fused list is; at the pool and primary_hits stages
      the same long-question leaf splits on the rare-token ratio or the
      bm25 top-2 score and sends to hybrid at +0.04 to +0.08 on the fit
      half. Held out that leaf is worth 0.3 of a point over the whole
      stream, because the long questions are a sixth of the set and dense
      is at 0.11 there whichever arm runs. So on musique a fixed rule finds
      the one region the hotpot tables did not have, a nameable one, and
      it is too small and too hard to move the rate: Stage 4 closes on
      musique as on hotpot, the stage-level reading unchanged. 2wiki waits
      for its log and the zip.

### Stage 5: boundaries that revise themselves (the claim under test)

- [x] per-lesson applicability model (predicate, then logistic / small tree),
      decide-then-reveal online protocol, K=5 seeded stream orders. Built,
      run on all three datasets with the paired contrasts and the
      shuffled-reveal control (5 Oct), and the claim fails as stated; the
      numbers follow. The
      protocol and every policy of the table are in `src/memory/stream.py`,
      run on the hop-0 records of an arms log: a policy reads the features
      of its stage (query / pool / primary_hits, the last meaning bm25 has
      already run and a switch costs a second arm), picks an arm or probes
      every arm, and is then told how each arm did; five seeded orders,
      every rate paired against the fixed default per order. A lesson is
      "where bm25 fails like this, arm A recovers it", created from a
      revealed recoverable failure with that question's features as the
      exemplar and a box of 1 or 2 running sd around it as its region; it
      fires when the question is inside and its utility (mean gain of A
      over bm25 on the revealed questions inside the region, fired or not,
      never the current one) is positive after five of them. `utility`
      keeps the region fixed; `boundary` revises it with the operators
      below; `+lr` adds a per-lesson online logistic inside the region.
      Hotpot, positions 0-1499, bm25 default, all gold in top-10
      (`results/hotpot_stream.csv`): fixed hybrid 0.634, probe 0.735 at
      cost 3. The shared router lands at 0.619-0.622 at every stage and the
      feature-space knn at 0.612, i.e. at always-dense, below always-hybrid,
      as the held-out tables of Stage 1b said. Among the lesson memories the
      plainest is the best: it is the fixed-region
      utility memory at the query stage with radius 2 (0.626 +- 0.004), and
      it gets there by firing on 1437 of 1500 questions, i.e. it has turned
      into "always switch"; the boundary memory is at 0.546-0.587 with
      380-700 lessons created, 30-230 alive at the end and 2000-4000 logged
      revisions per order. The per-lesson logistic does not help either:
      inside a fixed region it is the worst row (0.516-0.546), with the
      operators it stays where the box alone was (0.554-0.571).
      The static rules fixed from the Stage 3 table (thin or flat bm25 list
      -> hybrid, no title named -> dense) give 0.552 at cost 1.41, the veto
      changes nothing (0.550). Among the 344 recoverable failures, the
      fixed hybrid repeats 0.28 of them, the best memory 0.22, the boundary
      memory 0.47-0.70; false application is 0.79-0.83 for every policy
      that fires (mostly firings where bm25 would have hit too; the harm
      share, chosen arm missed where bm25 hit, is 0.06-0.08 throughout).
      So on hotpot the claim fails as stated: an explicit per-lesson
      boundary over the frozen process features does not beat the shared
      router, the fixed-region utility memory or always-hybrid, and the
      operators do not rescue it (they lower the rate in the plain radius-2
      rows and at the radius-1 query stage, and raise it only in the
      radius-1 pool / primary_hits and logistic rows, all below 0.58; see
      the ablation item), because there is no region of this feature
      space where one arm reliably beats another (Stage 1b). The unit test
      shows the instrument is not the problem: on a table with a planted
      boundary the boundary memory recovers it (0.94 against the oracle
      1.0, false application 0.09) where the fixed-region utility memory
      does not (0.63-0.69, 0.45-0.64).
      The twins as the stream (`--twin-records`, `results/hotpot_twin_stream.csv`,
      first 500 questions plus their twins of one kind in each seeded
      order, five orders): the pair the boundary study was built for, with
      the surface-keyed memory (`text:exact`, `text:0.6`: repeat what the
      nearest revealed question by text taught) as the baseline that by
      construction cannot see a changed pool, and per policy the hit rate
      on twins whose original came earlier, split into unchanged pairs and
      changed ones (some arm's outcome differs), beside the default, the
      repeated arm and the oracle. Distractor_para (865 questions, 180
      ordered pairs per order, 27 of them changed): on the unchanged pairs
      the text key is the oracle (0.630, the repeated arm is the best arm
      by definition) and every feature-keyed policy is 8-16 points below
      it (router 0.53, knn 0.55, utility memory 0.47-0.54, boundary memory
      0.48-0.51): the process features move with the pool (its size, the
      bm25 scores) while the outcome does not, so a feature boundary does
      not recognise the same question again. On the changed pairs nothing
      beats repeating the old arm either: repeat 0.396 against default
      0.255 and oracle 0.627; the best feature policy is the fixed-region
      utility memory at the query stage (0.387), the router 0.33-0.37, the
      boundary memories 0.27-0.34 (McNemar against the repeated arm p >=
      0.125 everywhere, 27 pairs per order, so this half is underpowered;
      the unchanged half is not). Alias (578 questions, 39 ordered pairs,
      10 changed): the exact key sees a new question and does nothing, the
      0.6 key matches every alias twin and equals the oracle on the
      unchanged pairs (0.767) where the feature policies sit at 0.58-0.76.
      So the twins say the same as the plain stream, and more sharply: the
      surface key is the better key for "when does this experience apply"
      on hotpot, and the boundary over frozen process features is worse
      than the key that cannot see the change at all. The fix, if there is
      one, is features that stay put when the question is the same and
      move when the arm's fortune does, not more operators.
      The same table on musique and 2wiki (Mac arms logs, 4 Oct;
      `results/{musique,2wiki}_stream.csv`). Musique, full dev, top-5: fixed
      dense 0.326, hybrid 0.281, probe 0.411. Routers 0.307-0.319, feature
      knn 0.316-0.319, the best lesson memory 0.321 (fixed-region utility,
      query stage, radius 2, firing on 2278 of 2417 questions), boundary
      memories 0.255-0.283, with the per-lesson logistic 0.193-0.225 and
      0.237-0.261, the hotpot-derived static rules 0.223. 2wiki, positions
      0-1499, top-10: fixed dense 0.573, hybrid 0.470, probe 0.626. Routers
      0.544-0.556, knn 0.550-0.556, best lesson memory 0.566 (the same row,
      firing on 1436 of 1500), boundary memories 0.410-0.518, static rules
      0.394 (all 408 firings go to hybrid; the no-title rule never fires on
      2wiki, so that row tests one rule). No policy reaches the best fixed
      arm on either dataset, so the hotpot result is not particular to
      hotpot. What the table cannot say:
      every gain and McNemar column is against the bm25 default, the weakest
      arm on all three datasets, so "below always-dense" is read off the
      rates and their spread over five orders, and a paired column against
      the best fixed arm is the missing piece. Rate follows how often a
      policy leaves bm25 (the best rows are the ones that fire almost
      always), and false application (0.73-0.88 on musique, 0.68-0.87 on
      2wiki) and harm (0.02-0.06, 0.01-0.06) sit where a blanket switch to
      dense would put them (by hand from the logs: 0.79 and 0.06, 0.74 and
      0.05), so no row shows a selective boundary. Among recoverable
      failures (566 and 398) fixed dense repeats 0.115 / 0.028 and the best
      memory 0.155 / 0.065. As on hotpot (routers +1.5 to +2.6 there against
      +0.4 to +1.2 for the fixed arms), most learned policies gain more on
      the second half than the fixed arms (routers +1.6 to +1.9
      points on musique and +0.8 to +1.4 on 2wiki, against +1.0 and -0.2 for
      fixed dense), but it is catch-up from a cold start: no policy's
      second-half gain exceeds fixed dense's (0.1532 against 0.1538, 0.2093
      against 0.2115).
      Twins as the stream on 2wiki (`results/2wiki_twin_stream.csv`, first
      500 questions plus their twins, five orders). Distractor_para, 192
      ordered pairs per order, 63 of them changed: on unchanged pairs the
      text key is the oracle (0.374, by construction), always-dense is at
      0.353, routers and knn at 0.34-0.36, the boundary memories at
      0.19-0.31. On changed pairs repeating the old arm gets 0.186,
      always-dense 0.309, routers 0.27-0.31, knn 0.29-0.31, the oracle
      0.377. So here the surface key loses on changed pairs, the opposite of
      hotpot, and it loses to a fixed arm, not to a memory: on the changed
      pairs no policy is above always-dense by more than 0.001. Two cautions
      on that row.
      `p_vs_repeat_changed` is the largest of five per-order p values (0.45
      for always-dense), and the orders reuse the same pairs; by hand,
      four of the five orders are under 0.05, so a pooled test over the
      changed pairs belongs in the table. And the repeat rule keeps bm25
      wherever bm25 hit the original, which is the weakest arm on 2wiki, so
      part of the surface key's loss is that rule and not the key. The
      hotpot reading above ("the surface key is the better key") rests on
      the same rule and should be re-read with it in mind. Alias: 28 pairs
      per order, 4 changed, too few. Still open: induced drift (not written
      in `stream.py`: one log, one permutation, no dataset marker), features
      that are not frozen process statistics, and qtype taken out of the
      query-stage features (it is dataset annotation, and on musique and
      2wiki it carries difficulty).
      The 2wiki twin stream rebuilt on the Mac (5 Oct) with the sorted
      twin builder (435 twins again, a few distractor_para twins pick other
      sentences; `results/2wiki_twin_stream.csv`, 192.2 pairs per order,
      60.6 changed, 251 distinct unchanged and 120 distinct changed pairs)
      and the pooled columns. On unchanged pairs the text key is the oracle
      (0.384) and above always-dense (0.352) by +0.028 (paired bootstrap
      0.008 to 0.052); nothing else is, apart from probe at cost 3. On
      changed pairs the pooled test settles the per-order caution:
      always-dense beats repeating the old arm by +0.133 (0.058 to 0.208),
      as do the routers and knn at every stage (router:query +0.122, knn20
      +0.137, lower bounds 0.002 to 0.058), while the radius-2 boundary
      memories do not separate from repeat (boundary:query:r2 +0.045, -0.029
      to 0.119). Against always-dense on changed pairs no cost-matched
      policy's interval excludes zero upward (largest knn20:query +0.004,
      -0.011 to 0.025); 29 of 44 rows are below it, every boundary memory
      among them (boundary:query:r2 -0.088, -0.139 to -0.041), and the text
      keys at -0.133 (-0.208 to -0.058). `repeat_best` answers the caution
      about the repeat rule: started from dense instead of bm25 the
      repeated arm gets 0.311 on changed pairs against always-dense 0.302
      and the oracle 0.391 (a rate, no paired column), so the surface key's
      loss on 2wiki changed pairs is its bm25 starting point, and the text
      rows in the table still test the bm25-started rule. Alias: 8 distinct
      changed pairs, nothing to read.
      Musique twins (first ones, 5 Oct, `results/musique_twins.csv`,
      `results/musique_twin_stream.csv`): 484 distractor_para twins of the
      first 500 questions, no alias kind (the alias rules read a
      paragraph's first sentence, which a paragraph pool does not have; the
      builder refuses them there now, after giving four, two wrong). The
      builder also skips a candidate whose text is already in the pool: on
      musique the same paragraph recurs across pools under different
      p{idx} ids, so the id check alone had let 54 of 484 twins add a copy
      of their own gold paragraph as a distractor. The twin changes some
      arm's outcome on 13.4% of pairs (74 hit-to-miss against 9
      miss-to-hit; bm25 24 / 3, dense 17 / 0, hybrid 33 / 6, McNemar p <
      1e-4 each), between hotpot's 14.5% and 2wiki's 31.8%, harder for
      every arm as elsewhere. The twin stream (233.4 pairs per order, 31.6
      changed; 403 distinct unchanged, 64 changed): on unchanged pairs the
      text key is the oracle (0.261) and above always-dense (0.202) by
      +0.055 (0.035 to 0.077), about half of hotpot's lead, with the
      routers and knn at 0.20-0.22 and the boundary memories at 0.15-0.20;
      on the 64 changed pairs the intervals are 0.06 to 0.31 wide and only
      probe separates from always-dense (repeat 0.284, always-dense 0.353,
      knn20:query 0.384 at +0.014 with -0.063 to 0.095, oracle 0.505), so
      the changed half is as underpowered as hotpot's. The twin streams
      carry the shuffled control too and no pair is under 0.05 (musique p
      0.11-1.0, 2wiki 0.34-1.0).
      The hotpot twin stream rebuilt with the `repeat_best` and pooled
      columns (13 Nov, linux, bm25s 0.3.10, bge-small on cpu;
      `results/hotpot_twin_stream.csv`, twins of the first 500 questions
      built here from the hf mirror's dev file: 78 alias twins, whose
      three arm rates equal the committed `hotpot_twins_para.csv` to the
      digit, and 364 distractor_para twins against the Mac's 365, a few
      drawing other distractor sentences, so the distractor half is the
      same rule over a slightly different twin set: 298 distinct unchanged
      pairs and 46 changed, against about 27 changed pairs per order in the
      Mac table). Distractor_para, unchanged pairs: the text key is the
      oracle (0.633) and above always-hybrid (0.549) by +0.084 (0.054 to
      0.121); the routers and knn are within noise of hybrid (router:query
      +0.007, -0.014 to 0.031; knn20 +0.018, -0.010 to 0.046), the
      fixed-region utility memory level (-0.002), the boundary memory
      below it (boundary:query:r2 -0.041, -0.073 to -0.010) and the static
      rules further below (-0.074). Changed pairs: the repeat rule started
      from bm25 gets 0.248 and started from hybrid 0.392, which is
      always-hybrid exactly (0.392; the oracle 0.588), so on hotpot too the
      surface key's loss on changed pairs was its bm25 starting point, as
      on 2wiki. Against the bm25-started repeat always-hybrid is +0.109
      (-0.065 to 0.283) and the text key -0.109 against always-hybrid
      (-0.283 to 0.065), neither separable on 46 pairs; the one row whose
      interval against repeat clears zero is boundary:primary_hits:r2
      (+0.137, 0.005 to 0.269, at cost 2), and against always-hybrid it is
      +0.028 (-0.108 to 0.168), so nothing is read from it. Alias (54
      unchanged, 21 changed distinct pairs; every rate equals the Mac
      table): on unchanged pairs the 0.6 key is the oracle (0.767) at
      +0.037 over always-hybrid (0.000 to 0.093) and the exact key, which
      sees a new question, sits at bm25 (0.568); on changed pairs
      always-hybrid 0.834 against repeat 0.546 (+0.238, 0.000 to 0.476),
      repeat_best 0.787. So the earlier reading is cut to what the pooled
      test supports: on unchanged pairs the surface key is the better key
      on hotpot (as on musique and 2wiki, by construction), and on changed
      pairs no key beats the best fixed arm, the feature boundaries
      included.
      qtype taken out (13 Nov, `--no-qtype` in `stream.py`,
      `results/hotpot_stream_noqtype.csv`, the same log and orders; the
      revision log of that run is not committed): every policy moves by at
      most 1.2 points against the table with the one-hots
      (boundary:primary_hits +0.012; the routers -0.001 to +0.004; the
      lesson memories at radius 2 +0.002 to +0.008, their +lr rows -0.002 to
      -0.009), the ranking is unchanged, nothing is above always-hybrid
      (router:query 0.626, -0.008, p 0.92; utility:query:r2 0.627, -0.007,
      p 0.92; every boundary memory -0.042 to -0.089, p < 1e-3) and no
      policy separates from its shuffled copy (p 0.07-1.0). The lesson
      memories at the box radius fire more and create fewer lessons without
      the two extra dimensions (a box holds more questions), which moves
      their rates by half a point. So no rate of the hotpot table rests on the question type; on
      musique and 2wiki, where the type carries difficulty, the check waits
      for the Mac's logs.
      Musique, 14 Nov, on the replayed log of the Stage 4 item (2359
      questions; `results/musique_replay_stream.csv` with the one-hots,
      which reproduces the committed Mac table within a point on every row
      and the ranking, and `results/musique_replay_stream_noqtype.csv`
      without them, the same orders; the revision logs are not committed):
      every policy moves by at most 1.8 points. The routers +0.001 to
      +0.003, the knn -0.003 to -0.004, the lesson memories at the box
      radius +0.001 to +0.015 (utility:pool +0.015, the largest, firing on
      1409 against 970 questions), their +lr rows -0.002 to -0.018
      (boundary+lr:query:r2 -0.018), and as on hotpot the box memories fire
      more and create fewer lessons without the six extra dimensions
      (utility:query:r2 8.6 lessons against 19.4). Nothing is above
      always-dense: utility:query:r2 0.322, -0.003, p 0.86; router:query
      0.319, -0.006, p 0.42; every boundary memory -0.043 to -0.068, p <
      1e-4. The first three of the ranking are unchanged (probe, dense,
      utility:query:r2) and the rows below swap places within a point. The
      shuffled control holds on 11 of 12 pairs (p 0.21-1.0); the one that
      separates is knn20:primary_hits, 0.3145 real against 0.2986 shuffled
      (p 0.011 on the largest order, 0.053 with the one-hots and 0.94 on
      the Mac's full set), a row 1.0 points under always-dense. So no rate
      of the musique table rests on the question type either, although
      here the type does carry difficulty (Stage 1b: each arm's hit at AUC
      0.70 from qtype alone); 2wiki waits for its log.
      5 Oct, hotpot rebuilt here (bm25s 0.3.10 on linux replays the
      committed arms trace on 1500 of 1500 hops, and every cell of the
      committed table reproduces), with the paired contrasts the table
      lacked. Against the best fixed arm (hybrid, 0.634) nothing is above
      it: the routers and the best lesson memory are within noise of it
      (router:query -0.012, largest p over the five orders 0.25;
      utility:query:r2 -0.008, p 0.70; utility:pool:r2 -0.017, p 0.61),
      every boundary memory is below it on every order (-0.047 to -0.088,
      p < 1e-4 on each), as are the +lr rows and the static rules (-0.082).
      Inside the utility / boundary pairs, per order: at radius 2 removing
      the operators raises the rate at the query stage (+0.061, p < 1e-4;
      158 questions the fixed region alone gets against 67) and the pool
      stage (+0.031, p 0.011), and the +0.023 at primary_hits is not
      separable (p 0.18); at radius 1 the operators raise it at
      primary_hits (+0.029, p 0.018), the pool stage is level (+0.019, p
      0.18) and the query stage goes the other way (-0.054, p < 1e-4); with
      the per-lesson logistic the operators help at query (+0.039, p
      0.0003) and pool (+0.037, p 0.003) and not separably at primary_hits
      (+0.019, p 0.32). So the ablation's reading stands with a test behind
      it, and every row the operators raise is 5-9 points under
      always-hybrid. The shuffled-reveal control (paper log 2026-10-05: the
      same policy, told after each decision the outcomes of a different
      question under one fixed permutation per order, scored on the true
      outcomes; `shuffled:` rows, paired against the policy they shuffle)
      is the sharpest line in the table: no learned policy separates from
      its shuffled copy. Query stage, real against shuffled: router 0.622 /
      0.620 (p 1.0), knn 0.612 / 0.606 (1.0), utility:r2 0.626 / 0.622
      (0.50), boundary:r2 0.565 / 0.563 (1.0); pool stage, differences of
      0.004 to 0.009 with p 0.57-1.0; primary_hits, -0.008 to +0.003 with
      p 0.88-1.0. The shuffled boundary memory creates and revises about as
      many lessons as the real one (595 against 574 created at the query
      stage, 264 against 246 narrows, 1069 against 1053 exceptions). So on
      hotpot every policy in the table gets its rate from the arms'
      marginal rates (how often it leaves bm25, and for which arm) and
      nothing from the pairing of situation with outcome, which is what
      Stage 1b's "no region where one arm reliably beats another" predicts,
      now at the level of the policies rather than of an AUC.
      Musique and 2wiki with the same columns (Mac, 5 Oct, from the 4 Oct
      arms logs; every rate of the 4 Oct paragraph above reproduces, the
      revision logs byte for byte). Against always-dense nothing is above
      it on either dataset. On musique the routers, knn and the best lesson
      memory are within noise of it (router:query -0.007, largest p over
      the orders 0.29; utility:query:r2 -0.005, p 0.52, firing on 2278 of
      2417), every boundary memory is 4-7 points below (p < 1e-4 each), the
      +lr rows 6-13 below and the static rules 10 below. On 2wiki no
      learned policy is within noise at 0.05: the closest is utility:query:r2
      at -0.008 (p 0.049; 4.6 questions it gets against 16.2 only dense
      gets), the routers and knn are -0.017 to -0.029 (p <= 0.014), the
      boundary memories -0.056 to -0.164, the rules -0.18. Inside the nine
      utility / boundary pairs per dataset, removing the operators raises
      the rate at radius 2 at every stage on both (musique +0.053 to +0.063
      at query, +0.043 pool, +0.022 primary_hits, p <= 0.0009; 2wiki +0.068
      to +0.076, +0.028, +0.020, p <= 0.018), is level at radius 1 at the
      pool stage (p 0.78 and 0.57), and lowers it at radius 1 at
      primary_hits (-0.052 and -0.027, p <= 0.0006) and in every +lr pair
      (musique -0.044 / -0.038 / -0.036, p <= 0.027; 2wiki -0.047 to
      -0.087, p < 1e-4); the two hotpot cells that were not separable
      (primary_hits r2, +lr primary_hits) separate here in the direction of
      the hotpot point estimates. Every row the operators raise is 6-16
      points under always-dense, and in all 18 pairs the member that fires
      more has the higher rate. The shuffled-reveal control says what it
      said on hotpot: no learned policy separates from the copy told the
      wrong question's outcomes. Musique, real / shuffled: router:query
      0.319 / 0.310 (p 0.25), knn 0.316 / 0.300 (0.94), utility:query:r2
      0.321 / 0.319 (0.71), boundary:query:r2 0.258 / 0.251 (0.65); pool and
      primary_hits differ by 0.002 to 0.020 with p 0.39-1.0, except
      boundary:primary_hits:r2 at 0.283 / 0.266 (p 0.041 on all five
      orders), which is one pair of 24 and sits 4.2 points under
      always-dense and 2.2 under its operators-off pair. 2wiki: eight
      shuffled copies are below the real policy (-0.001 to -0.026) and four
      above (+0.002 to +0.005), smallest p 0.35; utility:query:r2 real 0.566
      against shuffled 0.567. One difference from hotpot: on musique the
      shuffled boundary memory fires about as often (n_fired within 6%) but
      creates 17-43% fewer lessons and logs 20-55% fewer exceptions than
      the real one (1201 against 1450 at the query stage); on 2wiki it
      creates and revises about as many (1545 against 1339). I have not
      traced why. False application of every firing utility / boundary row
      is 0.725-0.790 on musique and 0.682-0.741 on 2wiki against 0.793 and
      0.742 for a blanket switch to dense (from the fixed:dense row's
      discordant counts), the lowest being the rows that fire least, and
      harm 0.045-0.061 against 0.058 and 0.045. Nothing out-learns fixed
      dense on the second half (musique 0.1532 against 0.1538, 2wiki
      0.2109 against 0.2115), and the shuffled copies gain as much on the
      second half as the real ones, so that gain is the cold start. So the
      claim fails as stated on all three datasets, with the same shape each
      time: no region of the frozen feature space where one arm reliably
      beats another (Stage 1b), policies whose rate is how often they leave
      the default (the pairs), and nothing read from which outcome went
      with which situation (the shuffled control). Levels are not
      comparable across datasets (musique is the full dev at top-5 of 20
      paragraphs, the others positions 0-1499 at top-10 of sentence pools);
      the shapes and signs are
- [x] operators narrow / expand / exception / split / retire with a revision
      log; never overwrite silently. `LessonMemory(revise=True)` in
      `stream.py`: narrow pulls one box edge to just inside a contradicting
      point on the dimension where it is farthest from the exemplar, only
      if no supporter is lost; exception excludes the point with a small
      ball when every cut would lose one; expand grows the box to a point
      just outside it where the arm beat the default; split cuts a lesson
      with more than three exceptions at the median of its supporters and
      replaces it with two; retire drops a lesson whose utility is not
      positive after five questions. Every step is a row of
      `results/hotpot_stream_revisions.csv` (first order) with the box
      width before and after; the test checks that narrowing never widens a
      box or loses a supporter
- [x] baselines on the same stream: shared online router, scalar utility,
      always-probe, fixed hybrid; induced drift (switch dataset mid-stream);
      a query-only router (the same router on the query-stage features
      alone), since matched query-only controls have erased retrieval-signal
      gains elsewhere; and a no-persistence control (decide each question
      from the current features with nothing kept across questions), which
      beat every persisted-skill method in the held-out skill study (paper
      log, 2026-10-01). All but the drift are rows of `hotpot_stream.csv`
      (numbers above): the query-only router equals the primary-hits router
      (0.622 vs 0.619), and the no-persistence rules are the `rule:` rows.
      The musique and 2wiki tables have the same rows, with the paired
      columns and the shuffled control since 5 Oct.
      Induced drift (12 Nov, `src/memory/drift.py`): the stream is 1500
      hotpot questions in a seeded order followed by 1500 musique ones
      (drawn per order from the 2417), and the reverse, five orders each;
      the question-type one-hots are left out so that nothing announces
      the dataset; every policy is run stale (carrying what it learned over
      the switch) and cold (started fresh at the switch on the same
      questions in the same order), and transfer is stale minus cold on the
      second segment, paired per order (`results/hotpot_to_musique_drift.csv`,
      `results/musique_to_hotpot_drift.csv`, and the first boundary
      memory's log on the first order with the switch position). On the
      second segment the fixed arms are bm25 0.174 / dense 0.329 / hybrid
      0.278 on musique and 0.505 / 0.617 / 0.634 on hotpot, as on the single
      streams. Transfer is within -0.025 and +0.023 for every policy in
      both directions, and the two cells whose largest p over orders is
      under 0.05 are the per-lesson logistic at the query stage, which
      carries a worse boundary into the new dataset (-0.021, p 0.003 into
      musique; -0.025, p 0.017 into hotpot), and the fixed-region memory at
      primary_hits radius 1 into musique (+0.023, p 0.015, at 0.22 against
      0.20, both under bm25+rule). The shared router and the knn adapt
      inside the 200-question window (window transfer +0.02 to +0.04,
      p 0.6-1.0) and land where their cold start lands. Why nothing moves:
      at the pool and primary_hits stages a lesson box built on one dataset
      never contains a question of the other, since the pool has 9.9 +- 0.7
      paragraphs on hotpot and 20 on musique and the list 10 against 5
      hits, so every carried lesson goes dormant (0 operator steps on the
      old lessons after the switch, 0 of 200 window questions inside an
      old box, 0 retirements) and the memory restarts from nothing: the
      switch is absorbed by the size features, nothing is learned about
      it. At the query stage every window question sits inside an old box,
      the boundary memory retires 27 of 38 carried lessons within the
      window going into musique and 61 of 67 going into hotpot (the
      revision rate rises from 2.0 to 2.5 and from 2.6 to 4.2 steps per
      question), the fixed-region memory retires none (its utilities
      just drift down), and the two end level with each other and with
      their cold starts (boundary transfer +0.013, p 0.42, and +0.007,
      p 1.0; utility +0.002, p 0.91, and +0.005, p 0.93). So the
      operators do remove the stale lessons, visibly in the log, and it
      is worth nothing in the rate because the lesson memory sits five to
      six points under always-dense or always-hybrid on either side of
      the switch (0.27 against 0.33 on musique, 0.57 against 0.63 on
      hotpot) whether it starts warm or cold. The no-persistence rules
      transfer exactly 0, as they must. The drift control does not
      rescue the claim: a regime change is the case the boundary memory
      was built for, and the only policies that lose anything across it
      are the ones with a learned per-lesson boundary
- [x] metrics: repeated-failure rate, false application, false rejection,
      boundary precision / recall against the known best arm, adaptation
      speed, revision rate, regret, retrieval cost. Every rate that
      conditions on a lesson firing is reported beside the unconditioned
      contrast over the whole stream at matched retrieval cost (paper log,
      2026-10-02: conditioning on the trigger does not identify the
      invocation effect). In `stream.py` (`boundary_metrics`, `evaluate`):
      each row carries the whole-stream rate, cost and gain over the fixed
      default with McNemar per order, then the conditioned rates; false
      application is split into useless firings (both arms hit) and harm
      (the chosen arm missed where the default hit). Adaptation speed is
      the gain over the default on the first half of each order against
      the second (`gain_half1`, `gain_half2`): a policy that learns gains
      more later. On hotpot nothing does, beyond what the stream's
      composition gives: the fixed arms themselves gain 0.4-1.2 points more
      on the second half (hybrid 0.123 -> 0.135), the routers and knn 1.2-2.6
      more, the lesson memories 0.6-3.7 more at the box radius and less or
      nothing with the per-lesson logistic (utility+lr 0.020 -> 0.001). The
      rebuilt linux log reproduces every cell of the committed table
- [x] ablation: remove the operators and the log, keep the features; if
      nothing changes, the structure is decoration. Done as the `utility`
      against `boundary` pairs at every stage and radius, and `utility+lr`
      against `boundary+lr`: on hotpot removing the operators does not
      merely leave the rate unchanged, it raises it (0.626 against 0.565 at
      the query stage, radius 2); with the per-lesson logistic inside the
      region the order reverses (0.516 against 0.554), both well below
      always-hybrid. The structure is worse than decoration
      on these features; whether it is on any features is what the twins
      stream and the musique run have to say.
      With musique and 2wiki in, and looking at every pair on all three
      datasets, that was too strong. Removing the operators raises the rate
      at radius 2 at every stage and at radius 1 at the query stage. At
      radius 1 the two are level at the pool stage (boundary 0.260 against
      0.253 on musique, 0.459 against 0.448 on 2wiki, 0.570 against 0.551 on
      hotpot) and the boundary memory is higher at primary_hits (0.258
      against 0.206, 0.410 against 0.383, 0.546 against 0.517) while paying
      for more second retrievals; with the per-lesson logistic the operators
      raise the rate at every stage. In every pair the member that fires on
      more questions has the higher rate, so this ablation measures how
      often a memory leaves bm25 more than where it puts the boundary, and
      the rows where the operators help are all well under the best fixed
      arm. What the tables support is "no evidence the structure helps".
      With the pair tests on all three datasets (5 Oct, numbers under the
      first Stage 5 item): removing the operators raises the rate at radius
      2 at every stage on every dataset (p <= 0.018 everywhere, the hotpot
      primary_hits cell p 0.18), is level at radius 1 at the pool stage
      (p 0.18-0.78), lowers it at radius 1 at primary_hits (p <= 0.018)
      and in every +lr pair (p <= 0.027, hotpot primary_hits p 0.32), and
      every row the operators raise is 5-16 points under the best fixed
      arm. The reading stands and is now tested: the operators change how
      often a lesson fires, not where one arm beats another

### Stage 5b: actions inside the loop (28 Oct)

Stages 1b-5 asked one decision of the memory, which retriever to run, and on
three datasets that decision has nothing stable to learn. The ircot loop
offers others, and the hop-level logs already hold the effect of two of them:
retrieve deeper on this hop, and switch retriever on this hop. Extra hops,
query rewrites and early stops need generation and are not in the logs. So
the order is the same as before: replay what can be replayed, gate on it,
and only then run the loop live.

- [x] replay ceilings, hotpot exploration fold, hybrid ircot k=10
      (`scripts/inloop_ceiling.py`, trajectory fixed: later queries, stop and
      hop count are the logged ones). All gold in context 0.772 as run;
      per-hop best of the three retrievers at k 0.832 (+0.060, 30 vs 0);
      always dense 0.754 (-0.018, p 0.22); always 2k 0.908 (+0.136, 68 vs
      0) at 42.2 chunks per question against 21.1; 2k only on the hops where
      it adds gold (oracle) 0.908 at 26.6, i.e. on 26% of hops (275 of
      1057). Pools hold 41.5 chunks per hop, so 2k is not the whole pool
      (2.5% of hops have pool <= 2k). Reranked logs are refused: there top_k
      is the first-stage depth.
- [x] gate: is "deepening helps on this hop" learnable from the frozen
      features? Held-out AUC (halves by question, 5 seeds) 0.557 query
      stage, 0.562 pool, 0.643 primary_hits, size control 0.515. At matched
      cost (deepen 30% of hops, 27.5 chunks per question) the learned
      policy reaches 0.833 against 0.816 for uniform k=13 and 0.824 for
      deepening a random 30%: +0.017 +- 0.014 and +0.009 +- 0.015 over 10
      half-splits of 250 questions (per-split McNemar p median 0.54 and
      0.72, smallest 0.012 and 0.041 unadjusted, one split each of ten;
      `--gate`). The same shape as routing: a large oracle gap (0.908
      against 0.816 at matched cost) that the pre-retrieval features do not
      reach. Uniform k computed from gold_ranks gives 0.768 at k=10, not
      0.772, because hybrid's full-pool ranking is fused differently.
- [x] confirmation, preregistration 2 D3 (3 Nov): the policy fit on the
      exploration log and applied once to `hybrid_ircot_hotpot_confirm`
      (1000 questions, 2177 hops), deepening 30% of hops: all gold 0.826
      against 0.825 for uniform k=13 at 28.3 chunks per question, +0.001
      [-0.016, +0.018] (36 vs 35, p 1.0), and against a random 30% -0.008
      [-0.025, +0.009] (p 0.42). Upper bound 0.018 < 0.046: passed. The
      oracle at the same cost is 0.920 (27.4 chunks), always-2k 0.920,
      per-hop best arm 0.847. Musique confirmation log, descriptive (D4,
      k=5 over 20-paragraph pools, so 2k is half the pool): as run 0.553,
      always-2k 0.770, oracle 0.770 at 17.9 chunks against 13.1, per-hop
      best arm 0.681
- [x] probe stage for deepening: features of ranks k+1..2k computed without
      gold (does a deeper chunk come from a paragraph whose title is named in
      the question or in the context so far, score gap at rank k, how many
      deeper chunks are from paragraphs not yet in context). Retrieving 2k
      is cheap; what costs is the generator reading it, so a policy may look
      before it hands chunks over. Same gate as above, same fold.
      10 Nov. `memory.log_deeper: 2` (0 by default) logs the primary's chunks
      at ranks k+1..2k of the same full-pool ranking that gold_ranks reads,
      and `src/memory/deepen.py` computes eight gold-free features over the
      window (how many of its chunks come from a paragraph the question, the
      query or the context text names, how many from a paragraph not yet in
      context, the score gap at the cut, the score mass below it, the
      largest query overlap). `inloop_ceiling.py --gate` adds the window as
      a stage and pairs the two learned policies per split. Backfilled on
      linux (bm25s 0.3.10, bge-small on cpu): 1057 hops replayed, 11 differ
      from the Mac trace (7 questions dropped, 493 of 500 kept); every
      ceiling cell is within 0.001 of the committed ones (deep 0.909, best_k
      0.830, dense 0.755; 2k adds gold on 272 of 1042 hops). Held-out AUC:
      primary_hits 0.643 +- 0.017 as before, the window alone 0.609 +-
      0.026, both together 0.652 +- 0.023. At matched cost (deepen 30% of
      held-out hops, 27.4 chunks per question, ten half-splits of 246
      questions): the primary_hits policy 0.831, with the window 0.832,
      +0.002 +- 0.009 (per-split McNemar p median 0.71, smallest 0.25);
      against uniform k=13 (0.815) +0.017 +- 0.013, against a random 30%
      (0.829) +0.004 +- 0.014. The strongest single window feature is a
      deeper chunk from a paragraph the question names (AUC 0.574; 2k adds
      gold on 0.32 of those hops against 0.26 overall), then the largest
      query overlap (0.571) and the gap at the cut (0.561). So looking at
      the window before handing it over finds nothing the earlier stages
      missed: the gate stays where D3 left it and the confirmation fold is
      not spent on it. What the replay cannot test is the generator looking
      at the window, which needs generation
- [x] answer level: does more context move EM in this loop at all? Hybrid
      ircot k=20 on the hotpot exploration fold (`configs/hybrid_ircot_k20.yaml`)
      against the committed k=10 run, paired. If EM does not move with
      +0.136 all-gold, a deepening memory is a retrieval-level result only
      and is reported as that. Mac, 28 Oct, n=500: retrieval_correct 0.772
      -> 0.906 (+0.134 [+0.104, +0.166], 70 vs 3, p 1e-17), which the
      fixed-trajectory replay put at 0.908; EM 0.486 -> 0.502 (+0.016
      [-0.010, +0.042], 26 vs 18, p 0.29); F1 0.619 -> 0.636; prompt tokens
      1518 -> 2609 per question, latency 1.81 -> 2.37 s, hops 2.11 -> 2.19.
      The cell drops 0.068 -> 0.038 (20 vs 5, p 0.004), which is the
      retrieval_correct column moving, not answers. So deepening is a
      retrieval-level result here, and the intervention arms on
      `illusion-analysis` say why: with only the answer-bearing gold the
      reader does as well as with all of it, so completing the evidence adds
      little EM. A memory that deepens is judged on retrieval outcomes and
      context cost, and EM is reported beside it as level
- [x] confirmation, preregistration 2 D1/D2 (4 Nov): hybrid ircot k=20 on
      hotpot confirm (n=1000) against the k=10 run. retrieval_correct 0.774
      -> 0.925 (+0.151 [+0.129, +0.174], 156 vs 5, Holm p 2e-39); EM 0.466
      -> 0.476 (+0.010 [-0.009, +0.029]). Both pass: deepening is a
      retrieval-level change only on this fold too
- [ ] if the probe stage clears the gate: the live loop with the memory
      choosing deepen / extra hop / stop, lessons and boundaries as in
      Stage 5, on the exploration fold first, then once on a confirmation
      fold with the closed-book control

### Stage 5c: what the reader is handed (3 Nov)

Where the answers go, hotpot exploration fold, paired on the same 500 ids.
Retrieval-side changes move EM little: deepening +0.134 retrieval_correct
for +0.016 EM (Stage 5b); choosing the better of the closed-book and the
context answer per question (oracle) adds +0.026 over ircot (+0.068 over
bm25 on 2wiki, +0.010 on musique). What does move it is the context after
retrieval: on the 453 questions whose ircot k=20 context already held every
gold sentence, the gold sentences alone answer 0.625 against 0.512 for the
whole context (+0.113 [+0.068, +0.157], 79 vs 28, p 8e-7; +0.083 at k=10,
p 0.0002). And the reader still misses 39% with gold only, 89 of those with
F1 >= 0.5. So the decision a memory could take that reaches answers is which
retrieved chunks to hand the reader, not which to retrieve.

The memory, if the gate below is passed: before the answer step, keep or drop
each chunk of the answer context. Lessons are about chunks in a situation
("a sentence from a paragraph the question does not name, entered at a late
hop, with a low cross-encoder score, distracted the reader"), learned from
the per-chunk gold label and the reader's outcome, with a per-lesson
applicability boundary revised online as in Stage 5. Outcomes: EM recovered
toward the gold-only oracle on the all-gold-in-context subset, at matched
context size; kept-gold recall beside it.

What is already done elsewhere (literature check on 3 Nov, keyword search,
the 2026 items not yet read in full): learned filtering before the reader
(FILCO, RECOMP, Provence, CompAct, EXIT, CRAG, BRIEF); distraction by
irrelevant context (Shi et al. 2023, Lost in the Middle, The Power of Noise,
and gold-only against full context as a standard oracle ablation, so the
+0.113 is a confirmation in this harness, not a discovery); rule memories
with add / edit / remove (ExpeL), context-conditioned guidelines
(AutoGuide), evolving playbooks (ACE, Dynamic Cheatsheet, ReasoningBank),
applicability scoring of memory units (MILES, 2607.06974). Nothing found does
scoped, online-revised lessons for context selection, but every part exists,
so the claim stays narrow: scoped lessons beat a reranker and a supervised
filter at matched size, and the scope is what does it (ablation against the
same lessons without boundaries). To read in full before writing: MILES,
2607.17108 (evidence interfaces), 2609.17043 (fact-grounding gap), 2205.03685.

- [x] gate (`src/memory/reader_context.py`): the k=20 answer contexts cut
      to gold / reranker top 3 / top 5 / a held-out logistic chunk filter on
      gold-free features (top 3) / random 3, against the full context.
      Selection only, before generation: every gold chunk of the context
      kept on 0.438 (rerank3), 0.612 (rerank5), 0.464 (learned3), 0.016
      (random3) of questions; context 24.1 chunks, gold 2.31. If neither the
      reranker nor the filter recovers any of the +0.113, the decision is
      open but hard; if the filter recovers most of it, a memory has nothing
      to add and that is the result.
      Mac, 3 Nov, n=500 (`results/reader_ctx_hotpot_explore_*`). The full
      arm gives the run's answer on 500 of 500 (EM 0.502 both). Against
      full, Holm over the five arms: gold +0.102 [+0.064, +0.142] (79 vs
      28, p 3e-6); rerank3 -0.090 [-0.132, -0.050] (34 vs 79, p 8e-5);
      rerank5 -0.062 [-0.098, -0.026] (29 vs 60, p 0.001); learned3 -0.084
      [-0.124, -0.046] (31 vs 73, p 9e-5); random3 -0.322. The same on the
      453 all-gold-in-context questions (gold +0.117; rerank3 and learned3
      -0.080, rerank5 -0.057, all p <= 0.004). So no cut short of the
      oracle recovers any of the gap; every one of them loses EM, and the
      filter fit on the same gold labels is level with the reranker. Split
      by whether the cut kept every gold chunk (not tested, the subsets are
      selected by the arm): where rerank3 kept them all (219) it ties the
      full context, 0.594 against 0.594, under gold's 0.621; where it
      dropped one (281) it falls to 0.270 against full's 0.431, and learned3
      and rerank5 look the same. Reading: dropping a gold sentence costs
      far more than a distractor does, and a three-chunk cut that keeps the
      gold still gains nothing over the whole context, so the +0.11 lives
      only at exact selection. The gate is not passed: a keep / drop memory
      would be judged against a decision where every non-oracle policy
      tried so far is below doing nothing
- [x] confirmation, preregistration 2 C0-C3 (4 Nov,
      `results/reader_ctx_hotpot_confirm_*`, n=1000, filter fit on the
      exploration contexts only). Full arm = k=20 run on 1000 of 1000. Gold
      +0.134 [+0.106, +0.161] (180 vs 46, Holm p 2e-19). rerank3 -0.031
      [-0.057, -0.005] (below); rerank5 -0.017 [-0.042, +0.008] and learned3
      -0.022 [-0.050, +0.005] (not above, the weaker pass). All pass, but the
      cuts lose a third of what they lost on exploration. Kept-gold rates
      0.424 / 0.567 / 0.459. The split by kept gold (selected subsets, not
      tested) does not repeat exploration's tie: where rerank3 kept every
      gold chunk (385) it is above full on the same questions, 0.634 against
      0.577, and rerank5 and learned3 look the same. So on this fold a short
      context does help when it holds all the gold, and the cuts lose it by
      dropping gold on about half the questions. The gate stays closed for
      these cuts; what would open it is a selector that keeps all the gold
      far more often, which neither the reranker nor the filter is
- [ ] baselines a memory must face, from the literature check: reranker
      truncated to the same size, a supervised filter on the same labels
      (the learned3 arm here, and Provence off the shelf), and a flat lesson
      memory without boundaries (ExpeL / ACE style)
- [ ] then the memory, online over the exploration fold in a seeded order,
      and once on a confirmation fold with the closed-book control

### Stage 6: the agent and its experience memory (4 Nov)

Stages 1-5c tested the memory offline, by replaying logged runs. This stage
builds the live loop: the reader model acts, the memory decides what the
model is told before it acts, and the outcome of each episode feeds the
memory. Built whether or not it beats the baselines; reported either way.

Why this shape. The confirmation fold puts the loss after retrieval (C1:
gold-only +0.134 EM over the full k=20 context) and says a short context
helps when it holds all the gold (C3), but every cut tried keeps all of it
on under 60% of questions. Retrieval itself is near its ceiling at k=20
(0.925). So the agent's main action is choosing its evidence, and the
memory's job is to make those choices better from experience.

The agent (`src/agent/`), one episode per question, Qwen2.5-7B 4-bit, greedy:

- state: the question, a notebook of kept sentences, the latest search
  results numbered, the lessons the memory hands it, the steps left
- each step is one completion with two lines: `KEEP: <numbers or none>`
  from the latest results, then `NEXT: SEARCH <query>` or `NEXT: ANSWER`
- the answer comes from the same answer prompt every run uses, over the
  notebook only; an empty notebook falls back to everything retrieved and
  is logged as a fallback
- retriever hybrid k=10 per search, at most 3 searches (the ircot settings),
  so retrieval alone matches `hybrid_ircot` and the comparison isolates
  selection and memory
- logged per step: prompt, completion, kept ids, the lessons shown and the
  frozen gold-free features of Stage 1 at that step

The memory (`src/agent/memory.py`):

- an episode is scored after it ends, gold enters only here: notebook gold
  recall, answer EM, and a typed error per episode (dropped a gold sentence
  that was shown / never saw a gold sentence / kept distractors with all
  gold kept / stopped with a gold paragraph unreached / right)
- a lesson is short advice plus where it applies. Advice comes from a
  reflection call to the same model over the failed episode with the gold
  shown (ExpeL style). Where it applies is a box over the frozen features,
  revised online by the Stage 5 operators (narrow / expand / exception /
  split / retire), every revision logged
- at each step the lessons whose region holds the current features are shown
  to the agent, at most three, by utility
- built on hotpot train questions (never dev), then frozen or kept online
  for evaluation on the exploration fold; confirmation fold once, after a
  preregistration

What it is judged against, same questions, paired:

- `hybrid_ircot` with the full context (the run the agent has to beat), and
  gold-only (the ceiling)
- the agent with no memory (does choosing evidence help at all)
- reranker cut to the agent's notebook size per question (matched-size
  selection without a model)
- flat lesson memory: the same lessons retrieved by question similarity, no
  regions (ExpeL / ACE style)
- every lesson shown every time (does the region matter)
- shuffled lessons: regions and advice reassigned at random (does the
  content matter)

Primary outcomes EM and notebook gold recall; notebook size and model calls
as cost. Steps:

- [x] A1 agent loop without memory (`src/agent/agent.py`, pipeline mode
      `agent`, `configs/agent.yaml`) and a test with a fake generator
      (`test_agent_episode`: notebook, fallback, lessons passed through,
      other modes' rows and traces unchanged). The smoke run became the
      full exploration run: run_qa ignored --n when --ids was given (fixed,
      the first n ids now)
- [x] A2 the no-memory agent on the exploration fold (500) against
      `hybrid_ircot` and the matched-size reranker cut. If its notebook
      keeps all gold less often than rerank5 did (0.61 on exploration), fix
      the prompt on exploration before any memory goes in.
      Mac, 4 Nov (`results/agent_hotpot_explore_*`; the control is
      `results/reader_ctx_agent_hotpot_explore_*`, the A2 contexts re-read
      as full / gold / rerank3 / rerank5 / learned3 / random3). Exploratory,
      paired, n=500. Agent EM 0.526 against ircot 0.486 (+0.040 [+0.004,
      +0.078], 55 vs 35, p 0.045). It searches more (2.55 hops against
      2.11, all gold retrieved 0.862 against 0.772; it rarely says ANSWER,
      most episodes end at max_hops or a repeated query), so the gain is
      split with the control: ircot to the full context of the agent's own
      retrieval +0.008 [-0.014, +0.030] (p 0.61), full context to the
      agent's notebook +0.032 [-0.002, +0.068] (48 vs 32, p 0.093). The
      notebook (3.6 sentences) keeps every gold chunk of its context on
      0.624 of questions, against 0.472 / 0.652 for rerank3 / rerank5 on the
      same contexts, and is above both by +0.108 and +0.086 EM (p 1e-7, 2e-5):
      the reranker cuts lose 5-8 points to the full context here as on
      the k=20 contexts, the model's own selection does not. Gold-only is
      still +0.074 above the notebook. By type the agent gains on bridge
      (0.450 -> 0.512) and loses on comparison (0.635 -> 0.583, n=96). The
      gate reading holds: the notebook keeps all gold about as often as
      rerank5, so no prompt change before the memory
- [x] A2b main agent over subagents (`src/agent/orchestrator.py`, pipeline
      mode `orchestrator`, `configs/orchestrator.yaml`): one planning call
      splits the question into at most 3 sub-questions, #k standing for an
      earlier sub-answer; each sub-question gets the A1 keep / search loop
      (at most 2 searches) and a sub-answer from its own notebook; the
      answer is read from the union of the notebooks. Lessons are asked for
      per role (planner / subagent). Motivated by A2's split: the single
      agent gains on bridge questions and loses on comparison ones, where
      one notebook has to hold both entities. Judged against the single
      agent on the same questions, by question type.
      Mac, 4 Nov (`results/orchestrator_hotpot_explore_*`, commit 3d24c65),
      exploratory, n=500. EM 0.540: +0.054 [+0.018, +0.090] over ircot (55
      vs 28, p 0.004), +0.014 [-0.022, +0.050] over the single agent (47 vs
      40, p 0.52). By type, as expected before the run: comparison +0.104
      [+0.031, +0.177] over the single agent (0.583 -> 0.688, 12 vs 2, p
      0.013, n=96 only), bridge level (-0.007 [-0.050, +0.035], n=404). Plans:
      2 sub-questions on 327, 1 on 123, 3 on 37; none unreadable. Answering
      with the last sub-answer instead of reading the union of notebooks is
      far worse (0.356, -0.184), so the final read stays. Cost per question:
      7.2 model calls and 4.1 s against 3.5 / 2.4 s for the single agent and
      2.8 / 1.8 s for ircot. Notebook 3.7 sentences, all gold 0.516. To
      confirm: the comparison gain on a fold with more comparison questions
      (the hotpot confirmation fold has about 200)
- [x] A3 episode scoring and error typing; the reflection call.
      `src/agent/memory.py` (`score_steps`, `ExperienceMemory._advice`).
      Errors per decision point: planner `unreached` (a gold paragraph never
      retrieved by any search), subagent step `dropped` (a gold sentence
      shown and not in the final notebook) and `noise` (two or more
      non-gold kept, more than the gold kept at that step). Reflection: the
      failed step with the needed sentences marked, one rule under 25 words,
      rejected for leaks (a capitalised word of the question or a gold
      title, or a digit) and replaced by a fixed sentence per type, logged.
      Code 4 Nov; tests pass 9 Nov (`test_experience_memory`,
      `test_memory_in_the_loop`: the memory inside the main agent with a
      fake model, lessons created by reflection for both roles and shown
      from the next episode). Run on the GPU in A5: 447 of 478 region
      lessons and 212 of 228 flat ones came from reflection, the rest from
      the fixed sentence after a leak
- [x] A4 lesson memory with regions, the flat, show-all and shuffled
      variants. Same file. Regions are boxes over the planner's question
      features (24, from the text) or the Stage 1 step features (19), revised
      by the Stage 5 operators. Feedback is only seen where a lesson is
      shown, so a new lesson is on probation (shown wherever it applies)
      for min_n=5 showings, then shown while its utility (success rate of
      its error type when shown, minus the role's rate where no lesson of
      the type was shown) is positive; at most three per decision. Modes
      region / flat (question-token similarity, no regions or operators,
      built separately) / global (top by utility everywhere) / shuffled
      (advice permuted among the role's lessons) / none. Hooks: pipeline
      builds it from `agent_memory` in the config, run_qa calls
      `on_episode` after scoring and saves the store; `dataset.split`
      picks the data file (default dev, so nothing else changes). Tests
      pass 9 Nov; running them found the store's list of lessons shadowing
      the `lessons()` method the agent calls (renamed `entries`). Run in A5
- [x] A5 build on 1000 hotpot train questions
      (`results/folds/hotpot_train_mem.txt`, seed 13, none in either dev
      fold), with regions (`configs/orchestrator_mem_build.yaml`) and flat
      (`_build_flat`); evaluate frozen on the exploration fold:
      `orchestrator_mem_{region,global,shuffled,flat}.yaml`, against the
      main agent without memory (`orchestrator_hotpot_explore`), paired.
      Mac, 9-10 Nov, commit b42d92f (`results/orchestrator_mem_*`; the
      stores and their revision logs in `results/agent_memory/`; scored by
      `scripts/memory_eval.py` into
      `results/orchestrator_mem_hotpot_explore_{paired,steps}.csv` and
      `results/agent_memory/build_summary.csv`). Exploratory, n=500, paired
      against the main agent without memory (EM 0.540), Holm over the four
      arms:

      | arm      | EM    | diff, 95% CI           | discordant | p     | Holm |
      |----------|-------|------------------------|------------|-------|------|
      | region   | 0.536 | -0.004 [-0.024, +0.014] | 13 / 11    | 0.84  | 1.00 |
      | shuffled | 0.546 | +0.006 [-0.014, +0.026] | 11 / 14    | 0.69  | 1.00 |
      | global   | 0.506 | -0.034 [-0.066, -0.002] | 42 / 25    | 0.050 | 0.20 |
      | flat     | 0.508 | -0.032 [-0.062, +0.000] | 39 / 23    | 0.056 | 0.20 |

      No arm gains. The two that hand advice to every decision lose about
      three points (F1 -0.038 and -0.033, both intervals below zero), with
      notebook gold recall down at about the same notebook size (0.711 and
      0.724 against 0.742; notebooks 3.7 and 3.5 against 3.7). Region and
      shuffled are level with no memory and with each other: lessons shown
      on 0.81 of questions, 0.42 of subagent steps and 0.04 of plans; the
      error rate at steps shown a lesson of that type matches the steps
      shown none (dropped 0.276 / 0.276, noise 0.130 / 0.129). The one hint
      the other way is small: at shown steps noise is 0.130 with the region
      advice and 0.171 with shuffled advice (about 460 steps each), while
      EM goes the other way. Cost: no extra calls (reflection happens only
      while building); global's longer prompts add 0.17 s per question.

      The build (regions): 478 lessons from 1000 questions, 1492 expands,
      294 narrows, 18 exceptions, 2 splits, 179 retired; 297 active at the
      end, 198 of them never shown and only 10 past probation (5 dropped,
      5 noise, no planner lesson: plans leave a gold paragraph unreached on
      about 4% of train questions, so the planner has little to learn). Lessons
      get shown more as the build goes on (0.13 of subagent steps in the
      first quarter, 0.42 in the last), and the errors do not fall (dropped
      0.244, 0.214, 0.303, 0.273 by quarter). EM falls by quarter (0.692 to
      0.604) in both builds, and the flat build shows lessons at 0.91 of
      steps from the first quarter on, so as far as these runs can say it
      is the questions in those quarters (there is no run without memory on
      the train questions). The flat build made 228 lessons and retired 206
      after probation, keeping 22.

      Why, from the stores. The advice that survives is about the question
      it came from, despite the leak filter: global's three subagent lessons
      are "Include all results mentioning the publisher in your analysis",
      "... the specific unincorporated community in the question" and "Only
      keep results that directly answer the question about pseudonyms",
      handed to every step. Utility is measured on the step's error type, so
      a lesson is credited for where it was shown, not for what it said, on
      a few showings (6 to 61 for the ten that passed) against base success
      rates of 0.75 (dropped) and 0.86 (noise); the ones that pass are the
      ones that happened to land on easy steps. The frozen step features describe the retrieval, not the
      topic, so a region gathers steps whose search statistics look alike,
      and the advice in it is about some other question.

      Reading: as built, the experience memory does not help the main
      agent on exploration, and advice at every step costs about three
      points. Together with routing, deepening and the reader cuts, that is
      the fourth memory gate that stays closed. A6 waits: the confirmation
      fold is not spent on a version that is level on exploration.
      The discordant column read as WikiTrace reports it (paper log,
      2026-11-11): gain is the share of questions a memory arm answers that
      no memory did not, regression the share it loses. Region 0.022 gain
      / 0.026 regression, shuffled 0.028 / 0.022, global 0.050 / 0.084,
      flat 0.046 / 0.078. So every arm changes about 5-13% of the answers,
      and the show-all arms lose nearly twice what they win; the region
      memory's regression (2.6%) is the degradation a boundary is meant to
      prevent, and it is not lower than shuffled's (2.2%). Reported here
      with the EM difference from now on
- [ ] A6 preregistration 3, then the hotpot confirmation fold; 2wiki and
      musique after. On hold after A5
- [x] A7 memory version 2, aimed at the two causes A5 found (written 11
      Nov, before any run). Switched on per config
      (`configs/orchestrator_mem2_*.yaml`); the defaults are version 1, so
      the A5 configs and results stand as they are.
      Advice: a lesson is written from 4 failures of its type on 4
      different questions (the nearest waiting ones: step features for
      regions, question words for flat) in one reflection call that must
      cover all 4, and a rule is rejected if it uses a content word that
      occurs in fewer than half of its source steps (one retry that names
      the words, then the fixed sentence). The box spans the 4 points plus
      one sd on each side.
      Credit: while building, a decision where lessons apply shows none
      with probability 0.5 (the lessons are logged as held out, `held_ids`
      in the trace) and otherwise one of them at random. Utility is step
      success where shown minus where held out, in the same region; success
      is a step with no error of any type of its role, so a lesson that
      trades dropped for noise earns nothing. Retired when both counts
      reach 20 and utility <= 0; frozen, a lesson is shown only if
      utility - se > 0, one per decision.
      Same 1000 train questions, same exploration fold, same four arms
      against the main agent without memory, paired, scored by
      `scripts/memory_eval.py --tag mem2`.
      What would count, fixed now: (a) the build's pooled randomized
      comparison, step success where a lesson was shown against where the
      applying lessons were held out, above zero with its 95% interval: the
      direct test that the advice helps where it applies; (b) region EM
      above no memory and above shuffled on exploration, paired; (c)
      region above global and flat. If (a) is level, showing the lessons is
      not worth it whatever the frozen runs say. If no lesson is proven
      after the build, the frozen runs are skipped (they would equal no
      memory) and that is the result.
      Power, before the run: in A5 lessons applied at about 40% of subagent
      steps by the end, so about 600 shown and 600 held-out decisions; at a
      success rate near 0.62 that resolves about 5-6 points of step success
      (95%). Per-lesson effects will be resolved only when large.
      Mac, 14 Nov, at 75d4efc (`results/orchestrator_mem2_*`, the stores
      and their logs in `results/agent_memory/*2*`, scored into
      `results/orchestrator_mem2_hotpot_explore_{paired,steps}.csv`,
      `results/agent_memory/build_summary2.csv` and the two
      `*2_lessons.csv`). Exploratory.
      (a) fails. Over the region build, subagent steps shown a lesson
      succeed at 0.624 (n 1456) and steps where the lessons that applied
      were held out at 0.624 (n 1537): -0.000 [-0.035, +0.034]. Plans:
      0.949 (n 137) against 0.972 (n 144), -0.023 [-0.069, +0.022]. The
      flat build says the same (+0.006 [-0.027, +0.039] and -0.024
      [-0.061, +0.014]). The intervals rule out a gain above about 3.5
      points of step success, not a smaller one.
      (b) fails. Frozen on exploration against no memory (EM 0.540), Holm
      over the four arms:

      | arm      | EM    | diff, 95% CI            | lost / won | p    | Holm |
      |----------|-------|-------------------------|------------|------|------|
      | region   | 0.546 | +0.006 [-0.022, +0.034] | 24 / 27    | 0.78 | 1.00 |
      | shuffled | 0.544 | +0.004 [-0.020, +0.030] | 20 / 22    | 0.88 | 1.00 |
      | global   | 0.528 | -0.012 [-0.042, +0.018] | 32 / 26    | 0.51 | 1.00 |
      | flat     | 0.516 | -0.024 [-0.052, +0.004] | 33 / 21    | 0.13 | 0.53 |

      (c) in part, and not corrected: region against global +0.018
      [+0.000, +0.036] (7 / 16, p 0.093), against flat +0.030 [+0.004,
      +0.056] (15 / 30, p 0.036); against shuffled +0.002 [-0.022, +0.026]
      (17 / 18). So where a lesson is shown matters a little (regions lose
      less than show-all or similarity), what it says does not (shuffled
      is level with region), and nothing is above no memory.
      What changed from version 1: the advice is general now ("Only keep
      results directly answering the specific question asked", "Always
      include results that provide specific details about the subject and
      context of the question"); the region build ends with 14 active
      lessons of 67 written, 4 of them proven (1 planner, 1 dropped, 2
      noise); frozen they reach 0.85 of subagent steps and 0.33 of plans
      (0.42 and 0.04 in A5). More answers change than in A5 and they
      cancel: gain / regression 0.054 / 0.048 for region, 0.044 / 0.040
      shuffled, 0.052 / 0.064 global, 0.042 / 0.066 flat. One reading:
      advice this general says what the keep prompt already says, so the
      keep step's errors are not for lack of a rule.
      Fifth memory gate closed. A6 stays on hold. The memory line stops
      here unless the agent changes what there is to learn; the agent
      itself is Stage 6b

### Stage 6b: the agent itself (15 Nov)

Five memory gates are closed, and each of them tested a memory on agents
that were built once (4 Nov) and then held fixed, so that a change could be
credited to the memory. The agents are the one thing on this branch that
moved answers (the main agent over subagents is +0.054 EM over ircot on
exploration, p 0.004, not confirmed; against the ircot chosen below for
the confirmation, whose answer read shows paragraph titles, the round 3
agent is +0.021 [-0.006, +0.049] over both folds and the round 4 agent
+0.029 [+0.001, +0.056], on the folds their changes were chosen on), and nothing has been
done to them since. This stage works on the agent directly: find where
its answers are lost, change one thing at a time against a matched
control, choose on the exploration fold, confirm once.

Where the answers are lost (`src/eval/agent_losses.py`: every wrong answer
placed by where the gold answer string was; the string test is exact, per
sentence, so "read" is a lower bound). Hotpot exploration, n=500, before
any change:

| run                         | EM    | wrong | read | dropped | never retrieved | yes/no gold | EM where read |
|-----------------------------|-------|-------|------|---------|-----------------|-------------|---------------|
| ircot, 13.1 sentences       | 0.486 | 257   | 206  | -       | 49              | 2           | 0.500 (n 412) |
| single agent, 3.6 sentences | 0.526 | 237   | 167  | 39      | 24              | 7           | 0.580 (n 398) |
| main agent, 3.7 sentences   | 0.540 | 230   | 142  | 59      | 26              | 3           | 0.618 (n 372) |

read: a sentence the answer was read from holds the string. dropped:
retrieved by some search and not kept. The agents read better from a short
notebook (0.500 -> 0.618 where the string is there) and pay with the
questions where the string was retrieved and not kept (62 for the main
agent, EM 0.048 on them).

Three things in the main agent's traces:

1. Yes or no to a question with a span answer: 34 of the 467 span
   questions, 13 of them with the string in the notebook. It is not the
   agent's own: ircot 31, bm25 k=13 30, single agent 32, closed book 52. It
   comes with the answer prompt ("If it is a yes/no question, reply yes or
   no") on questions worded as statements ("This singer of A Rather
   Blustery Day also voiced what hedgehog?"). In the main agent it also
   breaks chains: a lookup answered "no" is pasted into the next lookup.
2. The final read never sees the subagents' answers; they only fill the #k
   of later lookups. On 27 of the 230 wrong answers a subagent's answer
   equals the gold (the last one on 26), so final-or-last would be 0.592.
   The last answer alone is 0.356 (A2b), so it cannot replace the read.
3. 89% of lookups search twice (ANSWER after the first search on 8%), and
   the second search adds 0.26 gold and 0.75 other sentences to the
   notebook. 7.2 calls and 4.1 s per question against 2.8 and 1.8 s for
   ircot.

Most of the dropped ones read as broken chains (12 traces read): an earlier
lookup answered wrong or "no", the next lookup asks about the wrong thing,
and the sentence with the answer is shown to a subagent that was not
looking for it.

Changes, each a config flag that is off by default (the 4 Nov runs and
every other mode come out as before; `test_answer_guard_and_final_notes`):

- [x] B1 `pipeline.final_notes`: the final read gets the lookups and the
      subagents' answers, written under the kept sentences inside the same
      answer prompt. No extra call. Not kept (round 1 below).
- [x] B2 `pipeline.answer_guard`: when a question does not read as yes/no
      and the reply is yes or no, the same context is read once more with
      the prompt's yes/no sentence replaced ("The answer is a name, a date,
      a number or a short phrase, not yes or no"); the second reply stands.
      Whether a question reads as yes/no is decided from its wording (it
      opens with an auxiliary, or has no question word), written on the
      first 20,000 hotpot train questions: that holds for 1226 of the 1231
      with a yes/no gold answer and 269 of the 18,769 others, and the guard
      leaves those alone. It sits in the answer read every mode shares, so
      it applies to lookups, final answers, ircot and single-hop alike, and
      an agent with it is compared with ircot with it. Kept (round 1
      below).
- [x] B3 `pipeline.sub_max_hops: 1`: one search per lookup, the cost
      reading of point 3. Not kept (round 1 below).

Runs, hotpot exploration, the same 500 ids (`.local_notes/queue16.sh`):
`orchestrator_notes`, `orchestrator_guard`, `orchestrator_notes_guard` and
`orchestrator_s1` against `orchestrator_hotpot_explore`, `hybrid_guard`
against `hybrid_ircot_hotpot_dev`; scored by `python -m
src.eval.agent_losses` into
`results/agent_dev_hotpot_explore_{losses,paired}.csv`.

What would count, written before the runs:

- Exploration is for choosing. With 60 to 90 discordant pairs a comparison
  on 500 questions resolves 3 to 4 points of EM, and these changes are
  expected to be worth 1 to 3, so most will not clear their interval alone.
  A change is kept when its paired EM difference against the same agent
  without it is above zero and neither F1 nor capped containment falls; all
  three are reported with intervals, McNemar, and Holm over the four agent
  arms. B3 is kept as the cheaper setting when its EM difference is not
  below -0.01.
- The guard's own effect is read on ircot (with against without). What the
  agent adds is read with the same answer step on both sides: the main
  agent with the guard against ircot with the guard.
- Bounds, so the result can be checked against them: the guard can win at
  most the 34 yes/no replies (+0.068; 31 for ircot), the notes at most the
  26 (+0.052), less whatever the notes break among the 118 answers that
  were right while the last lookup's answer was not.
- The guard prompt and the notes line were written once. A change of
  wording is a new arm with its own name, not an edit.
- The kept changes are frozen by 23 Nov. Preregistration 3 is written
  before the confirmation fold is touched, then one run on hotpot_confirm
  (n=1000): the frozen agent, the 4 Nov main agent, ircot with and without
  the guard. Nothing about the agent is reported as a result before that.

Round 1, Mac, 15 Nov at 6513a4e (`results/orchestrator_{notes,guard,
notes_guard,s1}_hotpot_explore_*`, `results/hybrid_ircot_guard_hotpot_
explore_*`, scored into `results/agent_dev_hotpot_explore_{losses,
paired}.csv`). Exploratory, n=500, paired, Holm over the four agent arms
against the 4 Nov main agent (EM 0.540, F1 0.658, 7.2 calls):

| arm           | EM    | diff, 95% CI            | lost / won | p     | Holm  | F1     | contain | calls |
|---------------|-------|-------------------------|------------|-------|-------|--------|---------|-------|
| notes         | 0.536 | -0.004 [-0.032, +0.026] | 30 / 28    | 0.90  | 1.00  | -0.011 | +0.006  | 7.2   |
| guard         | 0.544 | +0.004 [-0.006, +0.016] | 3 / 5      | 0.73  | 1.00  | +0.011 | +0.008  | 7.4   |
| notes + guard | 0.548 | +0.008 [-0.022, +0.040] | 30 / 34    | 0.71  | 1.00  | +0.004 | +0.020  | 7.4   |
| one search    | 0.492 | -0.048 [-0.082, -0.014] | 48 / 24    | 0.006 | 0.025 | -0.050 | -0.048  | 5.6   |

ircot with the guard is 0.508 against 0.486: +0.022 [+0.010, +0.036], 0 / 11,
p 0.001.

By the rule written above:

- B2 kept: EM above zero and neither F1 nor containment falls. On the agent
  that is two questions net and its interval holds zero; the clear effect
  is on ircot.
- B1 not kept: -0.004 against the agent without it, F1 down; on top of the
  guard +0.004 [-0.024, +0.034] (27 / 29) with F1 down 0.008.
- B3 not kept: -0.048, far under the -0.01 it was allowed. Two searches
  stay.

Against the bounds: the guard could win 34 on the agent and won 5 (lost 3),
31 on ircot and won 11; the notes could win 26 and won 28 while losing 30.

What the round says:

- The guard does what it was written for, and for the agent that is worth
  little. Yes or no to a span question falls from 34 to 4 (ircot: 31 to
  2). On ircot 11 of the 29 second reads come out right, all of them among
  the 18 where the string was in the context, and nothing is lost. On the
  main agent 4 of 30 do: 14 of the 30 had the string retrieved and not
  kept and 6 never retrieved, and none of those 20 can come out right from
  the same notebook. For the agent a yes or no to a span question mostly
  says that the notebook does not hold the answer, not that the read
  slipped. The 54 second reads on lookups repair no chain by themselves
  (dropped stays at 59).
- The notes make the final read follow the lookups. On 20 of the 28
  questions they win, a lookup's answer equals the gold; on 25 of the 30
  they lose, the final answer repeats a lookup's answer. The two cancel, as
  the bound said they could. By type, which was not planned: bridge 0.505
  -> 0.507, comparison 0.688 -> 0.656, where a lookup's answer is a value
  to compare and not the answer.
- The second search is worth about five points. With one search never
  retrieved goes from 26 to 46 and dropped from 59 to 71, and the notebook
  holds all the gold on 0.362 of the questions against 0.516, for 1.6
  calls and 1.25 s less per question. Point 3 above read it wrong: the 0.26
  gold sentences a second search adds are what those points are made of.
- With the same answer step on both sides the agent's margin over ircot is
  smaller than the 4 Nov number: the main agent with the guard against
  ircot with the guard is +0.036 [+0.002, +0.068] (29 / 47, p 0.05, Holm
  over the two guarded agents 0.10), against +0.054 [+0.018, +0.088] with
  neither. A third of the margin was ircot's fixable yes/no replies. The
  guarded pair is the one to carry forward.

After round 1 the agent is the 4 Nov main agent with the guard: EM 0.544,
F1 0.669, 7.4 calls. Nothing so far clears its interval on the agent.

Round 2, written 15 Nov before its runs. Two changes on the agent as it
stands after round 1 (the 4 Nov main agent with the guard):

- [x] B4 `pipeline.guard_context: wide`: the guard's second read goes over
      the step's kept sentences followed by everything else its searches
      retrieved (for a lookup its own searches, for the final read every
      search) instead of the same sentences again. The first read does not
      change, so only questions where the guard fires can move. Read for
      it: the 14 final reads in round 1 where the guard fired with the
      answer string retrieved and not kept. In all 14 the sentence had been
      shown to a subagent; the notebook held 2.9 sentences against 18.5
      retrieved; several chains had gone to the wrong entity at the first
      lookup ("Philadelphia Eagles" for an Oregon player, "Can't Buy Me
      Love" for The Soloist) or carried a non-answer into the next lookup
      ("Which American rock band did No information provided. play for?").
      ircot and single-hop already read everything retrieved, so the flag
      changes nothing there (`test_guard_wide_and_chain_notes`); no ircot
      run. Kept (round 2 below).
- [x] B5 `pipeline.final_notes: chain`: B1's notes only when the plan is a
      chain (a planned lookup refers to an earlier one with #k): 175 of the
      500 explore plans, all of them bridge. Parallel plans (194, with 92 of
      the 96 comparison questions) and one-lookup plans (131) get none.
      From round 1 split by plan shape after its runs: chain won 16 / lost
      11 with the notes alone and 18 / 8 with the guard; parallel 9 / 14
      both times; one lookup 3 / 5 and 2 / 5. Not kept (round 2 below).

The runs repeat exactly: on the 439 explore questions where the guard never
fired, the runs with and without it have identical completions, and the
notes runs match the plain ones up to the final read on all 500. So B5 on
the explore fold is known without running it (the guard's answers on
non-chain plans, the notes-with-guard answers on chain plans): EM 0.564
against 0.544, +0.020 [+0.000, +0.040] (8 / 18, p 0.076), F1 +0.015 [-0.001,
+0.032], containment +0.030 [+0.012, +0.050]. That number was chosen after
seeing the split and is not evidence. Hence a fresh fold:
`results/folds/hotpot_train_hard.txt` (`python -m data.folds --train-hard`),
500 hard train questions (the dev set is all hard), seed 17, none from the
first 20,000 train records (where the guard's wording rule was written) or
from the memory's train fold; 386 bridge, 114 comparison. It is for
choosing, like the explore fold; the confirmation fold stays untouched.

Runs (`.local_notes/queue17.sh`), all with the guard, scored by
`src.eval.agent_losses` into `results/agent_dev2_hotpot_{explore,
train_hard}_{losses,paired}.csv`:

- explore: `orchestrator_guard_wide` against `orchestrator_guard` (round 1)
- train_hard (`--split train`): `orchestrator_guard` as the control,
  `orchestrator_guard_wide`, `orchestrator_guard_chain`

What would count, written before the runs:

- B4 is kept when its paired EM difference against the guard agent, pooled
  over both folds (n=1000), is above zero and neither pooled F1 nor
  containment falls, and neither fold is below -0.01. Bound: it can move
  only questions where the guard fires; on explore that is 30 final reads
  (14 with the string retrieved and not kept, 10 with it kept, 6 never
  retrieved) plus what repaired lookups change downstream. A smoke run on 8
  of those 30 (selected, not an estimate) turned 2 wrong answers right and
  broke none.
- B5 is kept when its paired EM difference against the guard agent on
  train_hard is above zero and neither F1 nor containment falls. Its
  explore number is not counted. If the explore split holds it should give
  about +0.02 where 35% of plans are chains; train_hard has more comparison
  questions (23% against 19%), whose plans are parallel and get no notes.
- 500 questions resolve about 3 to 4 points and both effects are expected
  under that, so "above zero" chooses, it does not claim. If both are kept,
  the agent with both gets one train_hard run before the freeze.

Round 2, Mac, 15 Nov at 688e5a7 (`results/orchestrator_guard_wide_hotpot_
explore_*`, `results/orchestrator_guard{,_wide,_chain}_hotpot_train_hard_*`,
scored into `results/agent_dev2_hotpot_{explore,train_hard,pooled}_
{losses,paired}.csv`). The pooled row joins the two folds' runs with +,
which `src.eval.agent_losses` reads as one run. The train_hard runs first
failed on the queue passing `--split train` as one word, and ran from
`.local_notes/queue17b.sh`. Exploratory, paired against the guard agent on
the same ids (Holm over the two train_hard arms: 0.36 and 0.85):

| arm         | fold       | n    | EM            | diff, 95% CI            | lost / won | p     | F1     | contain | calls |
|-------------|------------|------|---------------|-------------------------|------------|-------|--------|---------|-------|
| wide read   | explore    | 500  | 0.544 → 0.562 | +0.018 [+0.006, +0.032] | 1 / 10     | 0.012 | +0.015 | +0.014  | 7.4   |
| wide read   | train_hard | 500  | 0.506 → 0.516 | +0.010 [-0.002, +0.022] | 2 / 7      | 0.18  | +0.011 | +0.010  | 7.5   |
| wide read   | pooled     | 1000 | 0.525 → 0.539 | +0.014 [+0.006, +0.023] | 3 / 17     | 0.003 | +0.013 | +0.012  | 7.4   |
| chain notes | train_hard | 500  | 0.506 → 0.510 | +0.004 [-0.016, +0.024] | 12 / 14    | 0.85  | -0.002 | +0.016  | 7.5   |

By the rule written above:

- B4 kept: pooled +0.014, F1 +0.013 [+0.005, +0.023], containment +0.012
  [+0.003, +0.021]; neither fold below -0.01 (+0.018, +0.010).
- B5 not kept: EM above zero, F1 down (-0.0025 [-0.018, +0.013]; containment
  +0.016). With only B4 kept there is no run with both.

What the round says:

- The wide read moves only what it was written to move. Where the guard
  does not fire the runs are identical (439 of the 500 explore questions,
  436 of the 500 train_hard), so the whole difference sits on the 125 where
  it does. Of the 17 wins, 12 had the answer string retrieved and not
  kept, 4 had it in the notebook and 1 never retrieved it (a repaired
  lookup changed what the next search found); 9 came from the final read
  alone, 2 from a lookup's second read alone, 6 from both. The guard's
  final reads that end right go from 4 of 30 to 12 of 29 on explore and
  from 11 of 38 to 15 of 37 on train_hard; dropped goes from 59 to 53 and
  from 61 to 56. The 3 losses are final reads with the answer in the
  notebook, where the wider context offered a related entity ("The African
  Queen" for White Hunter Black Heart, "9×20mmSR Browning Long" for .45
  ACP, "Umpqua Community College" for Oregon).
- Cost: the same calls and generated tokens, 60 more prompt tokens per
  question (2%). Wall-clock latency rose 0.8 s pooled, more than 60 prompt
  tokens account for; the throughput of the same generated tokens fell
  with it (94 against 73 tokens/s on explore), and the chain arm ran 0.45 s
  faster than its control the same night, so latency here is not read as a
  cost of the change.
- The chain notes make the final read follow the lookups, as B1's did. All
  14 wins had a lookup with the gold answer (wrong answers with a lookup
  right fall from 29 to 16); on 10 of the 12 losses the final answer
  repeats a lookup's answer, 4 of them the gold inside a longer or shorter
  phrase ("Foxfire magazine" for Foxfire, "Perri" for Fortunato Natale
  Perri Jr.) and 6 another answer. The explore number chosen after the
  split (+0.020) is +0.004 on fresh questions, which is what "not evidence"
  meant.

After round 2 the agent is the 4 Nov main agent with the guard and the
wide second read: EM 0.562 and F1 0.685 on explore, 0.516 and 0.673 on
train_hard, 7.4 and 7.5 calls. Pooled, the wide read's interval is above
zero; on train_hard alone it holds zero. Like everything here it was
chosen on these folds and is reported only after the confirmation run.
Wrong answers by place now: explore 219 (138 in the notebook, 53 dropped,
25 never retrieved, 3 yes/no gold), train_hard 242 (158, 56, 27, 1).

Round 3, written 16 Nov before its runs. The largest loss of the round 2
agent is still the read: 138 of the 219 wrong answers on explore had the
gold string in the notebook (73 of them another answer, 38 a span inside
the gold, 24 a span around it, 3 yes or no). Read for it: 15 of the 73
(seed 3).

- 8 of the 15 are the gold under another form: the name the question uses
  against the full name in the gold (Larry Drake, Janet Jackson, Arthur
  Conan Doyle, Edward Albee), a diacritic (Ibn Arabi), a list written
  another way (Royce da 5'9", Eminem), Detective Inspector against DI,
  "over 150 million" against "150 million copies". EM counts them wrong
  and F1 in part; nothing in the agent should chase them.
- 7 are misreads: a sentence taken for another paragraph's (the first
  episode of a season and another episode both appear as "Written by ...
  and directed by ...", and the answer read shows no titles, so which
  director goes with which episode is a guess), a comparison read
  backwards, the intermediate entity given for the answer (Royal
  Australian Navy for the defence force it is part of), "raised in" read
  for "born in", the place a barn is in given for the mill town near it, a
  year that is in no sentence (1970 for 1962), and two names joined by the
  guard's second read. In 4 of the 7 a lookup had answered right.

Two things in that, each counted over all 500:

- The answer read shows sentences without their paragraph's title, while
  the keep step shows "(title) text"; 0.53 of the notebook's sentences are
  not their paragraph's first. For ircot, which reads every retrieved
  sentence, EM where a sentence holding the answer string opens its
  paragraph is 0.558 (n 294), and 0.449 where none does (n 118). For the
  agent it is 0.623 against 0.643 on explore (n 260, 112) and 0.595
  against 0.540 on train_hard (n 262, 113). So titles should matter more
  for ircot than for the agent.
- A final answer whose string is not in the notebook it was read from
  (yes and no aside) is mostly wrong: 67 on explore, EM 0.224, against
  0.597 for the 382 whose string is there; 77 on train_hard, EM 0.182
  against 0.555. Of the 52 wrong on explore, 29 had the gold string in the
  notebook, 13 retrieved and dropped, 10 never retrieved (train_hard 34,
  19, 10 of 63). Lookups: 156 of 906 answers are not in the lookup's own
  sentences (train_hard 180 of 922). Non-answers ("not specified", "the
  context does not provide ...") are rare: no final answer on explore, 4
  on train_hard, 20 and 9 lookups, so a trigger for them alone has a
  ceiling of a few questions, as the counts under "From the traces" below
  also say.

Changes, each a flag off by default (`test_guard_unsupported_and_titles`):

- [x] B6 `pipeline.guard_unsupported`: with the guard on, its second read
      is also given to an answer whose normalized string is not in the
      context it was read from, on a question that does not read as
      yes/no. Same GUARD_PROMPT, the same wide context under
      guard_context: wide, and the second reply stands, as for yes or no;
      the guard's records say which of the two fired. It sits in the
      shared answer read, so ircot gets it too. Kept (round 3 below).
- [x] B7 `pipeline.answer_titles`: every answer read (a lookup's, the final
      read, the guard's second reads, ircot's and single-hop's answer)
      shows each sentence as "(title) text", the form the keep step
      already uses. The keep step and the ircot loop's own step are
      unchanged. No extra call. Not kept (round 3 below).

Runs (`.local_notes/queue18.sh`) against the round 2 agent
(`orchestrator_guard_wide`), whose round 2 runs are the control (the runs
repeat exactly):

- explore and train_hard: `orchestrator_guard_wide_unsup`,
  `orchestrator_guard_wide_titles`
- explore: `hybrid_guard_unsup` and `hybrid_guard_titles` against ircot
  with the guard (round 1), so the agent is also read against ircot with
  the same answer step on both sides

scored by `src.eval.agent_losses` into `results/agent_dev3_hotpot_{explore,
train_hard}_{losses,paired}.csv` and `results/agent_dev3_hotpot_pooled_
paired.csv`.

What would count, written before the runs:

- Each change is kept when its paired EM difference against the round 2
  agent, pooled over both folds (n=1000), is above zero, neither pooled F1
  nor containment falls, and neither fold is below -0.01 (B4's rule). Holm
  over the two arms is reported, not used to decide.
- If both are kept, the agent with both runs on both folds before the
  freeze, and is taken over the better one alone only if its pooled EM is
  above it with neither F1 nor containment falling.
- The ircot runs do not decide. They say what each change does to the
  shared step where no notebook is kept, and give the agent's margin over
  ircot with the same answer step on both sides.
- Bounds. B6 can move only questions where it fires: on explore 67 final
  reads (at most 42 to win, the ones with the gold string retrieved; 15
  right answers at risk) and 156 lookups, plus what a changed lookup moves
  downstream; about 0.45 more calls per question. B7 changes every read,
  so any question can move; by the counts above it should do more for
  ircot than for the agent. B6 is wrong if its second reads mostly repeat
  the first (calls for nothing) or break right answers written otherwise
  than in the context ("Royce da 5'9", Eminem"). B7 is wrong if answers
  turn into a paragraph's title where the question asks for something
  about it.
- Smoke, 12 explore questions, 8 of them chosen for a final answer not in
  the notebook (not an estimate): B6 won 2 and lost none (1970 -> 1962 among
  them) for 0.75 more calls on those questions; B7 won 2 and lost 2.

Round 3, Mac, 16-17 Nov at cf74cce, 23:37 to 02:33
(`results/orchestrator_guard_wide_{unsup,titles}_hotpot_{explore,
train_hard}_*`, `results/hybrid_ircot_guard_{unsup,titles}_hotpot_
explore_*`, scored into `results/agent_dev3_hotpot_{explore,train_hard,
pooled}_{losses,paired}.csv`). Exploratory, paired against the round 2
agent on the same ids (Holm over the two arms, pooled: 0.070 and 0.51):

| arm              | fold       | n    | EM            | diff, 95% CI            | lost / won | p     | F1      | contain | calls |
|------------------|------------|------|---------------|-------------------------|------------|-------|---------|---------|-------|
| unsupported read | explore    | 500  | 0.562 → 0.570 | +0.008 [-0.002, +0.020] | 2 / 6      | 0.29  | -0.0005 | +0.006  | 7.8   |
| unsupported read | train_hard | 500  | 0.516 → 0.526 | +0.010 [+0.000, +0.022] | 1 / 6      | 0.125 | +0.003  | +0.006  | 7.9   |
| unsupported read | pooled     | 1000 | 0.539 → 0.548 | +0.009 [+0.002, +0.017] | 3 / 12     | 0.035 | +0.001  | +0.006  | 7.8   |
| titles           | explore    | 500  | 0.562 → 0.554 | -0.008 [-0.038, +0.022] | 30 / 26    | 0.69  | +0.0005 | -0.006  | 7.4   |
| titles           | train_hard | 500  | 0.516 → 0.508 | -0.008 [-0.038, +0.022] | 31 / 27    | 0.69  | -0.003  | +0.000  | 7.5   |
| titles           | pooled     | 1000 | 0.539 → 0.531 | -0.008 [-0.029, +0.013] | 61 / 53    | 0.51  | -0.001  | -0.003  | 7.4   |

By the rule written above:

- B6 kept: pooled +0.009, F1 +0.001 [-0.008, +0.010], containment +0.006
  [-0.002, +0.014]; neither fold below -0.01 (+0.008, +0.010). It passes
  on point estimates; the F1 interval sits across zero.
- B7 not kept: pooled -0.008. With only B6 kept there is no run with both.

ircot on explore against ircot with the guard (Holm over the two: 0.063
and 0.029), then the agent against ircot with the same answer step on
both sides:

| ircot arm        | EM            | diff, 95% CI            | lost / won | p     | F1     | contain | calls |
|------------------|---------------|-------------------------|------------|-------|--------|---------|-------|
| unsupported read | 0.508 → 0.518 | +0.010 [+0.002, +0.020] | 0 / 5      | 0.063 | +0.007 | +0.006  | 2.9   |
| titles           | 0.508 → 0.546 | +0.038 [+0.010, +0.066] | 18 / 37    | 0.014 | +0.037 | +0.034  | 2.8   |

| answer step, both sides | agent vs ircot | diff, 95% CI            | lost / won | p     | F1     | calls      |
|-------------------------|----------------|-------------------------|------------|-------|--------|------------|
| guard (round 2)         | 0.562 vs 0.508 | +0.054 [+0.020, +0.088] | 26 / 53    | 0.003 | +0.042 | 7.4 vs 2.8 |
| guard, unsupported read | 0.570 vs 0.518 | +0.052 [+0.018, +0.084] | 25 / 51    | 0.004 | +0.035 | 7.8 vs 2.9 |
| guard, titles           | 0.554 vs 0.546 | +0.008 [-0.026, +0.042] | 39 / 43    | 0.74  | +0.005 | 7.4 vs 2.8 |

What the round says:

- B6 moves only what it fires on. Where it does not fire the runs are
  identical (366 of the 500 explore questions, 350 of the 500
  train_hard). It fired on 55 final reads and 125 lookups on explore (54
  and 156 on train_hard), against the 67 and 156 counted on round 2's
  traces before the runs, and the second read repeats the first on 55 of
  180 and 66 of 210, so most of the extra calls change the reply. Of the
  12 wins, 7 are another answer made right (1970 -> 1962, 331,000,000 ->
  9,984, 2 -> 33, 92.1 -> 1170 kHz AM, "Not applicable" -> alternative
  rock, Princeton University -> Rutgers University after a repaired
  lookup, Art Is... The Permanent Revolution -> Marlene) and 5 the same
  entity written another way or more fully (7 -> seven, 3 -> Three, film
  directors -> film director, Catholic Church -> Catholic Church in
  Ireland, French -> Lithuanian-born French); 4 came from the final read
  alone, 2 from a lookup alone, 6 from both. Of the 3 losses, 1 is such a
  rewrite the other way (Donald Trump Jr. -> Donald John Trump Jr.), 1 a
  lookup's "Not specified in the context provided." read again into a
  wrong name that the next lookup followed (Wendell Berry lost), 1 a
  misread (Boston -> American). The final reads it gave a second read end
  right on 14 of 55 and 10 of 54.
- Its cost sits where EM does not look. The test is on the whole answer
  string, so a short or reordered form of the gold (Edward Albee for
  Edward Franklin Albee III) fires it too, and the second read sometimes
  gives another entity: Edward Albee -> J. M. Barrie, Thomas Friedman ->
  Henri Bergson, November 23, 1946 -> Diana Quick, "Royce da 5'9",
  Eminem" -> Bad Meets Evil, Ulf Merbold -> Vladimír Remek, John Feldmann
  -> Nicole Jung. EM counted these wrong before and after; answers with
  F1 at least 0.5 lose 19 and gain 11 (-0.008 [-0.019, +0.003] pooled),
  which is why F1 barely moves while EM rises. That is the bound's
  "breaks answers written otherwise than in the context", on answers EM
  had already failed. A trigger by words (fire only when some word of the
  answer is in no sentence read), counted on the traces, would have left
  28 of the 109 final reads and 69 of the 281 lookups alone, sparing 6 of
  the 19 and losing 2 of the 12 wins: not an arm on its own.
- Cost: 0.39 more calls per question pooled (7.44 -> 7.83; about 0.45 was
  expected), 209 and 238 more prompt tokens (7% and 8%), 3 more
  generated. The round 3 runs ran at 92 tokens/s against 73 and 70 for
  the round 2 runs, so the arm is faster than its control in wall-clock
  (4.44 against 5.27 s); the same night the titles arm, with the round 2
  agent's calls, took 4.25 s, so B6 costs about 0.2 s a question there.
- B7 changes every read and moves far more: 104 and 113 answers change
  (B6: 40 and 39), and wins and losses nearly cancel (53 against 61).
  Where the answer string was in the notebook it lost 51 and won 32. 26
  of the 61 losses are a paragraph title given as the answer: the title's
  form of the gold (Awake (film), Linfield F.C., Macbeth (character),
  Kelly Osbourne for Kelly Lee Osbourne, Bill Clinton for William
  Jefferson Clinton) or another paragraph's title (Third and Indiana for
  The Soloist, Dogo Cubano for German Spaniel), which is what the bound
  said would make B7 wrong.
- For ircot, which reads every retrieved sentence, titles help as the
  counts before the runs said they would: 37 won, 18 lost. 8 of the wins
  were never retrieved by the sentence test, 7 of them because the
  answer is only a retrieved paragraph's title; the place test reads
  sentences, so "never retrieved" counts such titles (9 of ircot's 49
  wrong ones on explore, 3 of the round 2 agent's 25). 6 of the 18 losses
  are a title given as the answer. B6 on ircot: 5 won, none lost, 39
  second reads, 18 of them repeating the first.
- Most of the agent's margin over ircot on explore closes when ircot's
  answer read shows titles. With the same step on both sides it is
  +0.054 (guard) and +0.052 (with B6), and +0.008 when both read titles,
  which take 0.008 from the agent and give 0.038 to ircot. Computed after
  the runs, not written before them (`results/agent_dev3_hotpot_explore_
  ircot_titles_paired.csv`): the round 3 agent against ircot with titles
  is +0.024 [-0.014, +0.062] (43 lost, 55 won, p 0.27), F1 +0.004
  [-0.031, +0.039], containment +0.020 [-0.018, +0.058], at 7.8 calls and
  4.4 s against 2.8 and 1.8 s; the round 2 agent is +0.016 [-0.022,
  +0.054]. ircot with titles had not yet run on train_hard (below).

After round 3 the agent is the round 2 agent with B6: EM 0.570 and F1
0.684 on explore, 0.526 and 0.676 on train_hard, 7.8 and 7.9 calls. Like
everything here it was chosen on these folds and is reported only after
the confirmation run. Wrong answers by place now: explore 215 (136 in the
notebook, 48 dropped, 28 never retrieved, 3 yes/no gold), train_hard 237
(151, 57, 28, 1).

ircot's answer read, written 17 Nov before its runs. Titles in the answer
read gave ircot +0.038 on explore and the agent nothing (round 3), and
against ircot with titles the round 2 agent's margin is +0.016, not
+0.054. The agent should be confirmed against the best ircot found with
the same care as the agent, so ircot's answer read is chosen on both
folds by the rule the agent's changes follow.

Candidates, all ircot with the guard: the guard alone (G, round 1), G+B6,
G+titles, and G+titles+B6 (`configs/hybrid_guard_titles_unsup.yaml`).
With both, an answer that is only a paragraph's title counts as in the
context and gets no second read (`test_guard_unsupported_and_titles`).
Explore has G, G+B6 and G+titles from rounds 1 and 3.

Runs (`.local_notes/queue19.sh`, about 75 minutes): G+titles+B6 on
explore; G, G+titles, G+titles+B6 and G+B6 on train_hard. Scored by
`src.eval.agent_losses` into `results/ircot_read_hotpot_{explore,
train_hard}_{losses,paired}.csv` and `results/ircot_read_hotpot_pooled_
paired.csv`: each candidate against G, and the round 3 agent against
each candidate.

What would count, written before the runs:

- A candidate passes when, against G pooled over both folds (n=1000), its
  EM difference is above zero, neither pooled F1 nor containment falls,
  and neither fold is below -0.01 (the agent's rule). Of those that pass,
  the one with the highest pooled EM is the ircot the confirmation reads
  the agent against; a tie goes to the one with fewer changes; if none
  passes, G. The agent's runs do not enter the choice.
- Preregistration 3 then reads the frozen agent against that ircot as its
  main contrast, with ircot given the agent's own answer step (G+B6)
  beside it.
- Expected: titles were predicted before round 3 to help ircot more than
  the agent, and did on explore; chosen there, they should help less on
  fresh questions. B6 added +0.010 to ircot on explore with nothing lost.
  By the rule G+titles can pass on explore's +0.038 with train_hard near
  zero, as long as train_hard is not below -0.01.
- Smoke, round 3's 12 explore questions (8 chosen for an agent answer not
  in its notebook; not an estimate): G+titles+B6 gives G+titles' answer
  on 11, with 5 second reads (4 for an answer not in the context, 1
  yes/no); on the twelfth a sentence-long reply ("Shirley Temple was the
  17-year ...") became a name, still wrong. 4 right, as for G, G+B6 and
  G+titles.

ircot's answer read, results (Mac, 17 Nov at 71c9d24, 18:15 to 19:31;
`results/hybrid_ircot_guard_titles_unsup_hotpot_{explore,train_hard}_*`,
`results/hybrid_ircot_guard{,_unsup,_titles}_hotpot_train_hard_*`,
scored into `results/ircot_read_hotpot_{explore,train_hard,pooled}_*`).
Each candidate against G on the same ids (Holm over the three, pooled:
0.001, 0.001, 0.0004):

| ircot answer read | fold       | n    | EM            | diff, 95% CI            | lost / won | p      | F1     | contain | calls |
|-------------------|------------|------|---------------|-------------------------|------------|--------|--------|---------|-------|
| G+B6              | explore    | 500  | 0.508 → 0.518 | +0.010 [+0.002, +0.020] | 0 / 5      | 0.063  | +0.007 | +0.006  | 2.9   |
| G+B6              | train_hard | 500  | 0.464 → 0.478 | +0.014 [+0.004, +0.024] | 0 / 7      | 0.016  | +0.013 | +0.014  | 2.9   |
| G+B6              | pooled     | 1000 | 0.486 → 0.498 | +0.012 [+0.006, +0.019] | 0 / 12     | 0.0005 | +0.010 | +0.010  | 2.9   |
| G+titles          | explore    | 500  | 0.508 → 0.546 | +0.038 [+0.010, +0.066] | 18 / 37    | 0.014  | +0.037 | +0.034  | 2.8   |
| G+titles          | train_hard | 500  | 0.464 → 0.500 | +0.036 [+0.008, +0.066] | 18 / 36    | 0.020  | +0.036 | +0.032  | 2.8   |
| G+titles          | pooled     | 1000 | 0.486 → 0.523 | +0.037 [+0.017, +0.057] | 36 / 73    | 0.0005 | +0.037 | +0.033  | 2.8   |
| G+titles+B6       | explore    | 500  | 0.508 → 0.548 | +0.040 [+0.010, +0.070] | 19 / 39    | 0.012  | +0.033 | +0.030  | 2.9   |
| G+titles+B6       | train_hard | 500  | 0.464 → 0.506 | +0.042 [+0.014, +0.072] | 17 / 38    | 0.006  | +0.043 | +0.040  | 2.9   |
| G+titles+B6       | pooled     | 1000 | 0.486 → 0.527 | +0.041 [+0.021, +0.061] | 36 / 77    | 0.0001 | +0.038 | +0.035  | 2.9   |

All three pass: each is above zero pooled with F1 and containment up,
and no fold is below zero. The highest pooled EM is G+titles+B6, 0.527
against G+titles' 0.523, so the confirmation reads the agent against
ircot with `configs/hybrid_guard_titles_unsup.yaml`, with G+B6 beside
it. Wrong answers by place for it: explore 226 (182 in the context, 42
never retrieved, 2 yes/no gold), train_hard 247 (194, 52, 1); G's were
246 (195, 49, 2) and 268 (213, 54, 1).

What the check says:

- The choice between the two title candidates is narrow. Computed after
  the runs and not part of the rule, which sets each candidate against G
  (`results/ircot_read_hotpot_{explore,train_hard,pooled}_b6_on_titles_
  paired.csv`): G+titles+B6 against G+titles is +0.004 [-0.001, +0.009]
  pooled, 1 lost and 5 won (p 0.22), F1 +0.001, containment +0.002; on
  explore +0.002 with F1 -0.005 and containment -0.004, on train_hard
  +0.006 (0 / 3). B6's second read changed 18 answers on explore and 22
  on train_hard. The six that moved EM: 7 -> seven, a sentence -> Sean
  Yseult, 3 -> Three, directors -> film director, 3.2 billion -> €3.2
  billion, and Essex, right but not in its context, -> Waltham.
- The expectation that titles would help ircot less on fresh questions
  was wrong: +0.036 on train_hard against +0.038 on explore, with nearly
  the same wins and losses (36 / 18 against 37 / 18). For the agent they
  were -0.008 on both folds (round 3). B6 on ircot held as well: +0.014
  on train_hard with nothing lost (0 / 7), as on explore (0 / 5); on top
  of titles it adds the +0.004 above.
- Against the chosen ircot the round 3 agent is +0.021 [-0.006, +0.049]
  pooled (88 lost, 109 won, p 0.15), F1 +0.015 [-0.010, +0.041],
  containment +0.018 [-0.010, +0.045]; +0.022 [-0.016, +0.060] on
  explore and +0.020 [-0.020, +0.060] on train_hard. All of it is on
  bridge questions (+0.027 [-0.005, +0.059], n 790; comparison 0.000, n
  210). It costs 7.8 calls against 2.9, 3,180 prompt tokens against
  1,656 and 4.4 s against 1.8 s a question (both at 92 tokens/s): 2.7,
  1.9 and 2.4 times the chosen ircot.
- Against ircot with its own answer step on both sides (G+B6) the agent
  is +0.050 [+0.026, +0.075] pooled (57 / 107, p 0.0001), and against G
  +0.062 [+0.037, +0.087]. With titles in ircot's read it is +0.025
  [-0.003, +0.053] (G+titles, p 0.090) and +0.021 (chosen). So titles in
  ircot's read take more than half of the agent's margin over ircot with
  the same answer step, on both folds. Both sides were chosen on these
  folds, so none of these is an estimate for the confirmation fold.
- For preregistration 3, planning and not a result: with the discordance
  seen here (197 of 1000 questions) and a true margin of +0.021, an
  exact paired test at 0.05 on the 1000 confirmation questions would
  find the agent ahead of the chosen ircot about three times in ten
  (simulated); a margin of +0.03 about half the time, +0.04 about eight
  times in ten. Against G+B6 (+0.050) it would nearly always. The
  confirmation run is round 1's list (the frozen agent, the 4 Nov main
  agent, ircot with and without the guard) with G+titles+B6 and G+B6
  added.

Round 4, written 19 Nov before its runs, on the dropped answers (48 of the
round 3 agent's 215 wrong on explore, 57 of 237 on train_hard: the gold
string retrieved and in no kept sentence). Counted on the round 3 traces
with no model, both folds:

- Read by hand, 14 of the 57 on train_hard (seed 4): 4 are the gold under
  another form that an unkept sentence spells out in full (Nigel Godrich
  for Nigel Timothy Godrich, George Jones, fencing for fencer, Poma-Otis
  Transit Company without "of Farmington, Connecticut"), 1 a comparison, 1
  the bridge entity given for the answer (Dev Patel for the character he
  played), and 8 broke upstream of the keep step: a lookup's answer that
  is not an answer pasted into the next lookup ("He", the whole cast list
  of a film, "BoybandPH did not win American Idol."), a lookup the planner
  wrote garbled ("In what year was a company formed?"), or a lookup that
  named the wrong entity, so the gold sentence no longer answered the next
  lookup's question and was rightly not kept. The keep step missing a
  sentence that answers its own lookup is the smaller part.
- The lookups mostly come back wrong as well: the last lookup's answer is
  the gold in 3 of the 48 and 2 of the 57, though the last lookup was shown
  a sentence holding the gold in 43 and 49.
- A malformed lookup answer pasted into a later lookup (a pronoun, a
  refusal, more than 8 words): 12 questions on explore, 13 on train_hard,
  7 and 9 of them wrong. A last lookup that kept nothing: 40 and 34
  questions, 8 and 9 of them dropped. Each is a ceiling of a few questions.
- 36 of the 48 dropped on explore and 37 of the 57 on train_hard never got
  the guard's wide second read, the one step that reads what was not kept
  (12 of B4's 17 wins were answers retrieved and not kept).
- Notebooks of one sentence do worst: 27 on explore at EM 0.148 (23 wrong,
  12 of them dropped), 44 on train_hard at 0.409 (26 wrong, 16 dropped).
  Empty notebooks, which already fall back to everything retrieved, are at
  0.615 (n 13) and 0.500 (n 8); notebooks of 2-3 sentences 0.585 and
  0.518. Different questions, so not a comparison, but a two-hop question
  answered from one kept sentence is mostly answered wrong.

Changes, each a flag off by default (`test_keep_answer_sentence_and_thin_
notebook`). Both change only what the final read is handed: the keep
steps, the lookups and their answers are as in the round 3 agent, so
neither adds a call.

- [x] B8 `pipeline.keep_answer_sentence`: when a lookup's answer (yes and
      no aside) is in no sentence of the notebook so far, the first
      sentence its own searches retrieved, in retrieval order, that holds
      the answer string joins the notebook after the lookup's kept ones.
      When no keep step kept anything the final read still reads
      everything retrieved, as before (the smoke found the first version
      narrowing that read to the one added sentence). Kept (results below)
- [x] B9 `pipeline.final_min_notebook: 2`: the final read falls back to
      everything retrieved when the keep steps kept fewer than two
      sentences, as it already does when they kept none; the kept sentence
      is read first. Its trace says "+thin", and `agent_losses` counts what
      such a read was given as read. Kept (results below)

Runs (`.local_notes/queue20.sh`, about 2.5 hours) against the round 3
agent (`orchestrator_guard_wide_unsup`), whose runs are the control:
`orchestrator_guard_wide_unsup_keepans` and `orchestrator_guard_wide_
unsup_thin` on explore and train_hard, scored by `src.eval.agent_losses`
into `results/agent_dev4_hotpot_{explore,train_hard}_{losses,paired}.csv`
and `results/agent_dev4_hotpot_pooled_paired.csv`.

What would count, written before the runs:

- Each change is kept when its paired EM difference against the round 3
  agent, pooled over both folds (n=1000), is above zero, neither pooled F1
  nor containment falls, and neither fold is below -0.01 (B4's rule). Holm
  over the two arms is reported, not used to decide.
- If both are kept, the agent with both runs on both folds before the
  freeze (23 Nov), and is taken over the better one alone only if its
  pooled EM is above it with neither F1 nor containment falling.
- Bounds. B8 changes the final read on 65 questions on explore and 54 on
  train_hard (71 and 56 lookups add a sentence), 21 and 18 of them right
  now; it adds a sentence on 12 and 6 more whose keep steps kept nothing,
  where the read is everything retrieved either way. The added sentence
  holds the gold in 6 and 6 of the dropped ones, so its wins are few
  unless an added sentence also helps a read whose gold was already in
  the notebook. B9
  fires on 27 and 44 questions, 4 and 18 of them right; at most the 12 and
  16 dropped among them can be won, plus any of the 4 and 6 wrong with the
  gold already read. The A2 control (the single agent reading everything
  retrieved on every question, 0.032 below its notebook) says a wider
  read can cost; B9 gives it only to one-sentence notebooks.
- Expected: both small. B8 between 0 and +0.006 with few losses. B9 more
  on explore, where its notebooks do worst, than on train_hard; a loss on
  train_hard's 18 right ones would say a lone kept sentence is often
  enough. B8 is wrong if the added sentences mostly repeat what the final
  read already had; B9 is wrong if the wide read turns right one-sentence
  answers into another entity from the extra sentences.
- Smoke, 12 explore questions, 6 where B8 adds a sentence and 6 with one
  kept sentence, half of each right in round 3 (not an estimate): the keep
  steps, lookups and lookup answers are the round 3 agent's on all 12 for
  both arms. B8 changed one answer, wrong both times (261 -> 25,949 for
  35,124); its first version also narrowed one empty notebook's read of
  everything to the one added sentence (1974 -> 1964, both wrong), fixed
  as written above. B9 fired on 7 and changed no answer, saving one second
  read.

Round 4 results (Mac, 19 Nov at f55b232, 20:05 to 22:33,
`results/agent_dev4_hotpot_*`), each arm paired against the round 3 agent
(0.570 explore, 0.526 train_hard, 0.548 pooled):

| arm        | explore        | train_hard     | pooled d EM [95% CI]    | lost/won | p (Holm)      | d F1   | d contain | calls       |
|------------|----------------|----------------|-------------------------|----------|---------------|--------|-----------|-------------|
| B8 keepans | 0.574 (+0.004) | 0.534 (+0.008) | +0.006 [+0.000, +0.012] | 2/8      | 0.109 (0.125) | +0.001 | +0.006    | 7.80 / 7.83 |
| B9 thin    | 0.572 (+0.002) | 0.534 (+0.008) | +0.005 [+0.001, +0.010] | 0/5      | 0.062 (0.125) | +0.003 | +0.004    | 7.82 / 7.83 |

Both are kept by the rule (pooled EM above zero, F1 and containment not
falling, no fold below -0.01). Neither is significant alone (Holm 0.125
for both); at n=1000 the rule reads direction, as it did for B4.

- The keep steps and the lookup answers are round 3's on all 1000
  questions in both arms, as designed; only the final read moved.
- B8 added a sentence on 77 and 60 questions and changed the final read
  on 63 and 54 (21 and 18 right in round 3); on the other 14 and 6 the
  read was everything retrieved either way. Of its 8 wins, the added
  sentence holds the gold on 4 (Darkroom, Norman Foster and .44 caliber
  Bulldog revolver were dropped and are now read; Dorothy and the Wizard
  in Oz was read already), 3 are the gold's longer form after a changed
  context (William Jefferson Clinton, Sun Valley, Idaho, much less
  common), and 1 is a comparison (Sierra Nevada National Park). So 3 of
  the 12 dropped answers whose added sentence held the gold were won.
  Both losses are the guard's second read no longer running: the added
  sentence made a wrong first reply look supported ("2" for 33 Senate
  seats; "Nothing" for the first two cymbal notes, where round 3's first
  reply "No" had gone to the second read). The guard's final second read
  ran on 72 and 79 questions against 83 and 92, one call fewer each.
- B9 fired on 27 and 44 questions (4 and 18 right in round 3), as counted
  before the run. Its 5 wins are dropped answers now read: 1 on explore
  (137th) and 4 on train_hard (la Manche, and three comparisons: Sierra
  Nevada National Park, Humbug, Hot Rod). It lost nothing, so none of
  train_hard's 18 right one-sentence answers turned into another entity.
  The fold was guessed wrong: the gain is on train_hard, not on explore,
  where 12 of the 27 were dropped and 1 was won. Most of what it moved
  out of dropped stays wrong with the string read: dropped 48 -> 36 and
  57 -> 41, read with another answer 136 -> 147 and 151 -> 163.
- As run, B8 counts a sentence it added as added when a later lookup keeps
  it as well. On 2 explore questions that was the only kept sentence, so
  the read was everything retrieved rather than that sentence (wrong
  either way: "He" -> "Semyon Aronovich Gershgorin", "Walt Disney"
  twice). The docstring and the test now say so. It is left as run, so
  the agent frozen is the one measured.
- Next by the rule: both on (`configs/orchestrator_guard_wide_unsup_
  keepans_thin.yaml`, `.local_notes/queue21.sh`, about 75 minutes) on both
  folds, against B8, the better one alone (0.554 pooled against B9's
  0.553, one question). Both is taken only if its pooled EM is above
  0.554 with neither F1 nor containment falling against B8; otherwise the
  agent is round 3 with B8. B8's sentences do not count toward B9's two,
  and are read with the notebook before the rest.
- Expected, counted on the runs above before this one: on the 977
  questions where the two do not overlap (the 2 above go with B8), both
  has to repeat the run of the arm that fires, 549 right. The other 23
  (12 on explore, all wrong in round 3; 11 on train_hard) are open: 6
  right in round 3, 8 under B8, 8 under B9. Both is above B8 if 6 or more
  of the 23 are right, so it fails only if the overlap does worse than
  round 3 did; 0.555 to 0.557 if it does as round 3 or either arm did.
- Smoke, 12 explore questions (6 where both fire, 3 B8 only, 3 B9 only):
  the keep steps and lookups are round 3's on all 12, and the 6 where one
  fires repeat that arm's run (answer, calls, notebook). The 6 where both
  fire stay wrong, one with an answer neither arm gave ("nominated" for
  "multiple awards", where round 3's reply held the gold string and
  neither arm's did).

Both on, results (Mac, 19 Nov at 420bdd5, 22:44 to 00:00,
`results/agent_dev4_both_hotpot_*`):

| run  | explore | train_hard | pooled | against B8, d EM [95% CI] | lost/won | p     | d F1   | d contain |
|------|---------|------------|--------|---------------------------|----------|-------|--------|-----------|
| B8   | 0.574   | 0.534      | 0.554  |                           |          |       |        |           |
| both | 0.576   | 0.536      | 0.556  | +0.002 [-0.003, +0.007]   | 2/4      | 0.688 | +0.002 | +0.002    |

Both is taken by the rule (pooled EM above 0.554, F1 and containment not
falling against B8). It is a narrow choice: 4 won, 2 lost. Against the
round 3 agent it is +0.008 [+0.000, +0.016], 4 lost / 12 won, p 0.077,
F1 +0.003, containment +0.008, at the same calls (7.80 against 7.83).

- The count before the run held on answers: the 977 questions outside the
  overlap give 549 right, each with the answer of the arm that fires. It
  missed 8 B8-only questions (7 explore, 1 train_hard) whose second kept
  sentence B8 had added first. Counted as one kept sentence (the as-run
  rule above), they read as thin under both, with the same answer on all
  8 and one call fewer on 2.
- The 23 overlap questions: 7 right (6 in round 3, 8 under B8, 8 under
  B9). Both keeps B8's William Jefferson Clinton and B9's 137th on explore
  and Sierra Nevada National Park on train_hard. It loses two that round 3
  and both arms had right: Marco Restrepo (a comparison answered with the
  other entity, Porno for Pyros) and 1170 kHz AM (cut to "1170 kHz").
  Against B8 its wins are 137th and B9's three train_hard wins outside
  the overlap (la Manche, Humbug, Hot Rod).
- Dropped answers, against round 3: 48 -> 32 on explore, 57 -> 36 on
  train_hard, most of them now read with another answer (136 -> 149,
  151 -> 167). The guard's final second read ran on 68 and 74 questions
  (83 and 92 in round 3).
- The agent for the freeze is now `configs/orchestrator_guard_wide_unsup_
  keepans_thin.yaml`: explore 0.576 / F1 0.685, train_hard 0.536 / F1
  0.681, 7.8 calls. It is the control for the next round.
- Against the confirmation's ircot (G+titles+B6), computed after the runs
  (`results/agent_dev4_vs_ircot_hotpot_pooled_paired.csv`) on the two
  folds the changes were chosen on, so optimistic and not a result:
  +0.029 [+0.001, +0.056], 83 lost / 112 won, F1 +0.019, containment
  +0.026, at 7.8 calls against 2.9; against G+B6 +0.058 [+0.033, +0.083].
  At this margin (195 of 1000 discordant) a simulated exact McNemar on
  1000 confirmation questions finds the agent ahead about 5 times in 10
  (3 in 10 for the round 3 agent), and the selection on these folds makes
  that an upper figure.

Next, in the order the losses table gives after these runs (one or two
changes a day, each written here with its control before it runs):

- [x] the read when the notebook does not hold the answer: B4 in round 2.
      The other arm (the notebook followed by every other retrieved
      sentence on every question) is not run: the A2 control already read
      the single agent's full context, and it was 0.032 below the notebook
      (p 0.093)
- [x] notes for chains only: B5 in round 2
- [x] ircot's answer read for the confirmation: G+titles+B6 by the rule
      (17 Nov, above); the round 3 agent is +0.021 [-0.006, +0.049]
      against it over both folds, +0.050 against G+B6
- [ ] dropped (59; 53 after round 2, 48 after round 3, mostly at the last
      keep step, "From the traces" below): a lookup whose answer is not
      an answer ("no", "the context does not say") still fills the next
      lookup; a subagent that kept nothing hands the final read nothing.
      Round 4 (19-20 Nov, B8 and B9) above: both kept and taken together;
      dropped 48 -> 32 on explore and 57 -> 36 on train_hard, most of them
      now read with another answer. The malformed lookup answers pasted
      into the next lookup (12 and 13 questions) are untouched
- [ ] the plan: the cap of 3 cutting the last lookup off, lookups that
      restate the question
- [ ] never retrieved (26, all bridge; 28 after round 3): what the second
      search asks. 3 of round 2's 25 are a retrieved paragraph's title,
      which the sentence test counts as never retrieved (round 3)
- [ ] read with the string there and another answer (68; 73 after round 2,
      71 after round 3): B6 kept and B7 not in round 3. Of the 15 read
      before round 3, 8 were the gold under another form, which nothing
      in the agent should chase
- [ ] the comparison gain (+0.104 over the single agent, n=96) on the
      confirmation fold's comparison questions
- [ ] 2wiki and musique once hotpot is frozen

#### From the traces

Counts read from committed traces with no model, on the explore fold only,
for the agent as it stands after round 2 (`orchestrator_guard_wide_hotpot_
explore`, n=500, EM 0.562, 219 wrong). `python -m src.eval.agent_traces
--run orchestrator_guard_wide_hotpot_explore --out agent_traces_hotpot_
explore` writes `results/agent_traces_hotpot_explore_{chain,dropped,plan,
keep,search2}.csv` (`test_agent_traces`). Gold is read only to place and
score. "Holds the answer" is the exact string test of `agent_losses`, so it
undercounts paraphrases and overcounts sentences that mention the string
in passing.

16 Nov, the dropped ones (53 wrong answers whose string was retrieved and
not kept):

- Who saw it. In 48 of the 53 the sentence with the string was shown to
  the subagent of the last lookup, in 5 only to an earlier one, never to
  none; in 17 it was shown to two or more. In 49 of the 53 no lookup came
  back with the gold answer (4 had the last lookup right and the final
  read still answered otherwise). So the loss is mostly one keep step:
  the last subagent was shown the sentence and kept something else or
  nothing.
- 9 of the 53 are comparison questions whose gold is one of the two
  names compared ("Ehretia", "Rolling Stone"). There the string is in
  nearly every sentence about that entity, so "dropped" says only that
  the final answer picked the other one; these are not keep-step losses
  and belong with the read. The 44 bridge questions are the dropped
  ones proper.
- What broke the chain first (`first_break`, in this order): the plan cut
  by the cap of three 0, a non-answer pasted into the next lookup 2, a
  pasted answer naming no gold paragraph 11, the last lookup answering
  with a non-answer 1, a bridge question planned with no #k 24 (12 one
  lookup, 12 parallel lookups), nothing found 15 (the 9 comparison ones
  among them). Over all 500 questions the cap cut 8 plans (3 wrong, none
  dropped) and a non-answer was pasted on in 9 (4 wrong). The two
  failures the Next item names are real ("Which American rock band did
  No information provided. play for?") and rare.
- The pasted-answer test is a word match against the gold paragraphs'
  titles, and read by hand on 20 of its 43 calls over all questions it
  is rough: 8 had clearly gone to another entity (Zoey Deutch for Lucy
  Fry, Rosie O'Donnell for Kelly Osbourne, Can't Buy Me Love for The
  Soloist), 4 were the right entity under another form (IBF, cranberries,
  George Harrison for two of his songs, Gollum), 8 could not be told from
  the strings. If the sample is representative, roughly half of its 11
  dropped are real wrong turns.
- Where the sentence sat in the list (every keep step over the 500
  questions that was shown a sentence holding the string): kept 300 of
  401 times at position 1 (0.75), 131 of 246 at 2-3 (0.53), 70 of 186 at
  4-6 (0.38), 32 of 106 at 7 or lower (0.30). Lists are 7 to 10
  sentences. Position here is also the retriever's score for the
  lookup, so the count cannot tell a keep step that reads the top of the
  list from a lookup whose answer sentence ranks low because it matches
  the lookup badly. The dropped questions' steps are spread over all
  four positions (26, 27, 28, 28 steps), so a low position is not most of
  the story either.

The plan's shape against the outcome (`_plan.csv`): bridge questions
planned as one lookup 127, EM 0.551; parallel 102, 0.500; chain 175,
0.531; comparison parallel 92, 0.696 (4 planned as one). Wrong answers by
place differ little by shape (dropped 12, 12, 20 for bridge one /
parallel / chain). Nothing here says one shape loses more; a bridge
question planned without a chain is not visibly worse off than one with
it, and the shapes are chosen by the model on different questions, so
this is not a comparison.

The never-retrieved ones (25 wrong, 42 lookups, `_search2.csv`): the
second search adds words the lookup did not have on 32 of the 42 (76%),
against 603 of 864 (70%) for every other question; it names a gold
paragraph's title on 18 of 42 (43%) against 449 of 864 (52%); 4 lookups
answered after one search. The second searches of these questions do not
look different in kind from the rest; what they lack is not visible in
these counts.

What this suggests for the next change, not decided here: of the 44
bridge questions that lost a retrieved answer, 31 show no break upstream
by these tests (24 planned without a chain, 6 with nothing found, 1 with
a last non-answer), and roughly half of the 11 rough off-path ones
probably belong with them; against that, 2 carried a non-answer forward
and the cap cut none. So a change aimed at the chain (refusing to paste a
non-answer, raising the cap) has a ceiling of a few questions on this
fold, and the larger target is the last keep step itself. The A2 control
already says that reading everything retrieved on every question cost
the single agent 0.032, so an arm there would have to be conditional, as
B4 is, and would need a control for position (the same list in another
order) before the keep step's misses are read as reading order.

17 Nov, the read with the string there (the largest loss after round 3:
136 of the round 3 agent's 215 wrong answers on explore,
`orchestrator_guard_wide_unsup_hotpot_explore`). `python -m
src.eval.agent_reads --run orchestrator_guard_wide_unsup_hotpot_explore
--out agent_reads_hotpot_explore` writes `results/agent_reads_hotpot_
explore_{kind,position}.csv` (`test_agent_reads`); the same for ircot with
the guard and B6 (`hybrid_ircot_guard_unsup_hotpot_explore`) into
`results/agent_reads_ircot_hotpot_explore_*`. Each wrong reply is given the
first kind that holds, by string tests (the module's docstring has the
order); I then read all 71 that are not yes/no or a span inside or around
the gold.

- By the string tests: 3 yes or no, 38 a span inside the gold and 24
  around it (40 of these 62 repeat a lookup's answer; not read by hand
  here), 9 the gold's words in another order or with words missing ("Leo
  Eric Varadkar", "West Point Foundry"), 9 one of two things the question
  names, 2 an entity of the question, 15 a gold paragraph's title, 3
  another retrieved paragraph's title, 15 in the sentence that holds the
  gold string, 15 in another sentence of the notebook, 3 not in the
  notebook. In 24 of the 136 a lookup's answer equals the gold.
- Read by hand, the 71 come to four groups. About 26 are the gold under
  another form or at another grain: the 9 by words, 2 of the 9 two-way
  picks (712 Fifth Avenue for "712 5th Avenue", a diacritic), 3 titles
  that are the answer's own name in another form (Joe Scarborough,
  Paige O'Hara), and 12 of the 38 in the notebook's sentences (Chicago
  for Wilmette, 19th for nineteenth, novelist for author, Civella for
  Kansas City crime family). About 16 stop one step short on the path:
  11 of the 15 gold titles (St James's Palace for the building opposite
  it, Diana Quick for her birthday, Portal for the series it is part of)
  and 5 of the 38 (Umpqua Community College for Oregon, the navy for the
  defence force). In 5 of the 11 the last lookup had come back with the
  gold, and the final read, which does not see lookups' answers, gave
  the entity of the first. 7 give the other of two compared things (one
  against a gold that is a year);
  in 4 of them the lookups' values settled it (47 against 48, 62 against
  81) and the read went the other way. About 22 are another entity of the
  asked type taken from what was read (the publisher for the developer,
  another band member, another episode's director).
- Position (`_position.csv`): EM where the string was read is 0.678 when
  the first sentence holding it opens the notebook (n 171), 0.616 at 2-3
  (146), 0.545 at 4 or later (55); by notebook size it is flat (0.627,
  0.633, 0.641 for 1-2, 3-4, 5+ sentences). Notebook order follows the
  lookups, so a late answer sentence is mostly a chain's last lookup;
  this is not a reading-order effect until the same notebook is read in
  another order.
- Against ircot (190 wrong with the string read, of 412): a gold title
  18 (the agent 15 of 372, about 4% of the questions where the string is
  read for both), a two-way pick 15 (9), another sentence read 36 (15).
  Stopping one step short is not the agent's own; reading the other
  compared thing and taking another sentence's entity happen less in the
  notebook than in everything retrieved.

What this suggests, not decided here: about 88 of the 136 (the 62 spans,
the 26 forms) are answers EM does not credit and nothing in the agent
should chase, so the read has some 45 questions that a change could move
on this fold. The two that are specific are 5 bridge answers where the
last lookup had the gold and the final read answered with the bridge
entity, and 4 comparisons whose lookups held the values that decide
them. Both are what B1's notes hand the final read, and B1 and B5 were
not kept because the read then follows wrong lookups as well; a ceiling
of about 9 questions on 500 is under what one fold resolves.

20 Nov, the never retrieved and the plan, on the agent as it stands after
round 4 (`orchestrator_guard_wide_unsup_keepans_thin_hotpot_explore`, EM
0.576, 212 wrong, 28 of them with the gold string in no retrieved
sentence). `python -m src.eval.agent_misses --run orchestrator_guard_wide_
unsup_keepans_thin_hotpot_explore --out agent_misses_hotpot_explore`
writes `results/agent_misses_hotpot_explore_{never,restate}.csv`
(`test_agent_misses`). Paragraph titles are gold-derived and read here only
to place.

- What the 28 are (all bridge). 4 have the gold as a retrieved paragraph's
  title (Compliance, "We'll Burn That Bridge", and two where the reply is
  the title's longer form: Donald John Trump Jr., Sir Frank P. Lowy). None
  is missing from the pool's sentences altogether. In the other 24 the
  string sits in a gold paragraph, and 18 of those paragraphs were
  retrieved, only not the sentence with the answer; 6 were not retrieved
  at all. The other gold paragraph was retrieved in 23 of the 24.
- The 18 partly retrieved: the answer sentence is not its paragraph's
  first in 14 of 18, and a sentence next to it was retrieved in 15. Read
  through, they are mostly sentences that do not name their subject ("She
  served as Chief of Protocol", "It was written by Max Martin, ..."), so a
  search naming the entity finds its first sentence and not the one that
  answers. Where the paragraph's title was first named: in the question 4,
  a lookup 1, a search query 2, a lookup's answer only 4, never 7. The 7
  never named are mostly chains that went to another entity first
  (Alonzo Mourning for Muggsy Bogues, Innerspace for Alien, Tariq Khan
  for Farooq Sheikh, I-85 on the way to Wytheville).
- The 6 never retrieved: named in no lookup or query in any of them, in a
  lookup's answer only once (Darling River: the right bridge entity, a
  one-lookup plan that answered after one search). 4 of the 6 are
  one-lookup plans, and in Hibbing the parallel plan found the town and
  had no lookup left to ask its county.
- So the second search is not where these are lost. On 23 of 24 the
  first hop's paragraph was found; what is missing is a search for the
  answer's entity (the 6, and the 7 never named) or the one sentence of an
  already retrieved paragraph that answers (most of the 18, 15 with a
  neighbour retrieved). The two overlap and a count cannot separate them.
- Lookups that restate the question (a planned lookup with at least 0.8
  of the question's content words): 105 of the 404 bridge plans, 73 of
  them one-lookup plans. EM 0.543 against 0.549 without one; one-lookup
  bridge plans 0.548 (n 73) against 0.593 (n 54), parallel 0.421 (19)
  against 0.530 (83), chain 0.692 (13) against 0.543 (162). The model
  chooses the plan on different questions, so these are not comparisons,
  and none of the gaps is outside what 500 questions resolve. The cap of
  three cut 8 plans on this run, 3 of them wrong.

What this suggests, not decided here: never retrieved is 28 of 212, and
the largest part of it (15 of 18 partly retrieved) is a sentence next to
one already retrieved. A change there would sit in what a search
returns, not in the agent's steps, and would lengthen every list a keep
step is shown; its ceiling on this fold is about 15 questions before the
keep step and the read take their share. The restating and cut plans give
no target by these counts.

22 Nov, the read with the string there on the agent after round 4, and what
round 4 moved into it (explore only; the largest loss of
`results/agent_dev4_both_hotpot_explore_losses.csv`, 149 of 212 wrong, 136 of
215 for the round 3 agent). `python -m src.eval.agent_reads --run
orchestrator_guard_wide_unsup_keepans_thin_hotpot_explore --out
agent_reads_dev4_hotpot_explore` and `python -m src.eval.agent_moves --base
orchestrator_guard_wide_unsup_hotpot_explore --arm orchestrator_guard_wide_
unsup_keepans_thin_hotpot_explore --out agent_moves_dev4_hotpot_explore`
write `results/agent_reads_dev4_hotpot_explore_{kind,position}.csv` and
`results/agent_moves_dev4_hotpot_explore_{moves,moved,kind}.csv`
(`test_agent_moves`). `agent_reads` now reads a "+thin" notebook as the
final read does (the notebook, then everything else retrieved), as
`place_of` already did; the round 3 and ircot tables it wrote come out
byte-identical.

- Outcome by outcome, round 3 to round 4 (`_moves.csv`): 284 right in both,
  1 read and 3 dropped now right, 1 right now dropped. Of the 48 dropped, 31
  stay dropped and 14 are now wrong with the string read; the 135 read and
  wrong in both and these 14 are the 149.
- The 14 (`_moved.csv`, 8 bridge, 6 comparison): the sentence with the
  string reached the read through B9's fallback on 8, through B8's added
  sentence on 5, through an empty notebook on 1. The reply is the same as
  round 3's on 9 of the 14, so on those the read was handed the sentence
  and answered as before.
- Read through: the 6 comparisons are the case set apart on 16 Nov (the
  gold is one of the two names compared, so the string is in the other
  entity's sentences): 4 give the other name (Xanthoceras for Ehretia,
  Shonen Jump for Rolling Stone), 1 the gold's longer form (J. G. Farrell
  for Farrell), 1 a name the question does not hold as written. Of the 8 bridge ones, 2 had the last lookup
  come back with the gold and its sentence added by B8, and the final read
  gave the first lookup's entity (Brian Doyle-Murray for Bill Murray;
  George Walter Hunter Macfarlane for Charles Hastings Judd, a question
  that asks for both); 2 are the gold at another grain or form
  ("requiring only men to register for the draft" for Conscription,
  "Centre" for Canadian professional ice hockey centre); 1 is cut short
  ("King" for King George IV); 3 are another entity from what was read
  (25,949 for 35,124, University of Phoenix Stadium for Mercedes-Benz
  Superdome, Newark for lower Manhattan).
- Over all 149 by the string tests (`_kind.csv`, round 3's in brackets):
  spans inside or around the gold 63 (62), the gold's words 9 (9), the
  other of two compared things 13 (9), a gold paragraph's title 16 (15),
  in the gold's sentence 16 (15), in another sentence read 20 (15), the
  rest 12 (11). A lookup's answer equals the gold on 26 (24).
- Position (`_position.csv`): EM where the string is read is 0.685 when the
  first sentence holding it opens what was read (n 165), 0.613 at 2-3 (160),
  0.500 at 4 or later (70; round 3 0.545, n 55). B9's reads put the string
  after the notebook, so the late bucket grew with them; as on 17 Nov this
  is not a reading-order effect without the same context in another order.

What this suggests, not decided here: round 4 moved 17 dropped answers on
this fold into what the final read sees and 3 of them became right; on 9 of
the other 14 the reply did not change at all. 6 of the 14 are comparisons
whose "dropped" was a label, and 2 more are the gold under another form, so
the move left about 6 bridge answers that a read could still get, 2 of them
the pattern of 17 Nov (the last lookup right, the read answering with the
bridge entity), beside the 5 read then on the round 3 agent. The read with the
string there is the largest loss, but the share of it a change could move
is about what 17 Nov found (some 45 of 136), with the comparison picks a
little larger (13 against 9) and nothing new in kind. These are counts on
the fold the changes were chosen on, and a reply that does not change
says nothing about why the read passed over the sentence.

23 Nov, final answers that repeat a lookup's answer, and whether the record
was in front of the read (explore only; prompted by arXiv 2611.27139 in the
paper log, where a senior agent rarely adopts a planted error when the
correct record is in front of it). B1 and B5 lost mostly where the final
answer repeats a lookup's; this asks whether those reads had the gold string
in the notebook. `python -m src.eval.agent_echoes --out agent_echoes_hotpot_
explore` (the run list is in `scripts/run_all.sh`) writes `results/agent_
echoes_hotpot_explore_{echo,paired,adopt}.csv` (`test_agent_echoes`). A
repeat is the reply equal (EM) to a lookup's answer, yes and no aside; the
notes runs match their controls up to the final read on all 500 questions,
notebooks included, so both reads of a pair were given the same sentences.

- B1's 30 losses against the 4 Nov agent (`_paired.csv`): 22 repeat a
  lookup's answer (14 the last one's, 8 an earlier one's), 3 are yes or no,
  5 repeat none (15 Nov counted 25 with the yes/no replies that repeat a
  lookup's "no"). The gold string was in the notebook on all 22: the record
  was there and the control read it right from the same sentences. Of the
  28 wins, 20 repeat the last lookup, which had the gold on all 20 (2 of
  them with the string dropped from the notebook). With the guard on both
  sides: 21 of 27 losses repeat a lookup, all 21 with the gold in the
  notebook; 22 of 29 wins repeat a right last lookup.
- As a rate (`_adopt.csv`): where the control read right, the gold string
  was in the notebook, and the last lookup came back with something else
  (not yes/no), the notes read gave a wrong lookup's answer on 20 of 78
  (0.26); bridge 19 of 54 (0.35), comparison 1 of 24, where the lookups'
  answers are values to compare rather than candidate answers. Where the
  control was wrong and the last lookup had the gold, the notes read took
  it on 20 of 26 (0.77). With the guard: 19 of 83 (bridge 18 of 57) and 22
  of 27. These are the agent's own lookups on its own questions, not a
  planted error, so the paper's 2.9% is no benchmark for them; the bridge
  read follows a wrong lookup about a third of the time with the right
  sentence in front of it.
- A test with no gold does not tell the two apart: the last lookup's
  answer string is in a sentence read on 58 of the 78 wrong-last exposures
  and 17 of the 20 adopted, against 22 of 26 and 18 of the 20 taken when
  the lookup was right. The wrong lookups are mostly read off the notebook
  too (another entity's sentence, the intermediate entity), which is the
  2611.25071 point about checking against the evidence the chain itself
  gathered.
- The agent as it stands after round 4 (no notes; `_echo.csv`): 212 wrong,
  74 equal the last lookup's answer (52 with the gold string read, 11
  dropped, 11 never retrieved), 22 an earlier lookup's (17 read; on 7 of
  the 22 some lookup had the gold, the stop-one-step-short pattern of 17
  Nov), 109 none (77 read), 7 yes or no. Of the 222 final answers that
  repeat the last lookup, 148 are right (0.667) against 74 of 183 that
  repeat none (0.404).

What this suggests, not decided here: the losses of B1 and B5 are reads
that had the record and followed the lookup, not lookups whose errors no
read could have caught, so a change there sits in how the lookups'
answers are offered to the read, not in the lookups. The bound is narrow:
on this fold handing the read the lookups gains about 20 questions and
loses about 20, and the one gold-free signal counted here (the lookup's
answer is in a sentence read) does not separate them. An instruction to
check each lookup's answer against its sentences, the arm the paper's
result points to, would be scored on the questions whose lookups were
right as well as on the wrong ones, since its duty prompts moved their
controls too. Counted on the fold B1 was chosen on, after its runs.

24 Nov, the dropped ones on the agent after round 4 (explore only; 32 of the
212 wrong in `results/agent_dev4_both_hotpot_explore_losses.csv`, 29 bridge
and 3 comparison, against 53 on the round 2 agent read on 16 Nov). `python
-m src.eval.agent_drops --run orchestrator_guard_wide_unsup_keepans_thin_
hotpot_explore --out agent_drops_hotpot_explore` writes `results/agent_
drops_hotpot_explore_{rows,asked,dangling}.csv` (`test_agent_drops`). It
reads the deciding lookup, the last one whose keep steps were shown a
sentence holding the gold string, and asks what that lookup was about; I
then read all 32 with their gold.

- By the tests (`_asked.csv`): the deciding lookup is the last lookup on 27
  of 32; every holder it was shown is in a gold paragraph; it opens its
  paragraph on 17 and sat first in the list on 8. The lookup names the
  holder's paragraph on 4, the holder's paragraph has the gold for its
  title on 5 (KKR, Comedy Film Nerds, Mr. Church, two comparisons), and on
  23, all bridge, the lookup is about something else. It kept another
  sentence of the holder's paragraph on 11, only other paragraphs on 19,
  nothing on 2. Its answer was read from a sentence it kept on 23, from
  the sentence B8 added for it on 5, from neither on 4. First breaks
  (`agent_traces.first_break`): off_path 9, no_chain 15, intact 8; no cut
  plan, no non-answer pasted on, no last non-answer. So the two chain
  failures this item names, and the cap, account for none of the 32.
- Read by hand, the 32 come to four groups. 7 are the gold under another
  form or a gold that is not an answer, with the right thing in the reply
  or in a lookup's answer (Kohlberg Kravis Roberts & Co. for KKR & Co,
  Oregon Ducks, NBC, Jackson, a gold that is a sentence, a gold that is
  the question's own words, and "U.S. Senator from Maryland" from the last
  lookup read as "D-Maryland"). 13 are plans that never ask the last hop:
  one-lookup or parallel bridge plans whose lookups stop at the bridge
  entity or ask the whole question (Buckingham Palace for the monarch
  whose residence it is, YouTube Rewind for who made it with YouTube, the
  player for the pinball machine named after him, the month of "the
  annual documentary film festival" asked before the festival was
  found), 2 of them with a later lookup pointing back as "that building"
  and "this island" without #k, and one that asks who Benvolio is when the
  question asks whom Romeo slays. 8 went to another entity or the wrong
  kind of answer upstream and pasted it on ("Who is the brother of 2007?",
  Eduardo Romero for Roberto De Vicenzo, Florida's seats for the national
  midterms, a phrase for a tribe). 4 asked the right lookup and answered
  it wrong from what it was shown: Mr. Church, "The Young Ones" given as
  the other series its own cast was in, and two comparisons whose two
  lookups came back with the same value ("World No. 51" twice, "3" twice).
- Pointing back without #k (`_dangling.csv`, a word list, so rough): 8 of
  the 500 plans, 7 of them real by reading ("This Is Elvis" is a title),
  EM 0.625, 3 wrong (2 dropped, 1 read).

What this suggests, not decided here: after round 4 the dropped answers
are no longer mostly the last keep step. 16 Nov read 48 of 53 as shown to
the last lookup's keep step and called that step the larger target; on
the round 4 agent the deciding lookup asked about the answer's own entity
and missed it on about 4 of 32, after B8 and B9 had moved 17 of the round
3 agent's 48 into what the final read sees (22 Nov). What is left is mostly the plan (13, all bridge) and wrong turns
upstream (8), with 7 that EM does not credit. A planner change aimed at
the 13 has a ceiling of about 2.6 points on this fold before the keep
step and the read take their share, and it would touch every plan: 16 Nov
found one-lookup and parallel bridge plans no worse by EM than chains
(0.551 and 0.500 against 0.531, n 127, 102, 175, on the round 2 agent;
chosen by the model on different questions, so not a comparison). The dangling references
are a ceiling of 3. The hand groups are one reading of 32 questions, the
counts are on the fold the round 4 changes were chosen on, and the ASKED
test is a word test on titles, so it calls a lookup that names the
entity in another form "other".

### Stage 6-7, only if Stage 5 shows something

- [ ] active probing when applicability is uncertain, on the cost-regret curve
- [ ] answer-level confirmation at n=2000 with the closed-book control
- [ ] a small evidence graph (title links over the pool, with the query-free
      degree control) as one more strategy the memory can choose

### Found while running on the Mac (4 Oct)

- [x] `failures.paragraph_of` and the twin builder read the paragraph title
      from the chunk id; musique ids are `p{idx}::{title}`, so failure
      typing never named a paragraph there and `make_twins` yielded no
      musique twins. Fixed 5 Oct: `paragraph_of` reads the title from
      either id form and the twin builder uses it. The musique failure
      table and twins were built on the Mac on 5 Oct (Stage 3 and Stage 5
      above); the hotpot and 2wiki tables are unchanged by it
- [x] `make_twins` iterated a set of tokens before its seeded tie-break, so
      the twin file depended on PYTHONHASHSEED. The three iterations are
      sorted now, and the hotpot twins of the first 500 questions are
      byte-identical under PYTHONHASHSEED 0 and 7 (339 distractor, 364
      distractor_para, 78 alias). Against the committed
      `hotpot_twins_para.csv` that is one distractor_para twin short (365
      there); whether a twin exists does not depend on the order, so the
      difference is in the dev file this machine prepared from the hf
      mirror, not in the sort. The 2wiki twins were rebuilt on the Mac on
      5 Oct: the sorted builder gives the same 435 twins under PYTHONHASHSEED
      0 and 7 but not the 4 Oct file (a few distractor_para twins draw other
      sentences inside a tie), so the twin arms log, `results/2wiki_twins.csv`
      and the twin stream were rebuilt from it; the 4 Oct twins file and
      arms log are kept outside the repo
- [x] `make_twins` checked a distractor candidate against the pool by chunk
      id only. Musique pools repeat the same paragraph under different
      p{idx} ids, so 54 of the 484 musique distractor_para twins carried a
      byte-identical copy of their own gold paragraph as a distractor, and
      retrieving it counted as a miss. Since 5 Oct a candidate is skipped
      when its id or its text is already in the pool; hotpot and 2wiki twins
      of the first 500 questions are byte-identical before and after. The
      alias twin is refused on paragraph pools (its rules read the sent_idx
      0 chunk, which on musique is whatever paragraph sits at pool position
      0: four musique aliases, two of them wrong). `run_qa`'s paragraph
      recall on a musique twins log is not reliable either: it maps musique
      ids to `p{idx}`, and the added paragraphs keep their source index, so
      a distractor `p9::Other` counts as gold paragraph `p9`; every twin
      table uses all_gold_in_topk on chunk ids, which is unaffected
- [x] `stream.py`: gains and tests were against the bm25 default only. Now
      every row carries the same paired contrast against the best fixed
      arm (`best_fixed`, `gain_vs_best`, `p_vs_best_max`), each utility /
      boundary pair its own paired test (`pair`, `gain_vs_pair`,
      `p_vs_pair_max`), the twin tables a repeat rule started from the best
      fixed arm (`repeat_best_*`) and the best arm's own hit on the twin
      (`best_*`), and a pooled test over the distinct pairs (each twin
      once, its mean hit over the orders that scored it, paired bootstrap
      against the repeated arm and against the best fixed arm,
      `diff_vs_repeat_*`, `diff_vs_best_*`) beside the largest per-order
      McNemar p. Numbers under Stage 5
- [x] `arms.py`: the routed table has a spread over half-splits and no paired
      test, and the fold-split table prints a spread of 0.0 for its one split.
      10 Nov: every routed row (the feature routers and the neighbour votes)
      now carries a per-split McNemar against the best single arm on the
      same score half (`p_vs_single_max`, `_median`) and a pooled paired
      bootstrap over questions on each question's mean outcome over the
      splits that scored it (`gain_lo`, `gain_hi`, `n_questions`), so the
      one-split fold tables get an interval and a p where gain_sd is 0.0.
      Hotpot rebuilt here (bm25s 0.3.10, bge-small on cpu: the replay
      matches the committed arms trace on 1500 of 1500 hops and every old
      cell of the four tables reproduces): against always-hybrid 0.634 no
      row's interval clears zero upward, the widest being the pool router
      +0.004 [-0.007, +0.015] and the dense discordant vote at k=20 +0.003
      [-0.007, +0.013]; the tf-idf vote at k=5 is below it, -0.009 [-0.017,
      -0.001]; the largest per-split p is 1.0 on every row. The musique and
      2wiki tables, and the fold-split tables, wait for the Mac logs
- [ ] `lookalike.py`: agreement counts both-miss and the random baseline is
      not matched on question type; add a row for pairs where the arms
      disagree on both questions, and a type-matched baseline
- [x] `backfill.py --allow-mismatch` exited 0 whatever the mismatch count;
      the exit code is now 1 whenever a hop differed, written or not
- [ ] 48 of the first 1500 2wiki pools hold one paragraph twice (same chunk
      ids), so a bm25 or dense top-10 can carry a repeated id there
- [ ] the 2wiki within-template pairs are not written (induced drift is,
      12 Nov: `src/memory/drift.py`, numbers under the Stage 5 baselines
      item; the 2wiki direction waits for the Mac's log and the zip)
- [x] on musique the shuffled boundary memory creates 17-43% fewer lessons
      than the real one while firing as often; on hotpot and 2wiki it
      creates about as many. Traced (12 Nov, boundary:primary_hits:r2 on
      the rebuilt logs, orders 13 and 17): the count is split children, not
      experiences. Fresh exemplars are fewer under the real reveals (221 /
      224 against 276 / 304 on musique), and the real memory makes up the
      difference with 836 / 756 splits against 395 / 348. The chain is
      expand -> exception -> split: with real outcomes the recoverable
      failures sit near each other in feature space, so a box grows by
      absorbing supporters just outside it (5208 / 4848 expansions against
      2674 / 2329; 13 supporters per lesson against 10), a grown box then
      holds more contradicting points, a contradiction at a would-fire
      moment with supporters on every side becomes an exception rather than
      a cut (3613 / 3249 against 1675 / 1475; the would-fire share of
      contradictions is 0.50 either way), and the fourth exception splits
      the lesson in two. Shuffled, the supporters are scattered, boxes grow
      half as often and the chain runs at half the rate. Hotpot runs the
      same chain with a smaller gap (expansions 2110 / 1875 against 1799 /
      1587, splits 289 / 241 against 247 / 193), which is why its counts
      looked level. So the larger lesson count on musique is the operators
      churning on the clustering of the real outcomes, worth about two
      points of rate there (0.286 / 0.288 against 0.264 / 0.270) and a
      higher mean lesson utility (0.14 / 0.11 against 0.07 / 0.06), and not
      more distinct lessons

## Writing

- [ ] measurement study: short paper draft from Part A; related work from
      `docs/paper_log.md`
- [ ] memory study: results notes per stage as they land, negative results
      included; decide on a write-up after Stage 5
- [ ] pin versions and model revisions; run manifest per results file;
      `scripts/run_all.sh` end to end with a short smoke path
      18 Nov, run_all.sh rewritten (the pins and the manifest are still
      open). Stages in order: data, retrieval, qa, interventions,
      reader_ctx, agents, analyses, memory, plots; qa to agents generate and
      are marked [model], the rest read data/processed and committed files
      only. `--no-model`, `--model-only`, `--only a,b`, and `--smoke`
      (SMOKE_N=5 questions per stage, every write to a scratch copy of
      results/, never to results/ itself). The runs are listed by config
      and fold as the committed files name them (Part A on the explore and
      confirm folds, the arms logs, Stage 6's agents and memory gates, the
      Stage 6b rounds on explore and train_hard; no confirmation agent run
      until preregistration 3). The queue files in .local_notes are not in
      the repo, so the model stages are read back from the results names,
      the configs' run names and the paired tables' base/arm columns, and
      only the stages without the model could be checked here. On linux
      (bm25s 0.3.16, numpy 2.5.5, pandas 3.1.1, torch 2.15 cpu):
      - data: the hotpot train file is now fetched by `data.download
        --datasets hotpotqa_train` (official host, hf mirror as fallback;
        the mirror's file came out 566,426,227 bytes), and `data.folds`
        skips a dataset with no processed file instead of failing on 2wiki.
        All the committed folds come out byte-identical, train_hard
        included.
      - analyses: 72 committed tables rewritten in place, all
        byte-identical (decomposition, criteria, paired, stop audit,
        interventions, reader context, every agent_dev*/ircot_read*
        losses and paired table, agent_traces, agent_reads, the A5 and A7
        memory_eval tables), 12 minutes. The four decomposition plots
        differ in bytes only (matplotlib build). The musique dev criteria
        table was made without --data-file (containment against the gold
        string, no aliases); the script keeps that and says so. Not run
        here: everything 2wiki, and `src.eval.prereg`, which reads the
        2wiki dev file.
      - plots: `scripts/plots.py` redraws the 63 2x2 and 41 per-hop plots
        and the pareto plot from the committed csv and json; the two I
        compared by eye match.
      - retrieval and memory: smoke only (5 questions; backfill replays
        them with 0 hops differing from the trace). The full memory stage
        (backfill of the 1500 hotpot and 2417 musique arms logs, then the
        Stage 1b-5 tables) is not run yet, and the twin and _mac tables are
        not in the script.
      19 Nov, pins and a manifest (`docs/pins.md`, `requirements-linux-cpu.txt`,
      `scripts/manifest.py` into `results/manifest.csv`, `test_manifest`).
      `run_all.sh --dry-run` lists every python step with its stage, and
      `--record FILE` logs each file a step wrote. The data, retrieval,
      analyses and plots stages ran recorded here (13 minutes): the folds
      and the 72 analysis tables came out byte-identical again, the plots in
      bytes only. The two bm25 retrieval tables did not: 16 of 7405 hotpot
      rows and 1 of 2417 musique rows change, the same rows with bm25s
      0.3.10 (the pin) and 0.3.16, so it is the platform and not the
      version; they stay the Mac's. The manifest has the 504 committed
      results files: 188 rows observed being written here, 344 credited by
      name to commands not run here (304 model rows from the dry run, 40
      for the 2wiki and memory steps), and 74 files that no command in
      run_all.sh writes: the 2wiki memory tables, the _mac, replay, twin,
      dense-lookalike and noqtype memory variants, `arms_foldsplit/`, the
      serving sweep and its two corners, `folds/hotpot_train_mem.txt` and
      the 2wiki folds. Pinned here: the linux environment, the three model
      ids with the hub's current revisions (none changed on the hub since
      2024), the sha256 of the raw and processed data. To fill on the Mac:
      its `pip freeze`, the model revisions in its hf cache, the 2wiki zip's
      hash (commands in `docs/pins.md`); then `--model-only --record` would
      turn the by-name rows into observed ones.
      20 Nov, `notebooks/demo.ipynb`, committed executed (python 3.12,
      linux): reads committed results only. The 2x2 of the four
      hotpot confirmation runs recomputed from their `_qa.csv` and asserted
      equal to `hotpot_confirm_decomposition.csv`'s cell, with Wilson
      intervals; the decomposition of the cell for the hotpot, musique and
      2wiki confirmation tables; per-hop precision and recall of hybrid
      ircot on hotpot and musique confirm (2wiki has no ircot run on its
      confirmation fold, only bm25 k=10 and k=13); the round 4 agent's
      losses on explore and train_hard and today's never-retrieved split,
      both marked exploration only. Next: `docs/readme_draft.md`.
      21 Nov, `docs/readme_draft.md`, written by `scripts/readme_draft.py`
      from 27 committed results files, each named beside its numbers
      (`test_readme_draft`); README.md is untouched. Scope first (distractor
      pools, one 7B model, one Mac), then the confirmation-fold tables and
      the preregistered tests, the decomposition, the interventions as the
      ablation, the stop audit, the five memory gates with their controls,
      the agent marked exploration only, reproduction and Limitations. The
      fixed text's directions (ircot halving the hotpot cell, shortcut
      largest under both orders, bridge-only at closed-book level on hotpot
      and musique, no memory arm above none) are checked against the files
      and stop the script if a new run breaks one. Two gaps the draft
      names: D3's deepening policy on the confirmation log has no committed
      table (only `docs/plan.md`; the draft shows D1/D2's answer-level
      side, paired from the two qa files), and the musique and 2wiki routed
      tables have a spread over half-splits but no interval or test. Not
      in it: the serving sweep (README's own section stands), the 2wiki
      agent and ircot runs that do not exist.
