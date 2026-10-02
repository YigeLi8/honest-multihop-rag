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
- [ ] partition tables and bars per dataset on the confirmation folds, with
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
      alternative order hotpot's is lower (0.053 against 0.064). Still open:
      the alternative order for every run in the decomposition tables, and
      the ladder.
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
- [ ] stop audit from the ircot traces: hop at which each gold sentence first
      entered context, stop reason, correctness
- [ ] README rewrite: scope in paragraph one (distractor pools, one model, one
      device), confirmation-fold tables, limitations, reproduction steps
- [ ] intervention arms on the exploration fold (gold-only, gold minus bridge,
      gold minus answer sentence, distractors only), if time

Cut from this branch: the selector arm, the model-size sweep, the concurrency
ladder, energy, the CUDA leg. The serving sweep stays as it is in the README,
relabelled with the corrected throughput definition once one corner re-run is
in. A small graph arm moves to Part B as one of the retrieval strategies.

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
- [ ] the same identity check with real generation on an ircot config, and
      with hybrid and reranked primaries (needs the Mac's GPU)
- [ ] backfill the hybrid and ircot traces with bm25 and dense shadow arms.
      Backfill the Mac's traces on the Mac: on linux with bm25s 0.3.11 the
      replay of bm25_hotpot_dev differs from the trace on 89 of 500 hops and
      bm25_musique_dev on 11, every one a different order inside a run of
      equal scores (16 of them at the top-k boundary, so the id set differs
      too); all-gold-in-top-k never flips. Pin bm25s in requirements.txt to
      the Mac's version once that is checked. A fresh bm25_arms_hotpot_dev
      run on linux (bm25s 0.3.12, bge-small on cpu) gives the same four
      Stage 1b tables as the Mac's to the last digit, so the question-level
      outcomes do not depend on the tie order.

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
      hybrid 0.825, oracle 0.891. 2wiki needs the manual zip.
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
      nothing either. Musique says the same: the router over the frozen
      features gains at most 0.004 over always-dense (0.326 vs oracle
      0.411). One caution the size control was for: on musique each arm's
      hit is predicted at AUC 0.69-0.71 from qtype alone, which is the
      composition shape, i.e. how many gold paragraphs must fit in top-5;
      the pairwise routing AUC stays at 0.46-0.62, so that is difficulty,
      not a routing signal.
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
      situation" pairs the boundary study needs. Next: a harder distractor
      twin (sentences sharing tokens with the whole bridge paragraph, more
      of them), then the 2wiki template pairs once the zip is here.

### Stages 2-4: the baselines

- [ ] 2. similarity memory over past experiences (top-k neighbours vote).
      At retrieval level on hotpot it is already in `src/memory/arms.py`
      (`question_knn` rows of `results/hotpot_arms_routed.csv`): the k
      nearest fit-half questions by tf-idf cosine vote with their own
      outcomes, and the arm with the most neighbour hits is chosen on the
      held-out half. k=5 lands at 0.625 and k=20 at 0.628 against
      always-hybrid 0.634 (gain -0.009 +- 0.007 and -0.006 +- 0.010), the
      same nothing as the feature router. On hotpot, look-alike questions
      by surface text do not say which retriever to use. Musique: k=5 gets
      0.334 against always-dense 0.326 (gain 0.008 +- 0.005 over ten
      halves, oracle 0.411), the only positive number in either table and
      too small to claim without the confirmation fold. Left to try before
      calling the stage: dense question neighbours, and the vote restricted
      to neighbours with a discordant outcome. Both tried on hotpot
      (`question_dense_knn*` and `*_discordant` rows of
      `results/hotpot_arms_routed.csv`): dense neighbours k=5 / 20 land at
      0.628 / 0.634, the discordant-only vote at 0.628 / 0.632 by tf-idf and
      0.633 / 0.637 by embedding, against always-hybrid 0.634; the best of
      the eight is +0.003 +- 0.006. The same nothing. Stage 2 on hotpot is
      closed: nearest past questions, by any key and with any vote, do not
      say which retriever to use. Musique with the two new votes is the
      remaining run before the tick.
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
- [ ] 4. static conditions: a fixed predicate per lesson, and the same as
      prose read by the generator; if a fixed rule captures most of the
      best-arm ceiling, stop here and report that. CAVE-Mem (paper log,
      2026-10-02) is this baseline in print with a cross-fitted utility veto
      and abstention, so the arm gets both, and any per-lesson utility
      reported anywhere here is cross-fitted (estimated from blocks that
      exclude the question being scored)

### Stage 5: boundaries that revise themselves (the claim under test)

- [ ] per-lesson applicability model (predicate, then logistic / small tree),
      decide-then-reveal online protocol, K=5 seeded stream orders
- [ ] operators narrow / expand / exception / split / retire with a revision
      log; never overwrite silently
- [ ] baselines on the same stream: shared online router, scalar utility,
      always-probe, fixed hybrid; induced drift (switch dataset mid-stream);
      a query-only router (the same router on the query-stage features
      alone), since matched query-only controls have erased retrieval-signal
      gains elsewhere; and a no-persistence control (decide each question
      from the current features with nothing kept across questions), which
      beat every persisted-skill method in the held-out skill study (paper
      log, 2026-10-01)
- [ ] metrics: repeated-failure rate, false application, false rejection,
      boundary precision / recall against the known best arm, adaptation
      speed, revision rate, regret, retrieval cost. Every rate that
      conditions on a lesson firing is reported beside the unconditioned
      contrast over the whole stream at matched retrieval cost (paper log,
      2026-10-02: conditioning on the trigger does not identify the
      invocation effect)
- [ ] ablation: remove the operators and the log, keep the features; if
      nothing changes, the structure is decoration

### Stage 6-7, only if Stage 5 shows something

- [ ] active probing when applicability is uncertain, on the cost-regret curve
- [ ] answer-level confirmation at n=2000 with the closed-book control
- [ ] a small evidence graph (title links over the pool, with the query-free
      degree control) as one more strategy the memory can choose

## Writing

- [ ] measurement study: short paper draft from Part A; related work from
      `docs/paper_log.md`
- [ ] memory study: results notes per stage as they land, negative results
      included; decide on a write-up after Stage 5
- [ ] pin versions and model revisions; run manifest per results file;
      `scripts/run_all.sh` end to end with a short smoke path
