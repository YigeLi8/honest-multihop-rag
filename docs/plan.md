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
relabelled: the two ends of its latency range (8-bit k=20 and 4-bit k=5,
rerank off) are re-run with the streaming runner in `results/serving_corner/`
and summarised by `scripts/serving_corner.py`. EM and retrieval_correct match
the old rows (0.53 / 0.85 and 0.46 / 0.45); the old tok/s column rebuilds to
4.4 and 9.5 against 4.7 and 10.1, i.e. it was answer tokens over prefill plus
decode and mostly prefill; per-step decode is 32.8 and 60.9 tok/s. What is
left for the README rewrite: the bullets still quote the old column, the
pareto x axis still says throughput, and the `tokens_per_s` that run_qa logs
counts n tokens over n-1 steps (43-45% high on five-token answers), which the
runner's field comment does not say. A small graph arm moves to Part B as one of the retrieval strategies.

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
      first 800. That is enough changed pairs for the stream (next item);
      the 2wiki template pairs still wait for the zip.

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
      decide-then-reveal online protocol, K=5 seeded stream orders. The
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
      as the held-out tables of Stage 1b said. The lesson memories do worse
      the more structure they have: the best of them is the fixed-region
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
      operators make it worse, because there is no region of this feature
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
      move when the arm's fortune does, not more operators. Still open:
      the same table on musique (the arms log has to be rebuilt; dense on
      cpu over 2417 questions with 20 paragraphs each is hours here, so on
      the Mac), induced drift, and features that are not frozen
      process statistics
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
- [ ] baselines on the same stream: shared online router, scalar utility,
      always-probe, fixed hybrid; induced drift (switch dataset mid-stream);
      a query-only router (the same router on the query-stage features
      alone), since matched query-only controls have erased retrieval-signal
      gains elsewhere; and a no-persistence control (decide each question
      from the current features with nothing kept across questions), which
      beat every persisted-skill method in the held-out skill study (paper
      log, 2026-10-01). All but the drift are rows of `hotpot_stream.csv`
      (numbers above): the query-only router equals the primary-hits router
      (0.622 vs 0.619), and the no-persistence rules are the `rule:` rows.
      Drift waits for the musique log
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
- [ ] ablation: remove the operators and the log, keep the features; if
      nothing changes, the structure is decoration. Done as the `utility`
      against `boundary` pairs at every stage and radius, and `utility+lr`
      against `boundary+lr`: on hotpot removing the operators does not
      merely leave the rate unchanged, it raises it (0.626 against 0.565 at
      the query stage, radius 2); with the per-lesson logistic inside the
      region the order reverses (0.516 against 0.554), both well below
      always-hybrid. The structure is worse than decoration
      on these features; whether it is on any features is what the twins
      stream and the musique run have to say

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
