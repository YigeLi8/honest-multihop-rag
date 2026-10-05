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

- [ ] `memory:` block in `configs/base.yaml`, off by default
- [ ] `src/memory/experience.py`: one record per (question, hop): strategy,
      retrieved ids/scores/ranks, gold hit/miss, hop / cumulative / aligned
      recall, features, final outcome row; run header with git sha and config
- [ ] `src/memory/features.py`: model-free features only, schema versioned and
      frozen (query length, hop, rare-token ratio, titles named in the query,
      top-1 score, margin, score entropy, bm25-dense disagreement, new chunks
      this hop, previous-hop top score)
- [ ] `src/memory/backfill.py`: rebuild the same records from committed traces
      by re-indexing each pool; assert recomputed top-k equals the trace
- [ ] hooks in the pipeline and `run_qa`, guarded by `memory.enabled`
- [ ] `scripts/check_baseline.py`: memory off vs memory logging must give
      identical traces and csvs (timing columns aside); tests for all of it

### Stage 1b: what is there to learn (no model calls)

- [ ] bm25 / dense / hybrid first-stage outcomes per question on full dev,
      three datasets; discordance table; best-arm-per-question ceiling
- [ ] how predictable is the best arm from the frozen features (held-out AUC),
      and how much is just the question template (2wiki)
- [ ] look-alike pairs: within-template pairs on 2wiki with discordant
      outcomes; natural pairs on hotpot by question similarity (expected
      sparse, reported as such); constructed twins (`data/make_twins.py`):
      distractor sentences sharing bridge-entity tokens, alias substitution

### Stages 2-4: the baselines

- [ ] 2. similarity memory over past experiences (top-k neighbours vote)
- [ ] 3. failure memory: failure type + what recovered it; rule-based failure
      typing with a 50-case hand audit
- [ ] 4. static conditions: a fixed predicate per lesson, and the same as
      prose read by the generator; if a fixed rule captures most of the
      best-arm ceiling, stop here and report that

### Stage 5: boundaries that revise themselves (the claim under test)

- [ ] per-lesson applicability model (predicate, then logistic / small tree),
      decide-then-reveal online protocol, K=5 seeded stream orders
- [ ] operators narrow / expand / exception / split / retire with a revision
      log; never overwrite silently
- [ ] baselines on the same stream: shared online router, scalar utility,
      always-probe, fixed hybrid; induced drift (switch dataset mid-stream)
- [ ] metrics: repeated-failure rate, false application, false rejection,
      boundary precision / recall against the known best arm, adaptation
      speed, revision rate, regret, retrieval cost
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
