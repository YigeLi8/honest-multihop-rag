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
- [ ] `docs/preregistration.md`: hypotheses, assignment order and one
      alternative order, correctness criteria, n, tests, pass/fail rules
- [ ] confirmation-fold baselines: hotpot_confirm (closed-book, bm25 k=10,
      bm25 k=13 matched context, hybrid ircot, + rerank), 2wiki_confirm and
      musique_confirm (closed-book, bm25, matched context); musique hybrid
      single-hop and ircot max_hops 4
- [ ] partition tables and bars per dataset on the confirmation folds, with
      the alternative assignment order and the evidence-criterion ladder (all
      gold sentences / answer-bearing gold / all gold paragraphs / any gold /
      answer string in context)
- [ ] correctness beyond EM: F1 >= 0.5 and capped containment with aliases,
      reported beside EM in every 2x2
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
question is known, and regret is exact. EM is secondary: it does not move with
retrieval in this harness.

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
- [ ] backfill the hybrid and ircot traces with bm25 and dense shadow arms

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
