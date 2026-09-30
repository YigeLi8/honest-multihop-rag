# Plan

Working notes for the `illusion-analysis` branch. Items get ticked as they land;
anything cut goes to FUTURE_WORK.md with a reason.

## Ground rules

- Measurement study. No new algorithm, no SOTA claim, no generator training.
- Every claimed difference is a paired comparison on identical question ids
  (McNemar exact + paired bootstrap), Holm-corrected across the primary tests.
- Positions 0-499 of each dev file are the exploration fold (everything looked
  at while building). Claims are tested once on the confirmation folds in
  `results/folds/` with the criteria in `docs/preregistration.md` written first.
- Nothing is tuned on any dev file. Graph knobs are frozen on train slices; the
  selector trains on hotpot train only and is a pure transfer test elsewhere.
- Gold-derived fields (`gold_chunk_ids`, `reasoning_path`, `evidence_triples`,
  musique `is_supporting`) never enter `src/graph/` or a selector. A leakage
  test greps for them.
- Local runs are the only place generation happens (Metal). Retrieval-only
  experiments, code, tests and notes can be done anywhere.

## Phase 1: controls and the partition (in progress)

- [x] closed-book control, per-run traces, gold recall / paragraph recall
- [x] partition of the "answer right, evidence incomplete" cell:
      parametric / yes-no / shortcut / paragraph / partial / none
- [x] paired statistics; cumulative and hop-aligned per-hop recall
- [x] one answer prompt for every config; loop stop reason logged
- [x] folds; `--ids` in the evals; musique shuffled at prep
- [ ] `docs/preregistration.md`: hypotheses, assignment order and one
      alternative order, correctness criteria, n, tests, pass/fail rules
- [ ] runner on `stream_generate`: prompt tokens, TTFT, prefill vs decode
      throughput, peak memory; ratio-of-sums aggregation; warm-up excluded
- [ ] confirmation-fold baselines (hotpot_confirm n=1000): closed-book, bm25
      k=10, bm25 k=13 (matched context), hybrid ircot, hybrid ircot + rerank
- [ ] evidence-criterion ladder (all gold sentences / answer-bearing gold /
      all gold paragraphs / any gold / answer string in context) reported side
      by side; partition under the alternative order
- [ ] intervention arms on the exploration fold: gold-only, gold minus bridge,
      gold minus answer sentence, distractors only, drop-one over retrieved gold
- [ ] correctness criteria beyond EM: F1 >= 0.5, capped containment with
      aliases; a local judge on a 100-item hand-checked subset with kappa
- [ ] entity-substitution counterfactual on the illusion cell
- [ ] illusion-vs-k curve for single-hop bm25/hybrid (k = 5, 10, 13, 15, 20)

## Phase 2: breadth

- [ ] musique_confirm (stratified n=1000): closed-book, bm25 k=5, k=7, hybrid,
      ircot max_hops 4; per-hop new-gold recall against the decomposition
- [ ] 2wiki_confirm (n=1000): closed-book, bm25 k=10, k=13, hybrid ircot;
      per-type tables; hop-aligned recall from the reasoning path
- [ ] stop-criterion audit from traces: hop at which each gold sentence first
      entered context, hop of the stop, correctness; one forced-continuation run
- [ ] full-dev bm25 single-hop on hotpot and 2wiki overnight
- [ ] model-size sweep, Qwen2.5-Instruct 1.5B / 3B / 7B / 14B (4-bit) plus one
      other family: closed-book EM, illusion, parametric share vs size
- [ ] three sampled repeats (T=0.7) of the headline configs on n=200

## Phase 3: two retrieval arms on the harness

Graph arm (retrieval-only, zero model calls):
- [ ] `src/graph/pool_graph.py`: paragraph nodes from chunk titles; title-mention
      edges (directed, hub-penalised, reverse weight), optional multi-word
      entity edges; anchors = titles named in the query; degree; link coverage
- [ ] `src/graph/graph_retriever.py`: seed retriever (bm25 | hybrid) + anchors
      + one-hop expansion + in-paragraph sentence ranking at equal sentence
      budget; controls `none | degree_only | random | anchor_only |
      para_complete`; `expansion: none` must reproduce bm25 to four decimals
- [ ] delete `src/graph/graph_index.py` (planned a graph over 2wiki evidence
      triples, which is gold)
- [ ] `tests/test_graph.py`, `tests/test_leakage.py`; configs per dataset and
      per control; matched-context `bm25 k=13` / `k=7 (musique)` references
- [ ] freeze knobs on 2000 hotpot-train + 1000 2wiki-train questions
- [ ] retrieval-only table on full dev, three datasets, with controls and the
      artifact share = (degree_only - bm25) / (graph - bm25)
- [ ] QA on the confirmation folds: graph vs bm25 and vs the controls; does the
      shortcut bucket move while the parametric bucket stays

Selector arm (one small cross-encoder):
- [ ] `src/retrieval/selector.py`: MiniLM-L6 sentence selector over the whole
      pool, pointwise and anchor-conditioned; usable as a reranker too
- [ ] `src/retrieval/selector_train.py`: pairs from hotpot train with bm25 hard
      negatives, 20k questions, 1k held out; train/dev disjointness assertion;
      checkpoint hash in the run name
- [ ] baselines: zero-shot MiniLM, zero-shot bge-reranker; retained gain on
      2wiki and musique; seen/unseen split; title-disjoint retrain
- [ ] QA on the confirmation folds; graph -> selector stack

## Phase 4: serving cost on one device

- [ ] corner configs at n=500 with the full 2x2 per point: {8-bit, 4-bit,
      4-bit + 4-bit KV} x {k=5, k=20, k=20 -> 5}; no-retrieval and gold-only
      anchors; EM and illusion vs TTFT and vs decode tok/s
- [ ] prefix-cache reuse across ircot hops: tokens prefilled vs recomputed,
      per-hop TTFT, identical outputs under greedy
- [ ] concurrency ladder via `mlx_lm.server` (c = 1, 2, 4, 8), reported as a
      separate runtime
- [ ] retrieval-stage time and memory for graph and selector on the frontier

## Phase 5: writing

- [ ] results tables per fold with controls and adjusted p; partition bars per
      dataset; decoupling plot; frontier with illusion as colour
- [ ] README rewrite: scope in paragraph one (distractor pools, one model, one
      device), results, ablations, limitations, reproduction from a fresh clone
- [ ] short paper draft; related work from `docs/paper_log.md`
- [ ] pin versions and model revisions; run manifest per results file;
      `scripts/run_all.sh` end to end with a short smoke path

## Cut list, in order, if time runs out

2wiki graph ircot; entity-edge ablation grid; title-disjoint selector retrain;
graph -> selector stack; concurrency ladder; sampled repeats. Never the
controls, the folds, or the closed-book runs.
