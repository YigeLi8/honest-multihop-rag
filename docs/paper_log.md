# Paper log

One entry per paper that changes what this repo should do or cite. Newest
first. Short: what it does, what it means here.

## 2026-10-01

Listings read in full (cs.IR 41, cs.CL 269, cs.AI 613 entries); the
September ids below were checked against their abstract pages and the three
closest read in full.

- Pu, Tang and Zhang 2026, CounterMem (arXiv 2609.31874). After a failed
  action the agent tests local alternatives against an executable checker
  (tests, proof checkers, solvers); verified fixes are stored as {situation,
  bad action, better action, evidence, condition, reuse statistic}. The
  condition is a free-text field matched by similarity plus a compatibility
  rule, the reuse statistic is an EMA frozen after construction, and a DQN
  over per-record statistics decides whether to use a record; nothing is
  revised at evaluation time. Baselines ExpeL, Voyager, MemGPT, ReasoningBank
  and a compute-matched best-of-N; proof, repair and SQL tasks, no retrieval.
  The nearest per-lesson condition on this scan, and it is my
  static-conditions + shared-router pair. Cite.
- Guo 2026, SkillApt (arXiv 2609.26863). Load/abstain for retrieved skills:
  k=5 nearest past states by hashed bag of words, vote over paired
  with/without outcomes, fixed threshold, offline. Similarity memory with
  counterfactual labels, no condition stored. The with/without pairing is the
  right way to label a lesson; gold per-hop hit/miss is the analogue here.
- Cheng et al. 2026, Scope Before You Persist (arXiv 2609.29144). Two
  decisions for a persistent skill edit: is it supported (a conjunction of
  one-sided paired-bootstrap bounds over three test channels) and where does
  it apply (one of nine code-repair families, given with the task, never
  learned, never revised). Scoped retrieval raises hidden-test utility and
  removes harmful deployments against seven baselines. The cleanest
  statement of the motivating claim, with a fixed label as the scope; cite,
  and keep the conjunction-of-lower-bounds acceptance rule in mind for the
  operators.
- Liang et al. 2026, UpliftMem (arXiv 2609.36805). Memory retrieval trained
  on set-level execution uplift over the same executor without memory, probe
  rollouts picked by a value-of-information criterion; scorer frozen at test,
  no conditions stored. Already does the Stage 6 probing item as VOI over a
  set utility; if that item stays, frame it as VOI over per-lesson
  boundaries.
- Bacellar 2026, Per-Query Gating of LLM Rerankers (arXiv 2609.22880). A
  per-(dataset, k) classifier over 27 pre-call features (score statistics of
  two retrieval lists, their agreement at several depths, query shape, an
  optional query-embedding PCA) decides whether to run the reranker, trained
  once per fold and frozen; evaluated on last-hop@k with one gold target
  passage per question on 2wiki, musique and hotpot under a pre-registered
  non-inferiority rule, against a score-gap gate, the best fixed action, a
  random gate at matched skip rate and the oracle. My shared router in
  published form, offline, with a single-target label; per-query-type gating
  is named as untested. Reuse the feature list and the non-inferiority
  framing; mine is online, per lesson and scored on full per-hop gold.
- Bacellar 2026, Predictable Failure in Multi-Hop Retrieval (arXiv
  2609.22056). Logistic confidence score over query-time score-distribution
  features gives calibrated abstention on musique, 2wiki and hover. The
  premise that process features predict hit/miss, with a feature set; cite.
- Li, Zhang and Ming 2026, Beyond the Query (arXiv 2609.12437). Matched
  query-only vs query-plus-retrieval-signal run/skip routers for adaptive
  multimodal RAG: the retrieval-state features do not beat the query-only
  control. Plan change: every Stage 5 comparison needs a matched query-only
  router on the query-stage features alone.
- Mondal et al. 2026, Before Answering (arXiv 2609.32269). A classifier that
  only counts paragraphs reaches 0.98 AUROC on musique evidence sufficiency,
  so they build a size-matched set before training the estimator (MemSafe).
  Plan change: any applicability or sufficiency model here gets a
  size-matched check (hop count, pool size and n_hits are trivial predictors).
- Anand et al. 2026, DRAG (arXiv 2609.17709). Per-query retriever and
  generator configuration selection from query-performance-prediction
  signals or a fine-tuned router, offline. Cite only.
- Sato et al. 2026, Evidence Sufficiency Boundaries (arXiv 2609.01687).
  Trains a small model to abstain under partial evidence and answer once the
  ordered chain is sufficient. Same object as the gold-completeness
  partition; cite next to Sufficient Context, I measure rather than train.
- Tian, Ganguly and Macdonald 2026 (arXiv 2609.16453). Partial answers after
  each agentic-RAG iteration plateau before natural termination; predicted
  utility drives early stopping. Prediction-side complement to the stop
  audit and HALT.
- Ye et al. 2026, CoEvo-Mem (arXiv 2608.01739). A residual router corrected
  online from task outcomes plus per-memory values updated from trajectory
  feedback, alternating updates: my shared online router and scalar utility
  baselines in one loop, no per-memory condition, no revision log. The
  strongest combined baseline. Also Iscan 2026 (arXiv 2604.27283, contextual
  bandit with abstention over a 16-feature retrieval-evidence state, updated
  after every decision, but evaluated on synthetic artefacts only) and Wu et
  al. 2026, U-Mem (arXiv 2602.22406, Thompson sampling over memory utility,
  +14.6 on HotpotQA over memory baselines): scalar utility with online
  feedback, published.
- Pu 2026, BeliefRAG (arXiv 2609.39139). Adaptive-RAG controller with an
  explicit per-episode belief state (sufficiency, reliability, conflict,
  uncertainty, gap, cost) from one verifier call and calibrated retrieval
  scores; only an answerability calibrator and a state-flip predictor are
  fitted. Hotpot, 2wiki, musique and three single-hop sets at n=100 each.
  Within-episode only, no cross-question memory, so the gap stays open; cite
  as the controller baseline. Two results to reuse: calibrated answerability
  dominates the other belief dimensions, and the same calibrator falls from
  0.78 to 0.65 AUC under source shift, which is the case for revising
  conditions online rather than fitting them once.
- Piao, Wang and Chen 2026, Do Self-Evolving Skills Generalize to Held-Out
  Tasks? (arXiv 2609.39148). Six skill-learning methods on six benchmarks
  with a fixed split: of 21 skills that improve on train, 5 keep the gain on
  test, 13 part of it, 3 none; the failures are rules written for one task
  and skills that never say when they apply. A no-persistence control
  (regenerate a skill per task from a meta-skill) wins all six. The cleanest
  external evidence that persisted lessons without an applicability boundary
  over-generalise; cite in the motivation, and add the no-persistence
  control to the Stage 5 baselines.
- Meng et al. 2026, U-Fuzz (arXiv 2609.38275). Memory-use failures as
  fuzzing: mutate the query or the memory state under stated obligations,
  validate the mutant, steer the next mutation by observed memory behaviour;
  finds "correct memory used wrongly" cases from final answers alone. A
  ready-made way to construct the contradicting probes the boundary tests
  need; cite in the evaluation and consider its mutation-obligation framing
  for the twins.
- Mao et al. 2026, TIDE (arXiv 2609.37544). Memory evolution from delayed,
  confounded feedback: responsibility credit spreads an outcome over the
  memories used, memories are reinforced, crossed, mutated or evicted, and
  memory evolution gain is utility over a no-memory baseline on strictly
  future tasks. Credit assignment to the lessons applied is the same problem
  here, and the prospective no-memory-relative metric is the one to report.
- Not entries, noted: 2607.05712 Scoring a Set (set-level evidence scorer,
  a candidate arm), 2609.39075 RAGScope (leakage-controlled gate evaluation,
  calibration collapses leave-source-out), 2609.39957 HiSentinel and
  2609.38822 SkillSeek (hindsight-trained gates, offline), 2609.38353
  TAGGraph (bm25 beats graph retrieval over agent histories), 2609.39578
  Box2-Bench (selective reliance on fallible guidance), 2609.11060 (memory
  curator with scope checks, scopes are text), 2609.32511 ShareMem,
  2609.32521 MemAgent, 2609.35808 MATE, 2609.32313 MemTransfer.

## 2026-09-30

- Tao et al. 2026, RealHop (arXiv 2609.36984). Behavioral necessity rate by
  dropping declared evidence. Same probe as the drop-one arm here; cite as the
  closest contemporary and keep my partition as the addition.
- Mo et al. 2026, EMNLP (arXiv 2609.17043). Per-hop retrieval-vs-extraction
  failure decomposition on MuSiQue/HotpotQA/2Wiki. Overlaps the stop audit;
  differentiate by the closed-book control and the paired config comparison.
- Roh and Han 2026, HALT (arXiv 2608.02009). Qwen2.5-7B on my three datasets:
  the loop stops before gold coverage on 51-79% of questions; forcing
  continuation raises EM on the under-covered subset. The forced-continuation
  run in Phase 2 replicates this with the correct-but-premature cell added.
- Asad et al. 2026, Faithfulness Is Not Free (arXiv 2608.30996). INT4 KV cache
  on Qwen2.5-7B/HotpotQA degrades judged faithfulness while accuracy holds.
  Phase 4 extends with gold grounding and weight quantization on-device.
- Ke et al. 2026 (MDPI AI 7(8):320). Answer quality x supporting-fact recovery
  on full HotpotQA dev with Qwen3-8B, BM25/BGE/hybrid, partial-fact buckets.
  The nearest 2x2; mine adds the closed-book split, the ordered partition, the
  shortcut bucket and paired tests across configs.
- Jain and Vedam 2026, CUE-R (arXiv 2604.05467). Remove gold supports on
  HotpotQA/2Wiki with Qwen3-8B; about half of correct answers survive.
- Bala 2026 (arXiv 2607.00725). Coverage rises while F1 stays flat on the same
  datasets and reader family. Same decoupling observation; cite.
- ONCU 2026 (arXiv 2606.06758). Matched no-evidence / retrieved / oracle
  conditions on HotpotQA and 2Wiki with open models. The intervention-arm
  template.
- Joren et al. 2025, ICLR, Sufficient Context (arXiv 2411.06037). Stratifies
  HotpotQA and MuSiQue by context sufficiency; models are right 35-62% of the
  time with insufficient context; Table 2 lists the causes I now measure per
  question. The paper a reviewer will name first.
- Chen et al. 2026, WWW, TRACE (arXiv 2602.21230). Source of the phrase
  "high-score illusion" for deep-research agents; credit and disambiguate.
- Jiang et al. 2025, ISCA, RAGO (arXiv 2503.14649). Iterative retrieval with
  per-hop prefill; TTFT and TPOT reported separately. The accounting to copy.
- 2609.37226 Follow the Entities (entity-resolved navigation for agentic
  search); 2609.37469 Relevance Is Not Sufficient Evidence; 2609.35774
  post-generation verification beats retrieval tweaks on long documents;
  2609.35782 evidence crowding under fixed context slots; 2609.35773
  Socrates-RAG. None report a query-free structural control or the
  gold-completeness cell with a closed-book split.
- Entity/passage-graph retrievers now crowded: ConRAG 2609.35193, NexusRAG
  2609.37661, STITCH-RAG 2609.34127, SentGraph 2601.03014, PAGE-RAG
  2608.29753, BDTR 2509.25530, LiteRAG 2609.10239. Set-wise / learned
  selectors: SETR 2507.06838, DPS 2508.09497, AdaGATE 2605.05245. The graph
  arm here is a re-implementation of HGN/DFGN-style title links with the
  artifact control as the point, not a new retriever.

## Older, to cite

Min et al. 2019 (single-hop reader reaches 67 F1 on HotpotQA); Chen and
Durrett 2019 (sentence-factored models solve more than half); Jiang and Bansal
2019 (adversarial distractors break word-matching shortcuts); Trivedi et al.
2020, DiRe (drop-one probe; only 18 of 72 F1 points from connected reasoning);
Trivedi et al. 2022, MuSiQue (HotpotQA > 2Wiki > MuSiQue in disconnected
reasoning); Petroni et al. 2021, KILT-EM (EM only with gold provenance);
Hofstaetter et al. 2022, FiD-Light (passage-correct x text-correct on a latency
frontier); Longpre et al. 2021 (memorisation ratio grows with size); Stolfo
2024 (retrieved- vs pretraining-grounded correct outputs across sizes); Qi et
al. 2019, GoldEn (per-hop recall); Xiong et al. 2021, MDR; Zhu et al. 2025,
ChainRAG; Kaushik and Lipton 2018 (input-ablated baselines); Card et al. 2020
(power analysis for NLP comparisons).
