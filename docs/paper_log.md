# Paper log

One entry per paper that changes what this repo should do or cite. Newest
first. Short: what it does, what it means here.

## 2026-10-05

Listings read in full (cs.IR 17, cs.CL 185, cs.AI 402 entries); about
thirty abstract pages checked, the closest read in full.

- Ji, Lu, Zhang and Olukotun 2026, Sentry (arXiv 2610.02994). Failure
  lessons are conditional knowledge: kept in the agent's context they
  misfire where their failure is absent (a frozen playbook as background
  context lowers WebShop reward 0.364 -> 0.296 and Mind2Web Replay 0.651 ->
  0.599), so the playbook stays outside the context and a lesson is exposed
  only when a rule-based detector over the last five reason-act-observe
  cycles sees a failure of its type (progress: repetition, stall;
  reasoning / grounding: unsupported claim, action-reasoning mismatch).
  Retrieval is a predicate, not an embedding: same failure type, at least
  one shared retrieval label, ranked by label overlap, k=5, falling back to
  the most recent entries of the type. A lesson (type, labels, trigger
  pattern, repair principle) is written only after a recovery judged from
  the next ten cycles without task reward; nothing is revised, narrowed or
  retired after admission. WebShop, AppWorld, SWE-bench Lite, Mind2Web
  Replay, online from scratch; +37% over the best runtime-intervention
  baseline, +39% over ACE where both run. This is the typed-trigger half of
  my failure memory (Stage 3) with the condition "the failure is present
  now" fixed in advance and never learned; the control they run (the whole
  playbook always visible) is the unconditioned baseline I should also
  have: every lesson exposed to a shared router, no per-lesson region. The
  gap they leave is the one under test here, whether the region a lesson
  applies in can be revised from later outcomes; cite as the strongest case
  that exposure conditions matter at all.
- Liu et al. 2026, Action Calibration (arXiv 2610.02769). Within one task,
  agents gain from interaction history even when the past actions are
  shuffled, and breaking the action-observation correspondence costs
  little, so the history is used as context and not as experience.
  Labelling each observation as the outcome of its action helps on its
  own; a learned calibrator then reassesses past actions and records only
  some of them. The point for me is the selective write: an experience
  that is kept is a decision, and a memory evaluation needs the
  shuffled-history control (does the policy use which past outcome went
  with which situation, or only that there were past outcomes). On the
  stream that is a permutation of the revealed outcomes across questions
  before they enter a lesson; a policy that does as well on shuffled
  reveals is not learning applicability. Add it as a control row.

Seen and set aside: 2610.02687 GraphMemory (strategies as a graph,
subgraph retrieved per query, token cost is the claim), 2610.02945
(Continual Graph Memory for a maths agent: negative findings recalled with
a scope, no learning of the scope), 2610.02932 PACE (when to compile a
repeated GUI procedure into a program, an online payback rule), 2610.02361
SEDIMA (insight memory across evolutionary-search runs), 2610.03190 Nautil
(when the evidence suffices to close a case, with an evidence-removal
control), 2610.03020 DyadMem, 2609.38021 (an audit of a long-term-memory
evaluation: reader variation and gains concentrated where the baseline
lacked evidence; no untouched holdout), 2610.02070 Causal Memory Policy
(already in the 2 Oct entry; cross-listed again).

## 2026-10-02

Listings read in full (cs.IR 24, cs.CL 280, cs.AI 568 entries); twenty
abstract pages checked, the two closest read in full.

- Li 2026, CAVE-Mem (arXiv 2610.00238). A past search lesson is stored as a
  typed operator with three hand-written predicates (memory substrate,
  answer contract, failure boundary) and a cached utility; the operator may
  change a base memory-search answer only if every predicate matches and its
  utility, cross-fitted from diagnostic blocks that exclude the target
  block, is positive, otherwise the system abstains. Training-free, nothing
  revised after construction, no revision log. Baselines R2Mem
  (relevance-only reuse), Mem0, A-Mem, MemoryOS, LightMem, Memory-R1, GAM
  and plain RAG; LoCoMo +6.5 F1 over R2Mem with GPT-4o-mini, HotpotQA
  128-question subsets +3.5 F1 at 56K context and about zero at 224K and
  448K, NarrativeQA +2.1; operators fire on 2-16% of questions. This is my
  Stage 4 static-conditions baseline in print, with a utility veto and
  abstention added, so Stage 4 gets both. Copy the cross-fitting: any
  per-lesson utility I report is estimated from blocks that exclude the
  question being scored.
- Behnam and Wang 2026, Causal Memory Policy (arXiv 2610.02070). A memory
  that is never retrieved has no identified utility, however the store is
  ablated; they reserve k of B=6 context slots for memories exposed with
  known, balanced propensities and estimate per-memory utility by
  self-normalised IPW, with a forget rule scaled by an irreversibility
  ratio. On LongMemEval and LoCoMo 54% and 67% of required memories are
  unidentified without the intervention; required-vs-not AUC 0.54
  (store-level ablation) to 0.66 (exposure); per-query utility reaches
  0.78 AUC on the query it was estimated for, and no query-independent
  aggregate (mean 0.55 AUC, max, recency) correlates above r=0.10 with
  contribution on unseen queries. Two things for me. The scalar per-memory
  utility baseline is shown to be the wrong object even when identified,
  which is the argument for conditions over features rather than a number
  per lesson; cite in the motivation. And the positivity point is why the
  retrieval-level outcome matters here: every arm is run on every
  question, so the counterfactual for a lesson that did not fire is
  observed, not estimated. Say that in the write-up, and use their
  balanced-exposure design if Stage 6 probing happens.
- Zhang 2026, What Does a Skill Actually Do? (arXiv 2609.33153). Critical
  review of 135 papers' tool and skill evaluation designs by treatment
  contrast, population, outcome, budget and identification assumption. The
  point that bites: pairing with/without runs on the same task does not
  identify the invocation effect when the comparison conditions on the
  treated run having triggered, and paired gain/regression counts measure
  discordance, not the share of tasks made worse. My false-application and
  repeated-failure rates condition on the lesson firing. Report them, but
  also the unconditioned contrast on the full stream at matched retrieval
  cost (the no-memory-relative gain of TIDE), and use the checklist.

Seen and set aside: 2610.01787 (lessons, locators, procedures and state
facts routed to weights or context by two properties fixed in advance),
2605.08386 SkillLens v2 (accept / decompose / rewrite / skip per skill
unit), 2610.00366 (memory evaluations should hold history access fixed and
report retention and selection apart), 2605.28009 MemGuard (typed memories
so that episodes do not become rules), 2610.00094 (finite-sample
certificate that a learned memory decision beats an incumbent), 2610.01161
FAULT (per-error-type costs updated online), 2609.39022 (verification
failures to reusable guidance, no conditions), 2610.01767 MatRAG
(hierarchical multi-hop retriever, retrieval quality only).

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
