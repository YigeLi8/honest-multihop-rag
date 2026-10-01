# Paper log

One entry per paper that changes what this repo should do or cite. Newest
first. Short: what it does, what it means here.

## 2026-10-01

arXiv was unreachable from today's session (listings and abstract pages
both), so these come from search-result excerpts of the 18-29 September
listings, not from the abstracts. Check each id and claim before citing or
acting on it; the two plan changes at the end are the ones to confirm first.

- Pu, Tang and Zhang 2026, CounterMem (arXiv 2609.31874). After a failed
  action the agent tests local alternatives against an executable checker;
  verified fixes are stored as {situation, bad action, better action, check
  result, applicability condition}, later records are filtered by that
  condition and a shared offline-trained selector picks one or skips memory.
  The nearest per-lesson applicability condition on this scan, but the
  condition is free text written once and matched by wording, with no
  revision from later outcomes: my static-conditions + shared-router pair,
  on coding and proof tasks, not retrieval. Cite.
- Guo 2026, SkillApt (arXiv 2609.26863). Load/abstain controller for
  retrieved skills from matched with/without executions, decided by a vote
  over similar past states. Per-lesson evidence by nearest neighbour, no
  boundary, no operators: my similarity-memory baseline with paired
  counterfactual labels. The with/without pairing is the right way to label
  applicability; gold per-hop hit/miss is the analogue here.
- Cheng et al. 2026, Scope Before You Persist (arXiv 2609.29144). Persisting
  a skill edit needs two decisions, is it supported and where does it apply;
  scoping each accepted skill to its originating task family raises held-out
  utility and removes harmful deployments on a code-repair stream. The
  cleanest statement of the motivating claim (evidence from one family does
  not justify global deployment). The scope is fixed at acceptance, never
  revised: static conditions with a result behind them. Cite.
- Liang et al. 2026, UpliftMem (arXiv 2609.36805). Learns memory retrieval
  from set-level execution uplift against the same executor without memory,
  with probe rollouts chosen by a value-of-information criterion. Already
  does the Stage 6 active-probing item as VOI over a per-set utility; if
  that item stays, frame it as VOI over per-lesson boundaries.
- Bacellar 2026, Per-Query Gating of LLM Rerankers (arXiv 2609.22880). A
  learned gate skips the reranker on multi-hop retrieval from pre-call
  features (score and lexical statistics of two retrieval lists plus a query
  embedding), evaluated on last-hop@k with gold hops on 2wiki, musique and
  hotpot under a pre-registered non-inferiority rule. A shared per-query
  router over retrieval-process features with gold per-hop labels, trained
  offline: my shared-router baseline in published form. Reuse the feature
  list and the non-inferiority framing.
- Bacellar 2026, Predictable Failure in Multi-Hop Retrieval (arXiv
  2609.22056). Logistic confidence score over query-time structural
  features gives calibrated abstention on musique, 2wiki and hover. The
  premise that process features predict hit/miss, with a feature set; cite.
- Li, Zhang and Ming 2026, Beyond the Query (arXiv 2609.12437). Matched
  query-only vs query-plus-retrieval-signal routers for adaptive RAG actions:
  retrieval-state features give no reliable gain over the query-only
  control. Plan change: every Stage 5 comparison needs a matched query-only
  control (the same router on the query-stage features alone).
- Mondal et al. 2026, MemSafe (arXiv 2609.32269). A classifier that only
  counts paragraphs reaches 0.98 AUROC on musique sufficiency, so they build
  a size-matched set before training the real estimator. Plan change: any
  applicability or sufficiency model here gets a size-matched check (hop
  count, pool size and n_hits are trivial predictors).
- Anand et al. 2026, DRAG (arXiv 2609.17709). Per-query retriever and
  generator configuration selection, offline, no feedback loop. Cite only.
- Sato et al. 2026, Evidence Sufficiency Boundaries (arXiv 2609.01687).
  Trains a small model to abstain under partial evidence and answer once the
  ordered chain is sufficient. Same object as the gold-completeness
  partition; cite next to Sufficient Context, I measure rather than train.
- Tian, Ganguly and Macdonald 2026 (arXiv 2609.16453). Intermediate answers
  after each agentic-RAG iteration plateau before natural termination and
  predicted utility drives early stopping. Prediction-side complement to the
  stop audit and HALT.
- Ye et al. 2026, CoEvo-Mem (arXiv 2608.01739). A residual router corrected
  online from task outcomes plus per-memory values updated by trajectory
  feedback: my shared online router and scalar utility baselines in one
  loop. The strongest combined baseline; it has no per-memory applicability
  model and no revision log. Also 2604.27283 (risk-sensitive bandit deciding
  whether to inject a memory from a feature state) and 2602.22406 (Thompson
  sampling over per-memory utility posteriors): scalar utility with online
  feedback, published.
- Not entries, noted: 2609.11060 (memory curator with scope checks, scopes
  are curator-written text), 2609.32511 ShareMem (declared scopes),
  2609.32521 MemAgent (routing over providers), 2609.35808 MATE (misleading
  successful trajectories, embodied), 2609.32313 MemTransfer (mismatched
  experience hurts, embodied).

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
