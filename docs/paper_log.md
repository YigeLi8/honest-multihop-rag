# Paper log

One entry per paper that changes what this repo should do or cite. Newest
first. Short: what it does, what it means here.

## 2026-11-11

Listings read in full (cs.IR 29, cs.CL 178, cs.AI 381 entries, replacements
included; 501 distinct ids); every title screened, about sixty abstracts
read, the four below read in the body.

- Tang et al. 2026, Falsification-Gated Self-Evolution (arXiv 2611.14326).
  Each candidate lesson is a structured hypothesis: a precondition, a
  behaviour change, an expected effect and a verifier. It is applied in a
  temporary state and scored on three probe sets before anything is
  written: target-transfer probes (does it help where it should),
  falsification probes (contrastive and archived cases where it should
  not fire) and a regression archive (solved tasks it must not break);
  then committed, refined or rejected by measured gain and risk. Scope
  refinement is the precondition refit on the probe outcomes, positives
  and negatives, with a complexity penalty, on a development split, with
  fresh probes for the final commit. WebArena 39.8, AppWorld 78.8,
  tau-retail 0.724; falsification failures 5.0% against 18.6% without the
  falsification probes, harmful commits 3.6% against 10.6%, regression
  damage 1.3% against 5.2% without the archive; dropping scope refinement
  costs 2.7-3.4 points on the task metrics, and an LLM-judge gate in
  place of the verifier probes costs 3.1-4.7. So a per-lesson applicability
  boundary fit from where the lesson helped and where it was falsified is
  in print, with the ablation that says the boundary is worth about three
  points on their tasks; the plan's Stage 4 to 5 step (a predicate per
  lesson, then narrowed on exceptions) is their scope refinement, and the
  item should say so. What stays open for me: their probes are run before
  the commit and the boundary is fixed after it (post-commit monitoring
  only retires), so a boundary that keeps moving as contradicting
  questions arrive in the stream, and whether that beats the one-shot
  refit, is not tested there; and every probe needs a verifier, which on
  retrieval is the gold label I have and an agent in the wild does not.
  Their no-falsification arm (commit whatever improved the source
  episode) is my version-1 memory. Cite as the strongest published form
  of the static-conditions baseline with a learned, refit precondition.
- Lee et al. 2026, WikiTrace (arXiv 2611.14320). Names what A5 saw:
  memory-induced degradation, an agent failing a task it solves without
  memory because it follows a wrong lesson or misapplies a right one.
  Lessons are kept with a link to the summary of why their trajectory
  succeeded or failed and to the raw trajectory, and the agent descends
  to the source when an insight lacks detail, conflicts with another or
  came from a failure. Eight suites over five benchmarks (ALFWorld,
  ScienceWorld, WebShop, AppWorld, tau-bench retail), DeepSeek-V4-Flash,
  three trials. No memory averages 79.5, the same insights handed over
  alone 87.0, WikiTrace 90.2 (+3.2 over the insights-alone arm, which is
  the matched control), and regression (tasks lost against no memory)
  goes 6.5% to 4.5%; it is the only memory arm at or above no memory on
  every suite, and the degradation causes are judge-labelled. For me the
  number is the regression rate as a reported endpoint beside the gain:
  my A5 table has the EM difference only, and the share of exploration
  questions the region memory loses against no memory is in the paired
  csv and should be in the plan with the gain. Their mechanism (check the
  lesson against its source before acting) is an applicability check made
  by the agent at use time instead of a boundary learned in advance, and
  it is the cheapest comparison arm I do not have: hand the subagent the
  failure episode a lesson came from, not the lesson. Cite beside
  frozen-weight self-improvement; both are lessons-from-failures with the
  "same insights" control.
- Lei and Liu 2026, SELD (arXiv 2611.12903). Silent evidence loss: a
  context edit removes the key evidence and the reader's answer does not
  change. SEL-Pairs is built from hotpot, 2wiki and musique: every
  evidence deletion is matched by a control deletion of exactly the same
  token length from the same context, as single sentences and inside
  LLMLingua-2 and Provence outputs, so a detector cannot win by noticing
  that the context got shorter. In their development data a key-sentence
  deletion and an equally long control deletion both leave the answer
  unchanged in 63-78% of pairs, and on musique about one in five correct
  answers comes back under an officially unanswerable context. The
  detector reads the shift in the reader's preference among the answers
  it did not give; 83% of the evidence-removing edits identified against
  70% for the best baseline, five readers. For Part A this is the cell
  measured by intervention instead of by gold labels: their 63-78% is
  "answer unchanged when a gold sentence is cut" on the same three sets,
  which is what my answer-only and bridge-only arms measured for one
  sentence at a time (dropping the bridge left EM level on hotpot), and
  their one-in-five on musique is a parametric-rate yardstick beside
  CARL's 7%. Cite in the cell's write-up with ChainCheck, and the
  length-matched control deletion is the control the reader-cut arms of
  Stage 5c should have had.
- Rayeed, Stewart and Ma 2026, CLADE (arXiv 2611.14355). Passages that
  name the broader family of the answer and not the answer itself
  (resolvable insufficiency): readers answer from them rather than
  abstain, abstention down 13.8 points against uninformative passages,
  the guesses wrong about half the time, and a counterfactual edit (family
  name in place of the exact name, against deleting the sentence) lowers
  abstention 17.4 points, on PrimeKG and GloBI. Confidence filters, a
  larger model and an abstain instruction do not remove it. One sentence
  in the write-up: the partial bucket of my cell (some gold in context,
  answer right) has this shape, and their counterfactual is the clean way
  to show a partial context causes the answer rather than merely
  accompanies it.
- Beesetty 2026, Agreeing on the answer but not the reason (arXiv
  2611.13140). Failure categories assigned by ordered tests: 50.1% of
  failures carry evidence for more than one category, so the reported
  share of a category depends on the test order (swapping two tests moves
  the plurality in eleven of twelve models; the share of one category
  spans 26.4% to 76.4% over the six orderings), and native speakers agree
  with the classifier at kappa 0.93 on correctness and 0.09 on the
  category. Four checks before comparing failure-mode rates: report label
  co-occurrence, permute the tie-break, validate on a shared independent
  sample, audit the systematic disagreements. My failure typing
  (`src/memory/failures.py`) is exactly five ordered rules, and the Stage
  3 item reports the type shares and the type x recovering-arm chi-square
  without either of the first two checks. Added to that item: the
  co-occurrence of the rules per failure and the type shares under the
  other rule orders, before the 50-case audit is read, so that the
  "near miss wants the fused list, named or bridge miss wants dense"
  reading is shown to survive the order of the rules.
- Huang et al. 2026, Synapse-Control (arXiv 2611.13615). Advice from past
  runs selected by a control plane: rules shortlist candidates and a 1.7B
  classifier fine-tuned on teacher-labelled (state, suggestion) pairs
  scores whether a suggestion applies to the current state, in place of
  Sentry's generated failure diagnosis; a table of earlier guidance and
  its feedback puts a cooldown on repeats. AppWorld and WebShop, 1308
  attempts per method, 270 successes against the base's 208, half the
  control time of Sentry. This is the shared router over advice: one
  classifier for every lesson, learned offline from a teacher's labels,
  no per-lesson boundary and nothing revised from outcomes. Cite beside
  Sentry as the cheap-router baseline for applicability.

Seen and set aside: 2611.13144 SAT (skills admitted only after a measured
gain on matched development tasks; libraries swapped between two student
models, each does better with its own: a negative on transfer across
readers, cite if the write-up claims reader specificity), 2611.12988
Third Eye (persistent self-updates in 3-8B models lose their value after
the next update; argues for a no-update baseline and independent scoring),
2610.31303 (decision-theoretic account of memory value, LoCoMo and
LongMemEval), 2611.07342 (a router trained from repair outcomes decides
when an agent may retrieve; retrieval helps some cases and hurts others,
building automation), 2606.02357 (93-96% of tool-solved problems also
solved without the tool: the illusion-cell argument for tool traces),
2611.13254 (72% of failed agent runs already held evidence against the
error; in 88% of those no agent acted on it), 2611.13135 (which past
success is the reference for a failure, by action-sequence alignment),
2611.14149 (survey of information-seeking capabilities incl. retrieval
routing), 2605.25379 StateRAG (replacement; sufficiency field in an
explicit state), 2610.36991 SpectRAG (multi-hop context selection under a
token budget, MultiHopRAG), 2611.14019 PASE and 2611.14009 (skill and file
memory evolved by feedback; not on these sets).

## 2026-11-10

Listings read in full (cs.IR 56, cs.CL 364, cs.AI 887 entries, replacements
included; 1114 distinct ids); every abstract screened, the five below read
in full or as far as their results tables.

- Chu et al. 2026, Peer Memory (arXiv 2611.09990). Six agents on hotpot
  distractor questions, each holding one passage (two gold, four
  distractors), three rounds of "pick a peer, read its answer, revise"; the
  pick is a LinUCB contextual bandit over nine process features, rewarded
  by the F1 of the revised answer. The memory is the saved bandit state per
  (selector, peer) pair after a 40-question block, restarted on a second
  block cold, warm (saved state) or shuffled (the same state permuted
  across peer ids), all three still learning online. 80 hard questions, 12
  paired lineages, Qwen3-32B and Gemma-4-31B. Warm lowers hindsight regret
  against cold by 7.4% of cold regret on Qwen (CI -12.8 to -2.1, 10 of 12
  lineages) and 6.0% on Gemma; warm against shuffled is -7.4% on Qwen and
  -4.7 (CI -11.2 to +1.8) on Gemma; final F1 does not move (warm minus cold
  -1.26 points on Qwen, +0.23 on Gemma, intervals through zero). A second
  study reassigns the gold passages to other agents: warm is then 1.50 F1
  points below cold (CI -2.79 to -0.20) and the warm agents consult
  themselves 5 points more often. This is my shuffled-reveal control on a
  scalar-utility routing memory, on hotpot, in print, with my result's
  shape: the memory changes who is consulted, the regret moves, the task
  metric does not, and on one of two models the shuffled copy keeps most
  of the regret gain. Nothing is scoped or revised per lesson, so it is
  the scalar-utility baseline of Stage 5 and not the claim. Two things to
  take: regret as the primary endpoint with F1 beside it is my
  retrieval-level / EM split in their words, and the role-reassignment
  block is the drift test my stream still lacks (a memory that was right
  and goes stale), cheaper than switching dataset mid-stream. Cite beside
  Action Calibration as the second shuffled-memory control.
- Mitra 2026, Frozen-weight self-improvement (arXiv 2611.11342). A frozen
  Qwen3.5-9B agent; a larger writer turns its failure traces into "when
  situation then action" lessons in a propose / validate / keep-or-revert
  loop. The control that matters is a matched no-evidence search: the same
  writer, prompts, loop, selection and budget with the episodes replaced by
  a fixed sentence. On 2997 sealed tau2 retail decisions the standard
  harness scores 70.3%, the no-evidence search 76.8%, the trace lessons
  80.9%, so the registered contrast is +4.0 [+2.3, +5.7] (under their
  preregistered 5-point bar) and the search without any trace gets 60% of
  the gain, all of it on the task whose rule the writer's prompt already
  stated. Full conversations: 82.9% without the lessons, 81.6% with
  (-1.3 [-5.9, +3.3]). A registered gate admits a single lesson only on
  randomised on / off exposure with per-context non-inferiority and a harm
  bound: 2 of 12 tau2 lessons pass (none tested held out), 0 of 8 BFCL
  trace lessons, and the one BFCL lesson that passed did not replicate
  (-0.31 [-3.65, +3.12]); a neutral note with no procedure beats both
  lesson sets there (+4.5 [+0.7, +8.4]), and a blind audit finds 146 of
  240 held-out BFCL items defective. For me this is the Stage 6 result
  stated the way it should be: my "region is level with no memory and the
  show-all arms lose three points" has no matched no-evidence control (the
  same reflection and probation loop fed a fixed sentence instead of the
  episode) and no neutral-note arm, so I cannot yet say whether the
  reflection content did anything at all; both are cheap to add to the
  A5 table on the Mac. Their gate is a one-shot offline applicability
  screen (contexts fixed in advance, nothing revised after), i.e. Stage 4
  with a harm bound, not the online boundary; cite it as the strongest
  recent negative on failure-derived lessons transferring.
- Han et al. 2026, ChainCheck (arXiv 2611.11965). An audit of
  evidence-sufficiency scorers on musique, 2wiki and hotpot. Negatives are
  built by substituting the bridge entity in the later-hop passage (chain
  broken, text edited); the control cell substitutes it in both passages
  (chain intact, text edited); the contrasts are a chain effect, an edit
  effect and their difference, chain selectivity. Probes and readouts on a
  frozen Qwen3.8-27B, two LoRA variants, a zero-shot judge and four more
  judge models. On 324 musique test pairs nominal AUCs are 0.970-0.996
  while every trained system scores the intact-but-edited cell almost as
  a negative and has negative selectivity; only the zero-shot judge is
  chain-selective (+0.21 [0.13, 0.29]), and on 2276 hotpot pairs the
  musique / 2wiki pattern did not replicate (judge selectivity -0.007 to
  +0.042). On a crossed hotpot arm an artifact-only per-passage log-prob
  scorer that never reads the chain reaches AUC 0.997. The number for me
  is the surface gate: negatives made by removing all support are
  separable from positives by a surface classifier at AUC 0.959, against
  0.568 for the substitution negatives, so any learned "is the chain in
  context" verifier judged on removal negatives is a relevance detector.
  My gold-only / full-context arms are not edited text and the cell is
  read off gold labels, so Part A is not hit; but a learned sufficiency
  check, which the stop audit would want, needs their intact-chain edit
  cell as a control. Cite in the shortcut paragraph, with their list of
  disconnected-reasoning work.
- Pardawala et al. 2026, CoRA-Bench (arXiv 2611.10238). Context selectors
  (head / tail truncation, bm25, Qwen3 dense, dense plus reranker,
  LLMLingua-2, Provence) at shared 32K / 64K / 128K budgets on BABILong,
  LOONG, a NoLiMa diagnostic and LongMemEval, scored on evidence recall
  and on reader accuracy over the identical selected context (Gemini 2.5
  Flash-Lite, three seeds, paired bootstrap). At 32K the full context gets
  31.1% on BABILong against bm25 46.2% and Provence 48.1% (paired +15.1
  [10.9, 19.6] and +17.0 [12.5, 21.5]), 0.6% on NoLiMa against Provence
  27.4%, 63.2% on LongMemEval against dense 71.1%; LLMLingua-2 is below
  the full context everywhere (12.1% on BABILong). So on long haystacks a
  bm25 or Provence cut beats reading everything, and my C2 result (no cut
  short of the oracle above the full context, rerank3 below it) is a
  property of these pools: forty sentences, gold a sentence away from its
  distractors, every cut dropping a gold hop on about half the questions.
  The write-up has to say that and cite this beside the filter baselines
  of Stage 5c; Provence off the shelf, which that item lists, has its
  long-context numbers here already. Their reporting (recall and reader
  accuracy on the same selected context, paired) is the Stage 5c table.
- Kumar, Wu and Suley 2026, CARL (arXiv 2605.27788, v2 of today). PPO
  for tool use with the critic read at tool-call boundaries, Qwen2.5-3B
  and 7B. What I take is one table: a question is "within parametric
  competence" when the base model answers it in at least one of five
  no-tool rollouts, and with bm25 search that leaves 24% of hotpot, 43% of
  2wiki and 7% of musique questions parametric for Qwen2.5-7B-Instruct
  (16 / 32 / 4% at 3B). Those are a published yardstick for my parametric
  bucket under the same reader family: my closed-book EM on the
  confirmation folds is 0.173 / 0.214 / 0.032, in the same order and
  below theirs, as one greedy answer against the best of five should be.
  The one-of-five definition is cheap and I could report it beside the
  closed-book control. The rest (EM 47.9 / 48.3 / 30.4 against Search-R1
  42.8 / 41.7 / 22.1, a competence boundary at AUC 0.93) is not my object.

Seen and set aside: 2611.11840 (correct answers from certified
insufficient histories: executable logs where sufficiency is provable, a
reader gives concrete answers on 196 of 203 insufficient inputs and matches
the hidden reference on 87; the Part A cell in a domain with a proof, cite
if the draft needs one), 2605.15721 NCCE (instance-wise prompt routing by
collaborative filtering, hotpot 60.1%; routes prompts, offline),
2611.11509 SimplEvo and 2611.11560 LiveAsset (text lessons and intent
assets, frozen at test, applicability by judge), 2611.12109 SkillDuet
(skills turn detrimental as the model evolves; RL co-training),
2611.10170 EviCover (complementary-evidence recall up, correctness not),
2611.11396 (dense retrievers favour passages with answers the model
already knows), 2611.11771 ARMOR and 2611.11010 Lightfoot (RAG RL and
GraphRAG on the three sets, no numbers I need), 2611.10477 PatternBloom
(reasoning-pattern memory for agentic RAG, no datasets in the abstract),
2605.25379 StateRAG (typed retrieval state, +9.8 F1 musique), 2611.10542
BadEvolve (failure memory as a poisoning channel), 2611.11929 SoMem,
2611.12068 MIRIX-2.0, 2611.10683 TransMem, 2611.12872 HiVe-Mem
(conversational memory systems, no applicability model), 2606.03056
SkillDAG, 2606.04391 SkillAny, 2605.06978 GoSkills, 2611.10022 SkillCache,
2611.12041 SMO, 2604.20133 EvoAgent (skill libraries, no outcome-revised
scope), 2605.09252 When2Tool (tool necessity decodable from the prefill),
2611.10193 PISA and 2611.10848 MultiTurnRoute (model routers), 2611.11842
(shuffling an evidence graph does not hurt belief updating, robotics).

## 2026-11-09

Listings read in full (cs.IR 25, cs.CL 146, cs.AI 397 entries); every
abstract read, the four below read in full.

- Vahdat and Vaezipoor 2026, Decision Context (arXiv 2611.08644). A
  tool-using agent's corrections (from a user) and its self-assessed
  tool-call failures (retry-then-succeed, recurring errors, read off the
  traces) are compiled into imperative rule directives, each wired at
  compile time to a trigger node: the point in the trace before the failing
  call where the rule would have prevented it. At run time a rule is
  injected only where its trigger matches; rules are deduplicated and
  superseded by later corrections, and nothing about where a rule applies
  is revised from what happens after it fires. DC-Bench, 71 scenarios in 9
  behavioural dimensions, corrections seeded on one company and tested on
  another, an Opus 4.7 agent, a GPT-4 rubric judge, 5 runs. Mean compliance:
  no injection 10%, mem0 12%, zep 15%, AWM 34%, Memp 49%, naive RAG over
  traces 55%, annotated chunks 63%, typed graph 70%, the same graph plus
  self-assessment rules injected unconditionally 82%, triggered directives
  95%. That last step moves scope precision 78% -> 96% (8 scenarios) and
  conditional activation 84% -> 95% (12 scenarios) at 159 injected tokens
  per query against 1641; a variant that attaches a reason to each rule is
  flat or lower on every model they tried. The authors say the stages are
  cumulative (compilation, triggering and conflict handling are not
  separated) and that benchmark, baselines and rubrics are all theirs. For
  me this is the Stage 6 pair "every lesson shown every time" against
  "lessons shown where their region holds", in print for tool agents and
  with the region fixed at write time: 82 against 95, confounded as they
  say. Three things to take. Their no-regression check (wherever the
  no-injection agent passes, the memory must still pass) belongs in the A4
  table beside false application. Keep the reflection advice to a bare
  directive. And their scope-precision and conditional-activation scenarios
  are the question-pair analogue of my twins. What they do not do is the
  claim under test: no trigger is narrowed, split or retired from later
  outcomes, and there is no shuffled-content control; the show-all baseline
  now has a published number.
- Schmid et al. 2026, FondueQA (arXiv 2611.09274). A multi-hop benchmark
  built so that every required fact lives in exactly one gold document and
  nowhere else in the question or the other documents (198,375 candidates,
  31,077 validated, 13,063 released; Wikipedia of October 2017), so
  removing a document removes one fact. Conditions: all gold, no gold,
  question only, bridge-only against target-only (the answer-bearing
  document), leave-one-out, and competitor distractors at fixed context
  size. Thirteen model configurations, Qwen3-8B among them. On FondueQA the
  bridge document matters more than the answer-bearing one for most
  models, more so for small ones; competitors that support a wrong answer
  cut EM by up to about 15 points and competitors that support the right
  one add more than 25. The same fact-isolation checks keep 3,424 of 7,404
  HotpotQA (46.2%), 7,740 of 12,576 2Wiki (61.5%) and 531 of 2,417 MuSiQue
  (22.0%) instances, and on those subsets (HotpotQA-Ext, 2Wiki-Ext) the
  role effect reverses: target-only beats bridge-only for most models,
  which matching entity popularity and restricting to easy questions do
  not remove (the popularity matching is run on 2Wiki-Ext). Two things for
  me. Their HotpotQA and 2Wiki result is my intervention arm (the reader
  does as well with the answer-bearing gold alone as with all of it) from an
  independent harness, and their FondueQA reversal says it is a property of
  how the sets were built, not of readers; cite beside Sufficient Context
  and the Stage 5c paragraph. And their retention rates are a caveat on
  the gold-completeness partition: on the standard sets a fact can be
  supported outside its gold paragraph (680 HotpotQA and 566 MuSiQue
  instances fail role isolation alone), so part of my "correct without
  gold" cell is a second source in the pool rather than parametric recall;
  report the share that passes a fact-isolation check, or cite theirs. The
  competitor result also names a distractor the reader-context arms did not
  try: a same-type candidate for the missing hop.
- Luo, Wen and Wang 2026, Agentic Abstention (arXiv 2606.28733, v3 of 6
  Nov). Abstention as a sequential decision (answer, abstain, or act again)
  over 28,000 instructions from WebShop, Terminal-Bench 2.0 and
  AbstentionBench QA sets; timely recall is abstaining at the first step
  where it is warranted. Thirteen systems, the best at 32.6% timely and
  84.2% overall. CONVOLVE is an ACE-style playbook: a reflection model reads
  each rollout for the observation that made stopping warranted, a curator
  writes stopping rules into fixed sections, the playbook goes into the
  system prompt. Built from 20 trajectories (8 more for validation),
  evaluated on 101 held-out abstention-only WebShop tasks: Llama-3.3-70B
  timely recall 26.7 -> 57.4, overall 83.2 -> 100.0; Llama-3-8B 6.9 -> 12.9
  with the 70B playbook. The row that matters is in their Table 2: the same
  model with in-context examples instead gets 55.4 and 97.0, so the
  distilled rules add two points over showing episodes (how many examples
  the ICL row uses is not stated). Never scoped, never revised. For me: the
  stop decision is learnable from a handful of episodes (the stop audit),
  and raw episodes in context is a control the Stage 6 table lacks; add it
  beside the flat and show-all lessons, since a lesson memory that does not
  beat its own source episodes is compression, not learning. One model
  family, n=101, WebShop only.
- Yang et al. 2026, ENGRAM (arXiv 2611.09472). Experience records are typed
  by skill; a skill enters LoRA training once mastered or demonstrated
  often enough, and its records leave retrieval only after a verification
  gate compares the updated model without retrieval against the base model
  with retrieval on reserved probes. Qwen3-8B, 3.1-8.4 points over
  frozen-weight retrieval on six benchmarks. The piece for me is the
  HotpotQA release study: 600 records, 3 of 5 skills pass, 331 archived;
  held out, retain-all EM/F1 54.4/71.1, immediate release (13 records left)
  50.9/66.8, random release at the same 269 retrievable records 51.6/67.6,
  verified release 54.0/70.9. Not my object (the lessons go into weights),
  but the budget-matched random-release row is the control my retire
  operator lacks: retiring the lessons the operator picks has to beat
  retiring the same number at random. Add it to the operator ablation if
  the operators run again.

Seen and set aside: 2611.09792 (randomised in-trajectory interventions for
causal effects on agent outcomes; the balanced-exposure design already noted
from 2610.02070), 2611.09028 Scout (routing from a frozen small model's
prefill activations, features that are not process statistics, for model
routing), 2604.23853 ClawTrace (skill patches by rule type: prune patches
are guardrails, preserve patches regress), 2611.09457 (knn over neighbour
labels recovers most of a retrieval-augmented annotation gain; the surface
key again), 2611.09609 (the wrong-user control is confounded by topic),
2611.08855 (a source's reliability transfers within a scope, in model
beliefs), 2611.09807 (compaction against retrieval memory on LoCoMo, McNemar
on the disagreeing questions), 2602.02865 MemSkill (evolving memory
routines, HotpotQA among the sets), 2611.09348 (poisoned reference answers
become trigger-conditioned rules in a skill optimiser), 2611.09084 ACTOR
(queried labels usable only for later samples, the decide-then-reveal
protocol in test-time adaptation), 2611.08498 (graph-first, dense-fallback
multi-hop RAG, no numbers in the abstract), 2605.13481 (GraphRAG on
hotpot / 2wiki / musique, judge gains), 2611.09554 (lost-in-the-middle by
positional modulation), 2611.09721 MemoryDojo (poisoning through agent
memory).

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
