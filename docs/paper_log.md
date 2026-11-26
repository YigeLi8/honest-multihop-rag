# Paper log

One entry per paper that changes what this repo should do or cite. Newest
first. Short: what it does, what it means here.

## 2026-11-26

New and cross listings read (cs.IR 14, cs.CL 116, cs.AI 248 entries,
replacements not read; 331 distinct ids, none already logged); every
title screened, 26 abstracts read, four bodies.

- 2026, When Rerankers Work Together: Better and Faster Reranking without
  Joint Training (arXiv 2611.33844). Chains a cheap cached first-pass
  scorer in front of a stronger reranker to prune long contexts, with
  paired bootstrap intervals throughout. The part that bears here is a
  control in their appendix (A.9): on 612 held-out HotpotQA examples,
  Gemma 3 12B is given all annotated evidence, with distractors added up
  to a 50% token cap, so that only the distractors differ. Evidence alone
  is 67.65%; every way of choosing the distractors (random, BM25, MiniLM,
  cross-encoder, their own) gives 61.76 to 63.40, a 4.25 to 5.88 point
  cost with recall held fixed, and the choice of selector makes no
  significant difference. On 2Wiki (n=600) their cascade raises F1 0.527
  -> 0.568 [+0.004, +0.078]. For this work: it is the same direction as
  the A2 control (the single agent reading everything it retrieved, 0.032
  below its notebook) and Part A's gold-only intervention, measured on a
  larger reader with evidence fixed by construction rather than by
  pairing runs. It is a reason the round 2 to 4 wide reads (B4, B9) are
  conditional and should stay so: their gain is on reads whose notebook
  lacks the string, and the other reads would pay this cost. Its
  repair/breakage rates per stage removed are the form the B-arm tables
  already take. Cite with A2.
- 2026, Constraint-Aware Test-Time Scoring for LLM and VLM Candidate
  Selection (arXiv 2611.33161). Picks one of several sampled candidates
  behind hard vetoes and a weighted blend of soft scores, on code, robot
  plans and HotpotQA. On HotpotQA (50 distractor-split questions, 25
  bridge and 25 comparison, gold paragraphs always in the context) one of
  the three vetoes is "answer-question compatible": a regex check that
  yes/no questions get yes/no answers and entity questions entity-shaped
  ones, which is B2's guard as a filter over samples rather than a
  second read. The gain is where the first sample is often broken
  (gpt-oss-20B joint score 0.58 -> 0.79) and nil or negative elsewhere
  (Phi-4 0.735 -> 0.713, Gemma and Qwen within noise at n=50); per-task
  weights never beat uniform ones; no paired test. For this work: cite
  beside B2; it agrees with round 1, where the guard moved ircot (+0.022,
  0 lost) and the agent hardly (+0.004), because on the agent a yes or no
  to a span question mostly marked a notebook without the answer. At
  n=50 with the gold in every context it says nothing about that case.

## 2026-11-25

New and cross listings read (cs.IR 21, cs.CL 126, cs.AI 300 entries,
replacements not read; 375 distinct ids, none already logged); every
title screened, 18 abstracts read, two bodies.

- 2026, SOARS: System One Assisted Reranking for Agentic Search (arXiv
  2611.32917). A deep search agent's every search is reranked in one call
  by a small scorer that is given the agent's sub-query together with the
  original question, over a passage chosen from the long document rather
  than its first tokens. Six API backbones on BrowseComp-Plus and
  WebShaper, paired two-tailed t-tests: GPT-5.4 Mini 73.9 -> 84.7
  accuracy with searches 23 -> 14. They trace gold evidence through the
  trajectory as surfaced (returned by a search), previewed (its snippet
  inspected) and behavior (opened or cited) recall; for GPT-5.5 surfaced
  gold recall is already 94.4 and barely moves, previewed goes 72.9 ->
  88.7. The gain is evidence already returned that is now looked at, not
  new retrieval. For this work: their three recalls are the never
  retrieved / dropped / read places of `agent_losses`, and their gain
  sits in the middle place. The agent's keep step and lookup answer are
  given only the lookup as asked (`src/agent/orchestrator.py`, `asked`
  with #k filled), never the question, so on a plan that stops at the
  bridge entity (13 of the 32 dropped on explore, 24 Nov) the step that
  is shown the answer sentence judges it against a lookup the sentence
  does not answer. The question beside the lookup at the keep step costs
  no call and is not among B1 to B9. Their evidence is reranker recall
  with frontier sub-queries over web documents, not a 7B keep step over
  distractor pools, and more context could move a keep step either way,
  so an arm there would need its control on the questions whose lookups
  were right. Cite beside 2611.28519 for the dropped place.

## 2026-11-24

New and cross listings read (cs.IR 36, cs.CL 204, cs.AI 485 entries,
replacements not read; 615 distinct ids, none already logged); every
title screened, 12 abstracts read, four bodies.

- 2026, SupportChain: Trace-Calibrated Aggregation of Quote-Grounded
  Reasoning Chains for Multi-Hop RAG with Small Readers (arXiv
  2611.28519). Decomposes a question into single-hop sub-questions; each
  hop retrieves, quotes one sentence verbatim, labels it explicit or
  implied and binds a short answer; a final reader sees 5 passages plus
  the hop notes. A logistic model over 16 trace features scores each
  execution with no model call, and several executions are merged by the
  pooled reliability of those that agree, adding executions until support
  passes a threshold (MuSiQue 63.2 F1 at 8.8 calls against 63.0 at 16.9
  with all). Qwen3.5-9B, 1,000 questions per set, paired bootstrap; +3.2
  F1 over HippoRAG 2 on HotpotQA, +17.2 on MuSiQue. The final-step
  ablation is the part that bears on Stage 6b: hop notes without passages
  lose 4.9 / 2.5 / 8.4 F1, passages without notes gain 1.9 on HotpotQA
  and lose 4.6 on MuSiQue, and dropping the final reader for the last
  hop's answer loses 45.8 on 2Wiki. In decomposed MuSiQue dev the first
  failing hop had its support passage retrieved and bound to a wrong
  answer in 33.2% of questions, missing in 5.2%. Several candidates per
  hop, a learned hop scorer with backtracking and gold-decomposition plan
  examples gave no significant gain. For this work: it already does what
  B1 and B5 tried (the final read given hop answers beside evidence) and
  finds it worth something on HotpotQA only together with the passages,
  which is how B1 was built; B1 lost what it won here because the read
  then follows wrong lookups (23 Nov). Their remedy is several executions
  merged by agreement, which multiplies calls, against the agent's 7.8.
  Their "retrieved and bound to a wrong answer" is the dropped place of
  `agent_losses`, and its size against "missing" matches ours on hotpot
  (32 dropped against 28 never retrieved after round 4). Cite in the
  agent section as the nearest design; its 9B reader and pools of about
  10k passages are not our distractor pools.
- 2026, Beyond Evidence Coverage: Encoding-Time Access in Multi-Agent
  Systems (arXiv 2611.29107). Attention masks split 20 paragraphs over
  two logical workers in frozen Qwen3-4B, Qwen3-8B and Llama-3.1-8B, 200
  two-hop MuSiQue-Ans and 200 HotpotQA bridge questions. Blocking
  cross-worker access costs 5.6 / 6.8 F1 against full access; exposing
  only the other worker's supporting paragraph recovers 7.6 / 6.7, and an
  equal-token non-support control gains nothing detectable. Oracle
  support labels, no selector. For this work: a subagent reads its own
  lookup's sentences only, and the final read is the one place both hops'
  sentences meet; their result says the gain is the bridging sentence
  itself, not more text, which is what B9's null on one-sentence
  notebooks (0 lost, 5 won) and A2's full-context loss would predict. A
  token-matched control is the one to copy if a later arm hands a
  subagent the earlier lookup's sentences.
- 2026, Can Search Agents Correct Their Own Errors? Evaluating
  Retrieval, Evidence Delivery, and Answer Completion (arXiv 2611.30243).
  BrowseComp-Plus, four open checkpoints, 709 questions: each model's own
  wrong closed-book answer is replayed as a prior, then what evidence
  reaches it is controlled. A confident assistant prior cut one model's
  paired accuracy by 16.1 points with the answer-bearing passage in
  context; gold at rank 1 often left the decisive page unopened, while
  delivering the passage beat neutral delivery by 51.0 and 22.4 points.
  For this work: the same separation as `agent_losses` (never retrieved,
  retrieved and not kept, read and still wrong), on a harder benchmark,
  and a second instance of the 23 Nov count (the read follows a wrong
  lookup with the right sentence in front of it about a third of the
  time on bridge questions). The search-agent samples are about 100
  questions each.
- 2026, CO-GRAFT: Attributing Reinforcement Learning Gains in Search
  Agents via Role-Level Policy Grafting (arXiv 2611.31185). Search-R1
  trajectories split into four roles (reasoning, search-or-answer,
  query, answer extraction), each grafted between the RL agent and its
  initialization over all 16 assignments with live retrieval; Shapley
  shares of the EM gain on NQ and HotpotQA, Qwen2.5 and Llama-3.2 up to
  7B. The best fixed search budget matches the learned stopping within 1
  EM point, and under lenient scoring the answer role's credit collapses
  for Qwen2.5 (much of its EM gain is format). For this work: a learned
  per-question stop is not worth more than a fixed budget, in line with
  the stop audit and with B3 (one search per lookup, -0.048); and answer
  form moves EM, as the 26 of 71 hand-read wrong answers of 17 Nov that
  were the gold under another form. Cite beside the stop audit.

## 2026-11-23

New and cross listings read (cs.IR 8, cs.CL 91, cs.AI 214 entries,
replacements not read; 265 distinct ids, none already logged); every
title screened, 16 abstracts read, three bodies.

- 2026, Do Senior Agents Do Their Job? Measuring Role Integrity in How
  LLM Agents Use Subordinate Material (arXiv 2611.27139). A senior agent
  turns subordinates' material into a decision, with one property of
  that material changed and the case held fixed: eight models, over
  12,000 episodes, three hierarchies (area chair and program chair over
  reviews, a judge over counsel, a committee chair relaying witnesses).
  A false procedural fact planted in one counsel's excerpt, with the
  correct record in front of the judge, is adopted in 111 of 3,832
  rulings (2.9%, CI [2.0, 3.9]), 81 of them from three open models; the
  authors read that as the error rarely reaching the ruling when the
  record is there, not as evidence that the judge checked it. A generic
  majority against the paper's outcome carries the decision about half
  the time, and the second tier mostly passes on what the first did.
  Witnesses' stated limitations are dropped in 20% to 65% of cited
  claims. Telling the senior agent its duties raises explicit correction
  of the planted error without changing how often it is adopted, and
  also moves the control arms. For Stage 6b: B1 and B5 showed the final
  read repeating a lookup's answer (25 of B1's 30 losses, 10 of B5's
  12); their split is what is not yet counted here. From the committed
  traces with no model: among final answers that repeat a wrong lookup
  answer, how many had the gold string in what the final read was given
  (the record present, as in their court arm) against absent. If most
  had it, the loss sits in the read's use of the notebook; if few, it
  is a lookup failure passed on. An instruction to the final read to
  check each lookup answer against its sentences would be an arm, and
  their result says to score it on questions whose lookups were right
  as well, since such prompts move the controls. Cite in the agent
  section beside 2611.17763 and 2611.18642.

## 2026-11-20

New and cross listings read (cs.IR 21, cs.CL 96, cs.AI 252 entries,
replacements not read; 313 distinct ids, none already logged); every
title screened, 18 abstracts read, four bodies.

- 2026, FinVeriQA: Financial Research Agents for Question Answering:
  Routing, Retrieval, MCP Tool Use, and Provenance-Aware Evidence
  Verification (arXiv 2611.25071). A multi-agent finance QA prototype
  with Mistral-7B as the writer and a validator that checks cited ids
  and figures against the notes the answer was written from. With one
  figure corrupted in the shared evidence, self-consistency agreement
  stays at 0.55-0.59 while accuracy falls from 0.50 to 0.00; a check
  against the shared evidence flags 35% [0.23, 0.47] of Mistral-7B's
  corrupted answers, a check against the original passage 77% [0.65,
  0.87] (DeepSeek-R1-8B: 68% against 90%). The second check reads the
  uncorrupted source, so it is an upper bound. For B6: its test asks
  whether the answer string is in the sentences it was read from, which
  is a check against the shared evidence in their sense. A chain that
  went to the wrong entity at its first lookup hands the final read
  sentences about that entity, and a wrong answer read from them passes
  the test. The count to take from the traces: among the wrong answers
  B6 left alone (string in the context), how many were read from a
  notebook built on an off-path lookup (`first_break` off_path). Cite
  beside the guard results; it is not finance-specific in what matters
  here.
- 2026, How Much Should an Agent Remember? Deployment Lessons from
  Self-Distilled Memory for LLM Agents (arXiv 2611.26361). Lessons
  mined from an agent's own trajectories are merged into guidelines with
  support counts and put into the prompt, on AppWorld. What memory buys
  depends on the model: DeepSeek-V3.2 79.8 -> 89.3 with all 191
  guidelines; gpt-oss-120b 39.9 -> 51.2 with all, 55.4 with about 29
  selected; GLM-5 flat at 87.5; Gemma-4-31B "too weak to apply memory".
  Two identical gpt-oss runs scored 56.0 and 54.8, about 32 of 168 tasks
  flip between identical runs, a high-support guideline was wrong and
  caused a regression, and an early null came from a write-path bug that
  kept the store at its seed. Against ACE most of the headline gap was
  scaffold. The closest published account to the five closed memory
  gates, and it reads a null on one 7B model as expected rather than odd.
  For the write-up of the gates: say per gate that the store grew and was
  read (store size, how often a lesson was injected), and put the
  run-to-run noise beside each difference. On this branch the runs
  repeat exactly (Stage 6b), so the noise there is selection noise over
  questions, not sampling; that should be said, since their flip count
  does not transfer. Cite in the negative results.
- 2026, Forget Freely, Grep Later: Lossless, Index-Free Memory for
  Long-Horizon LLM Agents (arXiv 2611.25123). Qwen3.5-9B trained with RL
  under five context strategies; keeping an append-only plain-text log
  of the whole history with a grep tool beside a lossy summary reaches
  95-97% on ContextBench against at most 21% for summarization or
  observation masking, +8.8 points over summarization on BCP-Multi, and
  0.9 on AMA-Bench after RL, where the summary keeps what the questions
  need. The keep step is such a lossy summary, and dropped answers are
  the case where the log should matter. B4 (the guard's wide second read)
  and B9 (thin notebooks read everything retrieved) already hand the
  final read what was not kept, conditionally, and won 17 and 5 questions
  on 1000; the paper's own result (the gap closes when the summary keeps
  what is needed) is the same shape as A2, where reading everything on
  every question cost 0.032. Cite with the dropped-answer counts.

## 2026-11-19

New and cross listings read (cs.IR 13, cs.CL 72, cs.AI 230 entries,
replacements not read; 277 distinct ids, none already logged); every
title screened, 14 abstracts read, four bodies.

- 2026, Safe to Omit, Worth Omitting? Keeping the Answer and Cutting
  Latency When an LLM Agent Skips Another's Message (arXiv 2611.23570).
  A receiver agent with and without a helper's message, on OpenBookQA,
  ARC-Challenge, MMLU-Pro (and SQuAD, GSM8K) with Qwen pairs. The part
  that matters here is a control: the receiver rerun on inputs with no
  answer-specific content (the options rotated by one, a message written
  for another question), resampled to change as many answers as the real
  message did. Such a null reproduces a median 76% [68, 89] of the
  oracle's gain over the receiver alone (85% stratified by whether the
  receiver was right), so oracle headroom over two paths is weak evidence
  that the second path carries anything. Skipping is gated on the
  receiver's own confidence, with a threshold certified by an exact
  binomial test so that at most 5% of skipped answers change; in a
  prospective test a verifier is skipped on 39.9% of 2,000 sealed
  questions keeping 12.55 of its 12.80 points. No multi-hop QA or
  retrieval. For Stage 6b: point 2 of the traces reads "final-or-last
  would be 0.592" and the 17 Nov read counts 24 of 136 where a lookup's
  answer equals the gold; both are oracle headroom of exactly this kind,
  and B1/B5 not being kept is what the paper would predict if much of it
  is answer volatility. Before a further arm is built on that headroom
  (the 5 bridge and 4 comparison questions), the matching null is a
  count I can take without a model: final-or-X where X is the lookup
  answer of another question of the same shape, against final-or-last.
  The same applies to reading B2/B4's wins as the second read's content
  rather than a second draw. Cite in the agent section beside
  2611.21651.
- 2026, Evidence-Based Credit Assignment for Search-Augmented Language
  Models (arXiv 2611.23824). Search-R1-style GRPO with an extra
  training-only reward: the length-normalized log-likelihood of the
  shortest gold alias under a frozen copy of the policy, given only the
  question and the deduplicated documents retrieved so far (no thoughts,
  no queries). E5-base-v2 over Wikipedia 2018, top-3 per search,
  Qwen2.5-7B-Instruct: EM over outcome-only GRPO on HotpotQA 38.96 ->
  43.05, 2Wiki 36.74 -> 39.35, MuSiQue 13.49 -> 16.88, Bamboogle 32.00
  -> 40.00. The authors call the single alias "an imperfect proxy" and
  do not report how the score relates to final correctness. Nothing
  here is trained, so the method does not transfer; two things do. The
  numbers are a published 7B trained search agent on our three sets
  (open Wikipedia, so not the same task as our pools) to set beside the
  4 Nov agent in the agent section. And the evidence-only score is a
  graded form of the exact string test behind "dropped" and "read",
  which the 17 Nov read showed undercounts (about 26 of 71 the gold
  under another form): scored on a lookup's shown list against its kept
  notebook, it would say how much answer support a keep step throws
  away without hand reading. It would be read by the same 7B, so it
  measures support for this reader, not sufficiency.

## 2026-11-18

New and cross listings read (cs.IR 20, cs.CL 102, cs.AI 259 entries,
replacements not read; 324 distinct ids, none already logged); every
title screened, 14 abstracts read, three bodies.

- 2026, Multi-Agent Memory Inconsistency: Where Factual Conflicts
  Disappear in LLM Agents (arXiv 2611.22245). Planted factual conflicts
  in document-review tasks, a lead agent with and without required
  delegation to workers, an evidence-flow profiler that follows each
  claim from source read to worker report to the lead's answer.
  Delegation retrieves both conflicting claims more often (50.2% ->
  68.7%) and detects the conflict less (16.2% -> 13.2%). The loss is at
  the handoff: workers that read only one side omit one or both claims
  in 73.2% of split cases against 50.5% when one worker reads both, and
  a lead flags 72.2% of conflicts that one report supplies whole against
  17.1% when the two arrive in separate reports. Not QA, but it is the
  same shape as two things on this branch: dropped answers lost at the
  last keep step after the subagent was shown the sentence ("From the
  traces", 16 Nov), and comparison questions planned as parallel
  lookups, where each subagent sees one side and the final read gets
  the two notebooks. Their split/same cohort is a count I can take from
  the committed traces with no model: for comparison questions, whether
  the sentences for both compared things ended in one lookup's notebook
  or in two, against EM. Cite in the agent section beside 2611.18642.
- 2026, Consistency Is Not Correction: Separating Confidence,
  Abstention, and Recovery in LLM Agents (arXiv 2611.21651). Splits a
  confidence gate's value into recoveries minus harms and notes that a
  gate which only replaces a proposal can act only on the base policy's
  errors, so the error count caps the discordant pairs; it recommends
  publishing that budget against the discordant count a paired test
  needs, and the gate's firing rate, before spending generation. This
  is what the plan already does for the guard and its second reads (the
  bounds written before rounds 1-3, firing counts in every round), so
  nothing changes; it is a citation for the method, and a reminder that
  B2/B4/B6 on the agent won 5, 17 and 12 questions where they fired
  on 30 final reads, 125 questions and 390 reads (round 1 explore,
  rounds 2 and 3 pooled).
- 2026, Done, But Still Working: Anytime Evaluation of When Coding
  Agents Should Stop (arXiv 2611.22901). Stopping rules evaluated by
  replaying recorded runs and scoring intermediate states, paired and
  without new trajectories, valid only for policies that stop or pick an
  earlier state and only where replay preserves the state; a
  preregistered replication saves 23% of cost and fails its
  non-inferiority test. For Stage 6b: a lookup's second search is a
  stopping decision (89% search twice), and the traces hold the keep
  step after each search, so "answer after the first search when the
  first keep already holds X" can be read from committed traces only to
  the extent the first keep's output is what the lookup would have
  answered from; the answer read after the first search is not in the
  trace, so this is a bound, not a replay. B3 (one search always, -0.048)
  says the unconditional rule loses; a conditional one has not been
  counted.
- 2026, The Resolution Hypothesis: Agent Evolution's Key Is Identifying
  Improvements (arXiv 2611.23068). Over seven agent-evolution methods
  and six models, helpful candidate prompts, skills and memories are
  proposed, but selection on noisy scores is no better than random at
  telling them from harmful ones, and method margins of a few points are
  smaller than two runs of the same method apart. Cite for why Stage 6b
  keeps changes by a written rule on fixed folds and confirms once.
  Here the runs repeat exactly (temperature 0, identical completions
  where a flag does not fire), so run-to-run noise is not ours; question
  sampling is, which is the 3 to 4 points one fold resolves.
- 2026, Surfacing Disagreement Beats Resolving It: A Gated Ablation
  Study of Provenance-Weighted Memory for Multi-Agent Debate (arXiv
  2611.22461). Five memory mechanisms removed one at a time, each gated
  on a threshold declared beforehand; one survives (passing the ranked
  conflict set rather than one winning fact, +0.340 F1), two cost
  accuracy, and a harness cap first hid three quarters of the planted
  contradictions. For the memory study's write-up, a published example
  of the same gated, mostly-negative design as the five closed gates,
  and of a harness bound found only after the first measurement. Cite
  in the memory section.

## 2026-11-17

New and cross listings read (cs.IR 36, cs.CL 236, cs.AI 428 entries,
replacements not read; 601 distinct ids, none already logged); every
title screened, 31 abstracts read, four bodies.

- 2026, DetouR: Plugging Association Routes into Search Agents for
  Multi-hop Question Answering (arXiv 2611.19967). Adds no search and no
  model call to an existing agent. Two parts: (1) a follow-up query that
  nearly repeats an earlier one (content-word Jaccard at least 0.8) is
  replaced by a query for a page linked from evidence already retrieved;
  (2) the reader's ten passages are chosen by personalized PageRank over
  the hyperlinks among the retrieved passages, seeded at the pages the
  question names. Setup: BM25, Qwen2.5-7B reader, 1,500 dev questions
  each of HotpotQA and 2Wiki, four agents (IRCoT, ReAct, Search-R1 and a
  plan-then-search agent). Mean EM rises 40.6 -> 42.0 on HotpotQA and
  31.6 -> 35.6 on 2Wiki; most of it comes from the context selection.
  Two of its side results bear on this work. A per-passage 7B relevance
  judge, which is the agent's keep step in another form, drops gold pages
  the question does not name; a reranker keeps only 81.2% of those on
  2Wiki against 97.8% for the link-based choice. And in an appendix, a
  selection that gave the same ten passages in another order lost answers
  that putting the order back mostly recovered (53 of 71). Their wrong
  answers split much as ours do: retrieved and left out of the context
  is rare (1.2% and 1.9%), and a complete context is read right only
  43.8% and 51.3% of the time. For Stage 6b: the order result is the
  reason "From the traces" asks for a same-list-other-order control
  before the keep step's position effect is read as anything; a keep
  rule from structure (a candidate's title named in a kept sentence or in
  the question), conditional like B4, is an arm the plan has not listed;
  and their IRCoT at 43-45 EM is a published point of comparison for our
  ircot on the same set and reader size. Cite in the agent section.
- 2026, Agents Cannot Hand Over What They Have Not Yet Understood (arXiv
  2611.18642). One agent with its full history against a relay of agents,
  each inheriting a 300-character note, over 56 rule-discovery worlds and
  six models: the relay learns less for every model (19% to 48% of what
  history gives is lost in the notes), the loss sits in summarizing a
  summary rather than in changing agents, letting the note's writer read
  the full record recovers most of it, and telling the writer to record
  evidence or keep a ledger changes little. Not QA. Cite beside B1 and B5
  (the final read follows a lookup's compressed answer once it is handed
  one) and B4 (the second read gets the retrieved record, not the
  notebook). Its null on writer instructions is one more reason not to
  spend an arm on rewording the keep prompt.

## 2026-11-16

New and cross listings read (cs.IR 7, cs.CL 61, cs.AI 164 entries,
replacements not read; 204 distinct ids); every title screened, 11
abstracts read, three of the four below searched in the body for their
numbers.

- 2026, More Agents, More Samples, or More Thinking? (arXiv 2611.17610).
  A synthesis of nine evaluations, 516 multi-agent configurations, each
  contrast labelled by its baseline: a single greedy pass (A), a baseline
  matched on sample count (B), on resource budget (C), or a frontier (D),
  with a six-part resource vector (tokens, rounds, samples, tool and
  retrieval calls among them) and a verification state per part.
  Against matched baselines multi-agent accuracy averages -0.042 and is
  at or below the baseline in 74% of contrasts; going from a greedy to a
  sampled baseline alone moves the difference by -0.038. The one HotpotQA
  row is +0.040 over 4 contrasts. Their argument for when there is a
  return: partitioned evidence, external verification or error diversity,
  since agents that see the same evidence add nothing by talking. For
  Stage 6b this is a check on how the agent's margin is written: the main
  agent against ircot is 7.4 calls against 2.8, a category A contrast in
  their terms, and nothing in the plan yet holds the budget fixed (ircot
  or the single agent at the agent's call count). The main agent's
  subagents do read partitioned evidence (each its own searches and
  notebook), which is the condition they name, so the comparison is
  worth making rather than assumed away. Cite in the agent section's
  cost paragraph, and report calls, prompt tokens and searches per
  question beside every agent number, which the paired table already
  half does.
- 2026, When Should LLM Agents Forget? (arXiv 2611.17338). Hard deletion
  of facts a resolver labels stale, against keeping everything, under
  controlled resolver error on a 530-item diagnostic: at perfect
  resolution hard-drop gains +0.19 to +0.23, at the lowest reliability
  it loses 0.18, and in paired targeted interventions dropping a
  query-critical fact costs much more than keeping a stale one; a gate
  chosen on a frozen 102-item calibration half reaches 0.655 against
  0.641 always-drop and 0.662 oracle. The agent's keep step is the same
  hard filter one level down: the 53 dropped answers on explore are
  query-critical sentences the subagent saw and did not keep, and B4's
  wide second read is a soft-retention fallback that fires only on a
  yes/no reply. Their asymmetry is what the round 2 numbers already
  showed (12 of the 17 wins were dropped strings, 3 losses where the
  wider context offered a related entity). It argues for counting the
  keep step's false drops directly, which the traces allow without a
  model (From the traces, Stage 6b).
- 2026, Knowledge After Collapse (arXiv 2611.17763). Freezes a dialogue at
  the first adoption of a planted false answer and branches probes from
  that exact history: on 46 selected adoption events with Llama 3.2 3B,
  16/46 right under the full history against 42/46 when the history is
  replaced by the question and a neutral acknowledgment; the matching
  contrasts for the Qwen and DeepSeek checkpoints do not survive
  correction, and twelve pressure templates move adoption a lot. This
  is the B1 / B5 finding from the outside: once a wrong lookup's answer
  is in the final read's context, the read follows it (25 of the 30 B1
  losses, 10 of the 12 B5 losses repeat a lookup's answer). Their
  branch-from-the-same-state design is what the plan already does by
  having the runs repeat exactly up to the final read; nothing to
  change, one citation for the notes paragraph.
- 2026, CarryOn (arXiv 2611.17808). A context manager for long-horizon
  tool agents that keeps a compact state plus source-linked evidence,
  chosen by a learned selector with no extra model calls, and reports
  cost per success beside success (AppWorld, OfficeBench, Tau2; 75.7%
  macro success, +6.7 over the best baseline). Abstract only. The keep
  step here is a model call per search; a selector that needs no call is
  a different design, not run. The one thing to take is the reporting:
  calls per correct answer is a single number that puts the agent's 7.4
  calls and ircot's 2.8 on one scale.

## 2026-11-13

Listings read in full (cs.IR 41, cs.CL 237, cs.AI 537 entries, replacements
included; 690 distinct ids); every title screened, 24 abstracts read, the
five below read in the body (the sixth body, Skill Adapter, is in the
set-aside list).

- Shang, Ge and Guo 2026, GraphSkillAA (arXiv 2609.20455, v3 of
  yesterday). Skills as graph nodes with three addressable fields,
  when_to_use, how_to_use and avoid, plus prereq / enhance edges; the
  student logs which nodes and edges it used; after a failure a frozen
  teacher contrasts it with related successes from the same four-example
  group and routes the repair to one field (a missed activation extends
  only when_to_use, a missing exclusion boundary completes only avoid, an
  execution lapse or insufficient evidence returns NO_PATCH); each repair
  group is retested on the examples whose usage touched the edited object
  (Local Gate) and the merged graph on the whole update pool (Big Gate),
  with rollback by group or by epoch. SearchQA, LiveMath, DocVQA, three
  seeds. GPT-5.6-sol on SearchQA: no skill 66.0, flat skill text 72.5,
  GraphSkillAA 81.5. Ablations (six-setting means): hiding when_to_use and
  avoid at use time with the optimised graph fixed costs 9.4 points (81.5
  to 70.5 on SearchQA, 66.7 to 42.7 on LiveMath), generic failure
  summaries in place of the structured attribution cost 8.3, no Local Gate
  4.9, no edges 1.8; merged candidates that passed local retest still
  regress 3.02-9.86% of out-of-scope examples. This is the plan's narrow /
  expand / exception step written as a routing table over text fields and
  revised offline in epochs with a gold verifier: the applicability
  predicate is prose that the student judges in context at use time, not
  a boundary over features, and the 9.4-point ablation measures showing
  the conditions, not learning them (the conditions are hidden, not
  unlearned). Nothing is compared against a shared router or a scalar
  utility, and nothing moves online as contradicting cases arrive. Cite
  beside Falsification-Gated Self-Evolution as the second published form
  of the static-conditions-with-refit baseline, and its NO_PATCH route is
  the one thing my version-1 memory lacked: an explicit decision that a
  failure was a lapse and writes nothing.
- Guo et al. 2026, MemCredit (arXiv 2611.15600). Per-memory decision
  utility defined by intervention: replay the same environment state with
  and without one memory under a frozen policy and fixed reference
  context, and take the change in return; 1,041 such interventions over
  45 ALFWorld tasks, then a lightweight estimator over state descriptors,
  memory content and failure cues is fitted to the credits and used at
  inference to arbitrate a relevance-recalled candidate pool under a token
  budget. The analysis is the part that matters here: 86.9% of
  interventions change nothing, mean credit 0.054, failure-context mean
  0.176 against 0.029 elsewhere, and semantic relevance against utility
  has Spearman -0.011 overall (-0.191 on the dense subset, +0.135 on the
  broad one). Task-disjoint prediction of the credit: relevance-only
  features Spearman 0.097 / AUROC 0.483, failure-only 0.184 / 0.658, full
  0.220 / AUROC 0.744, shuffled labels +0.048. 134 unseen ALFWorld tasks,
  Qwen3.6-27B policy, 500-token memory budget: BM25 memory 65.67 SR@50,
  dense 71.64, failure-aware 70.15, state-aware 76.12, MemArbiter 82.84,
  MemCredit 86.57 (90.30 at 750 tokens against BM25 81.34); a hybrid that
  mixes relevance into the final ranking drops to 78.36, and removing the
  explicit failure/state indicators from the estimator raises it from
  84.33 to 86.57, so the gain is not a failure shortcut. WebShop 0.612
  against MemArbiter 0.586 at 152 against 428 memory tokens. For me this
  is the shared router over gold-free state features, fitted offline to
  exact per-(memory, situation) credits, and it works: Spearman 0.220 and
  AUROC 0.744 against shuffled labels at 0.048 is the positive result my
  shuffled-reveal control did not produce. The 86.9% zero-utility share
  is the number I have not reported and should, before any boundary is
  fitted: the fraction of (lesson, question) pairs on which applying the
  lesson changes the retrieval outcome at all. If mine is of that order,
  the "no region of the feature space" result is a sparse-label problem
  as much as an absence, and the Stage 5 shared router should be given an
  offline fit to the exact credits as its ceiling before the online
  version is judged. Cite as the state-conditioned per-memory utility
  baseline (a scalar utility that depends on the situation, so it sits
  between my scalar-utility arm and the shared router).
- Abbas, Zia and Windridge 2026, Cited but Unused (arXiv 2611.17161).
  Parametric provenance by counterfactual value substitution: a value
  present in both answer and retrieved passages is replaced in every
  passage, titles included, and the question re-asked; a model that
  repeats the original answered from memory. ASQA, GTR top-5, 800
  questions, ALCE prompts; Qwen2.5-Instruct 0.5B to 14B and
  Llama-3.1-8B. Parametric rate 1.2% at 0.5B rising to 13.3% at 14B,
  Qwen2.5-7B 8.2% [6.0, 11.2] (4.6% with a near substitution, 16.3% far),
  Llama-3.1-8B 15.7%; ELI5 4.5% and QAMPARI 2.4% on the 7B. Parametric
  claims carry a citation as often as contextual ones (35/35 at 7B), so
  NLI support checking is at chance on every model (AUROC intervals all
  contain 0.5, also when restricted to the cited passage). Across ten
  attribution methods the signal is the change in the answer token's
  distribution when context is removed (mean AUROC 0.79; leave-one-out
  0.61; any quantity computed with the context present is at chance);
  asked with no context the model produces the value for 33-84% of
  parametric claims against 10-60% of contextual ones, and among claims
  it already knew 13.2% are parametric against 2.9%. Flagging does not
  pay at these base rates (precision peaks at 0.45). One artefact to copy
  into my own arms: substituting in passage bodies but not titles left
  the true value visible in 31-95% of cases and inflated the rate about
  twofold. For Part A this is a parametric-rate yardstick for my exact
  reader family on single-hop questions with the evidence present but
  contradicted: 8.2% for Qwen2.5-7B beside CARL's 7% on musique and
  SELD's one in five, and the within-family rise from 1.2% to 13.3% says
  the cell should grow with reader size, which is a check I can run at 3B
  and 7B. Two consequences for Stage 5c: any detector of the illusion
  cell read off the reader's with-context confidence is reading a
  quantity they show carries no provenance information, so the no-context
  log-probability of the produced answer is the feature to compute; and
  my reader-cut arms must verify the answer value does not survive in a
  paragraph title or the question after the gold sentence is removed.
  Cite with SELD and CARL in the cell's write-up.
- Pothuru, Setty and Vidra 2026, AgenticRAG-FP (arXiv 2608.20627, v2 of
  Nov 10). Causal failure attribution for multi-hop agentic RAG: a
  certified fault (empty, irrelevant, false-premise or stale evidence,
  query drift, or one corrupted fact span) is injected at hop k, the
  agent re-executes the suffix, and diagnosers are scored on whether they
  name hop k among trajectories that stay wrong. Three-hop MuSiQue, dense
  retrieval, Claude Haiku 4.5: gold-answer-coverage localisation
  (Doctor-RAG style) is 0.91 at hop 1 and 0.00 at hops 2 and 3 (n 43, 36,
  21) because it predicts hop 1 in 96% of failures, an earliest-hop rule
  no better than the always-hop-1 prior; an LLM judge and frozen-hop
  repair reach only 0.25-0.48 at hops 2 and 3. Content faults pooled over
  GPT-4o-mini and Haiku on HotpotQA and MuSiQue with BM25, depth 2 (n
  51): coverage 0.00, judge 0.49, frozen-hop repair 0.59, suffix
  regeneration 0.14. Recovery (corrupted trajectory still answers
  correctly) is 0.71 and 0.70 at depths 1 and 2 and 0.27 at depth 3; a
  corrupted answer fact is absorbed into the answer in 15% of cases, a
  corrupted bridge entity in 7%, a salient entity in 0%; the dense
  HotpotQA sweep had 9 and 2 failed trajectories because the agent
  recovered nearly every injected fault. Two things for me. My failure
  typing (`src/memory/failures.py`) assigns the hop from gold per-hop
  hit/miss, which is coverage localisation: after a hop-1 miss the later
  hops' misses are downstream consequences, so "bridge miss" against
  "named miss" is partly an earliest-miss artefact, and the Stage 3 item
  should say the hop label is observational while theirs is an
  intervention label; the lesson-level consequence is that a lesson keyed
  to a later hop is fitted on a population whose cause is often earlier.
  Second, their 0.70 recovery and 7% bridge absorption are the illusion
  cell measured by intervention on musique and hotpot with an agentic
  retriever (the context is damaged and the answer is still right), the
  same shape as my bridge-drop arm leaving EM level on hotpot, and the
  near-total recovery on hotpot is one more reason the hotpot stream has
  no headroom for a memory to find. Cite in the Stage 3 failure-typing
  item beside Beesetty, and in the Part A write-up with SELD.
- Wu, Liang and Li 2026, SAG2 (arXiv 2610.22588, v2 of Nov 10).
  Embedding-free multi-hop retrieval: a BM25 (SQLite FTS5) index built
  without any model call, a decision model that scores whether each
  candidate supplies an answer-chain fact given the question and the
  passages already selected, and two rounds in which the top seeds' rare
  terms retrieve unvisited passages. HippoRAG's 1,000-question subsets of
  MuSiQue, 2Wiki and HotpotQA with corpora of 11,656, 6,119 and 9,811
  passages: Recall@10 86.62, 97.43 and 99.00 against BM25 54.50, 71.58
  and 86.95, NV-Embed-v2 77.27, 80.75 and 97.70, HippoRAG 2 73.76, 93.40,
  97.15. MuSiQue ablation: scoring the BM25 pool alone adds 13.43 R@10,
  removing the shared context from the scorer costs 10.37, scoring an
  enlarged BM25 pool with the same number of judgments and no expansion
  stays 16.80 below. For me this is the retrieval ceiling on my three
  datasets in a small-corpus setting (not full-wiki, so not my numbers,
  but the same questions): hotpot and 2wiki are at 99 and 97 by a lexical
  index plus a reranker that sees the hop-1 passage, and the remaining
  musique misses are last-hop passages that share no term with anything
  activated (1.9% of supporting passages; 88.1% of lost four-hop last
  hops). The retrieval-level outcome table should quote these as the
  attainable ceiling per dataset, and the finding that the gain is in the
  later-hop passage, conditional on the earlier one, is the argument for
  the small graph arm in Part B being a context-conditioned reranker
  rather than another retriever: that is where the headroom is, and no
  per-question choice among bm25 / dense / hybrid gets at it.

Seen and set aside: 2611.17007 Skill Adapter (per-skill adapters that
carry their own applicability conditions, selected by a GRPO-trained
router from the current subtask and feedback, re-checked when inputs
arrive; ALFWorld 95.48 against SkillRL 87.14, routing policy alone 65.28
mean against 71.86 full; conditions are written text, nothing revised from
outcomes, not on my sets), 2611.16053 Wayfinder (resource router updated
online from the documents the downstream agent cites, beats contextual
bandit baselines when resources overlap; NFCorpus / FEVER testbeds, not
retrieval QA), 2611.14654 (a truthful note beside a preserved error raises
a receiver's recovery 51% to 66% while a clean-looking completed action
lowers it 50% to 44%, AgentCE-Bench: evidence for the "hand over the
failure episode, not the lesson" arm from the WikiTrace entry),
2611.16378 EGE (statistical gate for adopting self-generated updates,
fresh paired audits and non-inferiority tests; Monte Carlo only, no
agent), 2611.15572 (fine-tuning a decision head from logged bandit
feedback; proper score on the chosen action beats confirmed-answers-only
by 2.6 points on Banking77; no retrieval), 2611.15463 TRACE (source-level
counterfactual audit of audio-language models: 6.3% of correct answers
score the target-absent view at or above a target-present one; the Part A
audit in another modality), 2610.34872 (conformal claim filtering in
agentic RAG on HotpotQA; factual support against coverage, no retrieval
numbers), 2611.17068 MAPE and 2611.17316 CoReMem (memory designs evolved
or valued by next-task counterfactuals on LoCoMo and ChainSWE),
2611.16366 CURATOR (joint evidence-span selection with a decision model;
no numbers in the abstract), 2611.17241 Ballad (bandit over LLMs for
heuristic search, non-stationary rewards).

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
