# honest-multihop-rag

<!-- Written by scripts/readme_draft.py from committed results files; do not
edit the numbers by hand, re-run the script. README.md is edited separately. -->

How often a multi-hop RAG pipeline gets the answer right without the gold
supporting evidence in front of it, and what that cell is made of. Scope:
retrieval is over each question's own distractor pool (HotpotQA distractor,
MuSiQue, 2WikiMultihopQA), not an open corpus; one generator,
Qwen2.5-7B-Instruct at 4 bits through MLX; every run on one Apple M5 Pro.
Every difference below is paired on identical question ids (exact McNemar and
a paired bootstrap, Holm across the primary tests). The confirmation folds
(n=1000 per dataset) were read once, under criteria written first
(`docs/preregistration.md`, `docs/preregistration_2.md`). Nothing here is a
new method.

## The cell, on the confirmation folds

"Retrieval correct": every annotated gold sentence was in the context the
reader saw. "Cell": the share of all questions answered right (EM) with
retrieval not correct.

hotpot, confirmation fold (`results/hotpot_confirm_decomposition.csv`):

| config | n | EM | F1 | retrieval correct | cell [95% Wilson] | cell / right answers | closed book EM |
|---|---|---|---|---|---|---|---|
| bm25, single hop, k=10 | 1000 | 0.415 | 0.540 | 0.502 | 0.131 [0.112, 0.153] | 0.316 | 0.173 |
| bm25, single hop, k=13 | 1000 | 0.441 | 0.571 | 0.610 | 0.106 [0.088, 0.127] | 0.240 | 0.173 |
| hybrid ircot, k=10 | 1000 | 0.466 | 0.606 | 0.774 | 0.060 [0.047, 0.076] | 0.129 | 0.173 |
| hybrid ircot, rerank 20->5 | 1000 | 0.484 | 0.620 | 0.657 | 0.106 [0.088, 0.127] | 0.219 | 0.173 |

musique, confirmation fold (`results/musique_confirm_decomposition.csv`):

| config | n | EM | F1 | retrieval correct | cell [95% Wilson] | cell / right answers | closed book EM |
|---|---|---|---|---|---|---|---|
| bm25, single hop, k=10 | 1000 | 0.138 | 0.224 | 0.169 | 0.078 [0.063, 0.096] | 0.565 | 0.032 |
| bm25, single hop, k=7 | 1000 | 0.152 | 0.250 | 0.243 | 0.073 [0.059, 0.091] | 0.480 | 0.032 |
| hybrid, single hop | 1000 | 0.173 | 0.273 | 0.278 | 0.074 [0.059, 0.092] | 0.428 | 0.032 |
| hybrid ircot, k=10 | 1000 | 0.223 | 0.337 | 0.553 | 0.051 [0.039, 0.066] | 0.229 | 0.032 |

2wiki, confirmation fold (`results/2wiki_confirm_decomposition.csv`):

| config | n | EM | F1 | retrieval correct | cell [95% Wilson] | cell / right answers | closed book EM |
|---|---|---|---|---|---|---|---|
| bm25, single hop, k=10 | 1000 | 0.320 | 0.368 | 0.365 | 0.133 [0.113, 0.155] | 0.416 | 0.214 |
| bm25, single hop, k=13 | 1000 | 0.349 | 0.400 | 0.491 | 0.102 [0.085, 0.122] | 0.292 | 0.214 |

The preregistered tests on hotpot:

| test | metric | a -> b | value | diff [95% CI] | only a / only b | Holm p | verdict |
|---|---|---|---|---|---|---|---|
| H1 | illusion | bm25 -> hybrid_ircot | 0.131 -> 0.060 | -0.071 [-0.090, -0.052] | 88 / 17 | 2e-12 | pass |
| H4 | illusion | bm25_k13 -> hybrid_ircot | 0.106 -> 0.060 | -0.046 [-0.063, -0.029] | 63 / 17 | 2.3e-07 | pass |
| H2 | em | bm25_k13 -> hybrid_ircot | 0.441 -> 0.466 | +0.025 [+0.002, +0.048] | 57 / 82 | - | pass, small difference |
| H3a | retrieval_correct | hybrid_ircot -> hybrid_ircot_rerank | 0.774 -> 0.657 | -0.117 [-0.146, -0.088] | 178 / 61 | 5.5e-14 | pass |
| H3b | em | hybrid_ircot -> hybrid_ircot_rerank | 0.466 -> 0.484 | +0.018 [-0.005, +0.041] | 62 / 80 | - | pass |

(`results/confirm_prereg.csv`; the rules are in `docs/preregistration.md`.)

Better retrieval roughly halves the cell on hotpot while EM moves by a few
points; EM alone would hide most of the difference.

## What the cell is made of

Each right answer with incomplete evidence is given the first cause that
holds, in this order: the closed-book run on the same question is right too
(parametric); a yes/no question; the gold sentence that holds the answer was
retrieved and some other gold sentence, usually the bridge, was not
(shortcut); every gold paragraph was retrieved but not every gold sentence
(paragraph); some gold was retrieved (partial); none was (none). The last
three together are the unexplained residual (`src/eval/analyze.py`,
`bucket`). Counts, bm25 k=10, confirmation folds:

| dataset | n | cell | parametric | yesno | shortcut | paragraph | partial | none | unexplained | shortcut, evidence-first order | file |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hotpot | 1000 | 131 | 42 | 2 | 81 | 3 | 3 | 0 | 6 | 105 | `hotpot_confirm_decomposition.csv` |
| musique | 1000 | 78 | 9 | 0 | 67 | 0 | 2 | 0 | 2 | 71 | `musique_confirm_decomposition.csv` |
| 2wiki | 1000 | 133 | 56 | 0 | 57 | 8 | 9 | 3 | 20 | 94 | `2wiki_confirm_decomposition.csv` |

Most of the cell is an answer read off the one gold sentence that holds it,
with the hop that should lead there missing from the context; on 2wiki the
closed-book knowledge is about as large a part. The evidence-first order
(shortcut before parametric, `bucket_alt`) is in the same files and keeps
shortcut the largest cause on all three.

## The ablation: what the reader is handed

The same questions answered from controlled contexts: only the gold
sentences, only the gold sentence holding the answer, only the bridge
sentences (gold without the answer), and only distractors.

| dataset | context given | n | EM [95% Wilson] | closed book, same ids | sentences |
|---|---|---|---|---|---|
| hotpot | gold | 1000 | 0.630 [0.600, 0.659] | 0.173 | 2.430 |
| hotpot | answer_only | 825 | 0.550 [0.516, 0.584] | 0.126 | 1.050 |
| hotpot | bridge_only | 825 | 0.121 [0.101, 0.145] | 0.126 | 1.460 |
| hotpot | distractors | 996 | 0.139 [0.118, 0.161] | 0.171 | 9.990 |
| musique | gold | 1000 | 0.436 [0.406, 0.467] | 0.032 | 2.640 |
| musique | answer_only | 941 | 0.422 [0.391, 0.454] | 0.029 | 1.020 |
| musique | bridge_only | 941 | 0.027 [0.018, 0.039] | 0.029 | 1.630 |
| musique | distractors | 1000 | 0.015 [0.009, 0.025] | 0.032 | 5.000 |
| 2wiki | gold | 1000 | 0.550 [0.519, 0.581] | 0.214 | 2.420 |
| 2wiki | answer_only | 845 | 0.486 [0.453, 0.520] | 0.186 | 1.000 |
| 2wiki | bridge_only | 845 | 0.082 [0.065, 0.102] | 0.186 | 1.360 |
| 2wiki | distractors | 1000 | 0.144 [0.124, 0.167] | 0.214 | 9.720 |

(`results/interventions_{hotpot,musique,2wiki}_confirm_summary.csv`)

| dataset | a -> b | n | EM | diff [95% CI] | only a / only b | Holm p |
|---|---|---|---|---|---|---|
| hotpot | gold -> answer_only | 825 | 0.593 -> 0.550 | -0.042 [-0.068, -0.017] | 77 / 42 | 0.005 |
| hotpot | bridge_only -> answer_only | 825 | 0.121 -> 0.550 | +0.429 [+0.393, +0.466] | 16 / 370 | < 1e-4 |
| hotpot | closed_book -> bridge_only | 825 | 0.126 -> 0.121 | -0.005 [-0.029, +0.019] | 56 / 52 | 0.773 |
| musique | gold -> answer_only | 941 | 0.442 -> 0.422 | -0.020 [-0.049, +0.009] | 110 / 91 | 0.408 |
| musique | bridge_only -> answer_only | 941 | 0.027 -> 0.422 | +0.395 [+0.364, +0.427] | 4 / 376 | < 1e-4 |
| musique | closed_book -> bridge_only | 941 | 0.029 -> 0.027 | -0.002 [-0.015, +0.011] | 20 / 18 | 0.871 |
| 2wiki | gold -> answer_only | 845 | 0.542 -> 0.486 | -0.056 [-0.091, -0.020] | 144 / 97 | 0.003 |
| 2wiki | bridge_only -> answer_only | 845 | 0.082 -> 0.486 | +0.405 [+0.369, +0.439] | 12 / 354 | < 1e-4 |
| 2wiki | closed_book -> bridge_only | 845 | 0.186 -> 0.082 | -0.104 [-0.131, -0.077] | 117 / 29 | < 1e-4 |

(`results/interventions_{hotpot,musique,2wiki}_confirm_paired.csv`)

The answer-bearing sentence alone carries most of what the gold evidence is
worth; the bridge sentences alone are at closed-book level on hotpot and musique and
below it on 2wiki, where the closed-book run knows more of the answers. That is why
completing the evidence moves the cell far more than it moves EM.

## When ircot stops

The stop audit sorts the ircot runs by why the loop ended: the model said it
was done, it repeated a query, or it hit the cap. A stop is premature when
the model stopped itself (done or a repeated query) with the gold not
complete.

| run, confirm | stop | n | share | EM [95% Wilson] | retrieval correct | stopped with gold missing | gold never complete | hops |
|---|---|---|---|---|---|---|---|---|
| hybrid_ircot_hotpot | all | 1000 | 1.000 | 0.466 [0.435, 0.497] | 0.774 | 0.162 | 0.226 | 2.177 |
| hybrid_ircot_hotpot | repeated_query | 385 | 0.385 | 0.475 [0.426, 0.525] | 0.738 | 0.262 | 0.262 | 1.987 |
| hybrid_ircot_hotpot | max_hops | 378 | 0.378 | 0.442 [0.393, 0.492] | 0.831 | 0.000 | 0.169 | 3.000 |
| hybrid_ircot_hotpot | done | 237 | 0.237 | 0.489 [0.426, 0.553] | 0.743 | 0.257 | 0.257 | 1.173 |
| hybrid_ircot_musique | all | 1000 | 1.000 | 0.223 [0.198, 0.250] | 0.553 | 0.381 | 0.447 | 2.625 |
| hybrid_ircot_musique | repeated_query | 659 | 0.659 | 0.208 [0.179, 0.240] | 0.511 | 0.489 | 0.489 | 2.426 |
| hybrid_ircot_musique | done | 128 | 0.128 | 0.289 [0.218, 0.373] | 0.539 | 0.461 | 0.461 | 1.359 |
| hybrid_ircot_musique | max_hops | 213 | 0.213 | 0.230 [0.179, 0.291] | 0.690 | 0.000 | 0.310 | 4.000 |

(`results/hotpot_stop_audit_summary.csv`, `results/musique_stop_audit_summary.csv`)

On the confirmation folds a premature stop ends 0.162 of the hotpot
questions (EM 0.253 on them, against 0.524 where the gold was complete) and
0.381 of the musique questions (EM 0.113 on them, against 0.311 where the
gold was complete). The cap is not where the evidence is lost: a run that
hits it has the highest retrieval correct of the three on both datasets.

## Experience memory: five gates, all closed

The second half of the project asked whether a memory of past retrieval
outcomes can tell when a lesson applies. Each gate had a control fixed before
its run; none was passed, and the confirmation fold was spent only where the
plan said so (D1-D3, C0-C3).

| gate | what was tested | reference | difference [95% CI] | file |
|---|---|---|---|---|
| 1 routing, hotpot | which retriever per question, replayed arms | best single arm 0.634, oracle 0.735 | largest of 32 routed rows (pool): +0.004 [-0.006, +0.015] (paired bootstrap over questions) | `results/hotpot_arms_routed.csv` |
| 1 routing, musique | which retriever per question, replayed arms | best single arm 0.326, oracle 0.411 | largest of 14 routed rows (question_dense_knn5): +0.011 +- 0.004 sd over 10 half-splits, no test in this table | `results/musique_mac_arms_routed.csv` |
| 1 routing, 2wiki | which retriever per question, replayed arms | best single arm 0.573, oracle 0.626 | largest of 14 routed rows (size): +0.000 +- 0.000 sd over 10 half-splits, no test in this table | `results/2wiki_arms_routed.csv` |
| 1 routing, musique, fit on explore, scored on confirm | which retriever per question, replayed arms | best single arm 0.324, oracle 0.409 | largest of 14 routed rows (question_knn5_discordant): +0.009 (one split, no test in this table) | `results/arms_foldsplit/musique_mac_arms_routed.csv` |
| 2 deepening, hotpot | the answer side of deepening: ircot k=20 against k=10, confirm | all gold 0.774 -> 0.925, +0.151 [+0.128, +0.174] | EM 0.466 -> 0.476, +0.010 [-0.009, +0.029], 43 / 53, p 0.358 | `hybrid_ircot_k20_hotpot_confirm_qa.csv` against `hybrid_ircot_hotpot_confirm_qa.csv` (paired here) |
| 3 reader cut, hotpot | k=20 context cut to gold, confirm (n 1000) | full context EM 0.476 | +0.134 [+0.106, +0.161], Holm p < 1e-4 | `reader_ctx_hotpot_confirm_paired.csv` |
| 3 reader cut, hotpot | k=20 context cut to rerank3, confirm (n 1000) | full context EM 0.476 | -0.031 [-0.057, -0.005], Holm p 0.076 | `reader_ctx_hotpot_confirm_paired.csv` |
| 3 reader cut, hotpot | k=20 context cut to rerank5, confirm (n 1000) | full context EM 0.476 | -0.017 [-0.042, +0.008], Holm p 0.255 | `reader_ctx_hotpot_confirm_paired.csv` |
| 3 reader cut, hotpot | k=20 context cut to learned3, confirm (n 1000) | full context EM 0.476 | -0.022 [-0.050, +0.005], Holm p 0.255 | `reader_ctx_hotpot_confirm_paired.csv` |
| 4 agent memory v1 | region memory on the main agent, explore (n 500) | no memory EM 0.540 | -0.004 [-0.024, +0.014], 13 / 11, p 0.839 | `orchestrator_mem_hotpot_explore_paired.csv` |
| 4 agent memory v1 | global memory on the main agent, explore (n 500) | no memory EM 0.540 | -0.034 [-0.066, -0.002], 42 / 25, p 0.050 | `orchestrator_mem_hotpot_explore_paired.csv` |
| 4 agent memory v1 | shuffled memory on the main agent, explore (n 500) | no memory EM 0.540 | +0.006 [-0.014, +0.026], 11 / 14, p 0.690 | `orchestrator_mem_hotpot_explore_paired.csv` |
| 4 agent memory v1 | flat memory on the main agent, explore (n 500) | no memory EM 0.540 | -0.032 [-0.062, +0.000], 39 / 23, p 0.056 | `orchestrator_mem_hotpot_explore_paired.csv` |
| 5 agent memory v2 | region memory on the main agent, explore (n 500) | no memory EM 0.540 | +0.006 [-0.022, +0.034], 24 / 27, p 0.780 | `orchestrator_mem2_hotpot_explore_paired.csv` |
| 5 agent memory v2 | global memory on the main agent, explore (n 500) | no memory EM 0.540 | -0.012 [-0.042, +0.018], 32 / 26, p 0.512 | `orchestrator_mem2_hotpot_explore_paired.csv` |
| 5 agent memory v2 | shuffled memory on the main agent, explore (n 500) | no memory EM 0.540 | +0.004 [-0.020, +0.030], 20 / 22, p 0.878 | `orchestrator_mem2_hotpot_explore_paired.csv` |
| 5 agent memory v2 | flat memory on the main agent, explore (n 500) | no memory EM 0.540 | -0.024 [-0.052, +0.004], 33 / 21, p 0.134 | `orchestrator_mem2_hotpot_explore_paired.csv` |
| 5 agent memory v2 | region build, subagent steps shown a lesson vs held out | success 0.624 (n 1456) vs 0.624 (n 1537) | -0.000 [-0.035, +0.034] | `results/agent_memory/build_summary2.csv` |
| 5 agent memory v2 | region build, planner steps shown a lesson vs held out | success 0.949 (n 137) vs 0.972 (n 144) | -0.023 [-0.069, +0.022] | `results/agent_memory/build_summary2.csv` |

The routing rows are the most favourable row of each table, picked after
the fact. On musique every neighbour vote is a point or so above the best
single arm over the half-splits, a spread over resplits of the same
questions and not a test; fit on the exploration fold and scored once on
the confirmation fold the largest row is under a point, and it is not
claimed. Gate 2's policy (deepening 30% of hops, fit on the exploration log and
applied once to the hotpot confirmation log, preregistration 2 D3) is
reported in `docs/plan.md` (Stage 5b) and has no committed table of its
own; the row above is the answer-level side, read from the two runs. The
memory runs on the agent (gates 4 and 5) are exploration only: the
confirmation run was not made for a version that is level on exploration.

## The agent (exploration only)

A main agent that plans lookups and hands each to a subagent that searches,
keeps sentences and answers; the final read is over the kept sentences. It
has been changed one measured step at a time against a matched control
(`docs/plan.md`, Stage 6b). Every number in this section was chosen on the
folds it is measured on (hotpot explore, positions 0-499 of dev, and
train_hard); none is a result until the confirmation run in preregistration 3.

| comparison | folds | n | EM | diff [95% CI] | lost / won | p | calls | file |
|---|---|---|---|---|---|---|---|---|
| 4 Nov main agent vs ircot | explore | 500 | 0.486 -> 0.540 | +0.054 [+0.018, +0.088] | 28 / 55 | 0.004 | 2.8 -> 7.2 | `agent_dev_hotpot_explore_paired.csv` |
| round 4 agent vs round 3 agent | explore + train_hard | 1000 | 0.548 -> 0.556 | +0.008 [+0.000, +0.016] | 4 / 12 | 0.077 | 7.8 -> 7.8 | `agent_dev4_both_hotpot_pooled_paired.csv` |
| round 4 agent vs ircot G+titles+B6 (the confirmation's ircot) | explore + train_hard | 1000 | 0.527 -> 0.556 | +0.029 [+0.001, +0.056] | 83 / 112 | 0.045 | 2.9 -> 7.8 | `agent_dev4_vs_ircot_hotpot_pooled_paired.csv` |
| round 4 agent vs ircot G+B6 (same answer step) | explore + train_hard | 1000 | 0.498 -> 0.556 | +0.058 [+0.033, +0.083] | 51 / 109 | < 1e-4 | 2.9 -> 7.8 | `agent_dev4_vs_ircot_hotpot_pooled_paired.csv` |

Where the round 4 agent's wrong answers are, by where the gold string was
(an exact per-sentence string test, so "read" is a lower bound):

| fold | n | EM | wrong | string read | retrieved, not kept | never retrieved | yes/no gold | calls | file |
|---|---|---|---|---|---|---|---|---|---|
| explore | 500 | 0.576 | 212 | 149 | 32 | 28 | 3 | 7.7 | `agent_dev4_both_hotpot_explore_losses.csv` |
| train_hard | 500 | 0.536 | 232 | 167 | 36 | 28 | 1 | 7.9 | `agent_dev4_both_hotpot_train_hard_losses.csv` |

## Plots

- `results/hybrid_ircot_hotpot_confirm_2x2.png`
- `results/bm25_hotpot_confirm_2x2.png`
- `results/hotpot_confirm_decomposition.png`
- `results/hybrid_ircot_hotpot_confirm_per_hop.png`
- `results/hybrid_ircot_musique_confirm_per_hop.png`
- `results/pareto.png`

## Reproduce

On the Mac (Apple Silicon, Python 3.12, Metal):

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    scripts/run_all.sh --smoke       # a few questions per stage, into a scratch copy of results/
    scripts/run_all.sh               # everything; the [model] stages take days

Without the model (any machine; reads data/processed and committed results):

    pip install -r requirements-linux-cpu.txt
    scripts/run_all.sh --no-model
    python scripts/readme_draft.py   # this page

2WikiMultihopQA needs `data_ids_april7.zip` (link in the Alab-NII readme)
unzipped into `data/raw/2wiki/`. Pins and what is still to fill from the Mac
are in `docs/pins.md`; `results/manifest.csv` says which command, config and
fold wrote each results file. `notebooks/demo.ipynb` reads the committed
results only.

## Limitations

- Distractor pools, not open retrieval: every pool holds the gold
  paragraphs, so retrieval here is easier than over a corpus, and the cell
  would be measured differently there.
- One generator (a 7B model at 4 bits) on one machine. Nothing says the
  decomposition, the ablation or the agent's margins carry to other models.
- "Retrieval correct" is against the annotated gold sentences, mapped onto
  chunks heuristically (the prepare step reports the facts that failed to
  map; every precision number inherits that error). A right answer read
  from an unannotated but sufficient sentence counts in the cell, and
  nothing here measures how often that happens, so the cell is an upper
  bound on unsupported answers.
- EM against a single gold string undercounts right answers written another
  way. The preregistered H6 reads the cell's share of right answers under
  F1 at 0.5 and under containment as well, within 0.04 of EM's on every
  hotpot and 2wiki run (`results/confirm_prereg.csv`); the agent tables
  report F1 and containment beside EM.
- The memory gates are negative results for the memories as built, on one
  model; they do not show that no memory could help.
- The agent numbers are chosen on the folds they are reported on, and the
  agent makes several times ircot's model calls (the calls column above);
  its confirmation run has not been made.
- MuSiQue and 2wiki have no agent runs; 2wiki has no ircot run on its
  confirmation fold.

Author: Yige Li

<!-- files read: hotpot_confirm_decomposition.csv, 2wiki_confirm_decomposition.csv, interventions_hotpot_confirm_paired.csv, interventions_musique_confirm_paired.csv, interventions_2wiki_confirm_paired.csv, agent_dev_hotpot_explore_paired.csv, agent_dev4_both_hotpot_pooled_paired.csv, agent_dev4_vs_ircot_hotpot_pooled_paired.csv, agent_dev4_both_hotpot_explore_losses.csv, agent_dev4_both_hotpot_train_hard_losses.csv, musique_confirm_decomposition.csv, confirm_prereg.csv, interventions_hotpot_confirm_summary.csv, interventions_musique_confirm_summary.csv, interventions_2wiki_confirm_summary.csv, hotpot_stop_audit_summary.csv, musique_stop_audit_summary.csv, hotpot_arms_routed.csv, musique_mac_arms_routed.csv, 2wiki_arms_routed.csv, arms_foldsplit/musique_mac_arms_routed.csv, hybrid_ircot_hotpot_confirm_qa.csv, hybrid_ircot_k20_hotpot_confirm_qa.csv, reader_ctx_hotpot_confirm_paired.csv, orchestrator_mem_hotpot_explore_paired.csv, orchestrator_mem2_hotpot_explore_paired.csv, agent_memory/build_summary2.csv -->
