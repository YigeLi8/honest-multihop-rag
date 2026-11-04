# Paper plan

One paper from both branches, for arXiv (cs.IR, cross-list cs.CL) by
15 Nov 2026. A measurement paper with paired tests; no method is proposed.

Working title: *Better evidence, same answers: where multi-hop RAG answers
come from, and what a retrieval memory can and cannot change.*

## The argument in five steps

1. **The cell.** On distractor-pool multi-hop QA, a fixed share of right
   answers come with incomplete gold evidence. Its ordered partition
   (parametric / yes-no / shortcut / paragraph / partial / none) is mostly
   shortcut (an answer-bearing gold sentence in context) and parametric.
   Better retrieval shrinks it. Confirmed under preregistration 1 (H1-H6,
   `results/confirm_prereg.csv`).
2. **Retrieval moves more than answers do.** bm25 to ircot: +0.27
   retrieval_correct for +0.05 EM on hotpot (confirmation fold); the
   reranker loses evidence with EM inside +-0.05 (H3a/H3b); deepening ircot
   to k=20: +0.13 retrieval_correct, EM +0.016 n.s. on exploration
   (preregistration 2, D1/D2, pending).
3. **Why: the reader answers from the answer sentence.** Intervention arms:
   gold minus the bridge leaves EM level, the bridge alone is no better than
   closed-book, and on 2wiki any context without the answer lowers EM below
   closed-book (I1-I3, pending; exploration in plan.md Part A).
4. **The loss is after retrieval, and it is at exact selection.** Gold-only
   beats the full context by about 10 EM points even where the context held
   every gold sentence, and every non-oracle cut (reranker top 3/5, a filter
   trained on the same gold) is below the full context (C1/C2, pending).
   The distraction itself is known (Shi et al. 2023, Cuconasu et al. 2024);
   what is new here is that no tried cut gets any of it.
5. **A retrieval memory has three decisions to learn, and none of them is
   learnable from these features.** Which retriever (Stages 1b-5), when to
   deepen (5b) and what to hand the reader (5c) each show a large oracle gap
   over the best fixed choice, and in each a policy over gold-free features
   recovers none of it: routers and scoped lessons level with always-dense
   on three datasets, shuffled reveals change nothing; the deepening policy
   level with uniform k=13 on the confirmation fold (D3, passed: +0.001
   [-0.016, +0.018]); the chunk filter level with the reranker.

## What is confirmed, and how

| Claim | Evidence | Status |
|---|---|---|
| 1, cell and partition | prereg 1 H1, H4, H5, H6 | confirmed, n=1000 x 3 |
| 2, retrieval vs EM (ircot, rerank) | prereg 1 H2, H3a, H3b; EM rise not preregistered | confirmed; H2 a thin pass |
| 2, deepening | prereg 2 D1, D2 | queued 3 Nov |
| 3, interventions | prereg 2 I1-I3 | queued 3 Nov |
| 4, reader context | prereg 2 C1, C2 | queued 3 Nov |
| 5, routing | stream on positions 0-1499 (hotpot, 2wiki) and full dev (musique), online, 5 orders, held-out halves | not preregistered; report as held-out analysis over both folds |
| 5, deepening policy | prereg 2 D3 | passed 3 Nov |
| 5, context filter | prereg 2 C2 (learned3) | queued 3 Nov |

## Figures and tables

- Fig 1: the 2x2 and the cell's partition per dataset and config
  (`results/*_confirm_decomposition.png`), primary and alternative order.
- Fig 2: retrieval_correct against EM per config, paired arrows (bm25 ->
  k=13 -> ircot -> rerank -> k=20), hotpot confirmation fold.
- Fig 3: intervention arms per dataset with closed-book as the floor.
- Fig 4: the three oracle gaps and what each policy recovers, at matched
  cost (routing ceiling, deepening, context cuts), one panel each.
- Table 1: preregistered tests with pass/fail, both preregistrations.
- Table 2: context cuts against full (C1/C2), with kept-gold rates.
- Appendix: serving corners (`results/serving_corner/`), the tok/s
  correction; failure typing; the twins; preregistration deviations.

## Still to do, in order

- [ ] 4 Nov: read queue12's results against preregistration 2, write the
      outcomes and any deviations into it; regenerate Table 1
- [ ] Part A leftovers needed for Fig 1: alternative order for every run and
      the evidence ladder (CPU); criteria columns in run_qa's 2x2
- [ ] Part A stop audit (CPU, from the ircot traces) if it fits; otherwise
      cut and listed in FUTURE_WORK.md
- [ ] read in full: MILES (2607.06974), 2607.17108, 2609.17043, 2205.03685,
      and the 2026 neighbours in the venue sweep (Ke et al., RAG-X,
      RealHop, Mo et al., CUE-R, Bala; Joren et al. 2025; DiRe)
- [ ] arXiv endorser for cs.IR: ask this week
- [ ] figures 2-4 (script under `scripts/`, from committed csvs only)
- [ ] draft 8-12 Nov, README rewrite from the draft, limitations verbatim
- [ ] 14 Nov read-through for overclaiming; 15 Nov post

## Limitations (to carry into the paper)

- Distractor pools of 10-20 paragraphs, not open-corpus retrieval.
- One reader (Qwen2.5-7B-Instruct, 4-bit, greedy) on one Apple M5 Pro; no
  claim about other readers or sizes; dataset contamination not ruled out.
- Gold is supporting facts mapped to chunks heuristically; the cell is an
  upper bound on unsupported answers; EM is strict (gold-only F1 0.75
  against EM 0.61 on hotpot exploration).
- Every learned policy is one model family (logistic) over one frozen
  feature set; "not learnable" means not by these, not in general.
- Routing results were analysed over exploration and confirmation ids
  together (online and held-out halves), not as a separate preregistered
  confirmation.
- The first preregistration was written after its runs had finished (its
  Status line says so); the second before its runs.
- Context cuts at three sizes only; nothing between 5 chunks and the full
  context was tried.
