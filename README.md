# honest-multihop-rag

Measuring where multi-hop RAG gets the right answer from the wrong evidence.

Two questions, one pipeline:

1. Per-hop retrieval precision against gold supporting facts (HotpotQA distractor,
   2WikiMultihopQA): how often is the final answer correct while retrieval actually
   missed the gold evidence?
2. What does that precision cost to serve? Sweep over quantization, context size
   and reranking on Apple Silicon, plotting accuracy vs throughput.

Status: in progress. The HotpotQA and 2Wiki evaluations and the Metal serving
sweep run end to end; numbers below are dev-set runs on an Apple M5 Pro. Still to
come: an optional entity-graph retrieval arm over the 2Wiki evidence triples, and
maybe a CUDA/vLLM comparison on a rented GPU.

## Setup

Needs native arm64 Python 3.12 (Apple Silicon; GPU work goes through Metal):

    python -c "import platform; print(platform.machine())"   # arm64
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/check_env.py

Don't install vllm / bitsandbytes / auto-gptq / autoawq / faiss-gpu / flash-attn
in this venv. Those are CUDA-only and belong to a separate environment if I rent
a GPU at the end for the comparison run.

## Reproduce

    python -m data.download --datasets hotpotqa
    python -m data.prepare_hotpotqa
    python -m src.eval.run_retrieval --config configs/bm25.yaml
    python -m src.eval.run_qa --config configs/bm25.yaml --backend mlx --n 200
    python -m src.eval.run_qa --config configs/hybrid.yaml --n 200
    python -m src.eval.run_qa --config configs/rerank_bge.yaml --n 200

    # 2wiki: get data_ids_april7.zip (link in the Alab-NII/2wikimultihop readme),
    # unzip into data/raw/2wiki/, then:
    python -m data.prepare_2wiki
    python -m src.eval.run_retrieval --config configs/bm25_2wiki.yaml
    python -m src.eval.run_qa --config configs/bm25_2wiki.yaml --backend mlx --n 200

    # serving sweep (takes hours):
    python -m src.serving.sweep --config configs/serving_sweep.yaml --n 100 --label "M5 Pro (Metal)"

## Results so far

Generator: Qwen2.5-7B-Instruct via MLX (4-bit unless noted), everything local on
an M5 Pro. Hotpot QA rows are n=500 dev subsets (95% Wilson intervals in
parentheses on the headline column); the 2wiki QA row is still n=200.

Note, 1 Oct 2026: the two ircot rows below and the paragraph under the tables
are from before every config shared one answer prompt, and they are superseded.
With one prompt EM is not flat: on the confirmation fold (n=1000) bm25 k=10 to
hybrid ircot is +0.051 EM (paired bootstrap CI 0.027 to 0.075) beside +0.27
retrieval correct, and the cell goes 0.131 -> 0.060. The 2wiki row is still the
first 200 questions (at n=1000 the cell is 0.133). Current tables:
`results/*_confirm_decomposition.csv`, `results/*_confirm_paired.csv` and
`results/confirm_prereg.csv`, with the rules in `docs/preregistration.md`. This
section gets rewritten from those.

Answer vs retrieval, HotpotQA distractor dev (n=500):

| config                      | EM    | F1    | retrieval correct | answer right, evidence incomplete |
|-----------------------------|-------|-------|-------------------|-----------------------------------|
| bm25, single hop, k=10      | 0.428 | 0.559 | 0.512             | 0.148 (0.120-0.182)               |
| hybrid, ircot, k=10         | 0.432 | 0.568 | 0.798             | 0.062 (0.044-0.087)               |
| hybrid, ircot, rerank 20->5 | 0.436 | 0.576 | 0.718             | 0.084 (0.063-0.112)               |

2WikiMultihopQA dev (2-4 hops, harder; n=200):

| config                      | EM    | F1    | retrieval correct | answer right, evidence incomplete |
|-----------------------------|-------|-------|-------------------|-----------------------------------|
| bm25, single hop, k=10      | 0.345 | 0.408 | 0.410             | 0.105 (0.070-0.155)               |

How to read this: "retrieval correct" means every annotated gold sentence made it
into the context the model saw. The last column is the headline number, the share
of all questions answered correctly without complete gold evidence in context.
As a share of correct answers that is 35% (bm25 on hotpot), 30% (bm25 on 2wiki),
and 14-19% even with the stronger retrieval stacks. Two things hold at n=500:
the bm25 and ircot intervals on that column don't overlap, so better retrieval
really does cut the illusion roughly in half; and the three EM values are
statistically indistinguishable (0.428-0.436) while retrieval correctness swings
from 51% to 80%. Answer accuracy alone would hide the entire difference. The two
failure directions decouple, which is the point of measuring them separately.
The ircot runs average 2.2 retrieval hops; per-hop precision/recall plots are in
`results/*_per_hop.png`.

First-stage retrieval, full dev sets (bm25, k=10, sentence-level):

| dataset | n     | precision@10 | recall@10 | all gold in top 10 |
|---------|-------|--------------|-----------|--------------------|
| hotpot  | 7405  | 0.177        | 0.742     | 0.486              |
| 2wiki   | 12576 | 0.155        | 0.639     | 0.378              |

Gold mapping error (annotated facts that failed to map onto a chunk): 1/18005
(0.006%) on hotpot, 70/30687 (0.23%) on 2wiki. Every precision number above
inherits this error.

Serving sweep, single hop, n=100 per point (full table in
`results/sweep_metal.csv`, plot in `results/pareto.png`). The tok/s in that
table and on the plot's x axis is answer tokens over the wall time of the whole
generate call, prefill included. With prompts of a few hundred tokens and
~5-token answers that is mostly a prefill measurement, not decode throughput
(two corners re-run below):

- fastest corner: 4-bit, k=5, ~10.4 tok/s at EM 0.45
- best quality: 8-bit, k=20, EM 0.53 at 4.7 tok/s (about 2.2x slower)
- the interesting middle: 4-bit, k=20 with rerank down to 5 sentences, EM 0.51
  at 10.1 tok/s. Reranked short contexts recover almost all the speed at almost
  the best accuracy, because prefill length is what the k axis really costs.
- peak memory is weights-dominated: ~4.9 GB (4-bit) vs ~8.5 GB (8-bit), with a
  few hundred MB of KV-cache growth at k=20

Two corners re-run on 4 Oct with the streaming runner, which times prefill and
decode separately (`results/serving_corner/`, `scripts/serving_corner.py`;
rerank off, n=100, one run each, times without the model-load question):

| corner      | EM   | retrieval_correct | prompt tokens | TTFT (s) | decode (s) | retrieval (s) | latency (s) | decode tok/s per step | tok/s, old definition (sweep) |
|-------------|------|-------------------|---------------|----------|------------|---------------|-------------|-----------------------|-------------------------------|
| 8-bit, k=20 | 0.53 | 0.85              | 833           | 0.748    | 0.140      | 0.090         | 0.977       | 32.8                  | 4.4 (4.7)                     |
| 4-bit, k=5  | 0.46 | 0.45              | 275           | 0.305    | 0.063      | 0.083         | 0.450       | 60.9                  | 9.5 (10.1)                    |

EM and retrieval match the sweep rows. Rebuilt the old way the tok/s comes out
about 6% under the sweep's, so the old column is what I thought it was, and
time to first token is over 80% of its denominator. End to end the 8-bit k=20
corner is about 2.2x slower (0.977 s against 0.450 s), and most of that gap is
time to first token on a prompt three times as long; per-step decode differs
by 1.9x. The two corners differ in quantization and k at once, so they do not
split the cost between the two (at equal k the sweep's own latency column puts
8-bit at about 1.2x: 0.884 s against 0.740 s at k=20). One more caution: the
tokens_per_s column in the `*_qa.csv` files is mlx-lm's figure, n tokens over
the time of n-1 steps, which reads 43-45% high on answers this short.

Plots: `results/pareto.png`, `results/*_2x2.png`.

## Limitations

- One-week measurement artifact. No new algorithm, no training, no SOTA claims.
- Single device, single stream; latency and throughput are hardware-specific, so
  I report relative trade-offs and state the hardware. Batch/concurrency needs a
  continuous-batching server and is not measured yet.
- Gold retrieval labels come from mapping dataset supporting facts onto chunks
  heuristically. Measured mapping error: 0.006% (hotpot), 0.23% (2wiki).
- "Evidence incomplete" is measured against the annotated gold sentences. Some of
  those answers may rest on genuinely equivalent evidence from other retrieved
  sentences, so the illusion rate is an upper bound on unsupported answers.
- QA numbers are dev subsets (n=500 hotpot, n=200 2wiki, n=100 sweep points)
  with Wilson intervals reported; don't over-read single-point differences.
- RAGAS (LLM-judge) metrics are planned as an indicative companion, not a
  replacement for the gold-label metrics.
- The graph arm, if built, is plain retrieval over an entity graph, not graph
  learning.

Author: Yige Li
