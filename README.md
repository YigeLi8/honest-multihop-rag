# honest-multihop-rag

Measuring where multi-hop RAG gets the right answer from the wrong evidence.

Two questions, one pipeline:

1. Per-hop retrieval precision against gold supporting facts (HotpotQA distractor,
   2WikiMultihopQA): how often is the final answer correct while retrieval actually
   missed the gold evidence?
2. What does that precision cost to serve? Sweep over quantization, batch size,
   top-k and reranking on Apple Silicon, plotting precision vs throughput.

Status: day 1, scaffolding. Nothing to reproduce yet.

## Setup

Needs native arm64 Python 3.12 (Apple Silicon; GPU work goes through Metal):

    python -c "import platform; print(platform.machine())"   # arm64
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/check_env.py

Don't install vllm / bitsandbytes / auto-gptq / autoawq / faiss-gpu / flash-attn
in this venv. Those are CUDA-only and belong to a separate environment if I rent
a GPU at the end for the comparison run.

## Layout

    configs/    one yaml per run, merged over base.yaml
    data/       download + prep for HotpotQA, 2Wiki, BEIR subset
    src/
      retrieval/  bm25, dense (BGE + faiss), hybrid, reranker
      graph/      entity-graph retrieval over 2Wiki triples (optional arm)
      pipeline/   IRCoT-style multi-hop loop, generation
      serving/    mlx / llama.cpp runners, sweep
      eval/       gold mapping, per-hop precision, 2x2 decoupling, ragas, systems
    results/    csv + plots

`scripts/run_all.sh` will reproduce everything once the pieces exist.

## Results

TBD.

## Limitations

- One-week measurement artifact. No new algorithm, no training, no SOTA claims.
- Single device; latency and throughput are hardware-specific, so I report
  relative trade-offs and state the hardware.
- Gold retrieval labels come from mapping dataset supporting facts onto chunks
  heuristically. The mapping error is measured and reported, since every
  precision number downstream inherits it.
- RAGAS metrics are LLM-as-judge and reported as indicative only, next to the
  gold-label metrics.
- The graph arm is plain retrieval over an entity graph, not graph learning.
- Subset evals report sample sizes and confidence intervals.

Author: Yige Li
