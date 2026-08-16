"""Retrieval + generation over one example.

single_hop: retrieve once, answer from that context. The IRCoT loop
(multihop_ircot) comes next; every hop gets logged as a HopTrace either way so
per-hop precision can be scored afterwards.
"""
import time

from src.pipeline.generate import Generator
from src.retrieval.base import get_retriever
from src.retrieval.rerank import Reranker
from src.types import HopTrace, PipelineResult

PROMPT = """Answer the question using only the context below. Reply with just \
the answer, no explanation. If it is a yes/no question, reply yes or no.

Context:
{context}

Question: {question}
Answer:"""


class MultiHopPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.retriever = get_retriever(cfg)
        self.reranker = Reranker(cfg) if cfg.rerank.enabled else None
        self.generator = Generator(cfg) if cfg.generation.backend != "none" else None

    def _retrieve(self, query):
        hits = self.retriever.retrieve(query, self.cfg.retrieval.top_k)
        if self.reranker:
            hits = self.reranker.rerank(query, hits)
        return hits

    def run_example(self, ex):
        if self.cfg.pipeline.mode != "single_hop":
            raise NotImplementedError("ircot loop lands next")

        t0 = time.perf_counter()
        self.retriever.index(ex.chunks)
        hits = self._retrieve(ex.question)
        trace = HopTrace(hop=0, query=ex.question, retrieved=hits,
                         gold_chunk_ids=list(ex.gold_chunk_ids))

        answer, ntok, tps, peak = "", 0, 0.0, 0.0
        if self.generator:
            ctx = "\n".join(f"- {h.chunk.text}" for h in hits)
            text, m = self.generator.generate(
                PROMPT.format(context=ctx, question=ex.question))
            answer = text.strip().split("\n")[0].strip()
            ntok, tps, peak = m.completion_tokens, m.tokens_per_s, m.peak_memory_mb

        return PipelineResult(example_id=ex.id, answer=answer, hops=[trace],
                              reasoning_tokens=ntok,
                              latency_s=time.perf_counter() - t0,
                              tokens_per_s=tps, peak_memory_mb=peak)

    def run(self, examples):
        return [self.run_example(ex) for ex in examples]
