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

IRCOT_PROMPT = """You are gathering evidence to answer a multi-hop question.

Context so far:
{context}

Question: {question}

Think about what is still missing, then reply with exactly one line: either
SEARCH: <short query for the missing fact>
or, if the context is already enough,
ANSWER: <the answer and nothing else>"""


def parse_step(text):
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            return line[7:].strip(), None
        if line.upper().startswith("SEARCH:"):
            return None, line[7:].strip()
    return None, None


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
        t0 = time.perf_counter()
        self.retriever.index(ex.chunks)
        mode = self.cfg.pipeline.mode
        if mode == "single_hop":
            traces, answer, stats = self._single_hop(ex)
        elif mode == "multihop_ircot":
            if self.generator is None:
                raise ValueError("multihop_ircot needs a generation backend")
            traces, answer, stats = self._ircot(ex)
        else:
            raise ValueError(f"unknown pipeline mode {mode!r}")

        ntok, tps, peak = stats
        return PipelineResult(example_id=ex.id, answer=answer, hops=traces,
                              reasoning_tokens=ntok,
                              latency_s=time.perf_counter() - t0,
                              tokens_per_s=tps, peak_memory_mb=peak)

    def _single_hop(self, ex):
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
        return [trace], answer, (ntok, tps, peak)

    def _ircot(self, ex):
        # hotpot doesn't order its gold facts by hop, so every trace carries the
        # full gold set; 2wiki's reasoning_path will give real per-hop gold
        seen = {}
        traces = []
        query = ex.question
        answer = ""
        ntok, tps, peak = 0, 0.0, 0.0

        for hop in range(self.cfg.pipeline.max_hops):
            hits = self._retrieve(query)
            for h in hits:
                seen.setdefault(h.chunk.chunk_id, h)
            traces.append(HopTrace(hop=hop, query=query, retrieved=hits,
                                   gold_chunk_ids=list(ex.gold_chunk_ids)))
            if hop == self.cfg.pipeline.max_hops - 1:
                break
            ctx = "\n".join(f"- {h.chunk.text}" for h in seen.values())
            text, m = self.generator.generate(
                IRCOT_PROMPT.format(context=ctx, question=ex.question))
            ntok += m.completion_tokens
            tps, peak = m.tokens_per_s, max(peak, m.peak_memory_mb)
            ans, search = parse_step(text)
            if ans is not None:
                answer = ans
                break
            if not search or search.lower() == query.lower():
                break   # model is stuck, stop retrieving
            query = search

        if not answer:
            ctx = "\n".join(f"- {h.chunk.text}" for h in seen.values())
            text, m = self.generator.generate(
                PROMPT.format(context=ctx, question=ex.question))
            answer = text.strip().split("\n")[0].strip()
            ntok += m.completion_tokens
            tps, peak = m.tokens_per_s, max(peak, m.peak_memory_mb)
        return traces, answer, (ntok, tps, peak)

    def run(self, examples):
        return [self.run_example(ex) for ex in examples]
