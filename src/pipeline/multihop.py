"""Retrieval + generation over one example.

single_hop: retrieve once, answer from that context.
multihop_ircot: interleave retrieve -> reason -> retrieve until the model answers.
closed_book: no retrieval at all; the control that tells parametric answers
apart from answers that actually came from the context.

Every hop gets logged as a HopTrace so per-hop precision can be scored
afterwards, and every raw completion is kept on the result for later audits.
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

CLOSED_BOOK_PROMPT = """Answer the question from memory. Reply with just the \
answer, no explanation. If it is a yes/no question, reply yes or no.

Question: {question}
Answer:"""

IRCOT_PROMPT = """You are gathering evidence to answer a multi-hop question. \
Do not answer it yet.

Context so far:
{context}

Question: {question}

Think about what is still missing, then reply with exactly one line: either
SEARCH: <short query for the missing fact>
or, if the context already contains everything needed,
DONE"""


def parse_step(text):
    """(done, search_query). The loop only ever gathers evidence; the answer is
    produced afterwards by the same PROMPT every config uses, so answer
    formatting never differs between single-hop and multi-hop runs."""
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("DONE") or line.upper().startswith("ANSWER:"):
            return True, None
        if line.upper().startswith("SEARCH:"):
            return False, line[7:].strip()
    return False, None


def first_line(text):
    return text.strip().split("\n")[0].strip()


class MultiHopPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.retriever = get_retriever(cfg) if cfg.pipeline.mode != "closed_book" else None
        self.reranker = Reranker(cfg) if cfg.rerank.enabled else None
        self.generator = Generator(cfg) if cfg.generation.backend != "none" else None

    def _retrieve(self, query):
        hits = self.retriever.retrieve(query, self.cfg.retrieval.top_k)
        if self.reranker:
            hits = self.reranker.rerank(query, hits)
        return hits

    def run_example(self, ex):
        t0 = time.perf_counter()
        mode = self.cfg.pipeline.mode
        self._raw = []
        self._prompt_tokens, self._ttft = 0, 0.0
        self.stop_reason = ""
        if mode == "closed_book":
            if self.generator is None:
                raise ValueError("closed_book needs a generation backend")
            traces, answer, stats = self._closed_book(ex)
        elif mode == "single_hop":
            self.retriever.index(ex.chunks)
            traces, answer, stats = self._single_hop(ex)
        elif mode == "multihop_ircot":
            if self.generator is None:
                raise ValueError("multihop_ircot needs a generation backend")
            self.retriever.index(ex.chunks)
            traces, answer, stats = self._ircot(ex)
        else:
            raise ValueError(f"unknown pipeline mode {mode!r}")

        ntok, tps, peak = stats
        return PipelineResult(example_id=ex.id, answer=answer, hops=traces,
                              raw_outputs=list(self._raw),
                              stop_reason=self.stop_reason,
                              reasoning_tokens=ntok,
                              prompt_tokens=self._prompt_tokens,
                              ttft_s=self._ttft,
                              gen_calls=len(self._raw),
                              latency_s=time.perf_counter() - t0,
                              tokens_per_s=tps, peak_memory_mb=peak)

    def _gen(self, prompt):
        text, m = self.generator.generate(prompt)
        self._raw.append(text)
        self._prompt_tokens += getattr(m, "prompt_tokens", 0)
        if self._ttft == 0.0:
            self._ttft = getattr(m, "ttft_s", 0.0)
        return text, m

    def _closed_book(self, ex):
        text, m = self._gen(CLOSED_BOOK_PROMPT.format(question=ex.question))
        trace = HopTrace(hop=0, query=ex.question, retrieved=[],
                         gold_chunk_ids=list(ex.gold_chunk_ids))
        return [trace], first_line(text), (m.completion_tokens, m.tokens_per_s, m.peak_memory_mb)

    def _single_hop(self, ex):
        hits = self._retrieve(ex.question)
        trace = HopTrace(hop=0, query=ex.question, retrieved=hits,
                         gold_chunk_ids=list(ex.gold_chunk_ids))
        answer, ntok, tps, peak = "", 0, 0.0, 0.0
        if self.generator:
            ctx = "\n".join(f"- {h.chunk.text}" for h in hits)
            text, m = self._gen(PROMPT.format(context=ctx, question=ex.question))
            answer = first_line(text)
            ntok, tps, peak = m.completion_tokens, m.tokens_per_s, m.peak_memory_mb
        return [trace], answer, (ntok, tps, peak)

    def _ircot(self, ex):
        # hotpot doesn't order its gold facts by hop, so every trace carries the
        # full gold set; 2wiki and musique carry an ordered path for the scorer.
        # The loop decides only whether to search again; the answer comes from
        # PROMPT over the union of everything retrieved, like single_hop.
        seen = {}
        traces = []
        query = ex.question
        ntok, tps, peak = 0, 0.0, 0.0
        asked = set()
        self.stop_reason = "max_hops"

        for hop in range(self.cfg.pipeline.max_hops):
            hits = self._retrieve(query)
            for h in hits:
                seen.setdefault(h.chunk.chunk_id, h)
            traces.append(HopTrace(hop=hop, query=query, retrieved=hits,
                                   gold_chunk_ids=list(ex.gold_chunk_ids)))
            asked.add(query.lower())
            if hop == self.cfg.pipeline.max_hops - 1:
                break
            ctx = "\n".join(f"- {h.chunk.text}" for h in seen.values())
            text, m = self._gen(IRCOT_PROMPT.format(context=ctx, question=ex.question))
            ntok += m.completion_tokens
            tps, peak = m.tokens_per_s, max(peak, m.peak_memory_mb)
            done, search = parse_step(text)
            if done:
                self.stop_reason = "done"
                break
            if not search:
                self.stop_reason = "no_marker"
                break
            if search.lower() in asked:
                self.stop_reason = "repeated_query"
                break
            query = search

        ctx = "\n".join(f"- {h.chunk.text}" for h in seen.values())
        text, m = self._gen(PROMPT.format(context=ctx, question=ex.question))
        answer = first_line(text)
        ntok += m.completion_tokens
        tps, peak = m.tokens_per_s, max(peak, m.peak_memory_mb)
        return traces, answer, (ntok, tps, peak)

    def run(self, examples):
        return [self.run_example(ex) for ex in examples]
