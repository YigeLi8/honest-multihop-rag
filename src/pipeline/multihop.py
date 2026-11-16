"""Retrieval + generation over one example.

single_hop: retrieve once, answer from that context.
multihop_ircot: interleave retrieve -> reason -> retrieve until the model answers.
closed_book: no retrieval at all; the control that tells parametric answers
apart from answers that actually came from the context.
agent: the evidence agent of Stage 6 (src/agent/agent.py): the same searches
as ircot, but the model keeps sentences in a notebook and the answer is
read from the notebook only.
orchestrator: a main agent splits the question and runs that agent once per
sub-question (src/agent/orchestrator.py); the answer is read from the union
of the notebooks.

Every hop gets logged as a HopTrace so per-hop precision can be scored
afterwards, and every raw completion is kept on the result for later audits.

pipeline.answer_guard (off by default; docs/plan.md, Stage 6b): every answer
read goes through _answer, and with the guard on a question that does not
read as yes/no and came back yes or no is read once more over the same
context with GUARD_PROMPT. It applies to every mode that reads a context, so
a run with it is compared with runs that have it too. pipeline.guard_context:
wide makes that second read go over the step's kept sentences followed by
everything else its searches retrieved (the agents only; ircot and single-hop
already read everything retrieved, so for them it changes nothing).

With memory.enabled and memory.mode == log in the config, every retrieval step
is also handed to an ExperienceLogger (src/memory), together with the hits of
any shadow retrievers listed in memory.log_arms. That path only reads: the
hits, the context, the prompts and the answer are the same with it on or off.
The time it takes (shadow indexing and retrieval, the full-pool rankings, the
features) is measured and left out of latency_s.
"""
import re
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

# PROMPT with its yes/no sentence replaced: the second read of answer_guard
GUARD_PROMPT = """Answer the question using only the context below. Reply with just \
the answer, no explanation. The answer is a name, a date, a number or a short \
phrase, not yes or no.

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


_AUX = {"is", "are", "was", "were", "am", "do", "does", "did", "has", "have", "had",
        "can", "could", "will", "would", "should"}
_WH = re.compile(r"\b(what|which|who|whom|whose|when|where|why|how|name)\b", re.I)


def looks_yesno(question):
    """From the wording alone: the question opens with an auxiliary, or has
    no question word at all ("Both X and Y are plants?"). Written on the
    first 20,000 hotpot train questions, where it holds for 1226 of the 1231
    with a yes/no gold answer and 269 of the 18,769 others. The guard leaves
    such questions alone, so the rule errs towards calling a question yes/no:
    a span question it flags keeps its first reply, as without the guard."""
    words = question.split()
    return (bool(words) and words[0].strip("\"'(").lower() in _AUX) or not _WH.search(question)


def is_yesno(answer):
    return re.sub(r"[^a-z]", "", answer.lower()) in ("yes", "no")


def usage_of(ms, ntok=0, peak=0.0):
    """(completion tokens, last decode rate, peak memory) over some calls."""
    for m in ms:
        ntok += m.completion_tokens
        peak = max(peak, m.peak_memory_mb)
    return ntok, ms[-1].tokens_per_s, peak


class MultiHopPipeline:
    memory = None    # ExperienceLogger when the config asks for logging
    shadow = None    # {arm: retriever} run beside the primary one, logged and never used
    _memory_s = 0.0  # seconds the current question spent in the logging path
    lessons = None   # agent mode: what the experience memory shows the model
    guards = None    # answer_guard on: the second reads of the current question

    def __init__(self, cfg):
        self.cfg = cfg
        self.retriever = get_retriever(cfg) if cfg.pipeline.mode != "closed_book" else None
        self.reranker = Reranker(cfg) if cfg.rerank.enabled else None
        self.generator = Generator(cfg) if cfg.generation.backend != "none" else None
        mem = getattr(cfg, "memory", None)
        if (self.retriever is not None and getattr(mem, "enabled", False)
                and getattr(mem, "mode", "none") == "log"):
            from src.memory.experience import ExperienceLogger, shadow_retrievers
            self.memory = ExperienceLogger(cfg, cfg.run.name)
            self.shadow = shadow_retrievers(cfg)
        amem = getattr(cfg, "agent_memory", None)
        if cfg.pipeline.mode in ("agent", "orchestrator") and getattr(amem, "enabled", False):
            from src.agent.memory import ExperienceMemory
            self.lessons = ExperienceMemory.from_config(cfg, self)

    def _index(self, ex):
        self.retriever.index(ex.chunks)
        if self.shadow:
            t = time.perf_counter()
            for r in self.shadow.values():
                r.index(ex.chunks)
            self._memory_s += time.perf_counter() - t

    def _retrieve(self, query, hop=0, ex=None):
        first = hits = self.retriever.retrieve(query, self.cfg.retrieval.top_k)
        if self.reranker:
            hits = self.reranker.rerank(query, hits)
        if self.memory is not None and ex is not None:
            t = time.perf_counter()
            self._log_hop(ex, hop, query, hits, first)
            self._memory_s += time.perf_counter() - t
        return hits

    def _log_hop(self, ex, hop, query, hits, first):
        """Everything the log needs beyond the hits themselves. It runs after
        the primary hits are final and feeds only the logger: the shadow arms
        at the same top_k, and every arm's ranking of the whole pool (a second
        call to the primary retriever included) so that a missed gold chunk
        still gets a rank."""
        k, pool = self.cfg.retrieval.top_k, len(ex.chunks)
        shadow = self.shadow or {}
        self.memory.on_hop(
            ex, hop, query, hits,
            {arm: r.retrieve(query, k) for arm, r in shadow.items()},
            first_stage=first if self.reranker else None,
            primary_full=self.retriever.retrieve(query, pool),
            shadow_full={arm: r.retrieve(query, pool) for arm, r in shadow.items()})

    def run_example(self, ex):
        t0 = time.perf_counter()
        mode = self.cfg.pipeline.mode
        self._raw = []
        self._prompt_tokens, self._ttft = 0, 0.0
        self._memory_s = 0.0
        self.stop_reason = ""
        self.notebook, self.steps = [], []
        self.guards = [] if getattr(self.cfg.pipeline, "answer_guard", False) else None
        if mode == "closed_book":
            if self.generator is None:
                raise ValueError("closed_book needs a generation backend")
            traces, answer, stats = self._closed_book(ex)
        elif mode == "single_hop":
            self._index(ex)
            traces, answer, stats = self._single_hop(ex)
        elif mode == "multihop_ircot":
            if self.generator is None:
                raise ValueError("multihop_ircot needs a generation backend")
            self._index(ex)
            traces, answer, stats = self._ircot(ex)
        elif mode == "agent":
            if self.generator is None:
                raise ValueError("agent needs a generation backend")
            from src.agent.agent import run_episode
            self._index(ex)
            traces, answer, stats = run_episode(self, ex)
        elif mode == "orchestrator":
            if self.generator is None:
                raise ValueError("orchestrator needs a generation backend")
            from src.agent.orchestrator import run_orchestrated
            self._index(ex)
            traces, answer, stats = run_orchestrated(self, ex)
        else:
            raise ValueError(f"unknown pipeline mode {mode!r}")

        ntok, tps, peak = stats
        return PipelineResult(example_id=ex.id, answer=answer, hops=traces,
                              raw_outputs=list(self._raw),
                              stop_reason=self.stop_reason,
                              notebook=list(self.notebook), steps=list(self.steps),
                              guards=None if self.guards is None else list(self.guards),
                              reasoning_tokens=ntok,
                              prompt_tokens=self._prompt_tokens,
                              ttft_s=self._ttft,
                              gen_calls=len(self._raw),
                              latency_s=time.perf_counter() - t0 - self._memory_s,
                              tokens_per_s=tps, peak_memory_mb=peak)

    def _gen(self, prompt, max_tokens=None):
        text, m = self.generator.generate(prompt, max_tokens)
        self._raw.append(text)
        self._prompt_tokens += getattr(m, "prompt_tokens", 0)
        if self._ttft == 0.0:
            self._ttft = getattr(m, "ttft_s", 0.0)
        return text, m

    def _answer(self, question, ctx, where="final", wide=None):
        """The answer read every mode shares: PROMPT over the context, first
        line of the reply. Returns (answer, [metrics of each call]). With the
        guard on, yes or no to a question that does not read as yes/no gets a
        second read with GUARD_PROMPT; the second reply stands and the pair
        is kept on self.guards. wide = (context, n added sentences) is what
        that read goes over instead under guard_context: wide."""
        text, m = self._gen(PROMPT.format(context=ctx, question=question))
        answer, ms = first_line(text), [m]
        if self.guards is not None and is_yesno(answer) and not looks_yesno(question):
            rec = {"where": where, "question": question, "first": answer}
            if wide is not None and getattr(self.cfg.pipeline, "guard_context", "same") == "wide":
                ctx = wide[0]
                rec["wide"] = wide[1]
            text, m = self._gen(GUARD_PROMPT.format(context=ctx, question=question))
            rec["second"] = first_line(text)
            self.guards.append(rec)
            answer = first_line(text)
            ms.append(m)
        return answer, ms

    def _closed_book(self, ex):
        text, m = self._gen(CLOSED_BOOK_PROMPT.format(question=ex.question))
        trace = HopTrace(hop=0, query=ex.question, retrieved=[],
                         gold_chunk_ids=list(ex.gold_chunk_ids))
        return [trace], first_line(text), (m.completion_tokens, m.tokens_per_s, m.peak_memory_mb)

    def _single_hop(self, ex):
        hits = self._retrieve(ex.question, hop=0, ex=ex)
        trace = HopTrace(hop=0, query=ex.question, retrieved=hits,
                         gold_chunk_ids=list(ex.gold_chunk_ids))
        answer, ntok, tps, peak = "", 0, 0.0, 0.0
        if self.generator:
            ctx = "\n".join(f"- {h.chunk.text}" for h in hits)
            answer, ms = self._answer(ex.question, ctx)
            ntok, tps, peak = usage_of(ms)
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
            hits = self._retrieve(query, hop=hop, ex=ex)
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
        answer, ms = self._answer(ex.question, ctx)
        return traces, answer, usage_of(ms, ntok, peak)

    def run(self, examples):
        return [self.run_example(ex) for ex in examples]
