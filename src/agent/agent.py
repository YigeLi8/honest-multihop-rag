"""The evidence agent: one episode per question (docs/plan.md, Stage 6).

Each step the model sees the question, its notebook (sentences kept so far),
the latest search results numbered, and whatever lessons the memory hands
it, and replies with two lines:

    KEEP: <numbers of the results to keep, or none>
    NEXT: SEARCH <query>   or   NEXT: ANSWER

The first search is the question itself, as in ircot, with the same
retriever and k, and at most pipeline.max_hops searches, so retrieval alone
matches hybrid_ircot and a comparison against it isolates what the agent
keeps. The answer comes from the same PROMPT every run uses, over the
notebook only. An empty notebook falls back to everything retrieved
(stop_reason gets a "+fallback" suffix).

The memory is anything with lessons(step) -> list[str] and, optionally,
on_episode(ex, res, row); `pipe.lessons` is None for the no-memory agent.
Gold never enters an episode: the step log holds the prompt, the reply, the
kept ids, the lessons shown and the Stage 1 features of the search.
"""
import re

from src.memory.features import compute_features
from src.pipeline.multihop import PROMPT, first_line
from src.types import HopTrace

AGENT_PROMPT = """You are collecting evidence for a multi-hop question. Do not \
answer it here: choose which search results to keep as evidence, then decide \
whether to search again.

Question: {question}
{lessons}
Notebook (evidence kept so far):
{notebook}

Search results for "{query}":
{results}

Reply with exactly two lines:
KEEP: <numbers of the results needed to answer the question, comma-separated, or none>
NEXT: {next_hint}"""

NEXT_OPEN = "SEARCH <short query for a fact that is still missing>, or ANSWER if the notebook plus what you keep is enough"
NEXT_LAST = "ANSWER (this was the last search)"

STEP_TOKENS = 64


def show(rc):
    title = rc.chunk.title
    text = rc.chunk.text.strip()
    return f"({title}) {text}" if title else text


def format_lessons(lessons):
    if not lessons:
        return ""
    return "\nLessons from earlier questions:\n" + "\n".join(f"- {l}" for l in lessons) + "\n"


def parse_reply(text, n_results):
    """(kept 0-based indices, next action, query). next is "search", "answer"
    or "" when the NEXT line is missing or unreadable."""
    keep, action, query = [], "", None
    for line in text.strip().splitlines():
        s = line.strip()
        up = s.upper()
        if up.startswith("KEEP:"):
            rest = s[5:]
            for m in re.findall(r"\d+", rest):
                i = int(m) - 1
                if 0 <= i < n_results and i not in keep:
                    keep.append(i)
        elif up.startswith("NEXT:"):
            rest = s[5:].strip()
            if rest.upper().startswith("ANSWER"):
                action = "answer"
            elif rest.upper().startswith("SEARCH"):
                q = rest[6:].strip().lstrip(":").strip().strip('"')
                if q:
                    action, query = "search", q
    return keep, action, query


class Usage:
    """Completion tokens, last decode rate and peak memory over an episode."""

    def __init__(self):
        self.ntok, self.tps, self.peak = 0, 0.0, 0.0

    def add(self, m):
        self.ntok += m.completion_tokens
        self.tps, self.peak = m.tokens_per_s, max(self.peak, m.peak_memory_mb)

    def stats(self):
        return self.ntok, self.tps, self.peak


def search_loop(pipe, ex, question, max_hops, usage, traces, steps, seen,
                role="agent", sub=None):
    """The keep / search loop on one question (the original one, or a
    sub-question when a main agent runs this as a subagent). Hop numbers
    continue from the traces already there, so a question's traces stay one
    sequence. Returns (notebook, stop_reason); fills traces, steps, seen."""
    notebook = {}             # chunk_id -> RetrievedChunk, in the order kept
    asked = set()
    prev_feats = None
    query = question
    stop = "max_hops"
    for i in range(max_hops):
        hop = len(traces)
        hits = pipe._retrieve(query, hop=hop, ex=ex)
        feats = compute_features(query, hits, ex.chunks, hop, ex.question,
                                 prev=prev_feats, seen_ids=list(seen))
        prev_feats = feats
        for h in hits:
            seen.setdefault(h.chunk.chunk_id, h)
        traces.append(HopTrace(hop=hop, query=query, retrieved=hits,
                               gold_chunk_ids=list(ex.gold_chunk_ids)))
        asked.add(query.lower())

        shown = [h for h in hits if h.chunk.chunk_id not in notebook]
        last = i == max_hops - 1
        step = {"hop": hop, "query": query, "features": feats,
                "shown": [h.chunk.chunk_id for h in shown]}
        if role != "agent":
            step.update(role=role, sub=sub, sub_question=question)
        lessons = pipe.lessons.lessons(dict(step, question=ex.question,
                                            notebook=list(notebook))) if pipe.lessons else []
        prompt = AGENT_PROMPT.format(
            question=question, lessons=format_lessons(lessons),
            notebook="\n".join(f"- {show(h)}" for h in notebook.values()) or "(empty)",
            query=query,
            results="\n".join(f"[{j + 1}] {show(h)}" for j, h in enumerate(shown)) or "(nothing new)",
            next_hint=NEXT_LAST if last else NEXT_OPEN)
        text, m = pipe._gen(prompt, max_tokens=STEP_TOKENS)
        usage.add(m)
        keep, action, nxt = parse_reply(text, len(shown))
        for j in keep:
            notebook[shown[j].chunk.chunk_id] = shown[j]
        step.update(lessons=lessons, reply=text, kept=[shown[j].chunk.chunk_id for j in keep],
                    action=action, next_query=nxt)
        steps.append(step)
        if last:
            break
        if action == "answer":
            stop = "done"
            break
        if action != "search":
            stop = "no_marker"
            break
        if nxt.lower() in asked:
            stop = "repeated_query"
            break
        query = nxt
    return notebook, stop


def answer_from(pipe, question, hits, usage):
    ctx = "\n".join(f"- {h.chunk.text}" for h in hits)
    text, m = pipe._gen(PROMPT.format(context=ctx, question=question))
    usage.add(m)
    return first_line(text)


def run_episode(pipe, ex):
    """Returns (traces, answer, (ntok, tps, peak)); sets pipe.stop_reason,
    pipe.notebook and pipe.steps."""
    usage = Usage()
    traces, steps, seen = [], [], {}
    notebook, pipe.stop_reason = search_loop(pipe, ex, ex.question, pipe.cfg.pipeline.max_hops,
                                             usage, traces, steps, seen)
    ctx_hits = list(notebook.values())
    if not ctx_hits:
        ctx_hits = list(seen.values())
        pipe.stop_reason += "+fallback"
    answer = answer_from(pipe, ex.question, ctx_hits, usage)
    pipe.notebook = list(notebook)
    pipe.steps = steps
    return traces, answer, usage.stats()
