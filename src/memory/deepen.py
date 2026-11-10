"""The probe stage for deepening: what the chunks just below the cut look like.

Stage 5b (docs/plan.md) asks whether "retrieving 2k on this hop adds a
labelled chunk the hop missed at k" can be told in advance. From the frozen
features of src/memory/features.py it barely can (held-out AUC 0.56-0.64 on
the hotpot exploration log, a policy level with uniform k=13 at matched cost,
D3 passed on the confirmation fold). Those features are read before or at the
top-k list. This module reads one step later: the chunks at ranks k+1..m*k of
the primary retriever's own ranking. Retrieving them is cheap; what costs in
this harness is the generator reading them. So a policy may look at the
window before it hands the chunks over, and the hypothesis under test is that
the window carries the signal the earlier stages do not: a deeper chunk from a
paragraph the question or the context names, a paragraph not yet in context,
a small score gap at the cut.

Everything here is computed from the retriever's ranking, the pool, the
question, the query and the chunk ids of earlier hops. Nothing comes from the
dataset's labels; the function never receives the example, and
tests/test_memory.py reads this file as text and fails if a label field is
named in it. The window itself is logged beside the features (record key
"deeper") so that later features can be computed from the same records.

The features are kept out of src/memory/features.py on purpose: that schema
is frozen under its hash, and these cost a deeper retrieval, which is a
different stage ("deeper", after primary_hits). They are logged only when
memory.log_deeper is set (0, the default, logs nothing).
"""
from collections.abc import Collection, Sequence
from types import MappingProxyType
from typing import Mapping, Union

from src.memory.features import _jaccard, normalise_title, tokens
from src.types import Chunk, RetrievedChunk

Number = Union[int, float]

DEEPER_STAGE = "deeper"

# name -> what it reads; every one needs the ranking down to m*k
DEEPER_FEATURES: Mapping[str, str] = MappingProxyType({
    "deep_n": "chunks at ranks k+1..m*k (0 when the pool ends at k)",
    "deep_named_question": "of them, from a paragraph whose title the question names",
    "deep_named_query": "from a paragraph whose title this hop's query names",
    "deep_named_context": "from a paragraph whose title the context so far names in its text",
    "deep_new_paragraphs": "from a paragraph with no chunk in the context so far",
    "deep_gap_at_k": "(score at k - score at k+1) / (score at 1 - score at k+1); 0 without a gap",
    "deep_score_share": "positive scores below the cut over positive scores above it",
    "deep_max_query_overlap": "largest token Jaccard between the query and a chunk of the window",
})


def paragraph_of(chunk_id: str) -> str:
    return chunk_id.rsplit("::", 1)[0]


def deeper_window(full: Sequence[RetrievedChunk], k: int, mult: int) -> list[RetrievedChunk]:
    """Ranks k+1..mult*k of a ranking of the whole pool, as the retriever
    returned them (its scores and ranks included)."""
    if mult < 1 or k < 1:
        return []
    return list(full[k:mult * k])


def _named_in(title_forms: Sequence[str], text_tokens: Sequence[str]) -> int:
    padded = " " + " ".join(text_tokens) + " "
    return sum(1 for form in title_forms if form and f" {form} " in padded)


def deeper_features(full: Sequence[RetrievedChunk],
                    k: int,
                    mult: int,
                    hits: Sequence[RetrievedChunk],
                    chunks: Sequence[Chunk],
                    question: str,
                    query: str,
                    seen_ids: Collection[str] = (),
                    ) -> dict[str, Number]:
    """Features of the chunks at ranks k+1..mult*k, keyed in DEEPER_FEATURES order.

    full       the primary retriever's ranking of the whole pool at this hop
    k          the depth the pipeline used
    mult       the window ends at mult * k
    hits       what the pipeline used at this hop (after a reranker, if any)
    chunks     the question's candidate pool
    question   the original question
    query      the string sent to the retriever at this hop
    seen_ids   chunk ids returned at earlier hops of the same question

    Deterministic; floats rounded to 6 places so that a replay writes the
    same record. Scores are the full ranking's own, so for hybrid they are
    the fused values at full depth, not the top-k list's.
    """
    window = deeper_window(full, k, mult)
    by_id = {c.chunk_id: c for c in chunks}
    context_ids = [h.chunk.chunk_id for h in hits] + [i for i in seen_ids if i in by_id]
    context_paragraphs = {paragraph_of(i) for i in context_ids}
    context_tokens: list[str] = []
    for i in dict.fromkeys(context_ids):
        context_tokens += tokens(by_id[i].text) if i in by_id else []
    q_tokens, question_tokens = tokens(query), tokens(question)
    titles = [normalise_title(h.chunk.title or paragraph_of(h.chunk.chunk_id)) for h in window]

    above = [float(h.score) for h in full[:k]]
    below = [float(h.score) for h in window]
    gap = 0.0
    if window and above:
        s1, sk, sk1 = above[0], above[-1], below[0]
        gap = (sk - sk1) / (s1 - sk1) if s1 - sk1 > 0 else 0.0
    pos_above = sum(s for s in above if s > 0)
    pos_below = sum(s for s in below if s > 0)

    out: dict[str, Number] = {
        "deep_n": len(window),
        "deep_named_question": _named_in(titles, question_tokens),
        "deep_named_query": _named_in(titles, q_tokens),
        "deep_named_context": _named_in(titles, context_tokens),
        "deep_new_paragraphs": sum(1 for h in window
                                   if paragraph_of(h.chunk.chunk_id) not in context_paragraphs),
        "deep_gap_at_k": round(gap, 6),
        "deep_score_share": round(pos_below / pos_above, 6) if pos_above > 0 else 0.0,
        "deep_max_query_overlap": round(max((_jaccard(q_tokens, tokens(h.chunk.text))
                                             for h in window), default=0.0), 6),
    }
    assert tuple(out) == tuple(DEEPER_FEATURES)
    return out
