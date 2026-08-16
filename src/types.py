"""Shared record types."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    chunk_id: str          # "{title}::{sent_idx}"
    text: str
    title: str = ""
    sent_idx: int = -1


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int              # 1-based


@dataclass
class Example:
    id: str
    question: str
    answer: str
    hops: int
    chunks: list = field(default_factory=list)
    gold_chunk_ids: list = field(default_factory=list)
    gold_supporting_facts: list = field(default_factory=list)
    evidence_triples: list = field(default_factory=list)   # 2wiki only
    reasoning_path: list = field(default_factory=list)     # 2wiki only


@dataclass
class HopTrace:
    hop: int
    query: str
    retrieved: list = field(default_factory=list)
    gold_chunk_ids: list = field(default_factory=list)
    reasoning: str = ""


def example_from_json(d):
    return Example(
        id=d["id"], question=d["question"], answer=d["answer"], hops=d.get("hops", 2),
        chunks=[Chunk(chunk_id=c["chunk_id"], text=c["text"],
                      title=c.get("title", ""), sent_idx=c.get("sent_idx", -1))
                for c in d["chunks"]],
        gold_chunk_ids=d.get("gold_chunk_ids", []),
        gold_supporting_facts=d.get("gold_supporting_facts", []),
        evidence_triples=d.get("evidence_triples", []),
        reasoning_path=d.get("reasoning_path", []),
    )


@dataclass
class PipelineResult:
    example_id: str
    answer: str
    hops: list = field(default_factory=list)
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    tokens_per_s: float = 0.0
    peak_memory_mb: float = 0.0
