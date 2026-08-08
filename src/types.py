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


@dataclass
class PipelineResult:
    example_id: str
    answer: str
    hops: list = field(default_factory=list)
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    tokens_per_s: float = 0.0
    peak_memory_mb: float = 0.0
