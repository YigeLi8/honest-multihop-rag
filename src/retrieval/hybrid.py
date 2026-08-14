"""Score fusion: alpha * dense + (1 - alpha) * bm25, min-max normalized."""
from src.types import RetrievedChunk

from .bm25 import Bm25Retriever
from .dense import DenseRetriever


def minmax(scores):
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


class HybridRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bm25 = Bm25Retriever(cfg)
        self.dense = DenseRetriever(cfg)

    def index(self, chunks):
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def retrieve(self, query, top_k):
        alpha = self.cfg.retrieval.hybrid.alpha
        pool = top_k * 3   # retrieve wider from each side before fusing
        fused = {}         # chunk_id -> [chunk, score]
        for weight, hits in ((1 - alpha, self.bm25.retrieve(query, pool)),
                             (alpha, self.dense.retrieve(query, pool))):
            if not hits:
                continue
            for hit, s in zip(hits, minmax([h.score for h in hits])):
                entry = fused.setdefault(hit.chunk.chunk_id, [hit.chunk, 0.0])
                entry[1] += weight * s
        ranked = sorted(fused.values(), key=lambda e: -e[1])[:top_k]
        return [RetrievedChunk(chunk=c, score=s, rank=r + 1)
                for r, (c, s) in enumerate(ranked)]
