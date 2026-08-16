"""BGE cross-encoder reranker, top_k in -> top_n out."""
from src.types import RetrievedChunk


class Reranker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    def _load(self):
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(self.cfg.rerank.model, use_fp16=True)

    def rerank(self, query, candidates):
        if not candidates:
            return []
        if self.model is None:
            self._load()
        scores = self.model.compute_score([[query, c.chunk.text] for c in candidates])
        if not isinstance(scores, list):
            scores = [scores]   # single pair comes back as a bare float
        order = sorted(zip(candidates, scores), key=lambda t: -t[1])
        return [RetrievedChunk(chunk=c.chunk, score=float(s), rank=i + 1)
                for i, (c, s) in enumerate(order[: self.cfg.rerank.top_n])]
