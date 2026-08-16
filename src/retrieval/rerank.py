"""Cross-encoder reranker, top_k in -> top_n out.

Scores with sentence-transformers' CrossEncoder. FlagEmbedding's FlagReranker
segfaults on Metal (fp16 plus its multiprocess pool), same weights either way.
"""
from src.types import RetrievedChunk


class Reranker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    def _load(self):
        import torch
        from sentence_transformers import CrossEncoder
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CrossEncoder(self.cfg.rerank.model, device=device)

    def rerank(self, query, candidates):
        if not candidates:
            return []
        if self.model is None:
            self._load()
        scores = self.model.predict([(query, c.chunk.text) for c in candidates])
        order = sorted(zip(candidates, scores), key=lambda t: -t[1])
        return [RetrievedChunk(chunk=c.chunk, score=float(s), rank=i + 1)
                for i, (c, s) in enumerate(order[: self.cfg.rerank.top_n])]
