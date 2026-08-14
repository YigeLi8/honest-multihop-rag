"""Dense retrieval: BGE via sentence-transformers, flat faiss index.

The per-question candidate sets are small (tens of sentences) so a flat
inner-product index is plenty. Embeddings are normalized, so IP = cosine.
"""
import numpy as np

from src.types import RetrievedChunk


class DenseRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None      # lazy, BGE takes a moment to load
        self.faiss_index = None
        self.chunks = []

    def _load_model(self):
        import torch
        from sentence_transformers import SentenceTransformer
        d = self.cfg.retrieval.dense
        device = d.device
        if device == "mps" and not torch.backends.mps.is_available():
            print("mps not available, using cpu")
            device = "cpu"
        self.model = SentenceTransformer(d.model, device=device)

    def _encode(self, texts):
        if self.model is None:
            self._load_model()
        d = self.cfg.retrieval.dense
        emb = self.model.encode(texts, batch_size=d.batch_size,
                                normalize_embeddings=d.normalize,
                                convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def index(self, chunks):
        import faiss
        self.chunks = list(chunks)
        emb = self._encode([c.text for c in self.chunks])
        self.faiss_index = faiss.IndexFlatIP(emb.shape[1])
        self.faiss_index.add(emb)

    def retrieve(self, query, top_k):
        k = min(top_k, len(self.chunks))
        scores, ids = self.faiss_index.search(self._encode([query]), k)
        return [RetrievedChunk(chunk=self.chunks[i], score=float(s), rank=r + 1)
                for r, (i, s) in enumerate(zip(ids[0], scores[0]))]
