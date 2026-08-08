"""Dense retrieval: sentence-transformers (BGE on MPS) + faiss-cpu."""


class DenseRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None   # lazy, BGE takes a moment to load

    def index(self, chunks):
        raise NotImplementedError

    def retrieve(self, query, top_k):
        raise NotImplementedError
