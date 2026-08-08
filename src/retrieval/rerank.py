"""BGE cross-encoder reranker, top_k in -> top_n out."""


class Reranker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    def rerank(self, query, candidates):
        raise NotImplementedError
