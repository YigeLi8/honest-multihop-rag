"""Entity graph over 2Wiki evidence triples; retrieve by walking out from
query-linked entities. Plain retrieval, no learning. Optional arm, build last."""


class GraphIndex:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, examples):
        raise NotImplementedError

    def retrieve(self, query, seed_entities, top_k):
        raise NotImplementedError
