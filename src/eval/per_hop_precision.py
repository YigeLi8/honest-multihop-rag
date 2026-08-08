"""Precision/recall of retrieved chunks against gold supporting facts, per hop."""


def score(results):
    # per hop: precision = |retrieved & gold| / |retrieved|, recall analogous;
    # also track retrieved-chunk-count variance across examples
    raise NotImplementedError
