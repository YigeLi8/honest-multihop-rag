"""Map supporting-fact annotations onto chunk ids.

Exact (title, sent_idx) match first, normalized-text containment as fallback.
Facts that fail to map or map ambiguously get counted, because that error rate
bounds how much the per-hop precision numbers can be trusted.
"""
from dataclasses import dataclass


@dataclass
class MappingReport:
    n_facts: int = 0
    n_mapped: int = 0
    n_ambiguous: int = 0
    n_unmapped: int = 0

    @property
    def error_rate(self):
        return (self.n_ambiguous + self.n_unmapped) / self.n_facts if self.n_facts else 0.0


def map_supporting_facts(example):
    raise NotImplementedError
