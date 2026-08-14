"""Map supporting-fact annotations onto chunk ids.

For hotpot this is direct: chunks are sentences and chunk_id is
"{title}::{sent_idx}", so a fact maps unless its sent_idx points past the end
of the paragraph (annotation glitches, there are a handful in dev). Those get
counted rather than dropped silently, since the miss rate bounds how much the
downstream precision numbers can be trusted.
"""
from dataclasses import dataclass


@dataclass
class MappingReport:
    n_facts: int = 0
    n_mapped: int = 0
    n_unmapped: int = 0
    examples_with_misses: int = 0

    def add(self, mapped, unmapped):
        self.n_facts += len(mapped) + len(unmapped)
        self.n_mapped += len(mapped)
        self.n_unmapped += len(unmapped)
        if unmapped:
            self.examples_with_misses += 1

    @property
    def error_rate(self):
        return self.n_unmapped / self.n_facts if self.n_facts else 0.0

    def __str__(self):
        return (f"{self.n_mapped}/{self.n_facts} facts mapped, "
                f"{self.n_unmapped} unmapped ({self.error_rate:.4%}), "
                f"{self.examples_with_misses} examples affected")


def map_facts(facts, sents_by_title):
    """facts: [(title, sent_idx), ...]; sents_by_title: {title: [sentence, ...]}.

    Returns (chunk_ids, unmapped).
    """
    mapped, unmapped = [], []
    for title, i in facts:
        sents = sents_by_title.get(title)
        if sents is None or not 0 <= i < len(sents):
            unmapped.append((title, i))
        else:
            mapped.append(f"{title}::{i}")
    return mapped, unmapped
