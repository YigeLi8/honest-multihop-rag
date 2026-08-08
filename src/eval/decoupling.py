"""The 2x2: answer correct/incorrect x retrieval correct/incorrect.

The cell that matters is answer-correct & retrieval-wrong: right answer, wrong
or incomplete evidence. Decision gate for the week: if that cell comes out
under ~5%, the more interesting story is the serving frontier.
"""
from dataclasses import dataclass


@dataclass
class Decoupling2x2:
    ac_rc: int = 0
    ac_rw: int = 0   # answer correct, retrieval wrong
    aw_rc: int = 0
    aw_rw: int = 0

    @property
    def n(self):
        return self.ac_rc + self.ac_rw + self.aw_rc + self.aw_rw

    @property
    def illusion_rate(self):
        return self.ac_rw / self.n if self.n else 0.0


def build_table(results, gold):
    raise NotImplementedError


def plot_table(table, out_path):
    raise NotImplementedError
