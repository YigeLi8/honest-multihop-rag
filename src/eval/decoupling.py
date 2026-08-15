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

    def __str__(self):
        return (f"n={self.n}  ans+/retr+ {self.ac_rc}  ans+/retr- {self.ac_rw} "
                f"(illusion {self.illusion_rate:.2%})  ans-/retr+ {self.aw_rc}  "
                f"ans-/retr- {self.aw_rw}")


def build_table(pairs):
    """pairs: iterable of (answer_correct, retrieval_correct) booleans."""
    t = Decoupling2x2()
    for ans_ok, retr_ok in pairs:
        if ans_ok and retr_ok:
            t.ac_rc += 1
        elif ans_ok:
            t.ac_rw += 1
        elif retr_ok:
            t.aw_rc += 1
        else:
            t.aw_rw += 1
    return t


def plot_table(table, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [[table.ac_rc, table.ac_rw], [table.aw_rc, table.aw_rw]]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cells, cmap="Blues")
    for i in range(2):
        for j in range(2):
            pct = cells[i][j] / table.n if table.n else 0
            ax.text(j, i, f"{cells[i][j]}\n{pct:.1%}", ha="center", va="center")
    ax.set_xticks([0, 1], ["retrieval correct", "retrieval wrong"])
    ax.set_yticks([0, 1], ["answer correct", "answer wrong"])
    # the interesting cell
    ax.add_patch(plt.Rectangle((0.5, -0.5), 1, 1, fill=False, edgecolor="red", lw=2))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
