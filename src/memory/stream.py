"""Stage 5: the decide-then-reveal stream and the policies run on it.
(docs/plan.md, Part B)

The arms log of a single-hop run scores every first-stage retriever on every
question, so a question can be replayed as an online decision with the
counterfactual observed rather than estimated: a policy reads the features
of the question that are available at its stage, picks an arm (or every
arm), and only then is told how each arm did. Questions arrive in a seeded
order; the same policy on K orders gives the spread of its numbers.

Hypothesis under test (the plan's narrow claim): a per-lesson applicability
boundary over the frozen process features, revised online with logged
narrow / expand / exception / split / retire steps, gives lower
false-application and repeated-failure rates at matched retrieval cost than
the baselines on the same stream. The baselines are the policies that say
what each part of the structure is worth:

  fixed:<arm>       one arm for every question; cost 1
  probe             every arm on every question; the oracle rate at cost n_arms
  rule:<name>       the no-persistence control and the Stage 4 static
                    condition: a fixed predicate over the current features
                    picks the arm, nothing is kept across questions. With a
                    veto, the rule's utility (gain of its arm over the default
                    on the past questions where the predicate held, fired or
                    not) must be positive before it may fire
  router:<stage>    one shared online logistic model per arm over the stage's
                    features, argmax of the predicted hit; query stage gives
                    the query-only router
  knn:<stage>       similarity memory over past situations: the k nearest
                    revealed questions in standardised feature space vote
                    with their outcomes
  utility           lessons with a scalar utility and a fixed trigger region
                    (a ball around the exemplar that created the lesson, never
                    revised); a lesson fires when the question is in its
                    region and its utility is positive. The same creation
                    rule as the boundary memory, so the pair is the
                    operators-and-log ablation the plan asks for
  boundary          the claim: lessons whose region is a box in feature space
                    that the operators revise as supporting and contradicting
                    experiences arrive, every revision logged
  utility+lr, boundary+lr   the same two with a per-lesson online logistic
                    model inside the region (the plan's "predicate, then
                    logistic"); utility+lr against boundary+lr is the
                    operators ablation with the learned boundary kept

A lesson is "where the default arm fails like this, arm A recovers it": it is
created when a revealed question had the default arm miss and A hit, with
that question's features as the exemplar. Lessons are keyed on the process
features and never on the question text or on anything gold-derived; gold
enters only through the revealed outcomes, after the decision.

Cost is counted in arms run per question. A policy at the query, pool or
history stage decides before any retrieval and pays 1 for the arm it picks.
A policy at the primary_hits stage has already run the log's primary arm
(the default; its features are that arm's), so switching costs 2 and the
switched-to arm's list replaces the primary's; probing pays n_arms and is
scored as any arm hitting (a context of n_arms lists, so a ceiling anchor,
not a same-depth arm). Rates are all-gold-in-top-k unless --outcome says
otherwise.

Every rate that conditions on a lesson firing (false application, false
rejection, repeated failure, boundary precision and recall against the
known best arm) is reported beside the unconditioned contrast: the policy's
hit rate against the fixed default over the whole stream, paired per order
(McNemar on the discordant questions, the largest p over orders), with the
retrieval cost next to it.

    python -m src.memory.stream --records results/memory/bm25_arms_hotpot_dev_experience.jsonl \\
        --dataset hotpot [--ids ...] [--seeds 13 17 19 23 29] [--default bm25]

Writes results/<dataset>_stream.csv (one row per policy) and
results/<dataset>_stream_revisions.csv (the first boundary memory's revision
log on the first order).

The twins as the stream (--twin-records, the arms log of a
data/make_twins.py file): per twin kind, one stream of the originals and
their twins in a seeded order, the same policies plus the surface-keyed
memories

  text:exact, text:0.6   the nearest revealed question by text, if identical
                    or at Jaccard 0.6 or more, decides: the default where it
                    hit, else the arm that recovered it there

and, per policy, the pair metrics of twin_metrics: on twins whose original
came earlier, the hit rate on unchanged and on changed pairs (the twin's
outcomes differ from the original's on some arm) beside the default, the
arm a surface key repeats from the original, and the oracle. A distractor
twin is the original's text over a harder pool, so the surface key cannot
tell them apart and the process features are the only thing that can; an
alias twin is a new text over the same pool, where the surface key sees a
new question and the features almost the same one. Writes
results/<dataset>_twin_stream.csv and _twin_stream_revisions.csv.
"""
import argparse
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from src.eval.stats import mcnemar
from src.memory.arms import (OUTCOMES, arm_table, arms_of, hits, read_ids, read_records,
                             stage_columns, with_qtype_columns)
from src.memory.features import tokens

DECIDE_STAGES = ("query", "pool", "history", "primary_hits")
OPERATORS = ("create", "narrow", "expand", "exception", "split", "retire")
TWIN_SEP = "::twin:"
TWIN_COLUMNS = ("hit", "default", "repeat", "oracle", "same_decision")


# ---------------------------------------------------------------- the stream

@dataclass
class Decision:
    arm: Optional[str]            # the arm whose list is used; None when probing
    probe: bool = False
    lesson: Optional[str] = None  # the lesson that made the decision, if one did
    cost: int = 1


@dataclass
class Stream:
    """One question order over the arm table. X holds every feature column
    the stage may read (f:*, qtype:*), H the hit of each arm."""
    ids: list
    X: np.ndarray                 # questions x features, raw units
    columns: list                 # names of the columns of X
    H: np.ndarray                 # questions x arms, 0/1
    arms: list
    order: np.ndarray             # the positions in stream order
    text: list = field(default_factory=list)   # question text per position (surface-keyed policies)

    def column(self, name: str) -> int:
        return self.columns.index(name)


def seeded_orders(n: int, seeds: Sequence[int]) -> list[np.ndarray]:
    return [np.random.RandomState(s).permutation(n) for s in seeds]


def make_stream(table: pd.DataFrame, stage: str, order: np.ndarray) -> Stream:
    """The features a policy at this stage may read, in raw units, and every
    arm's hit. The qtype one-hot columns ride along at every stage (they are
    known from the question alone)."""
    if stage not in DECIDE_STAGES:
        raise ValueError(f"stage must be one of {DECIDE_STAGES}, not {stage!r}")
    full = with_qtype_columns(table)
    cols = stage_columns(full, stage) + [c for c in full.columns if c.startswith("qtype:")]
    arms = arms_of(table)
    H = np.stack([hits(table, a).to_numpy(dtype=int) for a in arms], axis=1)
    return Stream(ids=table["id"].tolist(), X=full[cols].to_numpy(dtype=float), columns=cols,
                  H=H, arms=arms, order=np.asarray(order, dtype=int),
                  text=table["question"].fillna("").tolist())


class Standardiser:
    """Running mean and variance (Welford) of the revealed questions, so that
    distances are in units of what has been seen so far and nothing from
    the future enters a decision."""

    def __init__(self, d: int) -> None:
        self.n = 0
        self.mean = np.zeros(d)
        self.m2 = np.zeros(d)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def sd(self) -> np.ndarray:
        var = self.m2 / max(self.n - 1, 1)
        return np.where(var > 1e-12, np.sqrt(var), 1.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.sd


# ---------------------------------------------------------------- policies

class Policy:
    """decide() sees the features of one question; reveal() is then told every
    arm's hit. Subclasses keep whatever state they persist across questions
    in the instance; one instance is used per order, so a new instance per
    order is a fresh memory."""
    stage: str = "query"
    name: str = "policy"

    def __init__(self, stream: Stream, default: str) -> None:
        self.stream = stream
        self.default = default
        self.arms = list(stream.arms)
        self.d_arm = self.arms.index(default)

    def decide(self, x: np.ndarray, t: int) -> Decision:
        raise NotImplementedError

    def reveal(self, x: np.ndarray, h: np.ndarray, decision: Decision, t: int) -> None:
        pass

    def cost_of(self, arm: str) -> int:
        """At the primary_hits stage the default has already been run."""
        return 1 if arm == self.default or self.stage not in ("primary_hits", "history") else 2

    def lesson_state(self) -> dict:
        """What the boundary metrics need: nothing, for a policy without lessons."""
        return {}


class FixedArm(Policy):
    def __init__(self, stream: Stream, default: str, arm: str) -> None:
        super().__init__(stream, default)
        self.arm = arm
        self.name = f"fixed:{arm}"

    def decide(self, x, t):
        return Decision(self.arm, cost=1)


class AlwaysProbe(Policy):
    name = "probe"

    def decide(self, x, t):
        return Decision(None, probe=True, cost=len(self.arms))


class StaticRule(Policy):
    """No-persistence control and the Stage 4 static condition: a fixed
    predicate on the current features chooses an arm, else the default.

    rules   [(name, predicate(features dict) -> bool, arm)], first match wins
    veto    when set, a rule fires only if the running mean gain of its arm
            over the default, on the past questions where its predicate held
            (fired or not; every arm is revealed), is positive after at least
            min_n such questions. That is the CAVE-Mem utility veto, with the
            full-information gain in place of a cross-fitted estimate: the
            current question never enters its own utility.
    """

    def __init__(self, stream: Stream, default: str, stage: str, rules: Sequence[tuple],
                 veto: bool = False, min_n: int = 10, label: str = "rule") -> None:
        super().__init__(stream, default)
        self.stage = stage
        self.rules = list(rules)
        self.veto = veto
        self.min_n = min_n
        self.name = f"{label}{'+veto' if veto else ''}"
        self.gain_sum = {r[0]: 0.0 for r in self.rules}
        self.gain_n = {r[0]: 0 for r in self.rules}

    def _held(self, x):
        feats = dict(zip(self.stream.columns, x))
        return [(name, arm) for name, pred, arm in self.rules if pred(feats)]

    def decide(self, x, t):
        for name, arm in self._held(x):
            if self.veto and (self.gain_n[name] < self.min_n or self.gain_sum[name] <= 0):
                continue
            return Decision(arm, lesson=name, cost=self.cost_of(arm))
        return Decision(self.default, cost=1)

    def reveal(self, x, h, decision, t):
        for name, arm in self._held(x):
            self.gain_sum[name] += h[self.arms.index(arm)] - h[self.d_arm]
            self.gain_n[name] += 1


class OnlineLogistic:
    """Plain SGD logistic regression on standardised inputs with an
    intercept; the standardiser is shared and updated by the caller."""

    def __init__(self, d: int, lr: float = 0.05, l2: float = 1e-3) -> None:
        self.w = np.zeros(d + 1)
        self.lr = lr
        self.l2 = l2

    def prob(self, z: np.ndarray) -> float:
        s = float(self.w[0] + self.w[1:] @ z)
        return 1.0 / (1.0 + math.exp(-max(min(s, 30.0), -30.0)))

    def update(self, z: np.ndarray, y: float) -> None:
        g = self.prob(z) - y
        self.w[0] -= self.lr * g
        self.w[1:] -= self.lr * (g * z + self.l2 * self.w[1:])


class OnlineRouter(Policy):
    """The shared router: one online hit model per arm over the stage's
    features, fit on every revealed question (every arm's hit is known),
    argmax of the predicted hit after a warm-up on the default. At the query
    stage this is the query-only router."""

    def __init__(self, stream: Stream, default: str, stage: str, warmup: int = 50,
                 lr: float = 0.05) -> None:
        super().__init__(stream, default)
        self.stage = stage
        self.warmup = warmup
        self.name = f"router:{stage}"
        self.std = Standardiser(stream.X.shape[1])
        self.models = [OnlineLogistic(stream.X.shape[1], lr=lr) for _ in self.arms]
        self.seen = 0

    def decide(self, x, t):
        if self.seen < self.warmup:
            return Decision(self.default, cost=1)
        z = self.std(x)
        probs = [m.prob(z) for m in self.models]
        best = max(range(len(self.arms)), key=lambda i: (probs[i], i == self.d_arm))
        arm = self.arms[best]
        return Decision(arm, cost=self.cost_of(arm))

    def reveal(self, x, h, decision, t):
        self.std.update(x)
        z = self.std(x)
        for m, y in zip(self.models, h):
            m.update(z, float(y))
        self.seen += 1


class OnlineKNN(Policy):
    """Similarity memory over situations: the k nearest revealed questions
    (euclidean in standardised feature space) vote with their hits per arm;
    the arm with the most votes wins, ties to the default."""

    def __init__(self, stream: Stream, default: str, stage: str, k: int = 20,
                 warmup: int = 50) -> None:
        super().__init__(stream, default)
        self.stage = stage
        self.k = k
        self.warmup = warmup
        self.name = f"knn{k}:{stage}"
        self.std = Standardiser(stream.X.shape[1])
        self.xs: list[np.ndarray] = []
        self.hs: list[np.ndarray] = []

    def decide(self, x, t):
        if len(self.xs) < max(self.warmup, self.k):
            return Decision(self.default, cost=1)
        Z = self.std(np.stack(self.xs))
        dist = np.linalg.norm(Z - self.std(x), axis=1)
        near = np.argsort(dist, kind="stable")[: self.k]
        votes = np.stack(self.hs)[near].mean(axis=0)
        best = max(range(len(self.arms)), key=lambda i: (votes[i], i == self.d_arm))
        arm = self.arms[best]
        return Decision(arm, cost=self.cost_of(arm))

    def reveal(self, x, h, decision, t):
        self.std.update(x)
        self.xs.append(np.asarray(x, dtype=float))
        self.hs.append(np.asarray(h, dtype=int))


class TextMemory(Policy):
    """Similarity memory keyed on the question's surface, online: the pair
    baseline of the twin stream. The nearest revealed question by token
    Jaccard over the question text, if at least `threshold` similar, decides:
    the default when it hit there, else the first other arm that hit there,
    i.e. the arm a "this question failed before, X recovered it" memory
    repeats. It decides at the query stage, before any retrieval, and reads
    nothing but the text, so a distractor twin (identical text) always
    matches its original and an alias twin matches only below threshold 1.

    Hypothesis it serves (plan, Stage 5, the twins as the stream): whether a
    per-lesson boundary over the process features tells a changed retrieval
    situation from an unchanged one where a surface key by construction
    cannot. The text key is never a feature of any other policy."""
    stage = "query"

    def __init__(self, stream: Stream, default: str, threshold: float = 1.0) -> None:
        super().__init__(stream, default)
        self.threshold = threshold
        self.name = f"text:{'exact' if threshold >= 1.0 else f'{threshold:g}'}"
        self.keys: list[frozenset] = []
        self.hs: list[np.ndarray] = []
        self.exact: dict[frozenset, int] = {}

    def _key(self, t: int) -> frozenset:
        return frozenset(tokens(self.stream.text[self.stream.order[t]]))

    def _nearest(self, key: frozenset) -> tuple[Optional[int], float]:
        if not key:
            return None, 0.0
        if key in self.exact:
            return self.exact[key], 1.0
        if self.threshold >= 1.0:
            return None, 0.0
        best, best_sim = None, 0.0
        for i, k in enumerate(self.keys):
            inter = len(key & k)
            if inter:
                sim = inter / len(key | k)
                if sim > best_sim:
                    best, best_sim = i, sim
        return best, best_sim

    def decide(self, x, t):
        i, sim = self._nearest(self._key(t))
        if i is None or sim < self.threshold:
            return Decision(self.default, cost=1)
        h = self.hs[i]
        if h[self.d_arm]:
            return Decision(self.default, cost=1)
        for j, arm in enumerate(self.arms):
            if j != self.d_arm and h[j]:
                return Decision(arm, lesson=f"match@{sim:.2f}", cost=self.cost_of(arm))
        return Decision(self.default, cost=1)

    def reveal(self, x, h, decision, t):
        key = self._key(t)
        self.exact[key] = len(self.keys)      # the latest question with this text
        self.keys.append(key)
        self.hs.append(np.asarray(h, dtype=int))


@dataclass
class Lesson:
    """Where the default fails like the exemplar, arm `arm` recovers it. The
    region is a box (lo, hi) in raw feature units over the lesson's
    dimensions; exceptions are revealed points inside the box where the
    lesson was wrong, each excluded with a small ball around it."""
    id: str
    arm: str
    exemplar: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    created: int
    support: list = field(default_factory=list)      # points where arm beat the default
    contra: list = field(default_factory=list)       # points inside where it did not
    exceptions: list = field(default_factory=list)   # (point, radius in sd units)
    gain_sum: float = 0.0
    gain_n: int = 0
    fired: int = 0
    retired: bool = False
    model: Optional["OnlineLogistic"] = None    # per-lesson refinement inside the region
    n_pos: int = 0
    n_neg: int = 0

    def contains(self, x: np.ndarray, sd: np.ndarray) -> bool:
        if not np.all((x >= self.lo) & (x <= self.hi)):
            return False
        return not any(np.linalg.norm((x - p) / sd) <= r for p, r in self.exceptions)

    def utility(self) -> float:
        return self.gain_sum / self.gain_n if self.gain_n else 0.0


class LessonMemory(Policy):
    """Lessons created from recoverable failures of the default arm.

    A revealed question where the default missed and arm A hit creates a
    lesson for A with the question as exemplar, unless an existing lesson
    for A already contains the question (then it supports that lesson). The
    initial region is a box of half-width `radius` running standard
    deviations around the exemplar on every feature dimension.

    Decision: among the lessons that contain the question and whose utility
    (mean gain of their arm over the default on the revealed questions inside
    their region, fired or not; the current question is never included) is
    positive after min_n questions, the one with the highest utility fires.
    Otherwise the default.

    revise=False is the scalar-utility baseline: the region is never
    changed, lessons never retire, nothing is logged beyond creation.
    refine=True adds the plan's "predicate, then logistic": each lesson
    carries an online logistic model over the standardised features, fit on
    the revealed questions inside its region (did the arm beat the default),
    and once it has seen min_n of each class the lesson fires only where the
    model says the arm wins. With revise=False that is a learned boundary
    inside a fixed region and no operators; with both it is the full model.
    revise=True is the boundary memory, with the operators applied at
    reveal time:

      narrow     the lesson fired (or would have, on utility) and its arm did
                 not beat the default: the box edge on the dimension where the
                 point is farthest from the exemplar, in sd units, is pulled to
                 just inside the point, provided no supporting point is lost
      exception  the same, when every dimension's cut would lose a supporter:
                 the point is excluded with a ball of radius `exception_radius`
      expand     the lesson did not contain the point, the arm beat the
                 default and the point is within `expand_margin` sd of the box
                 on every dimension: the box grows to include it
      split      a lesson with more than max_exceptions exceptions is cut at
                 the median of its supporters on the dimension that best
                 separates supporters from contradicting points; two lessons
                 with the halves' supporters replace it, exceptions dropped
      retire     utility <= 0 after min_n questions in the region, or a box
                 that no longer contains its own exemplar

    Every operator appends to `log` (t, lesson, operator, reason, the box
    before and after as a width per dimension); nothing is overwritten
    silently. The ablation the plan asks for is revise=False against
    revise=True with everything else equal.
    """

    def __init__(self, stream: Stream, default: str, stage: str, revise: bool,
                 radius: float = 1.0, min_n: int = 5, warmup: int = 20,
                 expand_margin: float = 0.5, exception_radius: float = 0.25,
                 max_exceptions: int = 3, refine: bool = False,
                 label: Optional[str] = None) -> None:
        super().__init__(stream, default)
        self.stage = stage
        self.revise = revise
        self.refine = refine
        self.radius = radius
        self.min_n = min_n
        self.warmup = warmup
        self.expand_margin = expand_margin
        self.exception_radius = exception_radius
        self.max_exceptions = max_exceptions
        self.name = label or ("boundary" if revise else "utility")
        self.std = Standardiser(stream.X.shape[1])
        self.lessons: list[Lesson] = []
        self.log: list[dict] = []
        self.seen = 0
        self._n_created = 0

    # -- bookkeeping
    def _record(self, t: int, lesson: Lesson, op: str, reason: str,
                before: Optional[tuple] = None) -> None:
        width = lambda lo, hi: float(np.mean((hi - lo) / self.std.sd))   # noqa: E731
        self.log.append({"t": t, "lesson": lesson.id, "arm": lesson.arm, "op": op,
                         "reason": reason,
                         "width_before": round(width(*before), 4) if before else math.nan,
                         "width_after": round(width(lesson.lo, lesson.hi), 4),
                         "n_support": len(lesson.support), "n_contra": len(lesson.contra),
                         "n_exceptions": len(lesson.exceptions),
                         "utility": round(lesson.utility(), 4), "gain_n": lesson.gain_n})

    def active(self) -> list[Lesson]:
        return [l for l in self.lessons if not l.retired]

    def _containing(self, x):
        sd = self.std.sd
        return [l for l in self.active() if l.contains(x, sd)]

    def _ready(self, lesson: Lesson, x: Optional[np.ndarray] = None) -> bool:
        if lesson.gain_n < self.min_n or lesson.utility() <= 0:
            return False
        if x is not None and lesson.model is not None and \
                min(lesson.n_pos, lesson.n_neg) >= self.min_n:
            return lesson.model.prob(self.std(x)) > 0.5
        return True

    # -- the policy
    def decide(self, x, t):
        if self.seen < self.warmup:
            return Decision(self.default, cost=1)
        ready = [l for l in self._containing(x) if self._ready(l, x)]
        if not ready:
            return Decision(self.default, cost=1)
        best = max(ready, key=lambda l: (l.utility(), -l.created))
        best.fired += 1
        return Decision(best.arm, lesson=best.id, cost=self.cost_of(best.arm))

    def reveal(self, x, h, decision, t):
        x = np.asarray(x, dtype=float)
        self.std.update(x)
        self.seen += 1
        sd = self.std.sd
        default_hit = int(h[self.d_arm])
        inside = self._containing(x)
        for lesson in inside:
            # judged before this question's gain enters the utility
            would_fire = decision.lesson == lesson.id or self._ready(lesson, x)
            gain = int(h[self.arms.index(lesson.arm)]) - default_hit
            lesson.gain_sum += gain
            lesson.gain_n += 1
            if self.refine:
                if lesson.model is None:
                    lesson.model = OnlineLogistic(len(x))
                lesson.model.update(self.std(x), float(gain > 0))
                lesson.n_pos += gain > 0
                lesson.n_neg += gain <= 0
            if gain > 0:
                lesson.support.append(x)
            else:
                lesson.contra.append(x)
                if self.revise and would_fire:
                    self._narrow_or_except(lesson, x, t, sd)
        if self.revise:
            for lesson in self.active():
                if lesson in inside:
                    continue
                gain = int(h[self.arms.index(lesson.arm)]) - default_hit
                if gain > 0 and self._near(lesson, x, sd):
                    self._expand(lesson, x, t)
            for lesson in list(self.active()):
                if len(lesson.exceptions) > self.max_exceptions:
                    self._split(lesson, t)
                elif lesson.gain_n >= self.min_n and lesson.utility() <= 0:
                    self._retire(lesson, t, "utility <= 0")
                elif not np.all((lesson.exemplar >= lesson.lo) & (lesson.exemplar <= lesson.hi)):
                    self._retire(lesson, t, "box lost its exemplar")
        # create: a recoverable failure of the default no active lesson covers
        if not default_hit:
            for i, arm in enumerate(self.arms):
                if i != self.d_arm and h[i] and not any(l.arm == arm for l in inside):
                    self._create(arm, x, t, sd)

    # -- operators
    def _create(self, arm, x, t, sd):
        self._n_created += 1
        half = self.radius * sd
        lesson = Lesson(id=f"L{self._n_created}", arm=arm, exemplar=x.copy(),
                        lo=x - half, hi=x + half, created=t, support=[x.copy()])
        self.lessons.append(lesson)
        self._record(t, lesson, "create", f"default missed, {arm} hit")

    def _narrow_or_except(self, lesson, x, t, sd):
        before = (lesson.lo.copy(), lesson.hi.copy())
        z = (x - lesson.exemplar) / sd
        for dim in np.argsort(-np.abs(z), kind="stable"):
            if abs(z[dim]) < 1e-9:
                break
            lo, hi = lesson.lo.copy(), lesson.hi.copy()
            eps = 1e-6 * sd[dim]
            if z[dim] > 0:
                hi[dim] = x[dim] - eps
            else:
                lo[dim] = x[dim] + eps
            keeps = all(lo[dim] <= p[dim] <= hi[dim] for p in lesson.support)
            if keeps:
                lesson.lo, lesson.hi = lo, hi
                self._record(t, lesson, "narrow", f"contradicted; cut {self.stream.columns[dim]}",
                             before)
                return
        lesson.exceptions.append((x.copy(), self.exception_radius))
        self._record(t, lesson, "exception", "contradicted; every cut loses a supporter", before)

    def _near(self, lesson, x, sd):
        """Outside the box (not merely inside an exception) and within the
        expansion margin of it on every dimension."""
        margin = self.expand_margin * sd
        outside = np.any((x < lesson.lo) | (x > lesson.hi))
        return bool(outside and np.all((x >= lesson.lo - margin) & (x <= lesson.hi + margin)))

    def _expand(self, lesson, x, t):
        before = (lesson.lo.copy(), lesson.hi.copy())
        lesson.lo = np.minimum(lesson.lo, x)
        lesson.hi = np.maximum(lesson.hi, x)
        lesson.support.append(x.copy())
        self._record(t, lesson, "expand", "arm beat the default just outside the box", before)

    def _split(self, lesson, t):
        sd = self.std.sd
        S = np.stack(lesson.support)
        C = np.stack([p for p, _ in lesson.exceptions] + lesson.contra)
        gap = np.abs(S.mean(axis=0) - C.mean(axis=0)) / sd
        dim = int(np.argmax(gap))
        cut = float(np.median(S[:, dim]))
        lesson.retired = True
        self._record(t, lesson, "split", f"{len(lesson.exceptions)} exceptions; cut "
                     f"{self.stream.columns[dim]} at {cut:.4g}")
        for side, pts in (("lo", S[S[:, dim] <= cut]), ("hi", S[S[:, dim] > cut])):
            if not len(pts):
                continue
            lo, hi = lesson.lo.copy(), lesson.hi.copy()
            if side == "lo":
                hi[dim] = cut
            else:
                lo[dim] = cut
            self._n_created += 1
            child = Lesson(id=f"{lesson.id}.{side}", arm=lesson.arm, exemplar=pts[0].copy(),
                           lo=lo, hi=hi, created=t, support=[p.copy() for p in pts])
            self.lessons.append(child)
            self._record(t, child, "create", f"split of {lesson.id}")

    def _retire(self, lesson, t, reason):
        lesson.retired = True
        self._record(t, lesson, "retire", reason)

    def lesson_state(self):
        ops = Counter(e["op"] for e in self.log)
        return {"n_lessons": self._n_created, "n_active": len(self.active()),
                **{f"op_{op}": ops.get(op, 0) for op in OPERATORS}}


# ---------------------------------------------------------------- running

@dataclass
class StreamResult:
    policy: str
    stage: str
    hit: np.ndarray          # per question, in stream order
    cost: np.ndarray
    arm: list                # chosen arm per question, "probe" when probing
    fired: np.ndarray        # a lesson or rule made the decision
    lesson_arm_ok: np.ndarray    # fired and the chosen arm beat the default
    harm: np.ndarray         # fired and the chosen arm missed where the default hit
    state: dict
    log: list


def run_stream(stream: Stream, policy: Policy) -> StreamResult:
    n = len(stream.order)
    hit = np.zeros(n)
    cost = np.zeros(n, dtype=int)
    fired = np.zeros(n, dtype=bool)
    ok = np.zeros(n, dtype=bool)
    harm = np.zeros(n, dtype=bool)
    chosen = []
    d = stream.arms.index(policy.default)
    for t, q in enumerate(stream.order):
        x, h = stream.X[q], stream.H[q]
        decision = policy.decide(x, t)
        if decision.probe:
            hit[t] = h.max()
            chosen.append("probe")
        else:
            hit[t] = h[stream.arms.index(decision.arm)]
            chosen.append(decision.arm)
        cost[t] = decision.cost
        fired[t] = decision.lesson is not None
        ok[t] = fired[t] and not decision.probe and hit[t] > h[d]
        harm[t] = fired[t] and not decision.probe and hit[t] < h[d]
        policy.reveal(x, h, decision, t)
    return StreamResult(policy.name, policy.stage, hit, cost, chosen, fired, ok, harm,
                        policy.lesson_state(), getattr(policy, "log", []))


def boundary_metrics(stream: Stream, result: StreamResult, default: str) -> dict:
    """Rates against the known best arm, over the stream.

    recoverable   default missed and some other arm hit
    repeated_failure   among recoverable questions after the first one, the
                  share the policy still missed (the memory had seen such a
                  failure and did not avoid it)
    false_application  fired and the chosen arm did not beat the default
                  (both hit counts: the lesson changed nothing, and at the
                  primary_hits stage it cost a retrieval)
    harm          fired and the chosen arm missed where the default hit: the
                  firings that lost a question, apart from the useless ones
    false_rejection    recoverable, after the first recoverable question, not
                  fired
    boundary_precision / recall   fired-and-beat-default against recoverable
    """
    H = stream.H[stream.order]
    d = stream.arms.index(default)
    other = np.delete(H, d, axis=1).max(axis=1)
    recoverable = (H[:, d] == 0) & (other == 1)
    first = int(np.argmax(recoverable)) if recoverable.any() else len(recoverable)
    later = np.zeros(len(H), dtype=bool)
    later[first + 1:] = True
    rec_later = recoverable & later
    fired = result.fired
    ok = result.lesson_arm_ok

    def rate(num, den):
        return round(float(num.sum() / den.sum()), 4) if den.sum() else math.nan

    return {"n_recoverable": int(recoverable.sum()), "n_fired": int(fired.sum()),
            "repeated_failure": rate(rec_later & (result.hit == 0), rec_later),
            "false_application": rate(fired & ~ok, fired),
            "harm": rate(result.harm, fired),
            "false_rejection": rate(rec_later & ~fired, rec_later),
            "boundary_precision": rate(ok, fired),
            "boundary_recall": rate(ok & rec_later, rec_later)}


# ---------------------------------------------------------------- twins as the stream

def twin_pairs(table: pd.DataFrame) -> dict[int, int]:
    """twin row -> original row, for every twin (data/make_twins.py ids,
    '<original>::twin:<kind>') whose original is in the table."""
    at = {i: n for n, i in enumerate(table["id"])}
    pairs = {}
    for n, i in enumerate(table["id"]):
        if TWIN_SEP in i and i.rsplit(TWIN_SEP, 1)[0] in at:
            pairs[n] = at[i.rsplit(TWIN_SEP, 1)[0]]
    return pairs


def with_twins(originals: pd.DataFrame, twins: pd.DataFrame) -> pd.DataFrame:
    """One arm table of the originals and the twins whose original is in it,
    so that one stream carries both and a policy meets a twin before or
    after its original as the order says. The same arms on both sides."""
    if arms_of(twins) != arms_of(originals):
        raise ValueError(f"twins carry arms {arms_of(twins)}, originals {arms_of(originals)}")
    have = set(originals["id"])
    keep = twins["id"].map(lambda i: TWIN_SEP in i and i.rsplit(TWIN_SEP, 1)[0] in have)
    cols = [c for c in originals.columns if c in twins.columns]
    out = pd.concat([originals[cols], twins.loc[keep, cols]], ignore_index=True)
    if out["id"].duplicated().any():
        raise ValueError("duplicate ids between originals and twins")
    return out


def twin_metrics(stream: Stream, result: StreamResult, pairs: dict, default: str) -> dict:
    """What a policy does on a twin whose original it has already seen: the
    pairs where the original came earlier in this order (about half of them
    under a random order; the rest are left out).

    changed    the twin's hit vector differs from the original's on some arm:
               by construction the surface is the same (distractor twins) or
               nearly so (alias twins), and the retrieval situation moved
    repeat     the arm a surface-keyed memory carries over from the original:
               the default if it hit there, else the first other arm that hit
               there. On an unchanged pair it is the best arm; on a changed
               pair it is what a memory that cannot see the change does
    Per unchanged / changed pairs: the policy's hit rate on the twin beside
    the fixed default's, the repeat arm's and the oracle's, and
    same_decision, the share of pairs where the policy chose on the twin the
    arm it chose on the original. p_vs_repeat_changed is McNemar between the
    policy's hits and the repeat arm's on the changed pairs."""
    pos = np.empty(len(stream.order), dtype=int)
    pos[stream.order] = np.arange(len(stream.order))
    d = stream.arms.index(default)
    rows = []
    for tw, orig in pairs.items():
        pt, po = pos[tw], pos[orig]
        if po >= pt:
            continue
        ht, ho = stream.H[tw], stream.H[orig]
        rep = d if ho[d] else next((j for j in range(len(ho)) if j != d and ho[j]), d)
        rows.append({"changed": bool(np.any(ht != ho)), "hit": int(result.hit[pt] >= 1),
                     "default": int(ht[d]), "repeat": int(ht[rep]), "oracle": int(ht.max()),
                     "same_decision": int(result.arm[pt] == result.arm[po])})
    df = pd.DataFrame(rows, columns=["changed", *TWIN_COLUMNS])
    out = {"n_pairs": len(df), "n_changed": int(df["changed"].sum())}
    for label, sub in (("unchanged", df[~df["changed"]]), ("changed", df[df["changed"]])):
        for c in TWIN_COLUMNS:
            out[f"{c}_{label}"] = round(float(sub[c].mean()), 4) if len(sub) else math.nan
    changed = df[df["changed"]]
    out["p_vs_repeat_changed"] = mcnemar(changed["repeat"].tolist(), changed["hit"].tolist())[2] \
        if len(changed) else math.nan
    return out


def text_policies() -> list[tuple]:
    """The surface-keyed memories of the twin stream: exact text, and the
    nearest question at Jaccard 0.6 or more (an alias twin sits above it,
    nearest natural neighbours on hotpot almost never do)."""
    return [("text:exact", "query", lambda s, d: TextMemory(s, d, 1.0)),
            ("text:0.6", "query", lambda s, d: TextMemory(s, d, 0.6))]


def default_policies(default: str) -> list[tuple]:
    """(label, stage, factory) for every learned policy of the Stage 5 table."""
    pols: list[tuple] = [("probe", "query", lambda s, d: AlwaysProbe(s, d))]
    for stage in ("query", "pool", "primary_hits"):
        pols.append((f"router:{stage}", stage,
                     lambda s, d, st=stage: OnlineRouter(s, d, st)))
        pols.append((f"knn20:{stage}", stage, lambda s, d, st=stage: OnlineKNN(s, d, st, k=20)))
        for radius in (1.0, 2.0):
            tag = "" if radius == 1.0 else f":r{radius:g}"
            pols.append((f"utility:{stage}{tag}", stage,
                         lambda s, d, st=stage, r=radius, lb=f"utility:{stage}{tag}":
                         LessonMemory(s, d, st, revise=False, radius=r, label=lb)))
            pols.append((f"boundary:{stage}{tag}", stage,
                         lambda s, d, st=stage, r=radius, lb=f"boundary:{stage}{tag}":
                         LessonMemory(s, d, st, revise=True, radius=r, label=lb)))
        for revise in (False, True):
            lb = f"{'boundary' if revise else 'utility'}+lr:{stage}:r2"
            pols.append((lb, stage, lambda s, d, st=stage, rv=revise, lb=lb:
                         LessonMemory(s, d, st, revise=rv, radius=2.0, refine=True, label=lb)))
    return pols


def rule_policies(stage_rules: dict) -> list[tuple]:
    """The static-condition policies, with and without the utility veto, for
    {stage: rules}."""
    pols = []
    for stage, rules in stage_rules.items():
        for veto in (False, True):
            pols.append((f"rule:{stage}{'+veto' if veto else ''}", stage,
                         lambda s, d, st=stage, r=rules, v=veto: StaticRule(
                             s, d, st, r, veto=v, label=f"rule:{st}")))
    return pols


def bm25_rules(k: int) -> list[tuple]:
    """Stage 4 static conditions over the primary (bm25) hits: fixed in
    advance from what the Stage 3 failure table says, not fit. A thin bm25
    list (few scored hits) or a flat one (high entropy, small margin) looks
    like a near miss, which hybrid recovers most often; a query that names
    no pool title looks like a bridge or named miss, which dense recovers
    more often than hybrid."""
    return [
        ("thin_list", lambda f: f["f:n_scored_hits"] < k, "hybrid"),
        ("flat_scores", lambda f: f["f:score_entropy"] > 0.95 and f["f:score_ratio"] > 0.9,
         "hybrid"),
        ("no_title", lambda f: f["f:query_title_mentions"] == 0, "dense"),
    ]


def evaluate(table: pd.DataFrame, default: str, seeds: Sequence[int] = (13, 17, 19, 23, 29),
             policies: Optional[Sequence[tuple]] = None,
             rules: Optional[dict] = None,
             pairs: Optional[dict] = None) -> tuple[pd.DataFrame, list[dict]]:
    """Every policy on every seeded order. Returns the summary table (mean
    and sd over orders; gain over the fixed default paired per order with
    the largest McNemar p) and the boundary memory's revision log on the
    first order. With `pairs` (twin_pairs of a with_twins table) every row
    also carries the twin metrics, averaged over orders, the McNemar p as
    its largest."""
    arms = arms_of(table)
    if default not in arms:
        raise ValueError(f"default {default!r} is not one of the arms {arms}")
    orders = seeded_orders(len(table), seeds)
    pols = list(policies if policies is not None else default_policies(default))
    pols = [(f"fixed:{a}", "query", lambda s, d, arm=a: FixedArm(s, d, arm)) for a in arms] \
        + pols + rule_policies(rules or {})
    streams = {stage: [make_stream(table, stage, o) for o in orders] for stage in DECIDE_STAGES}
    rows, logs = [], {}
    for label, stage, factory in pols:
        if stage in ("primary_hits", "history") and default != arms[0]:
            continue   # the primary-hits features belong to the log's primary arm
        per_order = []
        for s in streams[stage]:
            res = run_stream(s, factory(s, default))
            base = s.H[s.order][:, arms.index(default)]
            only_a, only_b, p = mcnemar(base.tolist(), (res.hit >= 1).astype(int).tolist())
            per_order.append({"rate": res.hit.mean(), "cost": res.cost.mean(),
                              "gain": res.hit.mean() - base.mean(), "oracle": s.H[s.order].max(axis=1).mean(),
                              "mcnemar_p": p, "only_policy": only_b, "only_default": only_a,
                              **boundary_metrics(s, res, default), **res.state,
                              **(twin_metrics(s, res, pairs, default) if pairs else {})})
            if res.log and label not in logs:
                logs[label] = [{"policy": label, **e} for e in res.log]
        df = pd.DataFrame(per_order)
        row = {"policy": label, "stage": stage, "default": default, "n_orders": len(df),
               "rate": round(df["rate"].mean(), 4), "rate_sd": round(df["rate"].std(ddof=0), 4),
               "oracle": round(df["oracle"].mean(), 4),
               "cost": round(df["cost"].mean(), 4),
               "gain": round(df["gain"].mean(), 4), "gain_sd": round(df["gain"].std(ddof=0), 4),
               "mcnemar_p_max": round(df["mcnemar_p"].max(), 4),
               "only_policy": round(df["only_policy"].mean(), 1),
               "only_default": round(df["only_default"].mean(), 1)}
        for c in ("n_recoverable", "n_fired", "repeated_failure", "false_application", "harm",
                  "false_rejection", "boundary_precision", "boundary_recall", "n_lessons",
                  "n_active", *[f"op_{o}" for o in OPERATORS]):
            if c in df:
                row[c] = round(float(df[c].mean()), 4)
        if pairs:
            for c in ("n_pairs", "n_changed", *[f"{c}_{w}" for w in ("unchanged", "changed")
                                                 for c in TWIN_COLUMNS]):
                row[c] = round(float(df[c].mean()), 4)
            row["p_vs_repeat_changed"] = round(float(df["p_vs_repeat_changed"].max()), 4)
        rows.append(row)
    boundary = [k for k in logs if k.startswith("boundary")]
    first_log = logs[boundary[0]] if boundary else next(iter(logs.values()), [])
    return pd.DataFrame(rows), first_log


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="experience jsonl with shadow arms")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outcome", default="all_gold_in_topk", choices=OUTCOMES)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--default", default=None, help="the default arm (the log's primary)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 17, 19, 23, 29])
    ap.add_argument("--twin-records", default=None,
                    help="experience jsonl of the twins run: one stream of originals and twins "
                         "per twin kind, with the pair metrics and the text-keyed memories")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args(argv)
    header, records = read_records(args.records)
    table = arm_table(records, outcome=args.outcome, ids=read_ids(args.ids) if args.ids else None)
    arms = arms_of(table)
    default = args.default or arms[0]
    k = int(header["strategy"]["top_k"])
    rules = {"primary_hits": bm25_rules(k)} if default == arms[0] else None
    print(f"{args.records}: {len(table)} questions, arms {arms}, default {default}, k={k}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    if args.twin_records:
        _, twin_records = read_records(args.twin_records)
        twins = arm_table(twin_records, outcome=args.outcome)
        kinds = sorted({i.rsplit(TWIN_SEP, 1)[1] for i in twins["id"] if TWIN_SEP in i})
        summaries, logs = [], []
        for kind in kinds:
            both = with_twins(table, twins[twins["id"].str.endswith(f"{TWIN_SEP}{kind}")])
            pairs = twin_pairs(both)
            print(f"\n{kind}: {len(pairs)} pairs, {len(both)} questions in the stream")
            summary, log = evaluate(both, default, seeds=args.seeds,
                                    policies=text_policies() + default_policies(default),
                                    rules=rules, pairs=pairs)
            summary.insert(0, "kind", kind)
            summaries.append(summary)
            logs += [{"kind": kind, **e} for e in log]
            print(summary.to_string(index=False))
        pd.concat(summaries, ignore_index=True).to_csv(out / f"{args.dataset}_twin_stream.csv",
                                                       index=False)
        pd.DataFrame(logs).to_csv(out / f"{args.dataset}_twin_stream_revisions.csv", index=False)
        return

    summary, log = evaluate(table, default, seeds=args.seeds, rules=rules)
    summary.to_csv(out / f"{args.dataset}_stream.csv", index=False)
    pd.DataFrame(log).to_csv(out / f"{args.dataset}_stream_revisions.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nrevision log: {len(log)} entries -> {out / f'{args.dataset}_stream_revisions.csv'}")


if __name__ == "__main__":
    main()
