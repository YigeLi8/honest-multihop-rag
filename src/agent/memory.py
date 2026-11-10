"""Experience memory for the agents (docs/plan.md, Stage 6, A3-A4).

A lesson is short advice for one role plus the region of situations where it
applies. The agent asks `lessons(step)` before each decision and is shown the
advice of at most three applicable lessons; after the episode,
`on_episode(ex, res, row)` scores what happened against the gold (the only
place gold enters), credits or blames the lessons that were shown, revises
their regions, and turns new failures into new lessons.

Roles and the decision point each lesson is attached to:

    planner    the main agent's split of the question; the point is
               planner_features(question), computed from the question text
    subagent   one keep / search step (also the single agent's steps); the
               point is the frozen Stage 1 features of that step's search

Errors, typed after the episode from gold:

    unreached  planner: a gold paragraph was never retrieved by any search
    dropped    subagent step: a gold sentence was shown and is not in the
               final notebook
    noise      subagent step: at least two non-gold sentences kept, more
               than the gold kept at that step

A step (or plan) is a failure of type T if T occurred there and a success of
T otherwise. A lesson is about one (role, type).

Regions (mode "region", the claim). A box in raw feature units over every
dimension, created at the exemplar with half-width `radius` running standard
deviations, revised by the Stage 5 operators (src/memory/stream.py), every
revision appended to `log`:

    narrow     shown, and its type failed there: the box edge on the dimension
               farthest from the exemplar is pulled inside the point, unless
               that loses a supporting point; then
    exception  the point is excluded with a small ball
    expand     not shown (outside the box), its type failed there, and the
               point is within `expand_margin` sd of the box: the box grows
    split      more than max_exceptions exceptions: cut at the supporters'
               median on the dimension that best separates them from the
               contradicting points
    retire     utility <= 0 after min_n showings
    create     a failure where no lesson of its type was shown and (with
               regions) none covered the point, once `warmup` points of the
               role have been seen

Feedback is bandit feedback: a lesson's effect is seen only where it is
shown. So a new lesson is on probation, shown wherever it applies until it
has min_n showings; after that it is shown only while its utility is
positive. Utility = smoothed success rate of its type when it was shown,
minus the role's running success rate for that type at points where no
lesson of the type was shown.

Advice comes from a reflection call to the same model over the failed step,
with the needed sentences marked (ExpeL style), checked for leaks (no
capitalised word of the question or of a gold title, no digits); a leaking
or empty reply falls back to a fixed sentence per type, logged as such.

Baselines on the same store (docs/plan.md, Stage 6):

    flat       no regions and no operators: the lessons of the role whose
               exemplar question is most similar to the current question
               (token Jaccard), probation and utility as above
    global     the role's top lessons by utility, shown everywhere
    shuffled   regions kept, advice permuted among the role's lessons at load
    none       nothing shown (the no-memory agent)

learn=False freezes the store (evaluation); learn=True revises and creates
(building on train questions, or online).
"""
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.memory.features import FEATURES, cap_spans, tokens
from src.memory.stream import Standardiser

MODES = ("region", "flat", "global", "shuffled", "none")
TYPES = {"planner": ("unreached",), "subagent": ("dropped", "noise")}

CUES = ("both", "same", "or", "which", "first", "older", "younger", "more", "less",
        "than", "whose", "what", "who", "where", "when", "how", "is", "was", "are", "did")
PLANNER_FEATURES = ("n_tokens", "n_cap_spans", "n_commas", "n_quotes", *(f"cue_{c}" for c in CUES))

FALLBACK = {
    "unreached": "Make one lookup for every person, place or work the question names, "
                 "and one for each fact that links them.",
    "dropped": "Keep every result that states a fact about an entity in the question, "
               "even if it does not answer it directly.",
    "noise": "Keep only results that state a fact the question needs; leave out "
             "background sentences about other entities.",
}

REFLECT_PROMPT = """You are improving an evidence-collecting assistant. Below is \
one step where it made a mistake, and what it should have done.

Question it was working on: {question}
{situation}
Mistake: {mistake}

Write one general rule, under 25 words, that would have avoided this mistake \
on similar questions. Do not mention any name, title, place, date or number \
from this question. Reply with the rule only."""

MISTAKE = {
    "unreached": "the lookups it planned never found the needed article{s} ({titles}).",
    "dropped": "it left out results it needed: {needed}.",
    "noise": "it kept results that were not needed: {extra}.",
}


def planner_features(question):
    toks = tokens(question)
    words = set(re.findall(r"[a-z]+", question.lower()))
    x = [len(toks), cap_spans(question), question.count(","), question.count('"')]
    x += [int(c in words) for c in CUES]
    return np.asarray(x, dtype=float)


def step_point(step):
    return np.asarray([float(step["features"][f]) for f in FEATURES], dtype=float)


def _words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def leaks(advice, question, titles):
    """Capitalised words of the question or of a gold title (not the first
    word of the advice), or any digit."""
    if re.search(r"\d", advice):
        return True
    named = set()
    for text in [question, *titles]:
        for i, w in enumerate(re.findall(r"[A-Za-z][\w'-]*", text)):
            if w[0].isupper() and (i > 0 or text in titles):
                named.add(w.lower())
    named -= {"the", "a", "an", "what", "who", "which", "where", "when", "how", "is", "was",
              "are", "were", "did", "do", "does", "in", "of"}
    return bool(named & _words(advice))


def score_steps(ex, res):
    """[(role, step_key, point, {type: failed}, detail)] for one finished
    episode. Gold enters here and nowhere earlier."""
    gold = set(ex.gold_chunk_ids)
    text = {c.chunk_id: c.text for c in ex.chunks}
    retrieved = {cid for h in res.hops for cid in (rc.chunk.chunk_id for rc in h.retrieved)}
    out = []
    for s in res.steps:
        if s.get("role") == "planner":
            missing = {g.split("::")[0] for g in gold} - {c.split("::")[0] for c in retrieved}
            out.append(("planner", "planner", planner_features(ex.question),
                        {"unreached": bool(missing)},
                        {"plan": s.get("plan", []), "missing_titles": sorted(missing)}))
        elif "features" in s:
            shown, kept = set(s["shown"]), set(s["kept"])
            dropped = (gold & shown) - set(res.notebook)     # shown here, never kept by anyone
            extra = kept - gold
            out.append(("subagent", s["hop"], step_point(s),
                        {"dropped": bool(dropped),
                         "noise": len(extra) >= 2 and len(extra) > len(kept & gold)},
                        {"question": s.get("sub_question", ex.question),
                         "shown": [(c, text.get(c, ""), c in gold) for c in s["shown"]],
                         "kept": sorted(kept), "dropped": sorted(dropped),
                         "extra": sorted(extra)}))
    return out


@dataclass
class Lesson:
    id: str
    role: str
    type: str
    advice: str
    exemplar: list
    lo: list
    hi: list
    question: str                      # the exemplar's question, for the flat baseline
    created: int
    source: str = "reflection"         # reflection | fallback
    shown: int = 0
    ok_shown: int = 0
    support: list = field(default_factory=list)
    exceptions: list = field(default_factory=list)   # [point, radius in sd]
    retired: bool = False

    def contains(self, x, sd):
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        if not np.all((x >= lo) & (x <= hi)):
            return False
        return not any(np.linalg.norm((x - np.asarray(p)) / sd) <= r for p, r in self.exceptions)

    def rate(self):
        return (self.ok_shown + 1) / (self.shown + 2)


class ExperienceMemory:
    def __init__(self, cfg, pipe, mode="region", learn=True, store=None, load=None,
                 radius=1.0, min_n=5, expand_margin=0.5, exception_radius=0.25,
                 max_exceptions=3, max_shown=3, warmup=20, reflect=True, seed=13):
        assert mode in MODES, mode
        self.cfg, self.pipe = cfg, pipe
        self.mode, self.learn = mode, learn
        self.store = Path(store) if store else None
        self.radius, self.min_n = radius, min_n
        self.expand_margin, self.exception_radius = expand_margin, exception_radius
        self.max_exceptions, self.max_shown = max_exceptions, max_shown
        self.warmup = warmup
        self.reflect, self.seed = reflect, seed
        self.entries: list[Lesson] = []      # every lesson, retired ones too
        self.log: list[dict] = []
        self.std = {"planner": Standardiser(len(PLANNER_FEATURES)),
                    "subagent": Standardiser(len(FEATURES))}
        self.base = {(r, t): [0, 0] for r, ts in TYPES.items() for t in ts}   # [ok, n] unshown
        self.t = 0
        self._n_created = 0
        self._shown = {}          # step_key -> lesson ids shown in the current episode
        if load:
            self.load(load)
        if mode == "shuffled":
            self._shuffle_advice()

    @classmethod
    def from_config(cls, cfg, pipe):
        m = cfg.agent_memory
        return cls(cfg, pipe, mode=getattr(m, "mode", "region"), learn=getattr(m, "learn", True),
                   store=getattr(m, "store", None), load=getattr(m, "load", None),
                   radius=getattr(m, "radius", 1.0), min_n=getattr(m, "min_n", 5),
                   warmup=getattr(m, "warmup", 20),
                   reflect=getattr(m, "reflect", True), seed=getattr(m, "seed", 13))

    # -- what the agent is shown
    def _active(self, role):
        return [l for l in self.entries if l.role == role and not l.retired]

    def _base_rate(self, role, typ):
        ok, n = self.base[(role, typ)]
        return (ok + 1) / (n + 2)

    def utility(self, lesson):
        return lesson.rate() - self._base_rate(lesson.role, lesson.type)

    def _eligible(self, lesson):
        return lesson.shown < self.min_n or self.utility(lesson) > 0

    def lessons(self, step):
        """Advice strings for one decision; remembers which lessons it showed."""
        self.last_ids = []
        if self.mode == "none":
            return []
        role = "planner" if step.get("role") == "planner" else "subagent"
        key = "planner" if role == "planner" else step["hop"]
        pool = [l for l in self._active(role) if self._eligible(l)]
        if self.mode in ("region", "shuffled"):
            x = planner_features(step["question"]) if role == "planner" else step_point(step)
            sd = self.std[role].sd
            pool = [l for l in pool if l.contains(x, sd)]
            pick = sorted(pool, key=lambda l: (-self.utility(l), l.created))
        elif self.mode == "flat":
            q = _words(step.get("sub_question") or step["question"])
            sim = lambda l: len(q & _words(l.question)) / max(1, len(q | _words(l.question)))  # noqa: E731
            pick = sorted([l for l in pool if sim(l) > 0], key=lambda l: (-sim(l), l.created))
        else:  # global
            pick = sorted(pool, key=lambda l: (-self.utility(l), l.created))
        pick = pick[:self.max_shown]
        self._shown[key] = self.last_ids = [l.id for l in pick]
        return [l.advice for l in pick]

    # -- after the episode
    def on_episode(self, ex, res, row):
        self.t += 1
        shown, self._shown = self._shown, {}
        if not self.learn:
            return
        by_id = {l.id: l for l in self.entries}
        for role, key, x, failed, detail in score_steps(ex, res):
            sd = self.std[role].sd
            ids = shown.get(key, [])
            covered = {l.type for l in self._active(role) if l.contains(x, sd)}
            for typ, bad in failed.items():
                of_type = [by_id[i] for i in ids if i in by_id and by_id[i].type == typ]
                if not of_type:
                    self.base[(role, typ)][0] += not bad
                    self.base[(role, typ)][1] += 1
                for l in of_type:
                    l.shown += 1
                    l.ok_shown += not bad
                    if not bad:
                        l.support.append(x.tolist())
                    elif self.mode == "region":
                        self._narrow_or_except(l, x, sd)
                grown = False
                if bad and self.mode == "region":
                    for l in self._active(role):
                        if l.type == typ and l.id not in ids and self._near(l, x, sd):
                            self._expand(l, x)
                            grown = True
                # a failure no lesson of its type was shown at and (with regions)
                # none covered before this episode's revisions or grew to cover
                if (bad and not of_type and self.std[role].n >= self.warmup
                        and (self.mode != "region" or (typ not in covered and not grown))):
                    self._create(role, typ, x, sd, ex, detail)
            self.std[role].update(x)
        for l in list(self.entries):
            if l.retired:
                continue
            if self.mode == "region" and len(l.exceptions) > self.max_exceptions:
                self._split(l)
            elif l.shown >= self.min_n and self.utility(l) <= 0:
                self._retire(l, "utility <= 0")

    # -- operators
    def _record(self, lesson, op, reason):
        self.log.append({"t": self.t, "lesson": lesson.id, "role": lesson.role,
                         "type": lesson.type, "op": op, "reason": reason,
                         "shown": lesson.shown, "utility": round(self.utility(lesson), 4),
                         "n_support": len(lesson.support), "n_exceptions": len(lesson.exceptions)})

    def _create(self, role, typ, x, sd, ex, detail):
        advice, source, prompt, reply = self._advice(typ, ex, detail)
        self._n_created += 1
        half = self.radius * sd
        lesson = Lesson(id=f"{role[0]}{self._n_created}", role=role, type=typ, advice=advice,
                        exemplar=x.tolist(), lo=(x - half).tolist(), hi=(x + half).tolist(),
                        question=detail.get("question", ex.question), created=self.t,
                        source=source, support=[x.tolist()])
        self.entries.append(lesson)
        self._record(lesson, "create", f"{typ} on {ex.id}")
        self.log[-1].update(advice=advice, source=source, reflect_prompt=prompt, reflect_reply=reply)

    def _narrow_or_except(self, l, x, sd):
        ex_pt, lo, hi = np.asarray(l.exemplar), np.asarray(l.lo), np.asarray(l.hi)
        z = (x - ex_pt) / sd
        for dim in np.argsort(-np.abs(z), kind="stable"):
            if abs(z[dim]) < 1e-9:
                break
            nlo, nhi = lo.copy(), hi.copy()
            eps = 1e-6 * sd[dim]
            if z[dim] > 0:
                nhi[dim] = x[dim] - eps
            else:
                nlo[dim] = x[dim] + eps
            if all(nlo[dim] <= p[dim] <= nhi[dim] for p in l.support):
                l.lo, l.hi = nlo.tolist(), nhi.tolist()
                self._record(l, "narrow", f"failed while shown; cut dim {int(dim)}")
                return
        l.exceptions.append([x.tolist(), self.exception_radius])
        self._record(l, "exception", "failed while shown; every cut loses a supporter")

    def _near(self, l, x, sd):
        lo, hi = np.asarray(l.lo), np.asarray(l.hi)
        margin = self.expand_margin * sd
        outside = np.any((x < lo) | (x > hi))
        return bool(outside and np.all((x >= lo - margin) & (x <= hi + margin)))

    def _expand(self, l, x):
        l.lo = np.minimum(l.lo, x).tolist()
        l.hi = np.maximum(l.hi, x).tolist()
        self._record(l, "expand", "its type failed just outside the box")

    def _split(self, l):
        sd = self.std[l.role].sd
        S = np.asarray(l.support)
        C = np.asarray([p for p, _ in l.exceptions])
        dim = int(np.argmax(np.abs(S.mean(axis=0) - C.mean(axis=0)) / sd))
        cut = float(np.median(S[:, dim]))
        l.retired = True
        self._record(l, "split", f"{len(l.exceptions)} exceptions; cut dim {dim} at {cut:.4g}")
        for side, pts in (("lo", S[S[:, dim] <= cut]), ("hi", S[S[:, dim] > cut])):
            if not len(pts):
                continue
            lo, hi = list(l.lo), list(l.hi)
            if side == "lo":
                hi[dim] = cut
            else:
                lo[dim] = cut
            child = Lesson(id=f"{l.id}.{side}", role=l.role, type=l.type, advice=l.advice,
                           exemplar=pts[0].tolist(), lo=lo, hi=hi, question=l.question,
                           created=self.t, source=l.source, support=pts.tolist())
            self.entries.append(child)
            self._record(child, "create", f"split of {l.id}")

    def _retire(self, l, reason):
        l.retired = True
        self._record(l, "retire", reason)

    # -- advice
    def _situation(self, typ, detail):
        if typ == "unreached":
            plan = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(detail["plan"]))
            return f"The lookups it planned:\n{plan}"
        lines = []
        for i, (cid, text, needed) in enumerate(detail["shown"]):
            mark = "KEPT" if cid in detail["kept"] else "left out"
            need = ", needed" if needed else ""
            lines.append(f"[{i + 1}] ({mark}{need}) {text.strip()}")
        return (f"It was looking up: {detail['question']}\nThe search results, what it did "
                f"with each, and which were needed:\n" + "\n".join(lines))

    def _mistake(self, typ, detail):
        idx = {cid: i + 1 for i, (cid, _, _) in enumerate(detail.get("shown", []))}
        if typ == "unreached":
            t = detail["missing_titles"]
            return MISTAKE[typ].format(s="s" if len(t) > 1 else "", titles=", ".join(t))
        if typ == "dropped":
            return MISTAKE[typ].format(needed=", ".join(f"[{idx[c]}]" for c in detail["dropped"]))
        return MISTAKE[typ].format(extra=", ".join(f"[{idx[c]}]" for c in detail["extra"]))

    def _advice(self, typ, ex, detail):
        """(advice, source, prompt, reply). The reflection call goes to the
        generator directly, so it never counts toward an episode's calls."""
        if not self.reflect or self.pipe is None or self.pipe.generator is None:
            return FALLBACK[typ], "fallback", "", ""
        prompt = REFLECT_PROMPT.format(question=ex.question, situation=self._situation(typ, detail),
                                       mistake=self._mistake(typ, detail))
        reply, _ = self.pipe.generator.generate(prompt, 64)
        advice = reply.strip().splitlines()[0].strip().strip('"') if reply.strip() else ""
        titles = sorted({g.split("::")[0] for g in ex.gold_chunk_ids})
        if not advice or len(advice.split()) > 40 or leaks(advice, ex.question, titles):
            return FALLBACK[typ], "fallback", prompt, reply
        return advice, "reflection", prompt, reply

    # -- persistence
    def _shuffle_advice(self):
        rng = random.Random(self.seed)
        for role in TYPES:
            ls = [l for l in self.entries if l.role == role]
            texts = [l.advice for l in ls]
            rng.shuffle(texts)
            for l, a in zip(ls, texts):
                l.advice = a

    def state(self):
        return {"t": self.t, "n_created": self._n_created, "mode": self.mode,
                "base": {f"{r}|{t}": v for (r, t), v in self.base.items()},
                "std": {r: {"n": s.n, "mean": s.mean.tolist(), "m2": s.m2.tolist()}
                        for r, s in self.std.items()},
                "lessons": [vars(l) for l in self.entries]}

    def save(self, path=None):
        path = Path(path or self.store)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state(), indent=1))
        with open(path.with_suffix(".log.jsonl"), "w") as f:
            for e in self.log:
                f.write(json.dumps(e) + "\n")

    def load(self, path):
        d = json.loads(Path(path).read_text())
        self.t, self._n_created = d["t"], d["n_created"]
        self.base = {tuple(k.split("|")): v for k, v in d["base"].items()}
        for r, s in d["std"].items():
            self.std[r].n, self.std[r].mean, self.std[r].m2 = s["n"], np.asarray(s["mean"]), \
                np.asarray(s["m2"])
        self.entries = [Lesson(**l) for l in d["lessons"]]


def summary(mem: Optional[ExperienceMemory]):
    if mem is None:
        return {}
    ops = {}
    for e in mem.log:
        ops[e["op"]] = ops.get(e["op"], 0) + 1
    return {"lessons": len(mem.entries), "active": sum(not l.retired for l in mem.entries),
            "fallback_advice": sum(l.source == "fallback" for l in mem.entries), **ops}
