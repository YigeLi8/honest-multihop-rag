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

Version 2 (docs/plan.md, Stage 6 A7), switched on per config; the defaults
are version 1, so the A5 configs run as they did:

    batch=k       a lesson is written from k failures of its type on k
                  different questions, grouped by the mode's own notion of
                  near (step features for regions, question words for flat),
                  in one reflection call that has to cover all of them; the
                  box spans the k points plus radius sd on each side
    generality    a rule is rejected if it uses a content word that occurs
                  in fewer than half of its source steps (it is about one of
                  them); one retry names the words, then the fixed sentence
    credit=heldout
                  while building, a decision where lessons apply shows none
                  with probability p_control (the lessons are logged as held
                  out) and otherwise one of them at random. A lesson's
                  utility is the step success rate where it was shown minus
                  where it applied and was held out, so it is compared with
                  the same situations, not with the role's average. Success
                  is the step free of every error type of its role, so a
                  lesson that trades one error for another gets no credit.
                  Retired when both counts reach min_n and utility <= 0;
                  shown when frozen only if utility - z * se > 0
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

REFLECT_BATCH_PROMPT = """You are improving an evidence-collecting assistant. \
Below are {k} steps, from different questions, where it made the same kind of \
mistake: {kind}.

{cases}

Write one general rule, under 25 words, that would have avoided this mistake \
in all {k} steps and would still make sense for a question on any other topic. \
Do not mention any topic, name, title, place, date or number from these steps. \
Reply with the rule only."""

RETRY = """

Your rule was: {advice}
It is too specific: it mentions {words}, which only some of these steps are \
about. Write one rule that holds for all {k} steps. Reply with the rule only."""

KIND = {
    "unreached": "its plan of lookups never reached an article the answer needed",
    "dropped": "it left out search results that were needed",
    "noise": "it kept search results that were not needed",
}

# words a rule about evidence may use whatever the topic
GENERIC = set("""
question questions result results search searches keep kept keeping include includes
including included leave left answer answers entity entities fact facts sentence sentences
article articles name names named mention mentions mentioned mentioning directly related
relevant irrelevant information detail details subject subjects step steps lookup lookups
always only every each both other others different same specific first second which that
this these those with from about when where what whom whose there their they them then than
more less also into does have been were will would should could must need needed needs ask
asks asked asking give gives given state states stated describe describes described
description background context unless even before after while until like such some most
many much make made look looks find finds found check checks part parts provide provides
provided support supports supporting evidence notebook enough missing link links linked
connect connects connecting between person people place places date dates year years time
work works title titles thing things kind kinds type types term terms query queries
attribute attributes property properties value values main core central actual""".split())


def _stem(w):
    return w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w


_GENERIC_STEMS = {_stem(g) for g in GENERIC}


def specific_words(advice, texts):
    """Content words of the advice that occur in some of the source texts but
    in fewer than half of them (with one source, in it at all)."""
    k = len(texts)
    bags = [{_stem(w) for w in re.findall(r"[a-z]+", t.lower())} for t in texts]
    out = []
    for w in sorted({_stem(w) for w in re.findall(r"[a-z]+", advice.lower())}):
        if len(w) <= 3 or w in _GENERIC_STEMS:
            continue
        n = sum(w in b for b in bags)
        if n and (k == 1 or 2 * n < k):
            out.append(w)
    return out


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
    held: int = 0                      # credit=heldout: applied, held out by the coin
    ok_held: int = 0
    members: list = field(default_factory=list)      # batch: [question id, question]

    def contains(self, x, sd):
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        if not np.all((x >= lo) & (x <= hi)):
            return False
        return not any(np.linalg.norm((x - np.asarray(p)) / sd) <= r for p, r in self.exceptions)

    def rate(self):
        return (self.ok_shown + 1) / (self.shown + 2)

    def held_rate(self):
        return (self.ok_held + 1) / (self.held + 2)

    def questions(self):
        return [q for _, q in self.members] or [self.question]


class ExperienceMemory:
    def __init__(self, cfg, pipe, mode="region", learn=True, store=None, load=None,
                 radius=1.0, min_n=5, expand_margin=0.5, exception_radius=0.25,
                 max_exceptions=3, max_shown=3, warmup=20, reflect=True, seed=13,
                 credit="base", batch=1, generality=False, p_control=0.5, z=1.0,
                 buffer_cap=100):
        assert mode in MODES, mode
        assert credit in ("base", "heldout"), credit
        self.cfg, self.pipe = cfg, pipe
        self.mode, self.learn = mode, learn
        self.store = Path(store) if store else None
        self.radius, self.min_n = radius, min_n
        self.expand_margin, self.exception_radius = expand_margin, exception_radius
        self.max_exceptions, self.max_shown = max_exceptions, max_shown
        self.warmup = warmup
        self.reflect, self.seed = reflect, seed
        self.credit, self.batch, self.generality = credit, batch, generality
        self.p_control, self.z, self.buffer_cap = p_control, z, buffer_cap
        self.rng = random.Random(seed)
        self._held = {}           # step_key -> lesson ids held out in the current episode
        self.last_held = []
        self._buffer = {}         # (role, type) -> failures waiting for a batch
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
                   max_shown=getattr(m, "max_shown", 3), warmup=getattr(m, "warmup", 20),
                   reflect=getattr(m, "reflect", True), seed=getattr(m, "seed", 13),
                   credit=getattr(m, "credit", "base"), batch=getattr(m, "batch", 1),
                   generality=getattr(m, "generality", False),
                   p_control=getattr(m, "p_control", 0.5), z=getattr(m, "z", 1.0))

    # -- what the agent is shown
    def _active(self, role):
        return [l for l in self.entries if l.role == role and not l.retired]

    def _base_rate(self, role, typ):
        ok, n = self.base[(role, typ)]
        return (ok + 1) / (n + 2)

    def utility(self, lesson):
        if self.credit == "heldout":
            return lesson.rate() - lesson.held_rate()
        return lesson.rate() - self._base_rate(lesson.role, lesson.type)

    def se(self, lesson):
        p1, p0 = lesson.rate(), lesson.held_rate()
        return float(np.sqrt(p1 * (1 - p1) / (lesson.shown + 2) + p0 * (1 - p0) / (lesson.held + 2)))

    def tested(self, lesson):
        return lesson.shown >= self.min_n and (self.credit == "base" or lesson.held >= self.min_n)

    def proven(self, lesson):
        """credit=heldout: enough trials both ways and utility - z * se > 0."""
        return self.tested(lesson) and self.utility(lesson) - self.z * self.se(lesson) > 0

    def _eligible(self, lesson):
        if self.credit == "heldout":
            # building: every active lesson is still under test; frozen: proven only
            return True if self.learn else self.proven(lesson)
        return lesson.shown < self.min_n or self.utility(lesson) > 0

    def _rank(self, lesson):
        if self.credit == "heldout":
            return self.utility(lesson) - self.z * self.se(lesson)
        return self.utility(lesson)

    def lessons(self, step):
        """Advice strings for one decision; remembers which lessons it showed
        and, with held-out credit while building, which it held out."""
        self.last_ids, self.last_held = [], []
        if self.mode == "none":
            return []
        role = "planner" if step.get("role") == "planner" else "subagent"
        key = "planner" if role == "planner" else step["hop"]
        pool = [l for l in self._active(role) if self._eligible(l)]
        if self.mode in ("region", "shuffled"):
            x = planner_features(step["question"]) if role == "planner" else step_point(step)
            sd = self.std[role].sd
            pool = [l for l in pool if l.contains(x, sd)]
            pick = sorted(pool, key=lambda l: (-self._rank(l), l.created))
        elif self.mode == "flat":
            q = _words(step.get("sub_question") or step["question"])
            sim = {l.id: max(len(q & _words(t)) / max(1, len(q | _words(t))) for t in l.questions())
                   for l in pool}
            pick = sorted([l for l in pool if sim[l.id] > 0], key=lambda l: (-sim[l.id], l.created))
            if self.credit == "heldout" and self.learn:
                pick = pick[:3]           # the lessons that apply: the three most similar
        else:  # global
            pick = sorted(pool, key=lambda l: (-self._rank(l), l.created))
        if self.credit == "heldout" and self.learn and pick:
            if self.rng.random() < self.p_control:
                self._held[key] = self.last_held = [l.id for l in pick]
                pick = []
            else:
                pick = [self.rng.choice(pick)]
        pick = pick[:self.max_shown]
        self._shown[key] = self.last_ids = [l.id for l in pick]
        return [l.advice for l in pick]

    # -- after the episode
    def on_episode(self, ex, res, row):
        self.t += 1
        shown, self._shown = self._shown, {}
        held, self._held = self._held, {}
        if not self.learn:
            return
        by_id = {l.id: l for l in self.entries}
        for role, key, x, failed, detail in score_steps(ex, res):
            sd = self.std[role].sd
            ids = shown.get(key, [])
            covered = {l.type for l in self._active(role) if l.contains(x, sd)}
            if self.credit == "heldout":
                ok = not any(failed.values())
                for i in ids:
                    by_id[i].shown += 1
                    by_id[i].ok_shown += ok
                for i in held.get(key, []):
                    by_id[i].held += 1
                    by_id[i].ok_held += ok
            for typ, bad in failed.items():
                of_type = [by_id[i] for i in ids if i in by_id and by_id[i].type == typ]
                if not of_type:
                    self.base[(role, typ)][0] += not bad
                    self.base[(role, typ)][1] += 1
                for l in of_type:
                    if self.credit == "base":
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
                    failure = self._failure(x, ex, detail)
                    if self.batch > 1:
                        self._enqueue(role, typ, failure, sd)
                    else:
                        self._create(role, typ, [failure], sd)
            self.std[role].update(x)
        for l in list(self.entries):
            if l.retired:
                continue
            if self.mode == "region" and len(l.exceptions) > self.max_exceptions:
                self._split(l)
            elif self.tested(l) and self.utility(l) <= 0:
                self._retire(l, "utility <= 0")

    # -- operators
    def _record(self, lesson, op, reason):
        self.log.append({"t": self.t, "lesson": lesson.id, "role": lesson.role,
                         "type": lesson.type, "op": op, "reason": reason,
                         "shown": lesson.shown, "utility": round(self.utility(lesson), 4),
                         "n_support": len(lesson.support), "n_exceptions": len(lesson.exceptions)})
        if self.credit == "heldout":
            self.log[-1].update(held=lesson.held, se=round(self.se(lesson), 4))

    @staticmethod
    def _failure(x, ex, detail):
        """One failed decision, as a lesson is written from it."""
        text = " ".join([ex.question, detail.get("question", ""),
                         *(t for _, t, _ in detail.get("shown", [])), *detail.get("plan", [])])
        return {"x": x, "id": ex.id, "question": detail.get("question", ex.question),
                "ex_question": ex.question, "detail": detail, "text": text,
                "titles": sorted({g.split("::")[0] for g in ex.gold_chunk_ids})}

    def _enqueue(self, role, typ, failure, sd):
        """batch > 1: hold the failure until batch - 1 others of its type, on
        other questions, are waiting; then write one lesson from it and its
        nearest ones (step features for regions, question words for flat)."""
        buf = self._buffer.setdefault((role, typ), [])
        if self.mode == "region":   # a waiting point a lesson has since come to cover is done
            act = [l for l in self._active(role) if l.type == typ]
            buf[:] = [f for f in buf if not any(l.contains(f["x"], sd) for l in act)]
        if self.mode == "region":
            dist = lambda f: float(np.linalg.norm((f["x"] - failure["x"]) / sd))  # noqa: E731
        else:
            q = _words(failure["question"])
            dist = lambda f: 1 - len(q & _words(f["question"])) / max(1, len(q | _words(f["question"])))  # noqa: E731
        chosen, ids = [], {failure["id"]}
        for f in sorted(buf, key=dist):
            if f["id"] not in ids:
                chosen.append(f)
                ids.add(f["id"])
            if len(chosen) == self.batch - 1:
                break
        if len(chosen) < self.batch - 1:
            buf.append(failure)
            del buf[:-self.buffer_cap]
            return
        buf[:] = [f for f in buf if not any(f is c for c in chosen)]
        self._create(role, typ, [failure, *chosen], sd)

    def _create(self, role, typ, members, sd):
        if len(members) == 1 and self.batch == 1:
            advice, source, prompt, reply = self._advice(typ, members[0])
        else:
            advice, source, prompt, reply = self._advice_batch(typ, members)
        self._n_created += 1
        half = self.radius * sd
        pts = np.asarray([m["x"] for m in members])
        first = members[0]
        lesson = Lesson(id=f"{role[0]}{self._n_created}", role=role, type=typ, advice=advice,
                        exemplar=(pts.mean(axis=0) if len(members) > 1 else first["x"]).tolist(),
                        lo=(pts.min(axis=0) - half).tolist(), hi=(pts.max(axis=0) + half).tolist(),
                        question=first["question"], created=self.t, source=source,
                        support=pts.tolist(),
                        members=[[m["id"], m["question"]] for m in members] if len(members) > 1
                        else [])
        self.entries.append(lesson)
        self._record(lesson, "create", f"{typ} on {', '.join(m['id'] for m in members)}")
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
                           created=self.t, source=l.source, support=pts.tolist(),
                           members=l.members)
            self.entries.append(child)
            self._record(child, "create", f"split of {l.id}")

    def _retire(self, l, reason):
        l.retired = True
        self._record(l, "retire", reason)

    # -- advice
    def _situation(self, typ, detail, clip=None):
        if typ == "unreached":
            plan = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(detail["plan"]))
            return f"The lookups it planned:\n{plan}"
        lines = []
        for i, (cid, text, needed) in enumerate(detail["shown"]):
            mark = "KEPT" if cid in detail["kept"] else "left out"
            need = ", needed" if needed else ""
            text = text.strip()
            if clip and len(text) > clip:
                text = text[:clip].rsplit(" ", 1)[0] + " ..."
            lines.append(f"[{i + 1}] ({mark}{need}) {text}")
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

    def _advice(self, typ, failure):
        """(advice, source, prompt, reply) from one failure. The reflection call
        goes to the generator directly, so it never counts toward an
        episode's calls."""
        if not self.reflect or self.pipe is None or self.pipe.generator is None:
            return FALLBACK[typ], "fallback", "", ""
        detail = failure["detail"]
        prompt = REFLECT_PROMPT.format(question=failure["ex_question"],
                                       situation=self._situation(typ, detail),
                                       mistake=self._mistake(typ, detail))
        return self._ask(typ, prompt, [failure])

    def _advice_batch(self, typ, members):
        """One rule from several failures of the same type on different
        questions; the situations are clipped to keep the prompt short."""
        if not self.reflect or self.pipe is None or self.pipe.generator is None:
            return FALLBACK[typ], "fallback", "", ""
        cases = "\n\n".join(
            f"Step {i + 1}\nQuestion: {m['ex_question']}\n"
            f"{self._situation(typ, m['detail'], clip=200)}\n"
            f"Mistake: {self._mistake(typ, m['detail'])}" for i, m in enumerate(members))
        prompt = REFLECT_BATCH_PROMPT.format(k=len(members), kind=KIND[typ], cases=cases)
        return self._ask(typ, prompt, members)

    def _ask(self, typ, prompt, members):
        """Reflection call, the leak check against every source question and
        gold title, and (with generality) the specific-word check with one
        retry that names the words."""
        def first(reply):
            return reply.strip().splitlines()[0].strip().strip('"') if reply.strip() else ""

        def problem(advice):
            if not advice or len(advice.split()) > 40:
                return "empty or long", []
            if any(leaks(advice, m["ex_question"], m["titles"]) for m in members):
                return "leak", []
            words = specific_words(advice, [m["text"] for m in members]) if self.generality else []
            return ("specific", words) if words else (None, [])

        reply, _ = self.pipe.generator.generate(prompt, 64)
        advice = first(reply)
        why, words = problem(advice)
        if why == "specific":
            retry = prompt + RETRY.format(advice=advice, words=", ".join(words), k=len(members))
            again, _ = self.pipe.generator.generate(retry, 64)
            reply = reply + "\n--- retry ---\n" + again
            advice = first(again)
            if problem(advice)[0] is None:
                return advice, "reflection_retry", retry, reply
            return FALLBACK[typ], "fallback", retry, reply
        if why:
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
    out = {"lessons": len(mem.entries), "active": sum(not l.retired for l in mem.entries),
           "fallback_advice": sum(l.source == "fallback" for l in mem.entries), **ops}
    if mem.credit == "heldout":
        act = [l for l in mem.entries if not l.retired]
        out.update(tested=sum(mem.tested(l) for l in act), proven=sum(mem.proven(l) for l in act),
                   retry=sum(l.source == "reflection_retry" for l in mem.entries))
    return out
