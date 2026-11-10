"""Main agent over subagents (docs/plan.md, Stage 6).

The main agent splits the question into sub-questions, runs one subagent per
sub-question in order, and answers the original question from the union of
the subagents' notebooks.

    plan        one completion: a numbered list of sub-questions, where #k
                stands for the answer to sub-question k (Self-Ask /
                least-to-most style). Comparison questions come out as
                parallel lookups, bridge questions as a chain.
    subagent    the evidence agent's keep / search loop (agent.search_loop)
                on the sub-question, with #k replaced by subagent k's answer,
                then the answer prompt over its own notebook for that answer
    answer      the same PROMPT every run uses, over the union of the
                notebooks in the order kept; empty union falls back to
                everything retrieved, as in the single agent

Every subagent searches the question's own pool. pipeline.max_subquestions
caps the plan (default 3) and pipeline.sub_max_hops each subagent's searches
(default 2). A plan that cannot be read becomes the original question as the
only sub-question, logged as plan_fallback.

The memory hook is the same as the agent's: `pipe.lessons.lessons(step)`,
where step["role"] says who is asking ("planner" or "subagent"), so lessons
can be kept per role.
"""
import re

from src.agent.agent import Usage, answer_from, format_lessons, search_loop

PLAN_PROMPT = """Break the question into the simple lookups needed to answer \
it, in order. Use #1, #2, ... for the answer of an earlier lookup. If one \
lookup is enough, give the question itself as the only line.

Example
Question: Which was released first, the film Blue Harbor or the film The Glass Road?
1. When was the film Blue Harbor released?
2. When was the film The Glass Road released?

Example
Question: In what city was the director of the film Night Orchard born?
1. Who directed the film Night Orchard?
2. In what city was #1 born?
{lessons}
Question: {question}
Reply with the numbered list only."""

PLAN_TOKENS = 128
_LINE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.+?)\s*$")
_REF = re.compile(r"#(\d+)")


def parse_plan(text, cap):
    """Sub-questions in order, at most cap; [] when nothing numbered."""
    subs = []
    for line in text.strip().splitlines():
        m = _LINE.match(line)
        if m and m.group(2):
            subs.append(m.group(2))
        if len(subs) == cap:
            break
    return subs


def fill_refs(sub, answers):
    """Replace #k with the k-th sub-answer; a reference to a later or missing
    answer is left as written."""
    def rep(m):
        k = int(m.group(1)) - 1
        return answers[k] if 0 <= k < len(answers) and answers[k] else m.group(0)
    return _REF.sub(rep, sub)


def run_orchestrated(pipe, ex):
    """Returns (traces, answer, (ntok, tps, peak)); sets pipe.stop_reason,
    pipe.notebook and pipe.steps (one plan step, then the subagents' steps,
    then one step per sub-answer)."""
    cfg = pipe.cfg.pipeline
    cap = getattr(cfg, "max_subquestions", 3)
    sub_hops = getattr(cfg, "sub_max_hops", 2)
    usage = Usage()
    traces, steps, seen = [], [], {}

    plan_step = {"role": "planner", "question": ex.question}
    lessons = pipe.lessons.lessons(dict(plan_step)) if pipe.lessons else []
    text, m = pipe._gen(PLAN_PROMPT.format(question=ex.question, lessons=format_lessons(lessons)),
                        max_tokens=PLAN_TOKENS)
    usage.add(m)
    subs = parse_plan(text, cap)
    plan_fallback = not subs
    if plan_fallback:
        subs = [ex.question]
    if lessons:
        plan_step["lesson_ids"] = list(getattr(pipe.lessons, "last_ids", []))
    plan_step.update(lessons=lessons, reply=text, plan=subs, plan_fallback=plan_fallback)
    steps.append(plan_step)

    notebook, answers, stops = {}, [], []
    for k, sub in enumerate(subs):
        asked = fill_refs(sub, answers)
        start = len(traces)
        kept, stop = search_loop(pipe, ex, asked, sub_hops, usage, traces, steps, seen,
                                 role="subagent", sub=k)
        for cid, h in kept.items():
            notebook.setdefault(cid, h)
        own = list(kept.values()) or list({h.chunk.chunk_id: h for t in traces[start:]
                                           for h in t.retrieved}.values())
        sub_answer = answer_from(pipe, asked, own, usage)
        answers.append(sub_answer)
        stops.append(stop + ("" if kept else "+fallback"))
        steps.append({"role": "subagent", "sub": k, "sub_question": asked, "planned": sub,
                      "sub_answer": sub_answer, "kept": list(kept), "stop": stops[-1]})

    ctx_hits = list(notebook.values())
    fallback = not ctx_hits
    if fallback:
        ctx_hits = list(seen.values())
    answer = answer_from(pipe, ex.question, ctx_hits, usage)
    pipe.stop_reason = (f"subs={len(subs)}" + ("+plan_fallback" if plan_fallback else "")
                        + ("+fallback" if fallback else ""))
    pipe.notebook = list(notebook)
    pipe.steps = steps
    return traces, answer, usage.stats()
