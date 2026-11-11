"""Frozen experience memory on hotpot explore against the main agent without
memory (docs/plan.md, Stage 6 A5 and A7).

    PYTHONPATH=. .venv312/bin/python scripts/memory_eval.py              # A5, version 1
    PYTHONPATH=. .venv312/bin/python scripts/memory_eval.py --tag mem2   # A7, version 2

Reads the no-memory run (orchestrator_hotpot_explore) and the frozen memory
runs (orchestrator_<tag>_<mode>_hotpot_explore, mode in region, global,
shuffled, flat; a mode that was not run is skipped), all on the same 500
explore questions, plus the two stores built on train and the two build
runs. Writes

    results/orchestrator_<tag>_hotpot_explore_paired.csv
        one row per memory arm against no memory: em / f1 differences with
        paired bootstrap intervals, exact McNemar with Holm over the arms,
        the same by question type, notebook gold metrics, calls and
        latency, and how often lessons were shown
    results/orchestrator_<tag>_hotpot_explore_steps.csv
        per arm and error type: how often the error occurred, at steps where
        a lesson of that type was shown and where none was
    results/agent_memory/build_summary<suffix>.csv
        per store: operators, advice sources, active lessons by role and
        type, how many passed probation (version 2: tested and proven); per
        quarter of the build stream (questions in random order, so the
        quarters are exchangeable): em, show rate and error rates; and with
        held-out credit, the pooled randomized comparison over the build:
        step success where a lesson was shown against where the lessons
        that applied were held out
    results/agent_memory/<store>_lessons.csv (version 2)
        every lesson with its shown / held-out counts, utility, se, lower
        bound and advice

Shown-versus-not splits on the frozen runs are descriptive: where a lesson is
shown there is not random. The held-out comparisons from a version 2 build
are randomized.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from src.agent.memory import TYPES, ExperienceMemory, score_steps
from src.config import load_config
from src.eval.stats import holm, mcnemar, paired_bootstrap, wilson_ci

R = Path("results")
DATA = Path("data/processed")
BASE = "orchestrator_hotpot_explore"
ARMS = ("region", "global", "shuffled", "flat")
TAGS = {
    "mem": {"region": ("orchestrator_mem_build", "hotpot_train_region"),
            "flat": ("orchestrator_mem_build_flat", "hotpot_train_flat"), "suffix": ""},
    "mem2": {"region": ("orchestrator_mem2_build", "hotpot_train_region2"),
             "flat": ("orchestrator_mem2_build_flat", "hotpot_train_flat2"), "suffix": "2"},
}


def rows_of(run):
    with open(R / f"{run}_qa.csv") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def traces_of(run):
    with open(R / f"{run}_trace.jsonl") as f:
        return {d["id"]: d for d in map(json.loads, f)}


def gold_of(split, ids):
    want, out = set(ids), {}
    with open(DATA / f"hotpotqa_{split}.jsonl") as f:
        for line in f:
            # records start {"id": "...", so the id is read without parsing
            # the whole line (the train file is 0.9 GB)
            if line[:80].split('"')[3] in want:
                d = json.loads(line)
                out[d["id"]] = d["gold_chunk_ids"]
    return out


def step_errors(trace, gold):
    """score_steps on a trace record: [(role, key, {type: failed}, shown ids, held ids)]."""
    hops = [SimpleNamespace(retrieved=[SimpleNamespace(chunk=SimpleNamespace(chunk_id=c))
                                       for c in h["retrieved"]]) for h in trace["hops"]]
    res = SimpleNamespace(hops=hops, steps=trace["steps"], notebook=trace["notebook"])
    ex = SimpleNamespace(gold_chunk_ids=gold, chunks=[], question=trace["steps"][0]["question"])
    shown, held = {}, {}
    for s in trace["steps"]:
        key = "planner" if s.get("role") == "planner" else s.get("hop")
        if s.get("role") == "planner" or "features" in s:
            shown[key], held[key] = s.get("lesson_ids", []), s.get("held_ids", [])
    return [(role, key, failed, shown.get(key, []), held.get(key, []))
            for role, key, _, failed, _ in score_steps(ex, res)]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def paired(base, arm, ids, field):
    a = [float(base[i][field]) for i in ids]
    b = [float(arm[i][field]) for i in ids]
    d, lo, hi = paired_bootstrap(a, b)
    return d, lo, hi, a, b


def memory_of(build_run, store):
    """The store as the build left it, read with the build's own settings."""
    cfg = load_config(f"configs/{build_run}.yaml")
    mem = ExperienceMemory.from_config(cfg, None)
    mem.load(R / "agent_memory" / f"{store}.json")
    return mem


def show_stats(traces):
    q_any = planner = sub = sub_n = n_shown = 0
    distinct = set()
    for t in traces.values():
        any_ = False
        for s in t["steps"]:
            ids = s.get("lesson_ids") or []
            if s.get("role") == "planner":
                planner += bool(ids)
            elif "features" in s:
                sub_n += 1
                sub += bool(ids)
            if ids:
                any_ = True
                n_shown += len(ids)
                distinct.update(ids)
        q_any += any_
    return {"q_any_shown": q_any / len(traces), "planner_shown": planner / len(traces),
            "sub_steps": sub_n, "sub_shown": sub / sub_n if sub_n else 0.0,
            "lessons_per_shown_step": n_shown / max(1, planner + sub),
            "distinct_shown": len(distinct)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mem", choices=sorted(TAGS))
    tag = ap.parse_args().tag
    stores = {k: v for k, v in TAGS[tag].items() if k != "suffix"}
    mems = {k: memory_of(*v) for k, v in stores.items()}

    base = rows_of(BASE)
    ids = list(base)
    gold = gold_of("dev", ids)
    by_type = defaultdict(list)
    for i in ids:
        by_type[base[i]["qtype"]].append(i)

    out, steps_out, pvals = [], [], []
    none_tr = traces_of(BASE)
    ncnt = defaultdict(lambda: [0, 0])
    for i in ids:
        for role, key, failed, _, _ in step_errors(none_tr[i], gold[i]):
            for typ, bad in failed.items():
                ncnt[(role, typ)][0] += bad
                ncnt[(role, typ)][1] += 1
    for (role, typ), (k, n) in sorted(ncnt.items()):
        steps_out.append({"arm": "none", "role": role, "type": typ, "where": "all",
                          "steps": n, "error_rate": k / n})

    for m in ARMS:
        run = f"orchestrator_{tag}_{m}_hotpot_explore"
        if not (R / f"{run}_qa.csv").exists():
            print(f"{m}: not run, skipped")
            continue
        arm, tr = rows_of(run), traces_of(run)
        assert set(arm) == set(ids), run
        d_em, lo, hi, a, b = paired(base, arm, ids, "em")
        only_base, only_arm, p = mcnemar(a, b)
        d_f1, flo, fhi, _, _ = paired(base, arm, ids, "f1")
        row = {"arm": m, "n": len(ids), "em_none": mean(a), "em": mean(b),
               "d_em": d_em, "d_em_lo": lo, "d_em_hi": hi,
               "only_none": only_base, "only_arm": only_arm, "p": p,
               "d_f1": d_f1, "d_f1_lo": flo, "d_f1_hi": fhi}
        for qt, qids in sorted(by_type.items()):
            dq, qlo, qhi, qa, qb = paired(base, arm, qids, "em")
            row.update({f"n_{qt}": len(qids), f"d_em_{qt}": dq, f"d_em_{qt}_lo": qlo,
                        f"d_em_{qt}_hi": qhi, f"p_{qt}": mcnemar(qa, qb)[2]})
        for field in ("notebook_size", "notebook_gold_recall", "notebook_all_gold", "fallback",
                      "gen_calls", "latency_s"):
            row[f"{field}_none"] = mean(float(base[i][field]) for i in ids)
            row[field] = mean(float(arm[i][field]) for i in ids)
        row.update(show_stats(tr))
        # questions where anything was shown versus nothing (descriptive)
        shown_q = [i for i in ids if any(s.get("lesson_ids") for s in tr[i]["steps"])]
        quiet_q = [i for i in ids if i not in set(shown_q)]
        for t, qs in (("shownq", shown_q), ("quietq", quiet_q)):
            row[f"n_{t}"] = len(qs)
            row[f"d_em_{t}"] = (mean(float(arm[i]["em"]) for i in qs)
                                - mean(float(base[i]["em"]) for i in qs)) if qs else float("nan")
        # lessons shown that had passed probation (version 2: proven) at the end of the build
        mem = mems["flat" if m == "flat" else "region"]
        lessons = {l.id: l for l in mem.entries}
        shown_ids = [x for t in tr.values() for s in t["steps"] for x in s.get("lesson_ids") or []]
        row["shown_past_probation"] = mean(mem.tested(lessons[x]) for x in shown_ids) \
            if shown_ids else float("nan")
        out.append(row)
        pvals.append(p)

        # error rates per type, at steps with a lesson of that type shown and without
        cnt = defaultdict(lambda: [0, 0])
        for i in ids:
            for role, key, failed, sids, _ in step_errors(tr[i], gold[i]):
                for typ, bad in failed.items():
                    has = any(x in lessons and lessons[x].type == typ for x in sids)
                    for where in ("all", "shown" if has else "not_shown"):
                        cnt[(role, typ, where)][0] += bad
                        cnt[(role, typ, where)][1] += 1
        for (role, typ, where), (k, n) in sorted(cnt.items()):
            steps_out.append({"arm": m, "role": role, "type": typ, "where": where,
                              "steps": n, "error_rate": k / n})

    for row, ph in zip(out, holm(pvals)):
        row["p_holm"] = ph
    if out:
        write(R / f"orchestrator_{tag}_hotpot_explore_paired.csv", out)
    write(R / f"orchestrator_{tag}_hotpot_explore_steps.csv", steps_out)

    build = []
    for mode, (build_run, store) in stores.items():
        mem = mems[mode]
        lessons = {l.id: l for l in mem.entries}
        ops = Counter(json.loads(line)["op"]
                      for line in open(R / "agent_memory" / f"{store}.log.jsonl"))
        sources = Counter(l.source for l in lessons.values())
        rec = {"store": store, "part": "store", "lessons": len(lessons),
               "active": sum(not l.retired for l in lessons.values()),
               "reflection": sources["reflection"] + sources["reflection_retry"],
               "fallback_advice": sources["fallback"],
               **({"reflection_retry": sources["reflection_retry"]}
                  if mem.credit == "heldout" else {}),
               **{f"op_{k}": v for k, v in sorted(ops.items())}}
        for role, types in TYPES.items():
            for typ in types:
                act = [l for l in lessons.values()
                       if not l.retired and l.role == role and l.type == typ]
                rec[f"active_{role}_{typ}"] = len(act)
                if mem.credit == "heldout":
                    rec[f"tested_{role}_{typ}"] = sum(mem.tested(l) for l in act)
                    rec[f"proven_{role}_{typ}"] = sum(mem.proven(l) for l in act)
                else:
                    rec[f"proven_{role}_{typ}"] = sum(mem.tested(l) and mem.utility(l) > 0
                                                      for l in act)
                    rec[f"probation_{role}_{typ}"] = sum(l.shown < mem.min_n for l in act)
                    ok, n = mem.base[(role, typ)]
                    rec[f"base_ok_{role}_{typ}"] = (ok + 1) / (n + 2)
        build.append(rec)

        run = build_run.replace("configs/", "") + "_hotpot_train"
        rows, tr = rows_of(run), traces_of(run)
        order = list(rows)             # file order = stream order
        g = gold_of("train", order)
        q = len(order) // 4
        pooled = defaultdict(lambda: [0, 0])     # (role, shown|held) -> [ok, n]
        for k in range(4):
            part = order[k * q:(k + 1) * q]
            errs = defaultdict(lambda: [0, 0])
            for i in part:
                for role, key, failed, sids, hids in step_errors(tr[i], g[i]):
                    for typ, bad in failed.items():
                        errs[typ][0] += bad
                        errs[typ][1] += 1
                    ok = not any(failed.values())
                    for arm_, on in (("shown", sids), ("held", hids)):
                        if on:
                            pooled[(role, arm_)][0] += ok
                            pooled[(role, arm_)][1] += 1
            st = show_stats({i: tr[i] for i in part})
            build.append({"store": store, "part": f"q{k + 1}", "n": len(part),
                          "em": mean(float(rows[i]["em"]) for i in part),
                          "q_any_shown": st["q_any_shown"], "sub_shown": st["sub_shown"],
                          "planner_shown": st["planner_shown"],
                          **{f"err_{t}": v[0] / v[1] for t, v in sorted(errs.items())}})
        if mem.credit == "heldout":
            for role in TYPES:
                (ks, ns), (kh, nh) = pooled[(role, "shown")], pooled[(role, "held")]
                if not ns or not nh:
                    continue
                ps, ph = ks / ns, kh / nh
                se = (ps * (1 - ps) / ns + ph * (1 - ph) / nh) ** 0.5
                build.append({"store": store, "part": f"heldout_{role}", "n_shown": ns,
                              "ok_shown": ps, "ok_shown_lo": wilson_ci(ks, ns)[0],
                              "ok_shown_hi": wilson_ci(ks, ns)[1], "n_held": nh, "ok_held": ph,
                              "ok_held_lo": wilson_ci(kh, nh)[0],
                              "ok_held_hi": wilson_ci(kh, nh)[1], "diff": ps - ph,
                              "diff_lo": ps - ph - 1.96 * se, "diff_hi": ps - ph + 1.96 * se})
            write(R / "agent_memory" / f"{store}_lessons.csv",
                  [{"id": l.id, "role": l.role, "type": l.type, "source": l.source,
                    "members": " ".join(m[0] for m in l.members), "created": l.created,
                    "shown": l.shown, "ok_shown": l.ok_shown, "held": l.held,
                    "ok_held": l.ok_held, "utility": mem.utility(l), "se": mem.se(l),
                    "lcb": mem.utility(l) - mem.z * mem.se(l), "tested": int(mem.tested(l)),
                    "proven": int(mem.proven(l)), "retired": int(l.retired), "advice": l.advice}
                   for l in mem.entries])
    write(R / "agent_memory" / f"build_summary{TAGS[tag]['suffix']}.csv", build)

    for row in out:
        print(f"{row['arm']:9s} em {row['em']:.3f} vs {row['em_none']:.3f}  "
              f"d {row['d_em']:+.3f} [{row['d_em_lo']:+.3f}, {row['d_em_hi']:+.3f}]  "
              f"{row['only_none']}/{row['only_arm']} p {row['p']:.3f} holm {row['p_holm']:.3f}  "
              f"bridge {row['d_em_bridge']:+.3f} comparison {row['d_em_comparison']:+.3f}  "
              f"shown q {row['q_any_shown']:.2f} sub {row['sub_shown']:.2f} "
              f"planner {row['planner_shown']:.2f}  past probation {row['shown_past_probation']:.2f}")
    for rec in build:
        if rec["part"].startswith("heldout"):
            print(f"{rec['store']} {rec['part']}: ok shown {rec['ok_shown']:.3f} (n {rec['n_shown']})"
                  f" held {rec['ok_held']:.3f} (n {rec['n_held']}) diff {rec['diff']:+.3f}"
                  f" [{rec['diff_lo']:+.3f}, {rec['diff_hi']:+.3f}]")


def write(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


if __name__ == "__main__":
    main()
