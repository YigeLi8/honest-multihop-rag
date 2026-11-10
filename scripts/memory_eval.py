"""Frozen experience memory on hotpot explore against the main agent without
memory (docs/plan.md, Stage 6 A5).

    PYTHONPATH=. .venv312/bin/python scripts/memory_eval.py

Reads the no-memory run (orchestrator_hotpot_explore) and the four frozen
memory runs (orchestrator_mem_<mode>_hotpot_explore, mode in region, global,
shuffled, flat), all on the same 500 explore questions, plus the two stores
built on train and the two build runs. Writes

    results/orchestrator_mem_hotpot_explore_paired.csv
        one row per memory arm against no memory: em / f1 differences with
        paired bootstrap intervals, exact McNemar with Holm over the four
        arms, the same by question type, notebook gold metrics, calls and
        latency, and how often lessons were shown
    results/orchestrator_mem_hotpot_explore_steps.csv
        per arm and error type: how often the error occurred, at steps where
        a lesson of that type was shown and where none was
    results/agent_memory/build_summary.csv
        per store: operators, advice sources, active lessons by role and
        type, how many passed probation; and per quarter of the build stream
        (questions in random order, so the quarters are exchangeable): em,
        show rate and error rates

Shown-versus-not splits are descriptive: where a lesson is shown is not
random, so they say where the lessons landed, not what they caused.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from src.agent.memory import TYPES, score_steps
from src.eval.stats import holm, mcnemar, paired_bootstrap

R = Path("results")
DATA = Path("data/processed")
BASE = "orchestrator_hotpot_explore"
ARMS = ("region", "global", "shuffled", "flat")
STORES = {"region": ("orchestrator_mem_build_hotpot_train", "hotpot_train_region"),
          "flat": ("orchestrator_mem_build_flat_hotpot_train", "hotpot_train_flat")}
MIN_N = 5


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
    """score_steps on a trace record: [(role, type, failed, shown_ids_of_that_type?)]."""
    hops = [SimpleNamespace(retrieved=[SimpleNamespace(chunk=SimpleNamespace(chunk_id=c))
                                       for c in h["retrieved"]]) for h in trace["hops"]]
    res = SimpleNamespace(hops=hops, steps=trace["steps"], notebook=trace["notebook"])
    ex = SimpleNamespace(gold_chunk_ids=gold, chunks=[], question=trace["steps"][0]["question"])
    shown = {}
    for s in trace["steps"]:
        if s.get("role") == "planner":
            shown["planner"] = s.get("lesson_ids", [])
        elif "features" in s:
            shown[s["hop"]] = s.get("lesson_ids", [])
    return [(role, key, failed, shown.get(key, []))
            for role, key, _, failed, _ in score_steps(ex, res)]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def paired(base, arm, ids, field):
    a = [float(base[i][field]) for i in ids]
    b = [float(arm[i][field]) for i in ids]
    d, lo, hi = paired_bootstrap(a, b)
    return d, lo, hi, a, b


def lesson_types(store):
    d = json.loads((R / "agent_memory" / f"{store}.json").read_text())
    return {l["id"]: l for l in d["lessons"]}, d


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
    base = rows_of(BASE)
    ids = list(base)
    gold = gold_of("dev", ids)
    by_type = defaultdict(list)
    for i in ids:
        by_type[base[i]["qtype"]].append(i)

    out, steps_out, pvals = [], [], []
    store_lessons = {m: lesson_types(STORES["flat" if m == "flat" else "region"][1])[0]
                     for m in ARMS}
    for m in ARMS:
        run = f"orchestrator_mem_{m}_hotpot_explore"
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
        for tag, qs in (("shownq", shown_q), ("quietq", quiet_q)):
            row[f"n_{tag}"] = len(qs)
            row[f"d_em_{tag}"] = (mean(float(arm[i]["em"]) for i in qs)
                                  - mean(float(base[i]["em"]) for i in qs)) if qs else float("nan")
        # lessons shown that had passed probation at the end of the build
        lessons = store_lessons[m]
        shown_ids = [x for t in tr.values() for s in t["steps"] for x in s.get("lesson_ids") or []]
        row["shown_past_probation"] = mean(lessons[x]["shown"] >= MIN_N for x in shown_ids) \
            if shown_ids else float("nan")
        out.append(row)
        pvals.append(p)

        # error rates per type, at steps with a lesson of that type shown and without
        cnt = defaultdict(lambda: [0, 0])
        for i in ids:
            for role, key, failed, sids in step_errors(tr[i], gold[i]):
                for typ, bad in failed.items():
                    has = any(lessons.get(x, {}).get("type") == typ for x in sids)
                    for tag in ("all", "shown" if has else "not_shown"):
                        cnt[(role, typ, tag)][0] += bad
                        cnt[(role, typ, tag)][1] += 1
        if m == ARMS[0]:
            none_tr = traces_of(BASE)
            ncnt = defaultdict(lambda: [0, 0])
            for i in ids:
                for role, key, failed, _ in step_errors(none_tr[i], gold[i]):
                    for typ, bad in failed.items():
                        ncnt[(role, typ)][0] += bad
                        ncnt[(role, typ)][1] += 1
            for (role, typ), (k, n) in sorted(ncnt.items()):
                steps_out.append({"arm": "none", "role": role, "type": typ, "where": "all",
                                  "steps": n, "error_rate": k / n})
        for (role, typ, tag), (k, n) in sorted(cnt.items()):
            steps_out.append({"arm": m, "role": role, "type": typ, "where": tag,
                              "steps": n, "error_rate": k / n})

    for row, ph in zip(out, holm(pvals)):
        row["p_holm"] = ph
    write(R / "orchestrator_mem_hotpot_explore_paired.csv", out)
    write(R / "orchestrator_mem_hotpot_explore_steps.csv", steps_out)

    build = []
    for mode, (run, store) in STORES.items():
        lessons, d = lesson_types(store)
        ops = Counter(json.loads(line)["op"]
                      for line in open(R / "agent_memory" / f"{store}.log.jsonl"))
        base_rate = {k: (v[0] + 1) / (v[1] + 2) for k, v in d["base"].items()}
        rec = {"store": store, "part": "store", "lessons": len(lessons),
               "active": sum(not l["retired"] for l in lessons.values()),
               "reflection": sum(l["source"] == "reflection" for l in lessons.values()),
               "fallback_advice": sum(l["source"] == "fallback" for l in lessons.values()),
               **{f"op_{k}": v for k, v in sorted(ops.items())}}
        for role, types in TYPES.items():
            for typ in types:
                act = [l for l in lessons.values()
                       if not l["retired"] and l["role"] == role and l["type"] == typ]
                br = base_rate[f"{role}|{typ}"]
                rec[f"active_{role}_{typ}"] = len(act)
                rec[f"proven_{role}_{typ}"] = sum(
                    l["shown"] >= MIN_N and (l["ok_shown"] + 1) / (l["shown"] + 2) > br
                    for l in act)
                rec[f"probation_{role}_{typ}"] = sum(l["shown"] < MIN_N for l in act)
                rec[f"base_ok_{role}_{typ}"] = br
        build.append(rec)

        rows, tr = rows_of(run), traces_of(run)
        order = list(rows)             # file order = stream order
        g = gold_of("train", order)
        q = len(order) // 4
        for k in range(4):
            part = order[k * q:(k + 1) * q]
            errs = defaultdict(lambda: [0, 0])
            for i in part:
                for role, key, failed, _ in step_errors(tr[i], g[i]):
                    for typ, bad in failed.items():
                        errs[typ][0] += bad
                        errs[typ][1] += 1
            st = show_stats({i: tr[i] for i in part})
            build.append({"store": store, "part": f"q{k + 1}", "n": len(part),
                          "em": mean(float(rows[i]["em"]) for i in part),
                          "q_any_shown": st["q_any_shown"], "sub_shown": st["sub_shown"],
                          "planner_shown": st["planner_shown"],
                          **{f"err_{t}": v[0] / v[1] for t, v in sorted(errs.items())}})
    write(R / "agent_memory" / "build_summary.csv", build)

    for row in out:
        print(f"{row['arm']:9s} em {row['em']:.3f} vs {row['em_none']:.3f}  "
              f"d {row['d_em']:+.3f} [{row['d_em_lo']:+.3f}, {row['d_em_hi']:+.3f}]  "
              f"{row['only_none']}/{row['only_arm']} p {row['p']:.3f} holm {row['p_holm']:.3f}  "
              f"bridge {row['d_em_bridge']:+.3f} comparison {row['d_em_comparison']:+.3f}  "
              f"shown q {row['q_any_shown']:.2f} sub {row['sub_shown']:.2f} "
              f"planner {row['planner_shown']:.2f}  past probation {row['shown_past_probation']:.2f}")


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
