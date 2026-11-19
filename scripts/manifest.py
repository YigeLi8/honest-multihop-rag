#!/usr/bin/env python3
"""Which command, config and fold each committed results file comes from.

    python scripts/manifest.py --record rec/cpu.tsv [rec/memory.tsv ...]

writes results/manifest.csv, one row per (file, command that writes it):

    file       the path under results/, as committed
    stage      the run_all.sh stage (data, retrieval, qa, ... plots)
    kind       model (generates on the Mac) or cpu (reads committed files only)
    how        observed: the file was written when this command ran under
               `scripts/run_all.sh --record`; by name: a command not run here
               (the [model] stages, and on linux the 2wiki and memory steps),
               credited with the files its arguments name by the module's own
               naming (outputs_by_name below)
    command    the python command as run_all.sh runs it
    config     the --config it was given, if any
    ids        the --ids fold file, if any
    sha256     of the committed file

Committed files that no command in run_all.sh writes get one row with stage
empty and how = "not in run_all.sh", so the gaps are listed, not hidden.

Every command is read from `scripts/run_all.sh --dry-run` (which lists the
2wiki steps whether or not the data is there); what a command wrote is taken
from the --record files where it ran, by name otherwise. A model stage can be
recorded too, on the Mac.
"""
import argparse
import csv
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUN_SUFFIXES = ("_qa.csv", "_trace.jsonl", "_per_hop.json", "_2x2.png", "_per_hop.png")


def parse_lines(lines):
    """run_all.sh --dry-run / --record lines -> [(stage, kind, command, file|None)].
    Lines without three tabs (the stages' own messages) are skipped."""
    out = []
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 4:
            continue
        stage, kind, cmd, f = parts
        out.append((stage, kind, cmd, None if f == "-" else f))
    return out


def arg(cmd, flag):
    toks = shlex.split(cmd)
    for i, t in enumerate(toks[:-1]):
        if t == flag:
            return toks[i + 1]
    return ""


def module(cmd):
    toks = shlex.split(cmd)
    if "-m" in toks:
        return toks[toks.index("-m") + 1]
    return toks[1] if len(toks) > 1 else ""


def config_of(cmd):
    cfg = arg(cmd, "--config")
    if not cfg:
        return None
    from src.config import load_config
    return load_config(ROOT / cfg)


def run_name(cmd):
    """The run name a [model] command writes under: --name, else the config's run.name."""
    name = arg(cmd, "--name")
    if name:
        return name
    cfg = config_of(cmd)
    return cfg.run.name if cfg else ""


def outputs_by_name(cmd):
    """The results files a command writes, from its arguments and the names the
    module gives its outputs (read off each module's writes). [] for a module
    with no rule here."""
    mod = module(cmd)
    sub = shlex.split(cmd)[3] if len(shlex.split(cmd)) > 3 else ""
    d = arg(cmd, "--dataset")
    if mod == "src.eval.run_qa":
        name = run_name(cmd)
        out = [f"{name}{s}" for s in RUN_SUFFIXES]
        mem = getattr(config_of(cmd), "agent_memory", None)
        if mem is not None and getattr(mem, "learn", False):   # the memory gates' build runs
            store = mem.store.removeprefix("results/")
            out += [store, store.removesuffix(".json") + ".log.jsonl"]
        return out
    if mod in ("src.eval.interventions", "src.memory.reader_context"):
        name = arg(cmd, "--name")
        if sub == "run":
            return [f"{name}_qa.csv", f"{name}_trace.jsonl"]
        if sub == "summarize":
            return [f"{name}_summary.csv", f"{name}_paired.csv", f"{name}_by_bucket.csv"]
    if mod == "src.eval.analyze":
        return [f"{d}_decomposition.csv", f"{d}_paired.csv", f"{d}_decomposition.png"]
    if mod == "src.eval.criteria":
        return [f"{d}_criteria.csv"]
    if mod == "src.eval.stop_audit":
        return [f"{d}_stop_audit.csv", f"{d}_stop_audit_summary.csv"]
    if mod == "src.eval.prereg":
        return ["confirm_prereg.csv"]
    if mod == "src.eval.run_retrieval":
        return [f"{run_name(cmd)}_retrieval.csv"]
    if mod == "src.memory.arms":
        return [f"{d}_arms_{t}.csv" for t in ("discordance", "ceiling", "predictability", "routed")] \
            + [f"{d}_arms_rules.txt"]
    if mod == "src.memory.failures":
        return [f"{d}_failure_{t}.csv" for t in ("types", "cooccurrence", "orders", "audit")]
    if mod == "src.memory.lookalike" and sub == "natural":
        return [f"{d}_lookalike_pairs.csv", f"{d}_lookalike_summary.csv"]
    if mod == "src.memory.stream":
        return [f"{d}_stream.csv", f"{d}_stream_revisions.csv"]
    if mod == "src.memory.drift":
        a, b = arg(cmd, "--dataset-a"), arg(cmd, "--dataset-b")
        return [f"{x}_to_{y}_drift{t}.csv" for x, y in ((a, b), (b, a)) for t in ("", "_revisions")]
    return []


def by_name(cmd, committed):
    """Committed files a command not run here is credited with by name."""
    return [f"results/{f}" for f in outputs_by_name(cmd) if f"results/{f}" in committed]


def build(dry, recorded, committed):
    """dry: parsed --dry-run lines; recorded: parsed --record lines; committed:
    set of results/ paths. -> rows (dicts), sorted by file then stage order."""
    order = {}
    for stage, _, _, _ in dry:
        order.setdefault(stage, len(order))
    rows, seen = [], set()

    def add(f, stage, kind, how, cmd):
        key = (f, cmd)
        if f not in committed or key in seen:
            return
        seen.add(key)
        rows.append({"file": f, "stage": stage, "kind": kind, "how": how, "command": cmd,
                     "config": arg(cmd, "--config"), "ids": arg(cmd, "--ids")})

    observed_cmds = {cmd for _, _, cmd, _ in recorded}
    for stage, kind, cmd, f in recorded:
        if f:
            add(f, stage, kind, "observed", cmd)
    for stage, kind, cmd, _ in dry:
        if cmd not in observed_cmds:
            for f in by_name(cmd, committed):
                add(f, stage, kind, "by name", cmd)
    claimed = {r["file"] for r in rows}
    for f in sorted(committed - claimed):
        rows.append({"file": f, "stage": "", "kind": "", "how": "not in run_all.sh",
                     "command": "", "config": "", "ids": ""})
    rows.sort(key=lambda r: (r["file"], order.get(r["stage"], 99)))
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", nargs="*", default=[], help="files written by run_all.sh --record")
    ap.add_argument("--out", default="results/manifest.csv")
    args = ap.parse_args()

    dry = parse_lines(subprocess.run(["bash", "scripts/run_all.sh", "--dry-run"], cwd=ROOT,
                                     check=True, capture_output=True, text=True).stdout.splitlines())
    recorded = []
    for p in args.record:
        with open(p) as f:
            recorded += parse_lines(f)
    committed = set(subprocess.run(["git", "ls-files", "results"], cwd=ROOT, check=True,
                                   capture_output=True, text=True).stdout.split())
    committed.discard(args.out)
    rows = build(dry, recorded, committed)
    for r in rows:
        r["sha256"] = sha256(ROOT / r["file"])
    with open(ROOT / args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    files = {r["file"] for r in rows}
    gaps = {r["file"] for r in rows if r["how"] == "not in run_all.sh"}
    by = {}
    for r in rows:
        by[r["how"]] = by.get(r["how"], 0) + 1
    print(f"{args.out}: {len(files)} committed files, {len(rows)} rows "
          f"({', '.join(f'{k} {v}' for k, v in sorted(by.items()))}); "
          f"{len(gaps)} written by no command in run_all.sh")


if __name__ == "__main__":
    main()
