#!/usr/bin/env python3
"""Write docs/readme_draft.md from committed results files.

    python scripts/readme_draft.py [--results results] [--out docs/readme_draft.md]

The draft is what README.md could become; README.md itself is edited by hand.
Every number in the draft is read here from a file under results/ and the
file is named next to it, so re-running this after the Mac commits a new run
rewrites the tables instead of leaving copied numbers behind. The prose is
fixed text; where it states a direction (lower, level, above) it is checked
against the numbers it sits on and the script stops if one no longer holds.

Nothing here needs the model, and no agent run on a confirmation fold is
read: the agent section is exploration only until preregistration 3 lands.
"""
import argparse
import csv
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.stats import mcnemar, paired_bootstrap  # noqa: E402

DATASETS = ("hotpot", "musique", "2wiki")


class Results:
    def __init__(self, root):
        self.root = Path(root)
        self.used = []

    def rows(self, name):
        path = self.root / name
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if name not in self.used:
            self.used.append(name)
        return rows

    def row(self, name, **match):
        hits = [r for r in self.rows(name) if all(r.get(k) == v for k, v in match.items())]
        if len(hits) != 1:
            raise SystemExit(f"{name}: expected one row for {match}, found {len(hits)}")
        return hits[0]

    def exists(self, name):
        return (self.root / name).exists()


def f3(x):
    return f"{float(x):.3f}"


def sd(x):
    """Signed, three places."""
    return f"{float(x):+.3f}"


def ci(lo, hi):
    return f"[{float(lo):+.3f}, {float(hi):+.3f}]"


def pv(x):
    if x in ("", None):
        return "-"
    x = float(x)
    if x == 0.0:
        return "< 1e-4"
    return f"{x:.2g}" if x < 0.001 else f"{x:.3f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def check(cond, what):
    if not cond:
        raise SystemExit(f"the draft's text no longer holds: {what}")


RUN_LABEL = {
    "bm25": "bm25, single hop, k=10",
    "bm25_k13": "bm25, single hop, k=13",
    "bm25_k7": "bm25, single hop, k=7",
    "hybrid": "hybrid, single hop",
    "hybrid_ircot": "hybrid ircot, k=10",
    "hybrid_ircot_rerank": "hybrid ircot, rerank 20->5",
}


def label(run, ds):
    return RUN_LABEL.get(run.replace(f"_{ds}_confirm", ""), run)


# ------------------------------------------------------------------ sections

def confirm_tables(R):
    parts = []
    for ds in DATASETS:
        name = f"{ds}_confirm_decomposition.csv"
        rows = []
        for r in R.rows(name):
            rows.append([label(r["run"], ds), r["n"], f3(r["em"]), f3(r["f1"]), f3(r["retrieval_correct"]),
                         f"{f3(r['illusion'])} [{f3(r['illusion_lo'])}, {f3(r['illusion_hi'])}]",
                         f3(r["illusion_share_of_correct"]), f3(r["closed_book_em"])])
        parts.append(f"{ds}, confirmation fold (`results/{name}`):\n\n" + table(
            ["config", "n", "EM", "F1", "retrieval correct", "cell [95% Wilson]", "cell / right answers",
             "closed book EM"], rows))
    return "\n\n".join(parts)


def prereg_table(R):
    name = "confirm_prereg.csv"
    rows = []
    for h in ("H1", "H4", "H2", "H3a", "H3b"):
        r = R.row(name, hypothesis=h)
        rows.append([h, r["metric"], f"{r['a'].replace('_hotpot_confirm', '')} -> {r['b'].replace('_hotpot_confirm', '')}",
                     f"{f3(r['value_a'])} -> {f3(r['value_b'])}", f"{sd(r['diff'])} {ci(r['ci_lo'], r['ci_hi'])}",
                     f"{int(float(r['only_a']))} / {int(float(r['only_b']))}", pv(r["holm_p"]), r["verdict"]])
    h1 = R.row(name, hypothesis="H1")
    check(float(h1["ci_hi"]) < 0, "H1's interval is below zero")
    return table(["test", "metric", "a -> b", "value", "diff [95% CI]", "only a / only b", "Holm p", "verdict"],
                 rows) + f"\n\n(`results/{name}`; the rules are in `docs/preregistration.md`.)"


CAUSES = ("parametric", "yesno", "shortcut", "paragraph", "partial", "none")


def decomposition_table(R):
    rows = []
    for ds in DATASETS:
        name = f"{ds}_confirm_decomposition.csv"
        r = R.row(name, run=f"bm25_{ds}_confirm")
        cell = round(float(r["illusion"]) * int(r["n"]))
        counts = [int(r[c]) for c in CAUSES]
        check(sum(counts) == cell, f"{name}: the causes add up to the cell")
        check(int(r["unexplained"]) == sum(int(r[c]) for c in ("paragraph", "partial", "none")),
              f"{name}: unexplained is paragraph + partial + none")
        rows.append([ds, r["n"], cell] + counts + [r["unexplained"], r["alt_shortcut"], f"`{name}`"])
    shortcut_largest = all(int(R.row(f"{ds}_confirm_decomposition.csv", run=f"bm25_{ds}_confirm")["shortcut"])
                           >= int(R.row(f"{ds}_confirm_decomposition.csv", run=f"bm25_{ds}_confirm")[c])
                           for ds in ("hotpot", "musique") for c in CAUSES)
    check(shortcut_largest, "shortcut is the largest cause on hotpot and musique")
    check(all(int(R.row(f"{ds}_confirm_decomposition.csv", run=f"bm25_{ds}_confirm")["alt_shortcut"])
              > int(R.row(f"{ds}_confirm_decomposition.csv", run=f"bm25_{ds}_confirm")[f"alt_{c}"])
              for ds in DATASETS for c in CAUSES if c != "shortcut"),
          "shortcut is the largest cause under the evidence-first order on all three")
    return table(["dataset", "n", "cell"] + list(CAUSES) + ["unexplained", "shortcut, evidence-first order", "file"],
                 rows)


def interventions_table(R):
    rows = []
    for ds in DATASETS:
        name = f"interventions_{ds}_confirm_summary.csv"
        for r in R.rows(name):
            rows.append([ds, r["arm"], r["n"], f"{f3(r['em'])} [{f3(r['em_lo'])}, {f3(r['em_hi'])}]",
                         f3(r["closed_book_em_same_ids"]), f3(r["n_ctx_mean"])])
    paired = []
    for ds in DATASETS:
        name = f"interventions_{ds}_confirm_paired.csv"
        for a, b in (("gold", "answer_only"), ("bridge_only", "answer_only"), ("closed_book", "bridge_only")):
            r = R.row(name, a=a, b=b)
            paired.append([ds, f"{a} -> {b}", r["n"], f"{f3(r['em_a'])} -> {f3(r['em_b'])}",
                           f"{sd(r['diff'])} {ci(r['ci_lo'], r['ci_hi'])}", f"{r['only_a']} / {r['only_b']}",
                           pv(r["holm_p"])])
    return (table(["dataset", "context given", "n", "EM [95% Wilson]", "closed book, same ids", "sentences"], rows)
            + "\n\n(`results/interventions_{hotpot,musique,2wiki}_confirm_summary.csv`)\n\n"
            + table(["dataset", "a -> b", "n", "EM", "diff [95% CI]", "only a / only b", "Holm p"], paired)
            + "\n\n(`results/interventions_{hotpot,musique,2wiki}_confirm_paired.csv`)")


def stop_table(R):
    rows = []
    for name, run in (("hotpot_stop_audit_summary.csv", "hybrid_ircot_hotpot_confirm"),
                      ("musique_stop_audit_summary.csv", "hybrid_ircot_musique_confirm")):
        for r in R.rows(name):
            if r["run"] != run:
                continue
            rows.append([run.replace("_confirm", ""), r["stop_reason"], r["n"], f3(r["share"]),
                         f"{f3(r['em'])} [{f3(r['em_lo'])}, {f3(r['em_hi'])}]", f3(r["retrieval_correct"]),
                         f3(r["premature"]), f3(r["never_complete"]), f3(r["mean_hops"])])
    return table(["run, confirm", "stop", "n", "share", "EM [95% Wilson]", "retrieval correct",
                  "stopped with gold missing", "gold never complete", "hops"], rows) + \
        "\n\n(`results/hotpot_stop_audit_summary.csv`, `results/musique_stop_audit_summary.csv`)"


def stop_note(R):
    out = []
    for name, run in (("hotpot_stop_audit_summary.csv", "hybrid_ircot_hotpot_confirm"),
                      ("musique_stop_audit_summary.csv", "hybrid_ircot_musique_confirm")):
        a = R.row(name, run=run, stop_reason="all")
        out.append(f"{f3(a['premature'])} of the {run.split('_')[2]} questions (EM {f3(a['em_premature'])} on them, "
                   f"against {f3(a['em_complete'])} where the gold was complete)")
        check(float(a["em_premature"]) < float(a["em_complete"]), f"{name}: premature stops answer worse")
        by = {r["stop_reason"]: float(r["retrieval_correct"]) for r in R.rows(name) if r["run"] == run}
        check(by["max_hops"] == max(by[k] for k in ("max_hops", "done", "repeated_query")),
              f"{name}: the capped runs have the highest retrieval correct")
    return textwrap.fill("On the confirmation folds a premature stop ends " + " and ".join(out) +
                         ". The cap is not where the evidence is lost: a run that hits it has the highest "
                         "retrieval correct of the three on both datasets.", 76)


def qa_column(R, name, col):
    return {r["id"]: float(r[col]) for r in R.rows(name)}


def paired_runs(R, a, b, col):
    xa, xb = qa_column(R, a, col), qa_column(R, b, col)
    ids = [i for i in xa if i in xb]
    va, vb = [xa[i] for i in ids], [xb[i] for i in ids]
    d, lo, hi = paired_bootstrap(va, vb)
    oa, ob, p = mcnemar([int(v) for v in va], [int(v) for v in vb])
    return len(ids), sum(va) / len(ids), sum(vb) / len(ids), d, lo, hi, oa, ob, p


def gates_table(R):
    rows = []
    # 1. routing: the best routed row of each dataset's arms table
    for ds, name in (("hotpot", "hotpot_arms_routed.csv"), ("musique", "musique_mac_arms_routed.csv"),
                     ("2wiki", "2wiki_arms_routed.csv"),
                     ("musique, fit on explore, scored on confirm", "arms_foldsplit/musique_mac_arms_routed.csv")):
        rs = R.rows(name)
        best = max(rs, key=lambda r: float(r["gain"]))
        if best.get("gain_lo"):
            spread = f"{ci(best['gain_lo'], best['gain_hi'])} (paired bootstrap over questions)"
            check(float(best["gain_lo"]) <= 0, f"{name}: no routed row's interval clears zero")
        else:
            spread = (f"+- {f3(best['gain_sd'])} sd over {best['n_splits']} half-splits, no test in this table"
                      if int(best["n_splits"]) > 1 else "(one split, no test in this table)")
        rows.append([f"1 routing, {ds}", "which retriever per question, replayed arms",
                     f"best single arm {f3(best['best_single'])}, oracle {f3(best['oracle'])}",
                     f"largest of {len(rs)} routed rows ({best['stage']}): {sd(best['gain'])} {spread}",
                     f"`results/{name}`"])
    fs = R.rows("arms_foldsplit/musique_mac_arms_routed.csv")
    check(max(float(r["gain"]) for r in fs) < 0.01, "musique's fold split: the largest routed row is under a point")
    # 2. deepening: retrieving twice as deep moves the evidence, not the answers
    a, b = "hybrid_ircot_hotpot_confirm_qa.csv", "hybrid_ircot_k20_hotpot_confirm_qa.csv"
    n, ma, mb, d, lo, hi, oa, ob, p = paired_runs(R, a, b, "retrieval_correct")
    n2, ea, eb, d2, lo2, hi2, oa2, ob2, p2 = paired_runs(R, a, b, "em")
    check(lo > 0 and lo2 <= 0 <= hi2, "k=20 moves retrieval_correct and not EM on hotpot confirm")
    rows.append(["2 deepening, hotpot", "the answer side of deepening: ircot k=20 against k=10, confirm",
                 f"all gold {f3(ma)} -> {f3(mb)}, {sd(d)} {ci(lo, hi)}",
                 f"EM {f3(ea)} -> {f3(eb)}, {sd(d2)} {ci(lo2, hi2)}, {oa2} / {ob2}, p {pv(p2)}",
                 f"`{b}` against `{a}` (paired here)"])
    # 3. reader cuts
    name = "reader_ctx_hotpot_confirm_paired.csv"
    for arm in ("gold", "rerank3", "rerank5", "learned3"):
        r = R.row(name, subset="all", arm=arm)
        rows.append(["3 reader cut, hotpot", f"k=20 context cut to {arm}, confirm (n {r['n']})",
                     f"full context EM {f3(r['em_full'])}",
                     f"{sd(r['diff'])} {ci(r['ci_lo'], r['ci_hi'])}, Holm p {pv(r['holm_p'])}", f"`{name}`"])
    check(all(float(R.row(name, subset="all", arm=a)["diff"]) < 0 for a in ("rerank3", "rerank5", "learned3")),
          "no non-oracle cut is above the full context")
    # 4-5. the agent's experience memory, versions 1 and 2
    for gate, name in (("4 agent memory v1", "orchestrator_mem_hotpot_explore_paired.csv"),
                       ("5 agent memory v2", "orchestrator_mem2_hotpot_explore_paired.csv")):
        for r in R.rows(name):
            rows.append([gate, f"{r['arm']} memory on the main agent, explore (n {r['n']})",
                         f"no memory EM {f3(r['em_none'])}",
                         f"{sd(r['d_em'])} {ci(r['d_em_lo'], r['d_em_hi'])}, {r['only_none']} / {r['only_arm']}, "
                         f"p {pv(r['p'])}", f"`{name}`"])
        check(all(float(r["d_em_lo"]) <= 0 for r in R.rows(name)), f"{name}: no arm's interval is above zero")
    name = "agent_memory/build_summary2.csv"
    for part in ("heldout_subagent", "heldout_planner"):
        r = R.row(name, store="hotpot_train_region2", part=part)
        rows.append(["5 agent memory v2", f"region build, {part.split('_')[1]} steps shown a lesson vs held out",
                     f"success {f3(r['ok_shown'])} (n {r['n_shown']}) vs {f3(r['ok_held'])} (n {r['n_held']})",
                     f"{sd(r['diff'])} {ci(r['diff_lo'], r['diff_hi'])}", f"`results/{name}`"])
        check(float(r["diff_lo"]) <= 0, f"{name} {part}: showing a lesson is not above holding it out")
    return table(["gate", "what was tested", "reference", "difference [95% CI]", "file"], rows)


def agent_tables(R):
    rows = []
    r = R.row("agent_dev_hotpot_explore_paired.csv", base="hybrid_ircot_hotpot_dev", arm="orchestrator_hotpot_explore")
    rows.append(["4 Nov main agent vs ircot", "explore", r["n"], f"{f3(r['em_base'])} -> {f3(r['em_arm'])}",
                 f"{sd(r['d_em'])} {ci(r['d_em_lo'], r['d_em_hi'])}", f"{r['lost']} / {r['won']}", pv(r["p"]),
                 f"{float(r['gen_calls_base']):.1f} -> {float(r['gen_calls_arm']):.1f}",
                 "`agent_dev_hotpot_explore_paired.csv`"])
    name = "agent_dev4_both_hotpot_pooled_paired.csv"
    for rr in R.rows(name):
        if rr["base"].startswith("orchestrator_guard_wide_unsup_hotpot"):
            rows.append(["round 4 agent vs round 3 agent", "explore + train_hard", rr["n"],
                         f"{f3(rr['em_base'])} -> {f3(rr['em_arm'])}", f"{sd(rr['d_em'])} {ci(rr['d_em_lo'], rr['d_em_hi'])}",
                         f"{rr['lost']} / {rr['won']}", pv(rr["p"]),
                         f"{float(rr['gen_calls_base']):.1f} -> {float(rr['gen_calls_arm']):.1f}", f"`{name}`"])
    name = "agent_dev4_vs_ircot_hotpot_pooled_paired.csv"
    for rr in R.rows(name):
        which = "ircot G+titles+B6 (the confirmation's ircot)" if "titles" in rr["base"] else "ircot G+B6 (same answer step)"
        rows.append([f"round 4 agent vs {which}", "explore + train_hard", rr["n"],
                     f"{f3(rr['em_base'])} -> {f3(rr['em_arm'])}", f"{sd(rr['d_em'])} {ci(rr['d_em_lo'], rr['d_em_hi'])}",
                     f"{rr['lost']} / {rr['won']}", pv(rr["p"]),
                     f"{float(rr['gen_calls_base']):.1f} -> {float(rr['gen_calls_arm']):.1f}", f"`{name}`"])
    paired = table(["comparison", "folds", "n", "EM", "diff [95% CI]", "lost / won", "p", "calls", "file"], rows)
    lrows = []
    for fold in ("explore", "train_hard"):
        name = f"agent_dev4_both_hotpot_{fold}_losses.csv"
        r = R.row(name, run=f"orchestrator_guard_wide_unsup_keepans_thin_hotpot_{fold}")
        parts = [int(r[c]) for c in ("in_context", "dropped", "not_retrieved", "yesno_gold")]
        check(sum(parts) == int(r["wrong"]), f"{name}: the places add up to the wrong answers")
        lrows.append([fold, r["n"], f3(r["em"]), r["wrong"], *parts, f"{float(r['gen_calls']):.1f}",
                      f"`{name}`"])
    losses = table(["fold", "n", "EM", "wrong", "string read", "retrieved, not kept", "never retrieved",
                    "yes/no gold", "calls", "file"], lrows)
    return paired, losses


def plots(R):
    want = ["hybrid_ircot_hotpot_confirm_2x2.png", "bm25_hotpot_confirm_2x2.png",
            "hotpot_confirm_decomposition.png", "hybrid_ircot_hotpot_confirm_per_hop.png",
            "hybrid_ircot_musique_confirm_per_hop.png", "pareto.png"]
    have = [w for w in want if R.exists(w)]
    check(len(have) == len(want), "the six plots the draft points at are committed")
    return "\n".join(f"- `results/{w}`" for w in have)


# ------------------------------------------------------------------ the page

TEMPLATE = """# honest-multihop-rag

<!-- Written by scripts/readme_draft.py from committed results files; do not
edit the numbers by hand, re-run the script. README.md is edited separately. -->

How often a multi-hop RAG pipeline gets the answer right without the gold
supporting evidence in front of it, and what that cell is made of. Scope:
retrieval is over each question's own distractor pool (HotpotQA distractor,
MuSiQue, 2WikiMultihopQA), not an open corpus; one generator,
Qwen2.5-7B-Instruct at 4 bits through MLX; every run on one Apple M5 Pro.
Every difference below is paired on identical question ids (exact McNemar and
a paired bootstrap, Holm across the primary tests). The confirmation folds
(n=1000 per dataset) were read once, under criteria written first
(`docs/preregistration.md`, `docs/preregistration_2.md`). Nothing here is a
new method.

## The cell, on the confirmation folds

"Retrieval correct": every annotated gold sentence was in the context the
reader saw. "Cell": the share of all questions answered right (EM) with
retrieval not correct.

{confirm}

The preregistered tests on hotpot:

{prereg}

Better retrieval roughly halves the cell on hotpot while EM moves by a few
points; EM alone would hide most of the difference.

## What the cell is made of

Each right answer with incomplete evidence is given the first cause that
holds, in this order: the closed-book run on the same question is right too
(parametric); a yes/no question; the gold sentence that holds the answer was
retrieved and some other gold sentence, usually the bridge, was not
(shortcut); every gold paragraph was retrieved but not every gold sentence
(paragraph); some gold was retrieved (partial); none was (none). The last
three together are the unexplained residual (`src/eval/analyze.py`,
`bucket`). Counts, bm25 k=10, confirmation folds:

{decomposition}

Most of the cell is an answer read off the one gold sentence that holds it,
with the hop that should lead there missing from the context; on 2wiki the
closed-book knowledge is about as large a part. The evidence-first order
(shortcut before parametric, `bucket_alt`) is in the same files and keeps
shortcut the largest cause on all three.

## The ablation: what the reader is handed

The same questions answered from controlled contexts: only the gold
sentences, only the gold sentence holding the answer, only the bridge
sentences (gold without the answer), and only distractors.

{interventions}

The answer-bearing sentence alone carries most of what the gold evidence is
worth; the bridge sentences alone are at closed-book level on hotpot and musique and
below it on 2wiki, where the closed-book run knows more of the answers. That is why
completing the evidence moves the cell far more than it moves EM.

## When ircot stops

The stop audit sorts the ircot runs by why the loop ended: the model said it
was done, it repeated a query, or it hit the cap. A stop is premature when
the model stopped itself (done or a repeated query) with the gold not
complete.

{stop}

{stop_note}

## Experience memory: five gates, all closed

The second half of the project asked whether a memory of past retrieval
outcomes can tell when a lesson applies. Each gate had a control fixed before
its run; none was passed, and the confirmation fold was spent only where the
plan said so (D1-D3, C0-C3).

{gates}

The routing rows are the most favourable row of each table, picked after
the fact. On musique every neighbour vote is a point or so above the best
single arm over the half-splits, a spread over resplits of the same
questions and not a test; fit on the exploration fold and scored once on
the confirmation fold the largest row is under a point, and it is not
claimed. Gate 2's policy (deepening 30% of hops, fit on the exploration log and
applied once to the hotpot confirmation log, preregistration 2 D3) is
reported in `docs/plan.md` (Stage 5b) and has no committed table of its
own; the row above is the answer-level side, read from the two runs. The
memory runs on the agent (gates 4 and 5) are exploration only: the
confirmation run was not made for a version that is level on exploration.

## The agent (exploration only)

A main agent that plans lookups and hands each to a subagent that searches,
keeps sentences and answers; the final read is over the kept sentences. It
has been changed one measured step at a time against a matched control
(`docs/plan.md`, Stage 6b). Every number in this section was chosen on the
folds it is measured on (hotpot explore, positions 0-499 of dev, and
train_hard); none is a result until the confirmation run in preregistration 3.

{agent_paired}

Where the round 4 agent's wrong answers are, by where the gold string was
(an exact per-sentence string test, so "read" is a lower bound):

{agent_losses}

## Plots

{plots}

## Reproduce

On the Mac (Apple Silicon, Python 3.12, Metal):

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    scripts/run_all.sh --smoke       # a few questions per stage, into a scratch copy of results/
    scripts/run_all.sh               # everything; the [model] stages take days

Without the model (any machine; reads data/processed and committed results):

    pip install -r requirements-linux-cpu.txt
    scripts/run_all.sh --no-model
    python scripts/readme_draft.py   # this page

2WikiMultihopQA needs `data_ids_april7.zip` (link in the Alab-NII readme)
unzipped into `data/raw/2wiki/`. Pins and what is still to fill from the Mac
are in `docs/pins.md`; `results/manifest.csv` says which command, config and
fold wrote each results file. `notebooks/demo.ipynb` reads the committed
results only.

## Limitations

- Distractor pools, not open retrieval: every pool holds the gold
  paragraphs, so retrieval here is easier than over a corpus, and the cell
  would be measured differently there.
- One generator (a 7B model at 4 bits) on one machine. Nothing says the
  decomposition, the ablation or the agent's margins carry to other models.
- "Retrieval correct" is against the annotated gold sentences, mapped onto
  chunks heuristically (the prepare step reports the facts that failed to
  map; every precision number inherits that error). A right answer read
  from an unannotated but sufficient sentence counts in the cell, and
  nothing here measures how often that happens, so the cell is an upper
  bound on unsupported answers.
- EM against a single gold string undercounts right answers written another
  way. The preregistered H6 reads the cell's share of right answers under
  F1 at 0.5 and under containment as well, within 0.04 of EM's on every
  hotpot and 2wiki run (`results/confirm_prereg.csv`); the agent tables
  report F1 and containment beside EM.
- The memory gates are negative results for the memories as built, on one
  model; they do not show that no memory could help.
- The agent numbers are chosen on the folds they are reported on, and the
  agent makes several times ircot's model calls (the calls column above);
  its confirmation run has not been made.
- MuSiQue and 2wiki have no agent runs; 2wiki has no ircot run on its
  confirmation fold.

Author: Yige Li

<!-- files read: {used} -->
"""


def prose_checks(R):
    """The directions the fixed text states, against the files it sits on."""
    h = {r["run"]: r for r in R.rows("hotpot_confirm_decomposition.csv")}
    check(float(h["hybrid_ircot_hotpot_confirm"]["illusion"]) < 0.6 * float(h["bm25_hotpot_confirm"]["illusion"]),
          "ircot roughly halves the hotpot cell")
    w = R.row("2wiki_confirm_decomposition.csv", run="bm25_2wiki_confirm")
    check(abs(int(w["parametric"]) - int(w["shortcut"])) <= 0.2 * int(w["shortcut"]),
          "on 2wiki parametric is about as large as shortcut")
    for ds in ("hotpot", "musique"):
        r = R.row(f"interventions_{ds}_confirm_paired.csv", a="closed_book", b="bridge_only")
        check(float(r["ci_lo"]) <= 0 <= float(r["ci_hi"]), f"{ds}: bridge only is level with closed book")
    r = R.row("interventions_2wiki_confirm_paired.csv", a="closed_book", b="bridge_only")
    check(float(r["ci_hi"]) < 0, "2wiki: bridge only is below closed book")
    for ds in DATASETS:
        r = R.row(f"interventions_{ds}_confirm_paired.csv", a="bridge_only", b="answer_only")
        check(float(r["ci_lo"]) > 0, f"{ds}: the answer sentence alone is far above the bridge alone")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "readme_draft.md"))
    args = ap.parse_args()
    R = Results(args.results)
    prose_checks(R)
    agent_paired, agent_losses = agent_tables(R)
    page = TEMPLATE.format(confirm=confirm_tables(R), prereg=prereg_table(R),
                           decomposition=decomposition_table(R), interventions=interventions_table(R),
                           stop=stop_table(R), stop_note=stop_note(R), gates=gates_table(R), agent_paired=agent_paired,
                           agent_losses=agent_losses, plots=plots(R), used="")
    page = page.replace("<!-- files read:  -->", "<!-- files read: " + ", ".join(R.used) + " -->")
    Path(args.out).write_text(page)
    print(f"wrote {args.out} from {len(R.used)} results files")


if __name__ == "__main__":
    main()
