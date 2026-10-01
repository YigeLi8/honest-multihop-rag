"""Experience records: what one retrieval step did and how it turned out.

One record per (question, hop). The strategy, the hits and the features are
what was known when the step ran; hop_outcome, shadow and outcome are filled
from the dataset's gold labels and the scored row afterwards. Features never
see gold (src/memory/features.py); everything derived from a label lives under
hop_outcome, shadow or outcome and nowhere else.

Every arm is scored by the same function (arm_outcome), the primary strategy
and the shadow arms alike, so that union, best-arm and regret can be computed
from the records without opening the data file again. Three things to know
when reading them:

* Shadow arms are first-stage retrievers at retrieval.top_k. When a reranker
  cut the primary list to top_n, a shadow arm is scored on its first top_n hits
  (the "depth" in its block), so arms are compared at the same context size.
  Its full top_k list is kept in "retrieved", and the primary's list before the
  reranker in "first_stage".
* gold_ranks comes from a second, full-pool call to the same retriever, so a
  missed gold chunk still has a rank and a score. Ties are ordered by the
  retriever and bm25s orders them differently at different depths, so a rank
  inside a tie is not exact; full_prefix_same says whether the full-pool list
  starts with the top-k list. For hybrid the fusion is normalised over three
  times the requested depth, so its full-pool ranking is a different ranking:
  log bm25 and dense as shadow arms to get full-depth ranks under a hybrid run.
* At hop >= 1 every arm is run on the query the primary strategy's trajectory
  produced, and new_gold / cumulative_recall / aligned_hit of a shadow arm are
  relative to the primary's context before the hop. Question-level regret is
  therefore exact for single_hop runs and for hop 0 only.

The logger only reads. It does not touch the hits, the example or the result,
so a run with logging on retrieves and answers exactly what it would with
logging off (scripts/check_baseline.py checks that on real runs). The pipeline
keeps the time spent here out of latency_s, but the timing columns of a logged
run are still not systems numbers: take those from a run with memory off.
"""
import copy
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Mapping, Optional, Sequence

from src.config import as_dict
from src.eval.per_hop_precision import ordered_gold, precision_recall
from src.eval.run_qa import paragraph_recall
from src.memory import features as feature_module
from src.memory.features import (FEATURE_SCHEMA_VERSION, FEATURE_STAGE, FEATURES,
                                 PROBE_FEATURES, compute_features, probe_features)
from src.retrieval.base import get_retriever
from src.types import Example, PipelineResult, RetrievedChunk

REPO = Path(__file__).resolve().parents[2]

# Layout of a record (the keys below and what they mean), separate from the
# feature schema: bump when a key is added, removed or changes meaning.
RECORD_SCHEMA_VERSION = 1

# question-level outcome copied from the scored row onto every hop record
OUTCOME_FIELDS: tuple[str, ...] = ("em", "f1", "retrieval_correct", "gold_recall",
                                   "para_recall", "stop_reason", "hops")

# what git_dirty looks at: code and configs, tracked or not; results do not count
CODE_PATHS: tuple[str, ...] = ("src", "configs", "scripts", "tests", "data")


@dataclass
class ExperienceRecord:
    """One retrieval step of one question."""
    run: str
    dataset: str
    id: str
    qtype: str
    hop: int
    query: str
    strategy: dict           # method, top_k, hybrid_alpha, rerank_*, pipeline_mode
    retrieved: list          # [{chunk_id, score, rank}], the hits the pipeline used
    feature_schema_version: int
    features: dict           # src.memory.features, model-free and gold-free
    hop_outcome: dict        # this hop against gold: arm_outcome plus n_gold, n_gold_para
    first_stage: Optional[list] = None   # the primary list before the reranker; None without one
    probe_features: dict = field(default_factory=dict)   # {arm: {topk_jaccard, top1_same, depth}}
    shadow: dict = field(default_factory=dict)    # {arm: {retrieved, depth, **arm_outcome}}
    outcome: dict = field(default_factory=dict)   # OUTCOME_FIELDS and failure_type, once scored


def strategy_of(cfg: Any) -> dict:
    """The retrieval strategy a config stands for. hybrid_alpha and
    rerank_top_n are None when they did not act on the hits."""
    retrieval = getattr(cfg, "retrieval", None)
    rerank = getattr(cfg, "rerank", None)
    method = getattr(retrieval, "method", "")
    rerank_on = bool(getattr(rerank, "enabled", False))
    return {
        "method": method,
        "top_k": getattr(retrieval, "top_k", None),
        "hybrid_alpha": getattr(getattr(retrieval, "hybrid", None), "alpha", None)
        if method == "hybrid" else None,
        "rerank_enabled": rerank_on,
        "rerank_top_n": getattr(rerank, "top_n", None) if rerank_on else None,
        "pipeline_mode": getattr(getattr(cfg, "pipeline", None), "mode", ""),
    }


def _arm_configs(cfg: Any) -> dict:
    """{arm: config} for memory.log_arms: the same config with retrieval.method
    swapped and the reranker off (shadow arms are first stage only)."""
    out = {}
    for arm in getattr(getattr(cfg, "memory", None), "log_arms", None) or []:
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg.retrieval.method = arm
        if getattr(arm_cfg, "rerank", None) is not None:
            arm_cfg.rerank.enabled = False
        out[arm] = arm_cfg
    return out


def shadow_retrievers(cfg: Any) -> dict:
    """{arm: retriever} for memory.log_arms. They run beside the primary
    retriever on the same pool and query, first stage only (no reranker), and
    their hits go to the log and nowhere else."""
    return {arm: get_retriever(arm_cfg) for arm, arm_cfg in _arm_configs(cfg).items()}


def shadow_strategies(cfg: Any) -> dict:
    """{arm: strategy} for the header: a shadow arm named like the primary
    method is still a different strategy when the primary is reranked."""
    return {arm: strategy_of(arm_cfg) for arm, arm_cfg in _arm_configs(cfg).items()}


def git_state() -> tuple[str, bool]:
    """(HEAD sha, code differs from it). Dirty means a modified or untracked
    file under CODE_PATHS. ("", False) outside a git checkout. Run without
    optional locks so that it never writes to the repository."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--", *CODE_PATHS))
    except (OSError, subprocess.CalledProcessError):
        return "", False


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ranked(hits: Sequence[RetrievedChunk]) -> list[dict]:
    return [{"chunk_id": h.chunk.chunk_id, "score": float(h.score), "rank": int(h.rank)}
            for h in hits]


def arm_outcome(ids: Sequence[str], gold: Sequence[str], order: Sequence[str], hop: int,
                seen_before: Collection[str],
                full: Optional[Sequence[RetrievedChunk]] = None,
                first_stage_ids: Optional[Sequence[str]] = None) -> dict:
    """One arm's ranked chunk ids at one hop against the gold labels. The same
    function scores the primary strategy and every shadow arm.

    ids              the arm's chunk ids, ranked, at the depth being compared
    gold             the question's gold chunk ids
    order            gold chunk ids in reasoning order ([] when the dataset has none)
    seen_before      chunk ids in the primary strategy's context before this hop
    full             the arm's ranking of the whole pool, for gold_ranks
    first_stage_ids  the arm's top-k list as its retriever returned it, to
                     check the full ranking against (defaults to ids)
    """
    gold_set, got = set(gold), set(ids)
    precision, recall = precision_recall(ids, gold)
    context = set(seen_before) | got
    out = {
        "hop_precision": round(precision, 4),
        "hop_recall": round(recall, 4),
        "hop_para_recall": round(paragraph_recall(got, gold_set)[2], 4),
        "cumulative_recall": round(len(context & gold_set) / len(gold_set), 4)
        if gold_set else 0.0,
        "all_gold_in_topk": bool(gold_set) and gold_set <= got,
        "gold_hit_ids": [i for i in ids if i in gold_set],
        # gold this hop added to what the primary strategy had already seen
        "new_gold": [g for g in dict.fromkeys(gold) if g in got and g not in seen_before],
        # was the hop-th gold chunk in context by this hop; None without an ordered path
        "aligned_hit": (order[hop] in context) if order and hop < len(order) else None,
        "first_gold_rank": next((r for r, i in enumerate(ids, 1) if i in gold_set), None),
    }
    if full is not None:
        at = {h.chunk.chunk_id: [r, float(h.score)] for r, h in enumerate(full, 1)}
        reference = list(ids if first_stage_ids is None else first_stage_ids)
        # {gold id: [rank, score] in the full-pool ranking}, None if it is not ranked at all
        out["gold_ranks"] = {g: at.get(g) for g in dict.fromkeys(gold)}
        out["full_prefix_same"] = [h.chunk.chunk_id for h in full[:len(reference)]] == reference
    return out


class ExperienceLogger:
    """Buffers one ExperienceRecord per (question id, hop) and writes them,
    after a header line, to <memory.store>/<run>_experience.jsonl on close()."""

    def __init__(self, cfg: Any, run_name: str, source: str = "live") -> None:
        self.cfg = cfg
        self.run = run_name
        self.source = source            # live | backfill
        mem = getattr(cfg, "memory", None)
        self.path = Path(getattr(mem, "store", "results/memory")) / f"{run_name}_experience.jsonl"
        self.dataset = getattr(getattr(cfg, "dataset", None), "name", "")
        self.strategy = strategy_of(cfg)
        self.shadow_arms = shadow_strategies(cfg)
        # provenance is taken here, before the run: the code and config that
        # produce the records, not whatever the checkout holds when it ends
        self.config = as_dict(cfg)
        self.git_sha, self.git_dirty = git_state()
        self.features_sha256 = sha256_of(Path(feature_module.__file__))
        self.data: dict = {}
        self.extra_header: dict = {}
        self._records: dict[str, list[ExperienceRecord]] = {}   # question id -> hops, in order
        self._state: dict[str, tuple[set, Optional[dict]]] = {}  # id -> (seen ids, prev features)

    def describe_data(self, data_file: Any, n_questions: int,
                      ids_file: Optional[str] = None, n: Optional[int] = None) -> None:
        """Which questions the log is about, for the header: the data file and
        its hash, how they were selected (a fold file or the first n) and how
        many there are."""
        path = Path(data_file)
        self.data = {
            "file": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
            "ids_file": str(ids_file) if ids_file else None,
            "n": n,
            "n_questions": n_questions,
        }

    def on_hop(self, ex: Example, hop: int, query: str, hits: Sequence[RetrievedChunk],
               shadow_hits: Optional[Mapping[str, Sequence[RetrievedChunk]]] = None,
               first_stage: Optional[Sequence[RetrievedChunk]] = None,
               primary_full: Optional[Sequence[RetrievedChunk]] = None,
               shadow_full: Optional[Mapping[str, Sequence[RetrievedChunk]]] = None) -> None:
        """Record one retrieval step. Called after the reranker.

        hits          the hits the pipeline is about to use
        shadow_hits   {arm: top-k hits} of the shadow retrievers on the same query
        first_stage   the primary retriever's hits before the reranker, when one acted
        primary_full  the primary retriever's ranking of the whole pool
        shadow_full   {arm: ranking of the whole pool}
        """
        if hop == 0 or ex.id not in self._state:
            self._state[ex.id] = (set(), None)
            self._records[ex.id] = []
        seen, prev = self._state[ex.id]
        shadow_hits = shadow_hits or {}
        shadow_full = shadow_full or {}

        features = compute_features(query, hits, ex.chunks, hop, ex.question,
                                    prev=prev, seen_ids=seen)
        before_rerank = hits if first_stage is None else first_stage

        ids = [h.chunk.chunk_id for h in hits]
        gold = list(ex.gold_chunk_ids)
        order = ordered_gold(ex)
        n_gold_para = paragraph_recall((), set(gold))[0]
        hop_outcome = arm_outcome(ids, gold, order, hop, seen, full=primary_full,
                                  first_stage_ids=[h.chunk.chunk_id for h in before_rerank])
        hop_outcome.update(n_gold=len(set(gold)), n_gold_para=n_gold_para)

        shadow = {}
        for arm, arm_hits in shadow_hits.items():
            arm_ids = [h.chunk.chunk_id for h in arm_hits]
            depth = min(len(ids), len(arm_ids))     # the context size the primary handed on
            shadow[arm] = {"retrieved": arm_ids, "depth": depth,
                           **arm_outcome(arm_ids[:depth], gold, order, hop, seen,
                                         full=shadow_full.get(arm), first_stage_ids=arm_ids)}

        record = ExperienceRecord(
            run=self.run, dataset=self.dataset, id=ex.id, qtype=ex.qtype, hop=hop,
            query=query, strategy=dict(self.strategy), retrieved=ranked(hits),
            feature_schema_version=FEATURE_SCHEMA_VERSION, features=features,
            hop_outcome=hop_outcome,
            first_stage=None if first_stage is None else ranked(first_stage),
            probe_features=probe_features(before_rerank, shadow_hits), shadow=shadow)
        hops = self._records[ex.id]
        del hops[hop:]                  # a hop logged twice keeps the later one
        hops.append(record)
        self._state[ex.id] = (seen | set(ids), features)

    def on_outcome(self, ex: Example, res: PipelineResult, row: Mapping[str, Any]) -> None:
        """Attach the question-level outcome (from run_qa.score_row) to every
        hop record of the question. em and f1 are None when nothing was
        generated (a retrieval-only run): there is no answer to be wrong.
        failure_type is left empty here; the failure-memory stage fills it."""
        outcome = {k: row.get(k) for k in OUTCOME_FIELDS}
        if outcome["stop_reason"] is None:
            outcome["stop_reason"] = res.stop_reason
        if outcome["hops"] is None:
            outcome["hops"] = len(res.hops)
        if not res.raw_outputs:
            outcome["em"] = outcome["f1"] = None
        outcome["failure_type"] = ""
        for record in self._records.get(ex.id, []):
            record.outcome = dict(outcome)
        self._state.pop(ex.id, None)

    def discard(self, example_id: str) -> None:
        """Drop the buffered records of one question."""
        self._records.pop(example_id, None)
        self._state.pop(example_id, None)

    def records(self) -> list[ExperienceRecord]:
        return [r for hops in self._records.values() for r in hops]

    def header(self) -> dict:
        return {
            "header": True,
            "run": self.run,
            "source": self.source,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "config": self.config,
            "data": self.data,
            "strategy": self.strategy,
            "shadow_arms": self.shadow_arms,
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "features": list(FEATURES),
            "feature_stage": dict(FEATURE_STAGE),
            "probe_features": list(PROBE_FEATURES),
            "features_sha256": self.features_sha256,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": {"platform": platform.platform(), "machine": platform.machine(),
                         "python": platform.python_version()},
            **self.extra_header,
        }

    def close(self) -> Path:
        """Write the header and every buffered record; returns the path. The
        file is written beside its final name and moved into place, so a
        failure leaves neither a half-written log nor a truncated older one."""
        header = json.dumps(self.header(), default=str)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(tmp, "w") as f:
                f.write(header + "\n")
                for record in self.records():
                    f.write(json.dumps(asdict(record)) + "\n")
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)
        return self.path
