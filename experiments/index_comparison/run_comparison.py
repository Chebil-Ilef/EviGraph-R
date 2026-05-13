"""
Orchestrator for the SQuAI vs EviGraph-R retrieval comparison.

Loads questions from questions.jsonl (single-paper only), retrieves from both
systems, and computes:

  Hit@k, MRR@k, NDCG@k, MAP@k, Precision@k, Avg rank of gold  (both systems)
  Section Hit@k  (EviGraph-R only — SQuAI has no section granularity)

  Latency: p50 and p95 per system (wall-clock, in ms)

Outputs:
  results/raw_results.json        — per-question detailed results for both systems
  results/comparison_table.md     — paper-ready markdown tables
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

K_VALUES = [1, 5, 10]


def _hit(gold_ids: list[str], results: list, k: int) -> bool:
    top_pids = {r.paper_id for r in results[:k]}
    return any(g in top_pids for g in gold_ids)


def _rank_of(gold_ids: list[str], results: list) -> Optional[int]:
    for r in results:
        if r.paper_id in gold_ids:
            return r.rank
    return None


def _mrr(gold_ids: list[str], results: list, k: int) -> float:
    for r in results[:k]:
        if r.paper_id in gold_ids:
            return 1.0 / r.rank
    return 0.0


def _ndcg(gold_ids: list[str], same_cluster_ids: set[str], results: list, k: int) -> float:
    def rel(pid: str) -> float:
        if pid in gold_ids:
            return 1.0
        if pid in same_cluster_ids:
            return 0.5
        return 0.0

    dcg = sum(rel(r.paper_id) / math.log2(r.rank + 1) for r in results[:k])
    # Ideal DCG: computed from the known relevance universe, not from what the system returned.
    # Gold papers first (rel=1.0), then same-cluster papers (rel=0.5), up to k positions.
    n_gold = len(gold_ids)
    n_cluster = len(same_cluster_ids)
    ideal_rels = [1.0] * min(n_gold, k) + [0.5] * min(n_cluster, max(0, k - n_gold))
    ideal_rels = ideal_rels[:k]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def _map(gold_ids: list[str], results: list, k: int) -> float:
    gold_set = set(gold_ids)
    # Number of relevant docs that can possibly appear in top-k
    n_relevant = min(len(gold_ids), k)
    hits = 0
    precision_sum = 0.0
    for i, r in enumerate(results[:k], start=1):
        if r.paper_id in gold_set:
            hits += 1
            precision_sum += hits / i
    return precision_sum / n_relevant if n_relevant > 0 else 0.0


def _precision(gold_ids: list[str], results: list, k: int) -> float:
    gold_set = set(gold_ids)
    hits = sum(1 for r in results[:k] if r.paper_id in gold_set)
    return hits / k if k > 0 else 0.0


def _section_hit(gold_ids: list[str], gold_section: Optional[str], results: list, k: int) -> Optional[float]:
    """1.0 if any top-k result matches gold paper AND gold section, else 0.0. None if no gold_section."""
    if not gold_section or not results:
        return None
    gold_set = set(gold_ids)
    gold_norm = gold_section.lower()
    for r in results[:k]:
        if r.paper_id not in gold_set:
            continue
        st = (getattr(r, "section_title", None) or "").lower()
        if st and (gold_norm in st or st in gold_norm):
            return 1.0
    return 0.0


def evaluate_single(
    question: dict,
    squai_results: list,
    evi_results: list,
    cluster_paper_ids: set[str],
) -> dict:
    gold_ids = question["gold_paper_ids"]
    gold_set = set(gold_ids)
    same_cluster = cluster_paper_ids - gold_set

    record: dict = {
        "question_id": question["question_id"],
        "type": "single",
        "source": question["source"],
        "domain": question["domain"],
        "cluster": question["cluster"],
        "squai": {},
        "evigraph": {},
    }

    for system, results in [("squai", squai_results), ("evigraph", evi_results)]:
        m: dict = {}
        for k in K_VALUES:
            m[f"hit@{k}"] = int(_hit(gold_ids, results, k))
            m[f"mrr@{k}"] = _mrr(gold_ids, results, k)
            m[f"ndcg@{k}"] = _ndcg(gold_ids, same_cluster, results, k)
            m[f"map@{k}"] = _map(gold_ids, results, k)
            m[f"precision@{k}"] = _precision(gold_ids, results, k)
        rank = _rank_of(gold_ids, results)
        m["rank_of_gold"] = rank
        # Section Hit — only meaningful for EviGraph-R (SQuAI has no section granularity)
        if system == "evigraph":
            gold_section = question.get("gold_section")
            for k in K_VALUES:
                m[f"section_hit@{k}"] = _section_hit(gold_ids, gold_section, results, k)
        record[system] = m

    return record


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _agg_single(records: list[dict], system: str) -> dict:
    agg: dict = {}
    for k in K_VALUES:
        agg[f"Hit@{k}"]       = _mean([r[system][f"hit@{k}"] for r in records])
        agg[f"MRR@{k}"]       = _mean([r[system][f"mrr@{k}"] for r in records])
        agg[f"NDCG@{k}"]      = _mean([r[system][f"ndcg@{k}"] for r in records])
        agg[f"MAP@{k}"]       = _mean([r[system][f"map@{k}"] for r in records])
        agg[f"Precision@{k}"] = _mean([r[system][f"precision@{k}"] for r in records])
    ranks = [r[system]["rank_of_gold"] for r in records if r[system]["rank_of_gold"] is not None]
    agg["Avg Rank of Gold"] = _mean(ranks) if ranks else None
    if system == "evigraph":
        for k in K_VALUES:
            vals = [r[system].get(f"section_hit@{k}") for r in records
                    if r[system].get(f"section_hit@{k}") is not None]
            agg[f"Section Hit@{k}"] = _mean(vals) if vals else None
    return agg


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def _build_report(
    single_records: list[dict],
    squai_latencies: list[float],
    evi_latencies: list[float],
) -> str:
    lines = ["# SQuAI vs EviGraph-R — Retrieval Comparison\n"]

    squai_s = _agg_single(single_records, "squai")
    evi_s   = _agg_single(single_records, "evigraph")

    primary_metrics = [
        (f"Hit@{k}", f"Hit@{k}") for k in K_VALUES
    ] + [
        (f"MRR@{k}", f"MRR@{k}") for k in K_VALUES
    ] + [
        (f"NDCG@{k}", f"NDCG@{k}") for k in K_VALUES
    ] + [
        (f"MAP@{k}", f"MAP@{k}") for k in K_VALUES
    ] + [
        (f"Precision@{k}", f"Precision@{k}") for k in K_VALUES
    ]

    lines.append(f"*N = {len(single_records)} questions*\n")
    lines.append("| Metric | SQuAI (E5+BM25) | EviGraph-R (BGE-M3) |")
    lines.append("|--------|-----------------|---------------------|")
    for label, key in primary_metrics:
        lines.append(f"| {label:<18} | {_fmt(squai_s.get(key)):>15} | {_fmt(evi_s.get(key)):>19} |")
    # Avg rank
    lines.append(f"| {'Avg Rank of Gold':<18} | {_fmt(squai_s.get('Avg Rank of Gold')):>15} | {_fmt(evi_s.get('Avg Rank of Gold')):>19} |")
    # Section Hit (EviGraph-R only)
    for k in K_VALUES:
        v = evi_s.get(f"Section Hit@{k}")
        lines.append(f"| {'Section Hit@'+str(k):<18} | {'—':>15} | {_fmt(v):>19} |")

    # Breakdown by domain
    lines.append("\n### Breakdown by Domain (Single-paper)\n")
    for domain in ["CS", "Physics", "Math"]:
        domain_recs = [r for r in single_records if r["domain"] == domain]
        if not domain_recs:
            continue
        sq = _agg_single(domain_recs, "squai")
        ev = _agg_single(domain_recs, "evigraph")
        lines.append(f"**{domain}** (N={len(domain_recs)})\n")
        lines.append("| Metric | SQuAI | EviGraph-R |")
        lines.append("|--------|-------|------------|")
        for k in K_VALUES:
            lines.append(f"| Hit@{k}  | {_fmt(sq.get(f'Hit@{k}'))} | {_fmt(ev.get(f'Hit@{k}'))} |")
            lines.append(f"| MRR@{k}  | {_fmt(sq.get(f'MRR@{k}'))} | {_fmt(ev.get(f'MRR@{k}'))} |")
        lines.append("")

    # Breakdown by source
    lines.append("\n### Breakdown by Question Source\n")
    lines.append("| Source | N | SQuAI Hit@5 | EviGraph Hit@5 | SQuAI MRR@10 | EviGraph MRR@10 |")
    lines.append("|--------|---|-------------|----------------|--------------|-----------------|")
    for source in ["section", "fullpaper", "abstract"]:
        src_recs = [r for r in single_records if r["source"] == source]
        if not src_recs:
            continue
        sq = _agg_single(src_recs, "squai")
        ev = _agg_single(src_recs, "evigraph")
        lines.append(
            f"| {source:<10} | {len(src_recs)} "
            f"| {_fmt(sq.get('Hit@5'))} | {_fmt(ev.get('Hit@5'))} "
            f"| {_fmt(sq.get('MRR@10'))} | {_fmt(ev.get('MRR@10'))} |"
        )

    # Latency
    lines.append("\n## Latency\n")
    lines.append("| Metric | SQuAI (E5+BM25) | EviGraph-R (BGE-M3) |")
    lines.append("|--------|-----------------|---------------------|")
    for label, sq_vals, ev_vals in [
        ("p50 latency (ms)", squai_latencies, evi_latencies),
        ("p95 latency (ms)", squai_latencies, evi_latencies),
    ]:
        if sq_vals and ev_vals:
            sq_v = float(np.percentile(sq_vals, 50 if "p50" in label else 95)) * 1000
            ev_v = float(np.percentile(ev_vals, 50 if "p50" in label else 95)) * 1000
            lines.append(f"| {label:<20} | {sq_v:>13.1f}ms | {ev_v:>17.1f}ms |")
        else:
            lines.append(f"| {label:<20} | {'—':>15} | {'—':>19} |")

    lines.append("\n---")
    lines.append("*Section Hit reported for EviGraph-R only — SQuAI indexes abstracts with no section granularity.*")

    return "\n".join(lines) + "\n"


def run_comparison(
    questions_path: str,
    output_dir: str,
    top_k_values: list[int],
    squai_alpha: float,
) -> None:
    questions = []
    with open(questions_path) as f:
        for line in f:
            questions.append(json.loads(line))

    logger.info("Loaded %d questions", len(questions))

    # Build cluster→paper_id lookup for NDCG same-cluster partial credit
    cluster_papers: dict[str, set[str]] = defaultdict(set)
    for q in questions:
        for pid in q["gold_paper_ids"]:
            cluster_papers[q["cluster"]].add(pid)

    # Lazy-load retrievers
    from experiments.index_comparison.retrieve_squai import SQuAIRetriever
    from experiments.index_comparison.retrieve_evigraph import EviGraphRetriever

    squai = SQuAIRetriever(alpha=squai_alpha, top_k=max(top_k_values))
    evigraph = EviGraphRetriever(top_k=max(top_k_values))

    single_records: list[dict] = []
    squai_latencies: list[float] = []
    evi_latencies: list[float] = []

    for i, q in enumerate(questions):
        logger.info(
            "[%d/%d] %s  source=%-10s  cluster=%s",
            i + 1, len(questions), q["question_id"], q["source"], q["cluster"],
        )

        # SQuAI retrieval
        t0 = time.perf_counter()
        squai_results = squai.retrieve(q["query"], top_k=max(top_k_values))
        squai_latencies.append(time.perf_counter() - t0)

        # EviGraph-R retrieval
        t0 = time.perf_counter()
        evi_results = evigraph.retrieve(q["query"], top_k=max(top_k_values))
        evi_latencies.append(time.perf_counter() - t0)

        same_cluster = cluster_papers[q["cluster"]]
        record = evaluate_single(q, squai_results, evi_results, same_cluster)
        single_records.append(record)

    # Save raw results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_results.json"
    with open(raw_path, "w") as f:
        json.dump({
            "single": single_records,
            "squai_latencies_s": squai_latencies,
            "evi_latencies_s": evi_latencies,
        }, f, indent=2)
    logger.info("Saved raw results → %s", raw_path)

    # Build and save markdown report
    report = _build_report(single_records, squai_latencies, evi_latencies)
    report_path = out / "comparison_table.md"
    report_path.write_text(report)
    logger.info("Saved comparison table → %s", report_path)

    # Print summary to console
    sq = _agg_single(single_records, "squai")
    ev = _agg_single(single_records, "evigraph")
    logger.info("=== Summary ===")
    for k in K_VALUES:
        logger.info(
            "  Hit@%-2d  SQuAI=%.3f  EviGraph=%.3f",
            k, sq.get(f"Hit@{k}", 0), ev.get(f"Hit@{k}", 0),
        )
    if squai_latencies:
        logger.info(
            "  Latency p50  SQuAI=%.0fms  EviGraph=%.0fms",
            np.percentile(squai_latencies, 50) * 1000,
            np.percentile(evi_latencies, 50) * 1000,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SQuAI vs EviGraph-R comparison")
    parser.add_argument(
        "--questions",
        default="experiments/index_comparison/results/questions.jsonl",
    )
    parser.add_argument(
        "--output",
        default="experiments/index_comparison/results/",
    )
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--squai-alpha", type=float, default=0.5,
                        help="Alpha weight for SQuAI dense score in fusion (default: 0.5)")
    args = parser.parse_args()

    run_comparison(
        questions_path=args.questions,
        output_dir=args.output,
        top_k_values=args.top_k,
        squai_alpha=args.squai_alpha,
    )


if __name__ == "__main__":
    main()
