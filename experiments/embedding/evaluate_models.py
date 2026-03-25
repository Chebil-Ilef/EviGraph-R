"""
Evaluate all embedding models on synthetic QA dataset and a sample.

Usage
-----
# 1. Embed a chunks JSONL → upsert into Qdrant (uses src.utils.qdrant directly)
python benchmark/evaluate_models.py index \
  --chunks benchmark/data/chunks_combined.jsonl \
  --model e5-base-v2 --recreate

# 2. Evaluate one model (Qdrant must already be populated)
python benchmark/evaluate_models.py eval \
  --qa-file benchmark/data/synthetic_qa.jsonl \
  --model e5-base-v2 --top-k 1 5 10 \
  --output benchmark/results.json --report benchmark/reports/report.md

# 3. Index + evaluate all models back-to-back (always recreates between models)
python benchmark/evaluate_models.py eval-all \
  --chunks benchmark/data/chunks_combined.jsonl \
  --qa-file benchmark/data/synthetic_qa.jsonl \
  --models e5-base-v2 qwen3-0.6b jina-v3-nano bge-m3

"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    PATHS,
    QDRANT_ACTIVE,
)
from retrieval.embedder import Embedder
from retrieval.retriever import HybridQueryRetriever, ChunkResult
from indexing.indexing_pipeline import run_indexing

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("src.core.retriever").setLevel(logging.DEBUG)

BENCHMARK_MODEL_KEYS = [key for key in EMBEDDING_MODELS]

@dataclass
class QARecord:

    question_id: str
    query: str
    gold_paper_id: str
    gold_chunk_uid: str
    gold_section_title: str
    gold_answer_strings: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> QARecord:
        return cls(
            question_id=d["question_id"],
            query=d["query"],
            gold_paper_id=d["gold_paper_id"],
            gold_chunk_uid=d["gold_chunk_uid"],
            gold_section_title=d["gold_section_title"],
            gold_answer_strings=d["gold_answer_strings"],
        )


@dataclass
class EvalMetrics:

    question_id: str
    recall_at_k: dict[int, bool] = field(default_factory=dict)      # k → bool
    mrr_at_k: dict[int, float] = field(default_factory=dict)         # k → float
    ndcg_at_k: dict[int, float] = field(default_factory=dict)        # k → float
    paper_hit_at_k: dict[int, bool] = field(default_factory=dict)    # k → bool
    section_hit_at_k: dict[int, bool] = field(default_factory=dict)  # k → bool
    answer_contain_at_k: dict[int, bool] = field(default_factory=dict)


@dataclass
class ModelResults:

    model_key: str
    total_queries: int
    metrics_per_query: list[EvalMetrics] = field(default_factory=list)
    
    evaluation_time_seconds: float = 0.0  
    
    recall_mean: dict[int, float] = field(default_factory=dict)
    mrr_mean: dict[int, float] = field(default_factory=dict)
    ndcg_mean: dict[int, float] = field(default_factory=dict)
    paper_hit_mean: dict[int, float] = field(default_factory=dict)
    section_hit_mean: dict[int, float] = field(default_factory=dict)
    answer_contain_mean: dict[int, float] = field(default_factory=dict)
    
    def compute_aggregates(self, top_ks: list[int]) -> None:

        for k in top_ks:
            recalls = [m.recall_at_k.get(k, False) for m in self.metrics_per_query]
            mrrs = [m.mrr_at_k.get(k, 0.0) for m in self.metrics_per_query]
            ndcgs = [m.ndcg_at_k.get(k, 0.0) for m in self.metrics_per_query]
            paper_hits = [m.paper_hit_at_k.get(k, False) for m in self.metrics_per_query]
            section_hits = [m.section_hit_at_k.get(k, False) for m in self.metrics_per_query]
            answer_contains = [m.answer_contain_at_k.get(k, False) for m in self.metrics_per_query]
            
            self.recall_mean[k] = float(np.mean(recalls))
            self.mrr_mean[k] = float(np.mean(mrrs))
            self.ndcg_mean[k] = float(np.mean(ndcgs))
            self.paper_hit_mean[k] = float(np.mean(paper_hits))
            self.section_hit_mean[k] = float(np.mean(section_hits))
            self.answer_contain_mean[k] = float(np.mean(answer_contains))


# Metric Computation

def compute_recall(results: list[ChunkResult], top_k: int, gold_chunk_uid: str) -> bool:

    top_k_results = results[:top_k]
    return any(r.chunk_uid == gold_chunk_uid for r in top_k_results)


def compute_mrr(results: list[ChunkResult], top_k: int, gold_chunk_uid: str) -> float:

    top_k_results = results[:top_k]
    for rank, r in enumerate(top_k_results, 1):
        if r.chunk_uid == gold_chunk_uid:
            return 1.0 / rank
    return 0.0


def compute_ndcg(
    results: list[ChunkResult],
    top_k: int,
    gold_chunk_uid: str,
    gold_paper_id: str,
) -> float:

    top_k_results = results[:top_k]
    
    # Compute DCG
    dcg = 0.0
    for i, r in enumerate(top_k_results, 1):
        if r.chunk_uid == gold_chunk_uid:
            relevance = 1.0
        elif r.paper_id == gold_paper_id:
            relevance = 0.5
        else:
            relevance = 0.0
        dcg += relevance / np.log2(i + 1)
    
    # Compute IDCG (ideal ranking: gold chunk at position 1)
    idcg = 1.0 / np.log2(2)  # relevance=1 at position 1
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_paper_hit(
    results: list[ChunkResult],
    top_k: int,
    gold_paper_id: str,
) -> bool:

    top_k_results = results[:top_k]
    return any(r.paper_id == gold_paper_id for r in top_k_results)


def compute_section_hit(
    results: list[ChunkResult],
    top_k: int,
    gold_section_title: str,
) -> bool:

    top_k_results = results[:top_k]
    return any(r.section_title == gold_section_title for r in top_k_results)


def compute_answer_contain(
    results: list[ChunkResult],
    top_k: int,
    gold_answer_strings: list[str],
) -> bool:

    top_k_results = results[:top_k]
    combined_text = " ".join(r.embed_text for r in top_k_results).lower()
    
    for answer in gold_answer_strings:
        answer_lower = answer.lower()
        if answer_lower in combined_text:
            return True
    return False


# Evaluation Pipeline

def load_synthetic_qa(file_path: Path) -> list[QARecord]:

    records = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                records.append(QARecord.from_dict(json.loads(line)))
    logger.info("Loaded %d synthetic Q&A records from %s", len(records), file_path)
    return records


def _build_embedder(model_key: str):
    return Embedder.from_model_key(model_key)


def evaluate_model(
    model_key: str,
    qa_records: list[QARecord],
    top_ks: list[int] = None,
) -> ModelResults:
    """Evaluate one embedding model on all Q&A records."""
    if top_ks is None:
        top_ks = [1, 5, 10]
    model_cfg = EMBEDDING_MODELS[model_key]
    
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("Evaluating model: %s", model_key)
    logger.info("=" * 70)
    
    # Initialize embedder and retriever
    embedder = _build_embedder(model_key)
    retriever = HybridQueryRetriever(model_key=model_key)
    
    logger.info("Embedder initialized: %s | dim=%d | device=%s",
                model_key, model_cfg.dim, model_cfg.device)
    
    results = ModelResults(model_key=model_key, total_queries=len(qa_records))
    
    # Evaluate each query
    for qa in tqdm(qa_records, desc=f"Evaluating {model_key}", unit="query"):
        # Embed query
        embed_result = embedder.embed_query(qa.query)
        
        # Extract dense and sparse embeddings based on model type
        if model_cfg.bge_produces_sparse:
            # BGE-M3: extract both dense and sparse
            query_dense = embed_result.dense
            query_sparse = embed_result.sparse[0] if embed_result.sparse else None
        else:
            # Normal models: only dense
            query_dense = embed_result
            query_sparse = None
        
        # Retrieve with appropriate embeddings
        retrieved = retriever.retrieve(
            embeddings=query_dense,
            query_text=qa.query,
            top_k=max(top_ks),  # get max top_k so we can evaluate all k values
            sparse_embeddings=query_sparse,
        )
        
        # Compute metrics
        metrics = EvalMetrics(question_id=qa.question_id)
        for k in top_ks:
            metrics.recall_at_k[k] = compute_recall(retrieved, k, qa.gold_chunk_uid)
            metrics.mrr_at_k[k] = compute_mrr(retrieved, k, qa.gold_chunk_uid)
            metrics.ndcg_at_k[k] = compute_ndcg(retrieved, k, qa.gold_chunk_uid, qa.gold_paper_id)
            metrics.paper_hit_at_k[k] = compute_paper_hit(retrieved, k, qa.gold_paper_id)
            metrics.section_hit_at_k[k] = compute_section_hit(retrieved, k, qa.gold_section_title)
            metrics.answer_contain_at_k[k] = compute_answer_contain(retrieved, k, qa.gold_answer_strings)
        
        results.metrics_per_query.append(metrics)
    
    end_time = time.time()
    results.evaluation_time_seconds = end_time - start_time
    
    # Aggregate
    results.compute_aggregates(top_ks)
    
    return results


def evaluate_all_models(
    qa_records: list[QARecord],
    model_keys: Optional[list[str]] = None,
    top_ks: list[int] = None,
) -> dict[str, ModelResults]:

    if model_keys is None:
        model_keys = BENCHMARK_MODEL_KEYS
    if top_ks is None:
        top_ks = [1, 5, 10]
    
    all_results = {}
    for model_key in model_keys:
        try:
            all_results[model_key] = evaluate_model(
                model_key,
                qa_records,
                top_ks,
            )
        except Exception as e:
            logger.error("Failed to evaluate model %s: %s", model_key, e)
    
    return all_results


def format_results(all_results: dict[str, ModelResults], top_ks: list[int]) -> str:

    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("RETRIEVAL EVALUATION RESULTS")
    lines.append("=" * 100)
    
    for model_key, result in all_results.items():
        lines.append(f"\n{model_key}")
        lines.append("-" * 100)
        
        # Header
        header_parts = ["Metric"]
        for k in top_ks:
            header_parts.append(f"@{k}")
        lines.append(" | ".join(f"{h:>20}" for h in header_parts))
        lines.append("-" * 100)
        
        # Metrics rows
        metric_names = [
            ("Recall", result.recall_mean),
            ("MRR", result.mrr_mean),
            ("NDCG", result.ndcg_mean),
            ("PaperHit", result.paper_hit_mean),
            ("SectionHit", result.section_hit_mean),
            ("AnswerContain", result.answer_contain_mean),
        ]
        
        for name, values in metric_names:
            row_parts = [name]
            for k in top_ks:
                row_parts.append(f"{values.get(k, 0.0):.4f}")
            lines.append(" | ".join(f"{v:>20}" for v in row_parts))
    
    lines.append("=" * 100)
    return "\n".join(lines)


def save_results_json(
    all_results: dict[str, ModelResults],
    output_path: Path,
) -> None:

    output_data = {}
    
    for model_key, result in all_results.items():
        # Serialize model results
        model_dict = {
            "model_key": result.model_key,
            "total_queries": result.total_queries,
            "evaluation_time_seconds": result.evaluation_time_seconds,
            "aggregates": {
                "recall_mean": result.recall_mean,
                "mrr_mean": result.mrr_mean,
                "ndcg_mean": result.ndcg_mean,
                "paper_hit_mean": result.paper_hit_mean,
                "section_hit_mean": result.section_hit_mean,
                "answer_contain_mean": result.answer_contain_mean,
            },
            "per_query": [
                {
                    "question_id": m.question_id,
                    "recall_at_k": m.recall_at_k,
                    "mrr_at_k": m.mrr_at_k,
                    "ndcg_at_k": m.ndcg_at_k,
                    "paper_hit_at_k": m.paper_hit_at_k,
                    "section_hit_at_k": m.section_hit_at_k,
                    "answer_contain_at_k": m.answer_contain_at_k,
                }
                for m in result.metrics_per_query
            ]
        }
        output_data[model_key] = model_dict
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Saved detailed results to %s", output_path)


def generate_markdown_report(
    all_results: dict[str, ModelResults],
    top_ks: list[int],
    batches: list[str],
    indexing_time_seconds: float = 0.0,
    qa_records: Optional[list] = None,
) -> str:
    """Generate comprehensive markdown report with all metrics and timing."""
    lines = []
    
    # Header
    batches_str = " + ".join(batches)
    lines.append(f"# Retrieval Evaluation Report")
    lines.append(f"**Batches**: {batches_str}")
    lines.append(f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Queries**: {len(qa_records) if qa_records else 'N/A'}")
    lines.append("")
    
    # Overall timing
    lines.append("## Timing Summary")
    lines.append("")
    if indexing_time_seconds > 0:
        minutes, seconds = divmod(indexing_time_seconds, 60)
        lines.append(f"- **Indexing Time**: {minutes:.1f}m {seconds:.1f}s")
    
    total_eval_time = sum(r.evaluation_time_seconds for r in all_results.values())
    eval_minutes, eval_seconds = divmod(total_eval_time, 60)
    lines.append(f"- **Total Evaluation Time**: {eval_minutes:.1f}m {eval_seconds:.1f}s")
    lines.append("")
    
    # Summary table
    lines.append("## Summary Metrics by Model")
    lines.append("")
    
    # Table header
    header = ["Model", "Eval Time (s)", "Recall@1", "Recall@5", "Recall@10", "MRR@5", "NDCG@10", "PaperHit@5", "SectionHit@5"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    
    # Table rows
    for model_key, result in sorted(all_results.items()):
        row = [
            f"**{model_key}**",
            f"{result.evaluation_time_seconds:.2f}",
            f"{result.recall_mean.get(1, 0.0):.4f}",
            f"{result.recall_mean.get(5, 0.0):.4f}",
            f"{result.recall_mean.get(10, 0.0):.4f}",
            f"{result.mrr_mean.get(5, 0.0):.4f}",
            f"{result.ndcg_mean.get(10, 0.0):.4f}",
            f"{result.paper_hit_mean.get(5, 0.0):.4f}",
            f"{result.section_hit_mean.get(5, 0.0):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")
    
    # Detailed results per model
    lines.append("## Detailed Results by Model")
    lines.append("")
    
    for model_key, result in sorted(all_results.items()):
        lines.append(f"### {model_key}")
        lines.append("")
        
        # Model stats
        lines.append(f"**Evaluation Time**: {result.evaluation_time_seconds:.2f}s")
        lines.append(f"**Queries Evaluated**: {len(result.metrics_per_query)}")
        lines.append("")
        
        # Metrics summary
        lines.append("#### Aggregate Metrics")
        lines.append("")
        
        metrics_summary = [
            ("Recall@k", result.recall_mean),
            ("MRR@k", result.mrr_mean),
            ("NDCG@k", result.ndcg_mean),
            ("PaperHit@k", result.paper_hit_mean),
            ("SectionHit@k", result.section_hit_mean),
            ("AnswerContain@k", result.answer_contain_mean),
        ]
        
        for metric_name, metric_values in metrics_summary:
            values_str = ", ".join(
                f"{metric_name.split('@')[0]}@{k}: {metric_values.get(k, 0.0):.4f}"
                for k in sorted(metric_values.keys())
            )
            lines.append(f"- {values_str}")
        
        lines.append("")
        
        # Per-query breakdown
        lines.append("#### Per-Query Breakdown")
        lines.append("")
        
        query_header = ["Query ID", "Recall@1", "Recall@5", "MRR@5", "NDCG@5", "Paper Hit@5", "Answer@5"]
        lines.append("| " + " | ".join(query_header) + " |")
        lines.append("|" + "|".join(["---"] * len(query_header)) + "|")
        
        for metrics in result.metrics_per_query:
            row = [
                metrics.question_id,
                "✓" if metrics.recall_at_k.get(1, False) else "✗",
                "✓" if metrics.recall_at_k.get(5, False) else "✗",
                f"{metrics.mrr_at_k.get(5, 0.0):.4f}",
                f"{metrics.ndcg_at_k.get(5, 0.0):.4f}",
                "✓" if metrics.paper_hit_at_k.get(5, False) else "✗",
                "✓" if metrics.answer_contain_at_k.get(5, False) else "✗",
            ]
            lines.append("| " + " | ".join(row) + " |")
        
        lines.append("")
    
    return "\n".join(lines)


def save_markdown_report(
    report_content: str,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_content)
    logger.info("Saved markdown report to %s", output_path)



def _load_chunks_jsonl(path: Path) -> list[dict]:

    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    logger.info("Loaded %d chunks from %s", len(chunks), path)
    return chunks


def _cmd_index(args) -> None:

    from src.utils.qdrant import (
        setup_collection, build_points,
        qdrant_client, check_qdrant_alive,
        get_collection_info,
    )

    chunks   = _load_chunks_jsonl(args.chunks)
    client   = qdrant_client()
    check_qdrant_alive(client)
    model_key =args.model

    setup_collection(
        client,
        model_key=model_key,
        recreate=args.recreate,
    )

    embedder = _build_embedder(model_key)
    cfg      = EMBEDDING_MODELS[model_key]
    logger.info("Embedder ready: %s  dim=%d  device=%s",
                model_key, cfg.dim, cfg.device)

    batch_size = QDRANT_ACTIVE.upsert_batch_size
    total      = len(chunks)
    logger.info("Upserting %d chunks in batches of %d …", total, batch_size)

    for i in tqdm(range(0, total, batch_size), desc="Upserting", unit="batch"):
        batch        = chunks[i : i + batch_size]
        texts        = [c["embed_text"] for c in batch]
        embed_result = embedder.embed_passages(texts)
        points = build_points(
            batch,
            embed_result,
            profile=QDRANT_ACTIVE,
            model_key=model_key,
        )
        client.upsert(
            collection_name=QDRANT_ACTIVE.collection_name,
            points=points,
            wait=True,
        )

    stats = get_collection_info(client, QDRANT_ACTIVE.collection_name)
    logger.info(
        "Done — points_in_db=%s  status=%s",
        stats["points_count"], stats["status"],
    )


def _cmd_eval(args) -> None:

    qa_records = load_synthetic_qa(args.qa_file)
    model_key = args.model

    result = evaluate_model(args.model, qa_records, args.top_k)
    all_results = {model_key: result}

    print(format_results(all_results, args.top_k))

    output_path = args.output or (PATHS.root / "evaluation" / f"{model_key}_results.json")
    save_results_json(all_results, output_path)

    report_path = args.report or (
        PATHS.root / "evaluation" / f"{model_key}_report.md"
    )
    report = generate_markdown_report(
        all_results, args.top_k,
        batches=[model_key],
        qa_records=qa_records,
    )
    save_markdown_report(report, report_path)
    print(f"\n✓ Results  → {output_path}")
    print(f"✓ Report   → {report_path}")


def _cmd_eval_all(args) -> None:

    qa_records = load_synthetic_qa(args.qa_file)
    all_results: dict[str, ModelResults] = {}
    total_t0 = time.time()
    model_keys = args.models

    for model_key in model_keys:
        logger.info("=" * 70)
        logger.info("MODEL: %s", model_key)
        logger.info("=" * 70)

        # Re-index for each model (different dims require recreate)
        _cmd_index(argparse.Namespace(
            chunks  = args.chunks,
            model   = model_key,
            recreate= True,         # always recreate when cycling models
        ))

        result = evaluate_model(model_key, qa_records, args.top_k)
        all_results[model_key] = result
        logger.info(
            "Recall@%d=%.4f  MRR@%d=%.4f",
            args.top_k[-1], result.recall_mean.get(args.top_k[-1], 0),
            args.top_k[-1], result.mrr_mean.get(args.top_k[-1], 0),
        )

    logger.info("Total wall time: %.1fs", time.time() - total_t0)
    print(format_results(all_results, args.top_k))

    save_results_json(all_results, args.output)

    report = generate_markdown_report(
        all_results, args.top_k,
        batches=model_keys,
        qa_records=qa_records,
    )
    save_markdown_report(report, args.report)
    print(f"\n✓ Results → {args.output}")
    print(f"✓ Report  → {args.report}")


def _cmd_legacy(args) -> None:

    indexing_time = 0.0
    model_keys = args.models
    if not args.no_index:
        logger.info("=" * 70)
        logger.info("INDEXING PHASE: batches %s", args.batches)
        logger.info("=" * 70)
        index_model_key = args.index_model or model_keys[0]
        logger.info("Indexing with model: %s", index_model_key)
        indexing_start = time.time()
        run_indexing(args.batches, model_key=index_model_key, recreate=args.recreate)
        indexing_time = time.time() - indexing_start

    logger.info("=" * 70)
    logger.info("EVALUATION PHASE")
    logger.info("=" * 70)
    qa_records  = load_synthetic_qa(args.qa_file)
    all_results = evaluate_all_models(qa_records, model_keys, args.top_k)

    print(format_results(all_results, args.top_k))
    save_results_json(all_results, args.output)

    batches_str    = "_".join(args.batches)
    markdown_output = PATHS.root / "evaluation" / f"{batches_str}_batches.md"
    report = generate_markdown_report(
        all_results, args.top_k, args.batches,
        indexing_time_seconds=indexing_time,
        qa_records=qa_records,
    )
    save_markdown_report(report, markdown_output)
    print(f"\n✓ Markdown report saved to: {markdown_output}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    _SUBCMDS = {"index", "eval", "eval-all"}

    if len(sys.argv) > 1 and sys.argv[1] in _SUBCMDS:
        # ── subcommand interface ─────────────────────────────────────────────
        parser = argparse.ArgumentParser(
            description="Embedding model evaluator for unarXive chunks.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        sub = parser.add_subparsers(dest="cmd", required=True)

        def _add_qdrant(p):
            p.add_argument("--host",      default="localhost")
            p.add_argument("--port",      type=int, default=6333)
            p.add_argument("--grpc-port", type=int, default=6334)

        # ── index ────────────────────────────────────────────────────────────
        p_idx = sub.add_parser(
            "index",
            help="Embed a chunks JSONL file and upsert into Qdrant.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p_idx.add_argument(
            "--chunks", required=True, type=Path,
            help="Path to chunks JSONL (e.g. chunks_combined.jsonl).",
        )
        p_idx.add_argument(
            "--model", default=DEFAULT_EMBEDDING_MODEL,
            choices=BENCHMARK_MODEL_KEYS,
            help="Embedding model to use for indexing.",
        )
        p_idx.add_argument(
            "--recreate", action="store_true",
            help="Drop and recreate the Qdrant collection. "
                 "Required when switching to a model with a different vector dimension.",
        )
        _add_qdrant(p_idx)

        # ── eval ─────────────────────────────────────────────────────────────
        p_ev = sub.add_parser(
            "eval",
            help="Evaluate one embedding model on a QA JSONL file.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p_ev.add_argument(
            "--qa-file", required=True, type=Path,
            dest="qa_file",
            help="Path to synthetic_qa.jsonl.",
        )
        p_ev.add_argument(
            "--model", default=DEFAULT_EMBEDDING_MODEL,
            choices=BENCHMARK_MODEL_KEYS,
        )
        p_ev.add_argument(
            "--top-k", nargs="+", type=int, default=[1, 5, 10],
            dest="top_k",
        )
        p_ev.add_argument("--output", type=Path, default=None,
                          help="JSON results output path.")
        p_ev.add_argument("--report", type=Path, default=None,
                          help="Markdown report output path.")
        _add_qdrant(p_ev)

        # ── eval-all ──────────────────────────────────────────────────────────
        p_all = sub.add_parser(
            "eval-all",
            help="Index + evaluate all (or selected) models sequentially.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p_all.add_argument(
            "--chunks", required=True, type=Path,
            help="Path to chunks JSONL.",
        )
        p_all.add_argument(
            "--qa-file", required=True, type=Path,
            dest="qa_file",
            help="Path to synthetic_qa.jsonl.",
        )
        p_all.add_argument(
            "--models", nargs="+", default=BENCHMARK_MODEL_KEYS,
            choices=BENCHMARK_MODEL_KEYS,
        )
        p_all.add_argument(
            "--top-k", nargs="+", type=int, default=[1, 5, 10],
            dest="top_k",
        )
        p_all.add_argument(
            "--output", type=Path,
            default=PATHS.root / "evaluation" / "results.json",
        )
        p_all.add_argument(
            "--report", type=Path,
            default=PATHS.root / "evaluation" / "report.md",
        )
        _add_qdrant(p_all)

        args = parser.parse_args()

        if args.cmd == "index":
            _cmd_index(args)
        elif args.cmd == "eval":
            _cmd_eval(args)
        elif args.cmd == "eval-all":
            _cmd_eval_all(args)

    else:
        # ── legacy flat CLI (backward-compatible) ────────────────────────────
        parser = argparse.ArgumentParser(
            description="Evaluate embedding models on synthetic QA dataset.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--batches", nargs="+", default=["batch_01", "batch_02"],
            metavar="STEM",
            help='Batch stems to index before evaluation, e.g. batch_01 batch_02, or "all".',
        )
        parser.add_argument(
            "--no-index", action="store_true",
            dest="no_index",
            help="Skip indexing step (chunks must already exist in Qdrant).",
        )
        parser.add_argument(
            "--index-model", type=str, default=None,
            dest="index_model",
            help="Embedding model to use for indexing (default: first model in --models).",
        )
        parser.add_argument(
            "--recreate", action="store_true",
            help="Drop and recreate Qdrant collection.",
        )
        parser.add_argument(
            "--models", nargs="+", default=BENCHMARK_MODEL_KEYS,
        )
        parser.add_argument(
            "--top-k", type=int, nargs="+", default=[1, 5, 10],
            dest="top_k",
        )
        parser.add_argument(
            "--qa-file", type=Path,
            default=PATHS.root / "evaluation" / "synthetic_qa.jsonl",
            dest="qa_file",
        )
        parser.add_argument(
            "--output", type=Path,
            default=PATHS.root / "evaluation" / "results.json",
        )
        args = parser.parse_args()
        _cmd_legacy(args)

