"""
Evaluate all embedding models on synthetic QA dataset.

Usage
-----
python evaluate_models.py                                # all models, default indexing with e5-base-v2
python evaluate_models.py --batches batch_01 batch_02 --models qwen3-0.6b --index-model qwen3-0.6b --recreate
python evaluate_models.py --models e5-base-v2 --top-k 5 10 --recreate
python evaluate_models.py --no-index --models bge-m3   # skip indexing, evaluate existing chunks
python evaluate_models.py --output eval/results.json

Important: Use --recreate when switching to a model with different embedding dimension!
Since different models have different output dimensions (e.g., e5=768, qwen=1024),
the Qdrant collection must be recreated when using a different model.

Metrics
-------
Recall@k:         1 if gold_chunk_uid in top-k retrieval results
MRR@k:            Mean Reciprocal Rank of gold_chunk_uid
NDCG@k:           Normalized Discounted Cumulative Gain (gold_chunk=1, gold_paper=0.5)
PaperHit@k:       1 if any chunk from gold_paper in top-k results
SectionHit@k:     1 if any chunk from gold_section in top-k results
AnswerContain@k:  1 if any gold_answer_string found in retrievals' embed_text
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    PATHS,
    QDRANT_ACTIVE,
)
from src.core.embedder import Embedder
from src.core.retriever import UniversalQueryRetriever, ChunkResult
from src.indexing_pipeline import run_indexing

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


@dataclass
class QARecord:
    """One synthetic Q&A record from synthetic_qa.jsonl."""
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
    """Per-query metrics."""
    question_id: str
    recall_at_k: dict[int, bool] = field(default_factory=dict)      # k → bool
    mrr_at_k: dict[int, float] = field(default_factory=dict)         # k → float
    ndcg_at_k: dict[int, float] = field(default_factory=dict)        # k → float
    paper_hit_at_k: dict[int, bool] = field(default_factory=dict)    # k → bool
    section_hit_at_k: dict[int, bool] = field(default_factory=dict)  # k → bool
    answer_contain_at_k: dict[int, bool] = field(default_factory=dict)


@dataclass
class ModelResults:
    """Aggregated results for one model."""
    model_key: str
    total_queries: int
    metrics_per_query: list[EvalMetrics] = field(default_factory=list)
    
    # Timing
    evaluation_time_seconds: float = 0.0  # seconds
    
    # Aggregated (computed on demand)
    recall_mean: dict[int, float] = field(default_factory=dict)
    mrr_mean: dict[int, float] = field(default_factory=dict)
    ndcg_mean: dict[int, float] = field(default_factory=dict)
    paper_hit_mean: dict[int, float] = field(default_factory=dict)
    section_hit_mean: dict[int, float] = field(default_factory=dict)
    answer_contain_mean: dict[int, float] = field(default_factory=dict)
    
    def compute_aggregates(self, top_ks: list[int]) -> None:
        """Aggregate metrics across all queries."""
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
    """Recall@k: 1 if gold_chunk_uid in top-k results."""
    top_k_results = results[:top_k]
    return any(r.chunk_uid == gold_chunk_uid for r in top_k_results)


def compute_mrr(results: list[ChunkResult], top_k: int, gold_chunk_uid: str) -> float:
    """MRR@k: 1/rank if found in top-k, else 0."""
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
    """
    NDCG@k: Normalized Discounted Cumulative Gain.
    Relevance: gold_chunk=1, gold_paper (other chunk from same paper)=0.5, else=0.
    """
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
    """PaperHit@k: 1 if any chunk from gold_paper in top-k."""
    top_k_results = results[:top_k]
    return any(r.paper_id == gold_paper_id for r in top_k_results)


def compute_section_hit(
    results: list[ChunkResult],
    top_k: int,
    gold_section_title: str,
) -> bool:
    """SectionHit@k: 1 if any chunk from gold_section in top-k."""
    top_k_results = results[:top_k]
    return any(r.section_title == gold_section_title for r in top_k_results)


def compute_answer_contain(
    results: list[ChunkResult],
    top_k: int,
    gold_answer_strings: list[str],
) -> bool:
    """AnswerContain@k: 1 if any gold_answer_string found in top-k embed_text."""
    top_k_results = results[:top_k]
    combined_text = " ".join(r.embed_text for r in top_k_results).lower()
    
    for answer in gold_answer_strings:
        answer_lower = answer.lower()
        if answer_lower in combined_text:
            return True
    return False


# Evaluation Pipeline

def load_synthetic_qa(file_path: Path) -> list[QARecord]:
    """Load synthetic Q&A records from JSONL."""
    records = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                records.append(QARecord.from_dict(json.loads(line)))
    logger.info("Loaded %d synthetic Q&A records from %s", len(records), file_path)
    return records


def evaluate_model(
    model_key: str,
    qa_records: list[QARecord],
    top_ks: list[int] = None,
) -> ModelResults:
    """Evaluate one embedding model on all Q&A records."""
    if top_ks is None:
        top_ks = [1, 5, 10]
    
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("Evaluating model: %s", model_key)
    logger.info("=" * 70)
    
    # Initialize embedder and retriever
    embedder = Embedder.from_model_key(model_key)
    retriever = UniversalQueryRetriever()
    
    logger.info("Embedder initialized: %s | dim=%d | device=%s",
                model_key, EMBEDDING_MODELS[model_key].dim, EMBEDDING_MODELS[model_key].device)
    
    results = ModelResults(model_key=model_key, total_queries=len(qa_records))
    
    # Evaluate each query
    for qa in tqdm(qa_records, desc=f"Evaluating {model_key}", unit="query"):
        # Embed query
        if isinstance(embedder.embed_query(qa.query), tuple):
            # BGE-M3 returns (dense, sparse)
            embed_result = embedder.embed_query(qa.query)
            query_embedding = embed_result.dense if hasattr(embed_result, 'dense') else embed_result[0]
        else:
            query_embedding = embedder.embed_query(qa.query)
        
        # Retrieve
        retrieved = retriever.retrieve(
            embeddings=query_embedding,
            query_text=qa.query,
            top_k=max(top_ks),  # get max top_k so we can evaluate all k values
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
    
    # Record timing
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
    """Evaluate all models."""
    if model_keys is None:
        model_keys = list(EMBEDDING_MODELS.keys())
    if top_ks is None:
        top_ks = [1, 5, 10]
    
    all_results = {}
    for model_key in model_keys:
        try:
            all_results[model_key] = evaluate_model(model_key, qa_records, top_ks)
        except Exception as e:
            logger.error("Failed to evaluate model %s: %s", model_key, e)
    
    return all_results


def format_results(all_results: dict[str, ModelResults], top_ks: list[int]) -> str:
    """Format results as readable table."""
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
    """Save detailed results to JSON."""
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
    """Save markdown report to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_content)
    logger.info("Saved markdown report to %s", output_path)



if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    
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
        help="Skip indexing step (chunks must already exist in Qdrant).",
    )
    parser.add_argument(
        "--index-model", type=str, default=None,
        help="Embedding model to use for indexing (default: first model in --models).",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and recreate Qdrant collection (required when switching models).",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(EMBEDDING_MODELS.keys()),
        help="Embedding model keys to evaluate.",
    )
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=[1, 5, 10],
        help="Top-k values for metrics.",
    )
    parser.add_argument(
        "--qa-file", type=Path, default=PATHS.root / "evaluation" / "synthetic_qa.jsonl",
        help="Path to synthetic QA JSONL file.",
    )
    parser.add_argument(
        "--output", type=Path, default=PATHS.root / "evaluation" / "results.json",
        help="Output JSON file for detailed results.",
    )
    args = parser.parse_args()
    
    # Step 1: Index batches if requested
    indexing_time = 0.0
    if not args.no_index:
        logger.info("=" * 70)
        logger.info("INDEXING PHASE: Preparing batches %s", args.batches)
        logger.info("=" * 70)
        
        # Determine which model to use for indexing
        index_model_key = args.index_model or args.models[0]
        logger.info("Indexing with model: %s", index_model_key)
        
        if args.recreate:
            logger.info("Recreating collection (--recreate flag set)")
        
        indexing_start = time.time()
        run_indexing(args.batches, model_key=index_model_key, recreate=args.recreate)
        indexing_time = time.time() - indexing_start
    
    # Step 2: Load data
    logger.info("=" * 70)
    logger.info("EVALUATION PHASE: Loading synthetic QA records")
    logger.info("=" * 70)
    qa_records = load_synthetic_qa(args.qa_file)
    
    # Step 3: Evaluate
    all_results = evaluate_all_models(qa_records, args.models, args.top_k)
    
    # Step 4: Print summary
    print(format_results(all_results, args.top_k))
    
    # Step 5: Save detailed results
    save_results_json(all_results, args.output)
    
    # Step 6: Generate and save markdown report
    batches_str = "_".join(args.batches)
    markdown_output = PATHS.root / "evaluation" / f"{batches_str}_batches.md"
    
    report = generate_markdown_report(
        all_results,
        args.top_k,
        args.batches,
        indexing_time_seconds=indexing_time,
        qa_records=qa_records,
    )
    save_markdown_report(report, markdown_output)
    print(f"\n✓ Markdown report saved to: {markdown_output}")
