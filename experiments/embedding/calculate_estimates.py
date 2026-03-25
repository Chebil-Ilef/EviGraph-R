import argparse
import logging
from pathlib import Path
from datetime import timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOTAL_PAPERS = 2_300_000 
ESTIMATED_CHUNKS_PER_PAPER = 20  


def format_time(seconds: float) -> str:

    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(secs)}s"


def calculate_estimates(
    sample_size: int,
    chunk_count: int,
    indexing_time_seconds: float,
    evaluation_time_seconds: float,
) -> dict:

    chunks_per_article = chunk_count / sample_size if sample_size > 0 else 0
    indexing_time_per_chunk = indexing_time_seconds / chunk_count if chunk_count > 0 else 0
    
    qa_per_chunk = 1.5  # Average Q&A pairs per chunk (conservative)
    eval_time_per_query = evaluation_time_seconds / (chunk_count * qa_per_chunk) if chunk_count > 0 else 0
    
    # Project to full dataset
    estimated_total_chunks = TOTAL_PAPERS * chunks_per_article
    estimated_qa_pairs = int(estimated_total_chunks * qa_per_chunk)
    
    # Single-model evaluation time (all models run sequentially currently)
    estimated_full_indexing_time = estimated_total_chunks * indexing_time_per_chunk
    estimated_full_eval_time = estimated_qa_pairs * eval_time_per_query * 4  # 4 models
    
    # Add overhead estimates
    overhead_factor = 1.1  # 10% for I/O, overhead
    estimated_full_indexing_time *= overhead_factor
    estimated_full_eval_time *= overhead_factor
    
    return {
        "sample_size": sample_size,
        "sample_chunks": chunk_count,
        "sample_indexing_time": indexing_time_seconds,
        "sample_eval_time": evaluation_time_seconds,
        "chunks_per_article": chunks_per_article,
        "indexing_time_per_chunk": indexing_time_per_chunk,
        "eval_time_per_query": eval_time_per_query,
        "total_papers": TOTAL_PAPERS,
        "estimated_total_chunks": estimated_total_chunks,
        "estimated_qa_pairs": estimated_qa_pairs,
        "estimated_full_indexing_time": estimated_full_indexing_time,
        "estimated_full_eval_time": estimated_full_eval_time,
        "estimated_total_time": estimated_full_indexing_time + estimated_full_eval_time,
    }


def generate_markdown_report(estimates: dict) -> str:

    lines = []
    
    lines.append("# HPC Evaluation Time Estimates")
    lines.append("")
    
    lines.append("## Sample Measurements")
    lines.append("")
    lines.append(f"**Date**: Measured on HPC (Capella GPU)")
    lines.append(f"**Sample size**: {estimates['sample_size']:,} articles")
    lines.append(f"**Resulting chunks**: {estimates['sample_chunks']:,}")
    lines.append("")
    
    lines.append("### Sample Run Timing")
    lines.append("")
    lines.append(f"| Phase | Time | Per-Item |")
    lines.append("|-------|------|----------|")
    lines.append(f"| **Indexing (chunk+embed)** | {format_time(estimates['sample_indexing_time'])} | {estimates['indexing_time_per_chunk']*1000:.2f}ms/chunk |")
    lines.append(f"| **Evaluation (all models)** | {format_time(estimates['sample_eval_time'])} | {estimates['eval_time_per_query']*1000:.2f}ms/query |")
    lines.append(f"| **Total** | {format_time(estimates['sample_indexing_time'] + estimates['sample_eval_time'])} | |")
    lines.append("")
    
    lines.append("## Full Dataset Projections (2.3M papers)")
    lines.append("")
    lines.append(f"**Estimated chunks**: {estimates['estimated_total_chunks']:,.0f} (@ {estimates['chunks_per_article']:.1f} chunks/article)")
    lines.append(f"**Estimated Q&A pairs**: {estimates['estimated_qa_pairs']:,}")
    lines.append("")
    
    lines.append("### Time Estimates")
    lines.append("")
    lines.append(f"| Phase | Estimated Time | Notes |")
    lines.append("|-------|-----------------|-------|")
    lines.append(f"| **Indexing Phase (GPU array job)** | **{format_time(estimates['estimated_full_indexing_time'])}** | Parallelizable across nodes |")
    lines.append(f"| **Evaluation Phase (sequential)** | **{format_time(estimates['estimated_full_eval_time'])}** | 4 models × evaluation |")
    lines.append(f"| **Total** | **{format_time(estimates['estimated_total_time'])}** | |")
    lines.append("")
    
    lines.append("## HPC Considerations")
    lines.append("")
    
    indexing_hours = estimates['estimated_full_indexing_time'] / 3600
    eval_hours = estimates['estimated_full_eval_time'] / 3600
    
    lines.append("### Indexing Phase (Phase A)")
    lines.append("")
    lines.append(f"- **Estimated duration**: {format_time(estimates['estimated_full_indexing_time'])}")
    lines.append(f"- **Parallelization**: GPU array job over {max(1, int(indexing_hours / 24))} nodes (7-day max window)")
    lines.append(f"- **Resources needed**: 1 GPU per node, {int(8 * max(1, int(indexing_hours / 24)))} CPUs total")
    lines.append(f"- **Suggested**: Use GPU array job with checkpoint recovery every 100k chunks")
    lines.append("")
    
    lines.append("### Evaluation Phase (Phase B)")
    lines.append("")
    lines.append(f"- **Estimated duration**: {format_time(estimates['estimated_full_eval_time'])}")
    lines.append(f"- **Parallelization**: Sequential (models evaluated one-by-one)")  
    lines.append(f"- **Resources needed**: 1 GPU, ~64GB RAM, 8 CPUs")
    lines.append(f"- **Suggested**: Single GPU job (fits in 7-day window easily)")
    lines.append("")
    
    lines.append("## Recommended Strategy")
    lines.append("")
    lines.append("### Option A: Sequential Full Pipeline (Simplest)")
    lines.append("")
    lines.append("1. Submit large GPU array job for indexing phase")
    lines.append("   - Checkpoint every 100k chunks")
    lines.append("   - Auto-restart from volumes if timeout")
    lines.append("2. After indexing complete, run evaluation phase separately")
    lines.append(f"   - Estimated time: {format_time(estimates['estimated_full_eval_time'])}")
    lines.append("3. Generate final report with timing analysis")
    lines.append("")
    
    lines.append("### Option B: Parallel Models (Faster Evaluation)")
    lines.append("")
    lines.append("1. Run indexing phase once")
    lines.append(f"   - Time: {format_time(estimates['estimated_full_indexing_time'])}")
    lines.append("2. After indexing, launch 4 concurrent evaluation jobs (one per model)")
    lines.append(f"   - Each job: ~{format_time(estimates['estimated_full_eval_time'] / 4)}")
    lines.append(f"   - Total wall time: ~{format_time(estimates['estimated_full_eval_time'] / 4)} (4× speedup)")
    lines.append("")
    
    lines.append("### Option C: Incremental Evaluation (Risk Minimization)")
    lines.append("")
    lines.append("1. Index in batches (e.g., 700k papers per batch)")
    lines.append(f"   - Per-batch time: ~{format_time(estimates['estimated_full_indexing_time'] / 4)}")
    lines.append("2. Evaluate after each batch for progress visibility")
    lines.append("3. Reduces blast radius of failures")
    lines.append("")
    
    lines.append("## Sample Run Details")
    lines.append("")
    lines.append("This report was generated from sample run #1:")
    lines.append("")
    lines.append(f"- Articles sampled: {estimates['sample_size']:,}")
    lines.append(f"- Chunks generated: {estimates['sample_chunks']:,}")
    lines.append(f"- Indexing time: {format_time(estimates['sample_indexing_time'])}")
    lines.append(f"- Evaluation time: {format_time(estimates['sample_eval_time'])}")
    if estimates['sample_indexing_time'] > 0:
        lines.append(f"- Throughput: {estimates['sample_chunks']/estimates['sample_indexing_time']:.1f} chunks/sec")
    else:
        lines.append("- Throughput: N/A (indexing skipped)")
    lines.append("")
    
    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. Choose strategy above (A/B/C recommended)")
    lines.append("2. Adjust HPC job parameters based on estimates")
    lines.append("3. Set up checkpoint/snapshot recovery (see hpc.md)")
    lines.append("4. Monitor job progress with Qdrant snapshots")
    lines.append("5. Generate final evaluation report after completion")
    lines.append("")
    
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description="Calculate HPC time estimates from sample measurements."
    )
    p.add_argument(
        "--sample-size", type=int, required=True,
        help="Number of articles in sample.",
    )
    p.add_argument(
        "--chunk-count", type=int, required=True,
        help="Number of chunks generated from sample.",
    )
    p.add_argument(
        "--indexing-time", type=float, required=True,
        help="Indexing time in seconds.",
    )
    p.add_argument(
        "--evaluation-time", type=float, required=True,
        help="Evaluation time in seconds.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output markdown file path.",
    )
    args = p.parse_args()

    estimates = calculate_estimates(
        args.sample_size,
        args.chunk_count,
        args.indexing_time,
        args.evaluation_time,
    )

    report = generate_markdown_report(estimates)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    
    logger.info("✓ Estimates saved to %s", args.output)
    
    # Also print summary to console
    print("\n" + "=" * 80)
    print("TIME ESTIMATES FOR FULL 2.8M PAPER DATASET")
    print("=" * 80)
    print(f"\nIndexing Phase (GPU):    {format_time(estimates['estimated_full_indexing_time'])}")
    print(f"Evaluation Phase (GPU):  {format_time(estimates['estimated_full_eval_time'])}")
    print(f"TOTAL TIME:              {format_time(estimates['estimated_total_time'])}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
