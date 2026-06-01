from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ENV_FILE = _ROOT / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from evaluation.config import EVALUATION_TABLE_LABELS, EVALUATION_TABLE_ORDER
from evaluation.utils.metrics import (
    METRIC_NAMES,
    aggregate_scores,
    build_metrics,
    evaluate_cases,
    result_scores,
)
from evaluation.utils import build_deepeval_model


_FALLBACK_JUDGE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Retry schedule: (truncate_inputs, judge_model_override)
#   attempt 0 — full inputs,     original judge
#   attempt 1 — truncated inputs, original judge      (C)
#   attempt 2 — truncated inputs, fallback judge      (C + D)
_RETRY_TRUNCATE_OUTPUT  = 1000   # chars for actual_output / expected_output
_RETRY_TRUNCATE_CTX     = 3      # max retrieval_context chunks on retry
_RETRY_TRUNCATE_CHUNK   = 500    # chars per chunk on retry


def _build_test_cases(records: list[dict], max_ctx_chunks: int, truncate: bool):
    from deepeval.test_case import LLMTestCase
    test_cases = []
    for r in records:
        ctx = r.get("retrieval_context", [])[:max_ctx_chunks]
        actual  = r.get("actual_output", "")
        expected = r.get("expected_output", "")
        if truncate:
            actual   = actual[:_RETRY_TRUNCATE_OUTPUT]
            expected = expected[:_RETRY_TRUNCATE_OUTPUT]
            ctx = [c[:_RETRY_TRUNCATE_CHUNK] for c in ctx[:_RETRY_TRUNCATE_CTX]]
        test_cases.append(LLMTestCase(
            input=r["input"],
            actual_output=actual,
            expected_output=expected,
            retrieval_context=ctx,
        ))
    return test_cases


def _dropped_indices(test_cases, result) -> set[int]:
    returned = {
        (tr.index if tr.index is not None else i)
        for i, tr in enumerate(result.test_results or [])
    }
    return {i for i in range(len(test_cases)) if i not in returned}


def _score_one(record: dict, judge_model, max_ctx_chunks: int) -> dict:
    """Score a single record, retrying with truncation then fallback judge on failure."""
    results_by_index: dict[int, object] = {}

    def _try(rec, truncate, judge):
        cases = _build_test_cases([rec], max_ctx_chunks, truncate=truncate)
        result = evaluate_cases(cases, build_metrics(judge))
        dropped = _dropped_indices(cases, result)
        if 0 not in dropped:
            results_by_index[0] = (result.test_results or [])[0]

    _try(record, truncate=False, judge=judge_model)
    if 0 not in results_by_index:
        _try(record, truncate=True, judge=judge_model)
    if 0 not in results_by_index:
        fallback = build_deepeval_model(_FALLBACK_JUDGE_MODEL)
        _try(record, truncate=True, judge=fallback)

    test_result = results_by_index.get(0)
    if test_result is None:
        scores = {name: None for name in METRIC_NAMES}
        metric_errors = {"evaluation": "dead — judge failed after 3 attempts"}
        scored = {**record, "scores": scores, "metric_errors": metric_errors}
    else:
        scores, metric_errors = result_scores(test_result)
        scored = {**record, "scores": scores}
        if metric_errors:
            scored["metric_errors"] = metric_errors
    return scored


def score_results(results_path: Path, judge_model, output_dir: Path) -> dict:
    records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        logger.warning("No records in %s", results_path)
        return {}

    variant = records[0].get("variant", results_path.stem)
    scores_path = output_dir / f"scores_{results_path.stem}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load already-scored golden_ids so we can resume after a crash.
    done_ids: set[str] = set()
    scored_records: list[dict] = []
    if scores_path.exists():
        for line in scores_path.open():
            line = line.strip()
            if not line:
                continue
            sr = json.loads(line)
            gid = sr.get("golden_id")
            # Only keep records that actually have scores (not all-None).
            if gid and not all(v is None for v in sr.get("scores", {}).values()):
                done_ids.add(gid)
                scored_records.append(sr)

    remaining = [r for r in records if r.get("golden_id") not in done_ids]
    logger.info(
        "[EVAL] Scoring %d records for variant=%s (%d already done, %d remaining)",
        len(records), variant, len(done_ids), len(remaining),
    )

    _MAX_CTX_CHUNKS = int(os.getenv("DEEPEVAL_MAX_RETRIEVAL_CHUNKS", "5"))

    # Score one record at a time and append immediately so progress survives crashes.
    with open(scores_path, "a") as f:
        for i, record in enumerate(remaining):
            gid = record.get("golden_id", f"idx={i}")
            logger.info(
                "[EVAL] [%d/%d] variant=%s golden_id=%s",
                len(done_ids) + i + 1, len(records), variant, gid,
            )
            scored = _score_one(record, judge_model, _MAX_CTX_CHUNKS)
            scored_records.append(scored)
            f.write(json.dumps(scored) + "\n")
            f.flush()

    logger.info("[EVAL] Wrote scores to %s", scores_path)

    answer_model = records[0].get("answer_model", "") if records else ""
    judge_model_name = judge_model.get_model_name() if hasattr(judge_model, "get_model_name") else str(judge_model)

    agg = aggregate_scores(scored_records, variant)
    agg["answer_model"] = answer_model
    agg["judge_model"] = judge_model_name
    return agg



def build_table(all_agg: list[dict], output_dir: Path) -> None:

    # Index by variant name (normalise dots to underscores)
    by_variant: dict[str, dict] = {}
    for agg in all_agg:
        key = agg["variant"].replace(".", "_")
        by_variant[key] = agg

    header = (
        "| Configuration | Answer Model | Cat.1 | Cat.2 | Cat.3 | Cat.4 | Overall Avg. | Latency (s) |"
    )
    separator = "|---|---|---|---|---|---|---|---|"

    rows_md = [header, separator]
    rows_csv = ["Configuration,Answer Model,Cat.1,Cat.2,Cat.3,Cat.4,Overall Avg.,Latency (s)"]

    for key in EVALUATION_TABLE_ORDER:
        label = EVALUATION_TABLE_LABELS.get(key, key)
        agg = by_variant.get(key)
        if agg is None:
            row_md = f"| {label} | — | — | — | — | — | — | — |"
            row_csv = [label, "", "", "", "", "", "", ""]
        else:
            cats = {int(k): v for k, v in agg.get("by_category", {}).items()}
            c1 = cats.get(1, {}).get("mean_5", "—")
            c2 = cats.get(2, {}).get("mean_5", "—")
            c3 = cats.get(3, {}).get("mean_5", "—")
            c4 = cats.get(4, {}).get("mean_5", "—")
            ov = agg.get("overall", {}).get("mean_5", "—")
            lat = agg.get("latency_mean", "—")
            mdl = agg.get("answer_model", "")

            fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
            row_md = (
                f"| {label} | {mdl} | {fmt(c1)} | {fmt(c2)} | {fmt(c3)} | {fmt(c4)} "
                f"| {fmt(ov)} | {fmt(lat)} |"
            )
            row_csv = [label, mdl, fmt(c1), fmt(c2), fmt(c3), fmt(c4), fmt(ov), fmt(lat)]

        rows_md.append(row_md)
        rows_csv.append(row_csv)

    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "ablation_table.md"
    md_path.write_text("\n".join(rows_md) + "\n")
    logger.info("Wrote %s", md_path)

    csv_path = output_dir / "ablation_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows_csv:
            if isinstance(row, str):
                writer.writerow(row.split(","))
            else:
                writer.writerow(row)
    logger.info("Wrote %s", csv_path)

    metrics_path = output_dir / "metric_summary.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Answer Model", "Judge Model", *METRIC_NAMES, "mean_5", "latency_s", "n_records", "n_scored"])
        for key in EVALUATION_TABLE_ORDER:
            agg = by_variant.get(key)
            label = EVALUATION_TABLE_LABELS.get(key, key)
            if agg is None:
                writer.writerow([label, "", "", *["" for _ in METRIC_NAMES], "", "", "", ""])
                continue
            overall = agg.get("overall", {})
            writer.writerow([
                label,
                agg.get("answer_model", ""),
                agg.get("judge_model", ""),
                *[overall.get(metric, "") for metric in METRIC_NAMES],
                overall.get("mean_5", ""),
                agg.get("latency_mean", ""),
                agg.get("n_records", ""),
                agg.get("n_scored", ""),
            ])
    logger.info("Wrote %s", metrics_path)

    print("\n" + "\n".join(rows_md) + "\n")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark results with DeepEval and produce ablation table"
    )
    parser.add_argument(
        "--results_dir",
        default=str(_ROOT / "_data" / "benchmark" / "results"),
        help="Directory containing one JSONL file per variant",
    )
    parser.add_argument(
        "--output_dir",
        default=str(_ROOT / "_data" / "benchmark" / "eval"),
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Score only this variant (stem of the JSONL filename). "
             "Omit to score all files in results_dir.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv(
            "LLM_EVAL_JUDGE_MODEL",
            os.getenv("LLM_ANSWER_GENERATOR_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        ),
    )
    parser.add_argument(
        "--table_only",
        action="store_true",
        help="Skip scoring; just (re)build the ablation table from existing score files.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if args.table_only:
        # Load all existing aggregation JSONs and rebuild the table
        all_agg = []
        for p in output_dir.glob("agg_*.json"):
            with open(p) as f:
                all_agg.append(json.load(f))
        build_table(all_agg, output_dir)
        return

    judge_model = build_deepeval_model(args.model)

    if args.variant:
        requested = args.variant.replace(".", "_")
        target_files = sorted({
            *results_dir.glob(f"{args.variant}*.jsonl"),
            *results_dir.glob(f"{requested}*.jsonl"),
        })
    else:
        target_files = sorted(results_dir.glob("*.jsonl"))

    if not target_files:
        logger.error("No result files found in %s", results_dir)
        sys.exit(1)

    all_agg = []
    for results_path in target_files:
        agg_path = output_dir / f"agg_{results_path.stem}.json"
        scores_path = output_dir / f"scores_{results_path.stem}.jsonl"

        # Skip if both agg and scores already exist with the same record count
        # and no record was silently dropped by DeepEval (all-None scores indicate a drop)
        if agg_path.exists() and scores_path.exists():
            n_results = sum(1 for _ in results_path.open())
            score_rows = [json.loads(l) for l in scores_path.open() if l.strip()]
            n_scores = len(score_rows)
            dropped = sum(
                1 for sr in score_rows
                if all(v is None for v in sr.get("scores", {}).values())
            )
            if n_scores >= n_results > 0 and dropped == 0:
                logger.info(
                    "[EVAL] Skipping %s — already scored (%d/%d records in %s)",
                    results_path.name, n_scores, n_results, scores_path,
                )
                with open(agg_path) as f:
                    all_agg.append(json.load(f))
                continue
            if dropped > 0:
                logger.warning(
                    "[EVAL] Re-scoring %s — %d/%d previously scored records have all-None scores (DeepEval drops)",
                    results_path.name, dropped, n_scores,
                )

        logger.info("[EVAL] Processing %s", results_path.name)
        agg = score_results(results_path, judge_model, output_dir)
        if agg:
            all_agg.append(agg)
            # Persist aggregation
            with open(agg_path, "w") as f:
                json.dump(agg, f, indent=2)
            logger.info("[EVAL] Aggregation saved to %s", agg_path)

    # Load any previously computed aggregations not re-scored this run
    existing_agg = {}
    for p in output_dir.glob("agg_*.json"):
        stem = p.stem[4:]  # strip "agg_"
        if not any(a["variant"].replace(".", "_") == stem for a in all_agg):
            with open(p) as f:
                existing_agg[stem] = json.load(f)

    build_table(all_agg + list(existing_agg.values()), output_dir)


if __name__ == "__main__":
    main()

# Example usage:
# uv run python -m evaluation.full_evaluation \
#   --results_dir evaluation/_data/results_test3 \
#   --output_dir evaluation/_data/eval_test3_v3 \
#   --model Qwen/Qwen3-Coder-30B-A3B-Instruct 2>&1 | grep -E "INFO|score|Pass|Claim|Attrib|mean_5|overall|ERROR"