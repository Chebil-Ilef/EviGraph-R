from __future__ import annotations
import argparse
import json
import logging
import sys
import time
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


def load_goldens(path: Path) -> list[dict]:
    goldens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                goldens.append(json.loads(line))
    return goldens


def run_variant(
    goldens: list[dict],
    variant_name: str,
    output_path: Path,
) -> None:
    import os
    from evaluation.ablation_study import VARIANTS, build_variant_services
    from workflow.graph import build_workflow_graph
    from schemas.state import WorkflowState

    if variant_name not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant_name}'. Available: {list(VARIANTS)}")

    variant = VARIANTS[variant_name]
    logger.info("[RUNNER] Variant: %s — %s", variant.name, variant.description)

    answer_model = os.getenv("LLM_ANSWER_GENERATOR_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

    services = build_variant_services(variant)
    graph = build_workflow_graph(services)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out:
        for i, golden in enumerate(goldens):
            query = golden["input"]
            category = golden["category"]
            domain = golden["domain"]
            golden_id = golden.get("golden_id", f"g_{i:04d}")

            logger.info(
                "[RUNNER] [%d/%d] variant=%s cat=%d domain=%s query=%r",
                i + 1, len(goldens), variant_name, category, domain, query[:80],
            )

            # Build initial state with variant overrides
            state_kwargs: dict = {
                "query": query,
                **variant.state_overrides,
            }
            state = WorkflowState(**state_kwargs)

            t0 = time.perf_counter()
            try:
                final_state = graph.invoke(state)
            except Exception as exc:
                logger.error("[RUNNER] Pipeline failed for golden_id=%s: %s", golden_id, exc)
                final_state = state
                final_state.errors.append(str(exc))

            latency = round(time.perf_counter() - t0, 3)

            # Extract actual_output
            actual_output = ""
            if final_state.final_answer:
                actual_output = final_state.final_answer.text or ""

            # Build retrieval_context: the embed_text of every retrieved document
            retrieval_context: list[str] = [
                doc.embed_text
                for doc in (final_state.retrieved_documents or [])
                if doc.embed_text
            ]

            record = {
                "golden_id": golden_id,
                "variant": variant_name,
                "answer_model": answer_model,
                "category": category,
                "domain": domain,
                "input": query,
                "expected_output": golden.get("expected_output", ""),
                "actual_output": actual_output,
                "retrieval_context": retrieval_context,
                "latency_s": latency,
                "errors": final_state.errors,
                "chunk_ids": golden.get("chunk_ids", []),
            }
            out.write(json.dumps(record) + "\n")

    logger.info("[RUNNER] Done. Results written to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EviGraph-R variant over benchmark goldens")
    parser.add_argument(
        "--goldens",
        default=str(_ROOT / "_data" / "benchmark" / "goldens.jsonl"),
    )
    parser.add_argument(
        "--variant",
        default="full",
        help="Variant name: full | A1.1 | A1.2 | R1 | R2 | R3 | G1 | G2 | J1 | J2 | J3",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: _data/benchmark/results/<variant>.jsonl)",
    )
    args = parser.parse_args()

    goldens_path = Path(args.goldens)
    if not goldens_path.exists():
        logger.error("Goldens file not found: %s", goldens_path)
        sys.exit(1)

    output_path = Path(args.output) if args.output else (
        _ROOT / "_data" / "benchmark" / "results" / f"{args.variant.replace('.', '_')}.jsonl"
    )

    goldens = load_goldens(goldens_path)
    logger.info("[RUNNER] Loaded %d goldens from %s", len(goldens), goldens_path)

    run_variant(goldens, args.variant, output_path)


if __name__ == "__main__":
    main()
