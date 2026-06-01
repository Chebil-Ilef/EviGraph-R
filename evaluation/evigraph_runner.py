from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
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

_VARIANTS_THAT_REUSE_RETRIEVAL: frozenset[str] = frozenset(
    {"G1", "G2", "J1", "J2", "J3"}
)

def load_goldens(path: Path) -> list[dict]:
    goldens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                goldens.append(json.loads(line))
    return goldens


def _retriever_mode(variant_name: str) -> str:
    if variant_name == "R1":
        return "dense"
    if variant_name == "R2":
        return "sparse"
    if variant_name == "R3":
        return "hybrid_no_boost"
    return "hybrid"


def _cache_key(query: str, retriever_mode: str, top_k: int) -> str:
    raw = f"{query}\x00{retriever_mode}\x00{top_k}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(cache_dir: Path, key: str) -> dict | None:
    p = cache_dir / f"{key}.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _cache_put(cache_dir: Path, key: str, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{key}.json"
    with open(p, "w") as f:
        json.dump(data, f)


def run_variant(
    goldens: list[dict],
    variant_name: str,
    output_path: Path,
    shared=None,
    cache_dir: Path | None = None,
) -> None:
    import os
    from evaluation.ablation_study import VARIANTS, build_variant_services
    from evigraph.workflow.graph import build_workflow_graph
    from evigraph.schemas.state import WorkflowState

    if variant_name not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant_name}'. Available: {list(VARIANTS)}")

    variant = VARIANTS[variant_name]
    logger.info("[RUNNER] Variant: %s — %s", variant.name, variant.description)

    answer_model = os.getenv("LLM_ANSWER_GENERATOR_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

    if cache_dir is None:
        cache_dir = _ROOT / "evaluation" / "_data" / "cache"

    use_cache = variant_name in _VARIANTS_THAT_REUSE_RETRIEVAL
    rmode = _retriever_mode(variant_name)

    services = build_variant_services(variant, shared=shared)
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

            # For variants that only differ post-retrieval, inject cached
            # decomposer output + retrieved docs to skip Qdrant entirely.
            cache_hit = False
            if use_cache:
                key = _cache_key(query, "hybrid", state.top_k)
                cached = _cache_get(cache_dir, key)
                if cached is not None:
                    try:
                        from evigraph.schemas.objects import SubQuery, RetrievedDocument, IMRaDSection
                        sub_queries = [SubQuery(**sq) for sq in cached["sub_queries"]]
                        retrieved = [RetrievedDocument(**d) for d in cached["retrieved_documents"]]
                        state.sub_queries = sub_queries
                        state.decomposition_done = True
                        state.retrieved_documents = retrieved
                        state.retrieval_done = True
                        cache_hit = True
                        logger.info(
                            "[RUNNER] Cache HIT for golden_id=%s (%d sub-queries, %d docs)",
                            golden_id, len(sub_queries), len(retrieved),
                        )
                    except Exception as exc:
                        logger.warning("[RUNNER] Cache entry invalid, re-running: %s", exc)
                        cache_hit = False

            t0 = time.perf_counter()
            try:
                result = graph.invoke(state)
                # LangGraph always returns a dict when the state schema is a Pydantic model;
                # reconstruct WorkflowState so downstream code can use attribute access.
                if isinstance(result, dict):
                    final_state = WorkflowState(**result)
                else:
                    final_state = result
            except Exception as exc:
                logger.error("[RUNNER] Pipeline failed for golden_id=%s: %s", golden_id, exc)
                final_state = state
                final_state.errors.append(str(exc))

            latency = round(time.perf_counter() - t0, 3)

            # For "full" (and any other hybrid variant), write decomposer+retrieval
            # results to cache so downstream variants can reuse them.
            if not use_cache and not cache_hit and rmode == "hybrid":
                if final_state.decomposition_done and final_state.retrieval_done:
                    key = _cache_key(query, "hybrid", state.top_k)
                    try:
                        cache_data = {
                            "sub_queries": [sq.model_dump() for sq in final_state.sub_queries],
                            "retrieved_documents": [d.model_dump() for d in (final_state.retrieved_documents or [])],
                        }
                        _cache_put(cache_dir, key, cache_data)
                    except Exception as exc:
                        logger.warning("[RUNNER] Cache write failed for golden_id=%s: %s", golden_id, exc)

            # Extract actual_output
            actual_output = ""
            if final_state.final_answer:
                actual_output = final_state.final_answer.text or ""

            # Build retrieval_context: the text content of every retrieved document
            retrieval_context: list[str] = [
                doc.content
                for doc in (final_state.retrieved_documents or [])
                if doc.content
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
        default=None,
        help="Single variant name: full | A1.1 | A1.2 | R1 | R2 | R3 | G1 | G2 | J1 | J2 | J3",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        metavar="VARIANT",
        help="One or more variant names to run sequentially, sharing loaded models.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (only valid with --variant).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for results (used with --variants).",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Directory for decomposer+retrieval cache (reused by G1/G2/J1/J2/J3). "
             "Defaults to evaluation/_data/cache/.",
    )
    args = parser.parse_args()

    goldens_path = Path(args.goldens)
    if not goldens_path.exists():
        logger.error("Goldens file not found: %s", goldens_path)
        sys.exit(1)

    goldens = load_goldens(goldens_path)
    logger.info("[RUNNER] Loaded %d goldens from %s", len(goldens), goldens_path)

    variant_names = args.variants or ([args.variant] if args.variant else ["full"])

    if len(variant_names) > 1:
        # Load heavy models once and share across all variants
        from evaluation.ablation_study import load_shared_resources
        logger.info("[RUNNER] Pre-loading shared models (embedder + retriever + LLM client)…")
        shared = load_shared_resources()
        logger.info("[RUNNER] Shared models ready. Running %d variants.", len(variant_names))
    else:
        shared = None

    output_dir = Path(args.output_dir) if args.output_dir else (
        _ROOT / "_data" / "benchmark" / "results"
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        _ROOT / "evaluation" / "_data" / "cache"
    )

    n_goldens = len(goldens)
    for variant_name in variant_names:
        if args.output and len(variant_names) == 1:
            output_path = Path(args.output)
        else:
            output_path = output_dir / f"{variant_name.replace('.', '_')}.jsonl"

        # Skip variants already fully completed (exact line count match).
        if output_path.exists():
            existing = sum(1 for _ in output_path.open())
            if existing >= n_goldens:
                logger.info(
                    "[RUNNER] Skipping variant=%s — already complete (%d/%d records in %s)",
                    variant_name, existing, n_goldens, output_path,
                )
                continue
            else:
                logger.info(
                    "[RUNNER] Resuming variant=%s — found %d/%d records, re-running from scratch",
                    variant_name, existing, n_goldens,
                )

        run_variant(goldens, variant_name, output_path, shared=shared, cache_dir=cache_dir)


if __name__ == "__main__":
    main()
