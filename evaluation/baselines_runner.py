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


_RAG_SYSTEM = (
    "You are a scientific question-answering assistant. "
    "Answer the question concisely and accurately based solely on the provided context. "
    "If the context does not contain enough information, say so."
)


def _rag_answer(query: str, chunks: list[str], llm_client, model: str) -> str:
    context = "\n\n---\n\n".join(chunks[:10])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    try:
        return llm_client.chat_text(
            model=model,
            system_prompt=_RAG_SYSTEM,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=512,
            timeout=120,
        )
    except Exception as exc:
        logger.warning("[BASELINE] LLM answer failed: %s", exc)
        return ""

# Standard RAG

def run_standard_rag(goldens: list[dict], output_path: Path) -> None:
    from retrieval.embedder import Embedder
    from retrieval.retriever import HybridQueryRetriever
    from config.settings import LLM
    from utils.llm import get_llm_client

    embedder = Embedder.from_model_key()
    retriever = HybridQueryRetriever()
    llm = get_llm_client()
    model = LLM.answer_generator_model

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out:
        for i, golden in enumerate(goldens):
            query = golden["input"]
            golden_id = golden.get("golden_id", f"g_{i:04d}")
            logger.info(
                "[STD-RAG] [%d/%d] golden_id=%s query=%r",
                i + 1, len(goldens), golden_id, query[:80],
            )

            t0 = time.perf_counter()
            try:
                emb_out = embedder.embed_query(query)
                # BGEOutput has .dense; plain ndarray otherwise
                dense_vec = emb_out.dense.tolist() if hasattr(emb_out, "dense") else emb_out.tolist()

                chunks = retriever._retrieve_dense_only(dense_vec, top_k=15)
                retrieval_context = [c.embed_text for c in chunks if c.embed_text]
                actual_output = _rag_answer(query, retrieval_context, llm, model)
            except Exception as exc:
                logger.error("[STD-RAG] Failed: %s", exc)
                retrieval_context = []
                actual_output = ""

            latency = round(time.perf_counter() - t0, 3)

            record = {
                "golden_id": golden_id,
                "variant": "standard_rag",
                "category": golden["category"],
                "domain": golden["domain"],
                "input": query,
                "expected_output": golden.get("expected_output", ""),
                "actual_output": actual_output,
                "retrieval_context": retrieval_context,
                "latency_s": latency,
                "errors": [],
                "chunk_ids": golden.get("chunk_ids", []),
            }
            out.write(json.dumps(record) + "\n")

    logger.info("[STD-RAG] Done. Results written to %s", output_path)


# SQuAI baseline is intentionally not implemented yet.

def run_squai(goldens: list[dict], output_path: Path) -> None:
    raise NotImplementedError(
        "SQuAI baseline is not implemented yet. "
        "Run with --baseline standard_rag or set BASELINES=standard_rag in the Capella launcher."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline pipelines over benchmark goldens")
    parser.add_argument(
        "--goldens",
        default=str(_ROOT / "_data" / "benchmark" / "goldens.jsonl"),
    )
    parser.add_argument(
        "--baseline",
        choices=["standard_rag", "squai"],
        required=True,
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    goldens_path = Path(args.goldens)
    if not goldens_path.exists():
        logger.error("Goldens file not found: %s", goldens_path)
        sys.exit(1)

    output_path = Path(args.output) if args.output else (
        _ROOT / "_data" / "benchmark" / "results" / f"{args.baseline}.jsonl"
    )

    with open(goldens_path) as f:
        goldens = [json.loads(l) for l in f if l.strip()]
    logger.info("Loaded %d goldens", len(goldens))

    if args.baseline == "standard_rag":
        run_standard_rag(goldens, output_path)
    else:
        run_squai(goldens, output_path)


if __name__ == "__main__":
    main()
