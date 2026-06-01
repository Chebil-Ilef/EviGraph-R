from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from evaluation.config import SQUAI_DIR  # noqa: F401 — kept for reference

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
            max_tokens=1024,
            timeout=120,
        )
    except Exception as exc:
        logger.warning("[BASELINE] LLM answer failed: %s", exc)
        return ""

# Standard RAG

def run_standard_rag(goldens: list[dict], output_path: Path) -> None:
    from evigraph.retrieval.embedder import Embedder
    from evigraph.retrieval.retriever import HybridQueryRetriever
    from evigraph.config.settings import LLM
    from evigraph.utils.llm import get_llm_client

    embedder = Embedder.from_model_key()
    retriever = HybridQueryRetriever()
    llm = get_llm_client()
    model = LLM.answer_generator_model
    answer_model = model

    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        done_ids.add(rec["golden_id"])
                    except Exception:
                        pass
        logger.info("[STD-RAG] Resuming — %d records already done, skipping them", len(done_ids))

    with open(output_path, "a") as out:
        for i, golden in enumerate(goldens):
            query = golden["input"]
            golden_id = golden.get("golden_id", f"g_{i:04d}")
            if golden_id in done_ids:
                continue
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
                "answer_model": answer_model,
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


# SQuAI baseline — delegates to evaluation/utils/squai_runner.py which imports SQuAI directly.
# Called only from main() below; squai_runner.run() is invoked directly (same process)
# so all env vars loaded from .env above are already in scope.


def main() -> None:
    parser = argparse.ArgumentParser(description="EviGraph-R baseline runner")
    parser.add_argument("--goldens", required=True, type=Path, help="Path to goldens.jsonl")
    parser.add_argument(
        "--baseline", required=True, choices=["standard_rag", "squai"],
        help="Which baseline to run"
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write results JSONL")
    args = parser.parse_args()

    goldens: list[dict] = []
    with open(args.goldens) as f:
        for line in f:
            line = line.strip()
            if line:
                goldens.append(json.loads(line))
    logger.info("[MAIN] Loaded %d goldens", len(goldens))

    if args.baseline == "standard_rag":
        run_standard_rag(goldens, args.output)
    elif args.baseline == "squai":
        # squai_runner.py needs plyvel/faiss from SQuAI's own venv, not EviGraph's .venv.
        # uv would ignore VIRTUAL_ENV and use .venv, so we invoke SQuAI's Python directly.
        import subprocess
        _squai_python = (
            Path(__file__).resolve().parent.parent.parent / "SQuAI" / "env" / "bin" / "python"
        )
        _squai_runner = Path(__file__).resolve().parent / "utils" / "squai_runner.py"
        subprocess.run(
            [str(_squai_python), str(_squai_runner),
             "--goldens", str(args.goldens),
             "--output",  str(args.output)],
            check=True,
            env=os.environ,
        )


if __name__ == "__main__":
    main()
