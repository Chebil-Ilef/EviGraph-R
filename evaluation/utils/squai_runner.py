from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent        # evaluation/utils/
_EVAL_DIR = _UTILS_DIR.parent                       # evaluation/
_EVIGRAPH_DIR = _EVAL_DIR.parent                    # EviGraph-R/
_SQUAI_DIR = _EVIGRAPH_DIR.parent / "SQuAI"        

for _p in (str(_SQUAI_DIR), str(_EVIGRAPH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.chdir(_SQUAI_DIR)  # run_SQuAI.py writes logs/results relative to cwd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_VARIANT = "squai"
_MODEL = os.environ.get("SCADS_MODEL", "google/gemma-4-31B-it")
_RETRIEVER_TYPE = "hybrid"


def _load_squai_pipeline():
    import plyvel
    from config import E5_INDEX_DIR, BM25_INDEX_DIR, DB_PATH
    from run_SQuAI import Enhanced4AgentRAG, initialize_retriever

    db_path = os.environ.get("SQUAI_DB_PATH", DB_PATH)
    logger.info("[SQuAI] Opening database at %s", db_path)
    try:
        db = plyvel.DB(db_path, create_if_missing=False)
        logger.info("[SQuAI] Database opened successfully")
    except Exception as e:
        logger.error("[SQuAI] DB open failed: %s", e)
        raise

    e5_dir = os.environ.get("SQUAI_E5_INDEX_DIR", E5_INDEX_DIR)
    bm25_dir = os.environ.get("SQUAI_BM25_INDEX_DIR", BM25_INDEX_DIR)

    retriever = initialize_retriever(
        retriever_type=_RETRIEVER_TYPE,
        e5_index_dir=e5_dir,
        bm25_index_dir=bm25_dir,
        db_path=db_path,
        top_k=5,
        alpha=0.65,
    )

    # Enhanced4AgentRAG pre-warms in __init__
    ragent = Enhanced4AgentRAG(
        retriever,
        agent_model=_MODEL,
        n=0.5,
        max_workers=4,
    )

    return ragent, db


def _passages_to_context(passages_used: list[dict]) -> list[str]:
    texts = []
    for p in passages_used or []:
        text = p.get("context_passage") or p.get("full_text") or p.get("abstract") or ""
        if text:
            texts.append(text)
    return texts


def run(goldens_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load goldens
    goldens: list[dict] = []
    with open(goldens_path) as f:
        for line in f:
            line = line.strip()
            if line:
                goldens.append(json.loads(line))
    logger.info("[SQuAI] Loaded %d goldens from %s", len(goldens), goldens_path)

    # Resume: collect already-done golden_ids
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["golden_id"])
                    except Exception:
                        pass
        if done_ids:
            logger.info("[SQuAI] Resuming — %d records already done, skipping them", len(done_ids))

    remaining = [g for i, g in enumerate(goldens)
                 if g.get("golden_id", f"g_{i:04d}") not in done_ids]

    if not remaining:
        logger.info("[SQuAI] All %d questions already done.", len(goldens))
        return

    # Initialise pipeline (slow — models load here)
    ragent, db = _load_squai_pipeline()

    with open(output_path, "a") as out:
        for i, golden in enumerate(goldens):
            golden_id = golden.get("golden_id", f"g_{i:04d}")
            if golden_id in done_ids:
                continue

            query = golden["input"]
            logger.info(
                "[SQuAI] [%d/%d] golden_id=%s  query=%r",
                i + 1, len(goldens), golden_id, query[:80],
            )

            t0 = time.perf_counter()
            errors: list[str] = []
            actual_output = ""
            retrieval_context: list[str] = []

            try:
                should_split, sub_questions = ragent.question_splitter.analyze_and_split(query)
                cited_answer, _references, debug_info = ragent.answer_query(
                    query, db, should_split, sub_questions
                )
                actual_output = cited_answer or ""
                retrieval_context = _passages_to_context(debug_info.get("passages_used", []))
            except Exception as exc:
                logger.error("[SQuAI] Failed golden_id=%s: %s", golden_id, exc, exc_info=True)
                errors.append(str(exc))

            latency = round(time.perf_counter() - t0, 3)

            record = {
                "golden_id": golden_id,
                "variant": _VARIANT,
                "answer_model": _MODEL,
                "category": golden.get("category"),
                "domain": golden.get("domain"),
                "input": query,
                "expected_output": golden.get("expected_output", ""),
                "actual_output": actual_output,
                "retrieval_context": retrieval_context,
                "latency_s": latency,
                "errors": errors,
                "chunk_ids": golden.get("chunk_ids", []),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()  # save progress after every question

    ragent.close()
    db.close()
    logger.info("[SQuAI] Done. Results written to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="SQuAI baseline runner for EviGraph-R evaluation")
    parser.add_argument("--goldens", required=True, type=Path, help="Path to goldens.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Path to write results JSONL")
    args = parser.parse_args()
    run(args.goldens, args.output)


if __name__ == "__main__":
    main()
