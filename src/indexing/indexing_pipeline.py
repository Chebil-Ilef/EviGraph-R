from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    PATHS,
    QDRANT_ACTIVE,
    _QdrantProfile,
)
from indexing.index_builder import (
    build_paper_chunks,
    save_chunks,
    load_chunks,
    chunks_file_exists
)
from retrieval.embedder import Embedder
from utils.qdrant import (
    setup_collection, build_points, get_collection_info,
    qdrant_client, check_qdrant_alive,
)

logger = logging.getLogger(__name__)




# Stage 1 — Chunking

def run_chunking(
    batch_stems: list[str],
    model_key: str = DEFAULT_EMBEDDING_MODEL,
    *,
    overwrite: bool = False,
) -> list[str]:
    

    if batch_stems == ["all"]:
        batch_paths = sorted(PATHS.batches.glob("batch_*.jsonl"))
    else:
        batch_paths = [PATHS.batches / f"{s}.jsonl" for s in batch_stems]

    chunked: list[str] = []
    for batch_path in batch_paths:
        stem = batch_path.stem
        if not overwrite and chunks_file_exists(stem):
            logger.info("Chunks for %r already exist — skipping (use overwrite=True).", stem)
            continue
        if not batch_path.exists():
            logger.warning("Batch file not found: %s — skipping.", batch_path)
            continue
        logger.info("Chunking %s …", batch_path.name)
        papers = [
            json.loads(line)
            for line in batch_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        all_chunks: list[dict] = []
        for paper in tqdm(papers, desc=f"Chunking {stem}", unit="paper"):
            all_chunks.extend(build_paper_chunks(paper, model_key=model_key))
        save_chunks(all_chunks, stem)
        logger.info("Saved %d chunks → %s_chunks.jsonl", len(all_chunks), stem)
        chunked.append(stem)

    logger.info("Chunking done. %d batch(es) processed.", len(chunked))
    return chunked



# Stage 2 — Embedding + Indexing

def run_indexing(
    batch_stems: list[str],
    model_key: str = DEFAULT_EMBEDDING_MODEL,
    *,
    recreate: bool = False,
    profile: _QdrantProfile = QDRANT_ACTIVE,
) -> None:
   
    if batch_stems == ["all"]:
        batch_stems = sorted(
            p.stem.replace("_chunks", "")
            for p in PATHS.chunks.glob("*_chunks.jsonl")
        )
        if not batch_stems:
            logger.warning("No chunk files found in %s. Run chunking first.", PATHS.chunks)
            return

    client = qdrant_client()
    check_qdrant_alive(client)

    #  Collection setup (idempotent; recreate=True drops + rebuilds) 
    setup_collection(client, model_key=model_key, profile=profile, recreate=recreate)

    #  Embedder 
    embedder = Embedder.from_model_key(model_key)
    cfg      = EMBEDDING_MODELS[model_key]
    logger.info(
        "Embedder ready: %s  dim=%d  sparse=%s  device=%s",
        model_key, cfg.dim, cfg.bge_produces_sparse, cfg.device,
    )

    #  Upsert loop 
    total_upserted = 0
    for stem in batch_stems:
        if not chunks_file_exists(stem):
            logger.warning("No chunks file for %r — skipping. Run chunking first.", stem)
            continue

        chunks = load_chunks(stem)
        if not chunks:
            logger.info("Batch %r has 0 chunks — skipping.", stem)
            continue

        logger.info("Indexing %r — %d chunks …", stem, len(chunks))
        for i in tqdm(
            range(0, len(chunks), profile.upsert_batch_size),
            desc=f"Upserting {stem}",
            unit="batch",
        ):
            batch         = chunks[i : i + profile.upsert_batch_size]
            texts         = [c["embed_text"] for c in batch]
            embed_result  = embedder.embed_passages(texts)
            points        = build_points(batch, embed_result, profile, model_key)
            client.upsert(
                collection_name=profile.collection_name,
                points=points,
                wait=True,
            )
        total_upserted += len(chunks)
        logger.info("Batch %r — done.", stem)

    #  Final stats 
    stats = get_collection_info(client, profile.collection_name)
    logger.info(
        "Collection %r — total_upserted=%d  points_in_db=%s  status=%s",
        profile.collection_name,
        total_upserted,
        stats["points_count"],
        stats["status"],
    )


# Full pipeline (chunk + index)

def run_pipeline(
    batch_stems: list[str],
    model_key: str = DEFAULT_EMBEDDING_MODEL,
    *,
    recreate: bool = False,
    overwrite_chunks: bool = False,
    profile: _QdrantProfile = QDRANT_ACTIVE,
) -> None:

    run_chunking(batch_stems, model_key, overwrite=overwrite_chunks)
    run_indexing(batch_stems, model_key, recreate=recreate, profile=profile)

    


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="arXiv indexing pipeline: chunk → embed → upsert ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batches", nargs="+", default=["all"],
        metavar="STEM",
        help='Batch stems to process, e.g. batch_01 batch_02, or "all".',
    )
    parser.add_argument(
        "--model", default=DEFAULT_EMBEDDING_MODEL,
        choices=list(EMBEDDING_MODELS),
        help="Embedding model key.",
    )
    parser.add_argument("--recreate",      action="store_true", help="Drop and rebuild the Qdrant collection.")
    parser.add_argument("--overwrite",     action="store_true", help="Re-chunk even if chunks already exist.")
    parser.add_argument("--chunk-only",    action="store_true", help="Only run the chunking stage.")
    parser.add_argument("--index-only",    action="store_true", help="Only run the indexing stage (chunks must exist).")
    args = parser.parse_args()

    if args.chunk_only:
        run_chunking(args.batches, model_key=args.model, overwrite=args.overwrite)
    elif args.index_only:
        run_indexing(args.batches, model_key=args.model, recreate=args.recreate)
    else:
        run_pipeline(
            args.batches,
            model_key=args.model,
            recreate=args.recreate,
            overwrite_chunks=args.overwrite,
        )
