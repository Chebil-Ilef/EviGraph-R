"""
src/core/indexer.py
~~~~~~~~~~~~~~~~~~~
Orchestrates the embed-and-upsert pipeline.

Collection schema and point-building live in src/utils/qdrant.py.
This module owns only the high-level loop and CLI.

Entry points
------------
index_batches(batch_stems, model_key, recreate=False)
    Setup collection → for each batch stem: load chunks → embed → upsert.

CLI
---
python -m src.core.indexer --batches batch_01 batch_02
python -m src.core.indexer --batches all --recreate
python -m src.core.indexer --setup-only
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from tqdm import tqdm

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    PATHS,
    QDRANT_ACTIVE,
    _QdrantProfile,
)
from src.core.builder import load_chunks, chunks_file_exists
from src.core.embedder import Embedder
from src.utils.qdrant import setup_collection, build_points, get_collection_info, qdrant_client, check_qdrant_alive

logger = logging.getLogger(__name__)


def index_batches(
    batch_stems: list[str],
    model_key: str = DEFAULT_EMBEDDING_MODEL,
    *,
    recreate: bool = False,
    profile: _QdrantProfile = QDRANT_ACTIVE,
) -> None:
    """
    Full embed + upsert loop for a list of batch stems.

    Parameters
    ----------
    batch_stems:
        Chunk file stems, e.g. ``["batch_01", "batch_02"]``.
        Pass ``["all"]`` to auto-discover every ``*_chunks.jsonl`` in PATHS.chunks.
    model_key:
        Key into ``EMBEDDING_MODELS``.
    recreate:
        If True, drop and rebuild the Qdrant collection before indexing.
    profile:
        Qdrant profile to use (default: ``QDRANT_ACTIVE``).
    """
    if batch_stems == ["all"]:
        batch_stems = sorted(
            p.stem.replace("_chunks", "")
            for p in PATHS.chunks.glob("*_chunks.jsonl")
        )
        if not batch_stems:
            logger.warning("No *_chunks.jsonl files found in %s", PATHS.chunks)
            return
        logger.info("Resolved --batches all → %s", batch_stems)

    client = qdrant_client()
    check_qdrant_alive(client)
    setup_collection(client, model_key, profile, recreate=recreate)
    embedder = Embedder.from_model_key(model_key)

    for stem in batch_stems:
        if not chunks_file_exists(stem):
            logger.warning("Chunks file for %r not found — skipping.", stem)
            continue

        chunks = load_chunks(stem)
        if not chunks:
            logger.info("Batch %r has no chunks — skipping.", stem)
            continue

        logger.info("Batch %r — %d chunks to index …", stem, len(chunks))

        for i in tqdm(
            range(0, len(chunks), profile.upsert_batch_size),
            desc=f"Upserting {stem}",
            unit="batch",
        ):
            sub_chunks   = chunks[i : i + profile.upsert_batch_size]
            texts        = [c["embed_text"] for c in sub_chunks]
            embed_result = embedder.embed_passages(texts)
            points       = build_points(sub_chunks, embed_result, profile, model_key)

            client.upsert(
                collection_name=profile.collection_name,
                points=points,
                wait=True,
            )

        logger.info("Batch %r — upserted %d points.", stem, len(chunks))

    stats = get_collection_info(client, profile.collection_name)
    logger.info(
        "Collection %r — points=%s  indexed_vectors=%s",
        profile.collection_name,
        stats["points_count"],
        stats["indexed_vectors_count"],
    )



if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    for _noisy in ("transformers", "sentence_transformers", "urllib3", "filelock"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Embed chunks and upsert into Qdrant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default=DEFAULT_EMBEDDING_MODEL,
        choices=list(EMBEDDING_MODELS),
        help="Embedding model key",
    )
    parser.add_argument(
        "--batches", nargs="+", required=False, default=None,
        metavar="STEM",
        help=(
            "Chunk batch stems to index, e.g. batch_01 batch_02. "
            "Pass 'all' to index every *_chunks.jsonl in _data/chunks/."
        ),
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and recreate the Qdrant collection before indexing",
    )
    parser.add_argument(
        "--setup-only", action="store_true",
        help="Only create the collection schema, do not embed or upsert",
    )
    args = parser.parse_args()

    try:
        if args.setup_only:
            c = qdrant_client()
            check_qdrant_alive(c)
            setup_collection(c, args.model, recreate=args.recreate)
        else:
            if not args.batches:
                parser.error("--batches is required unless --setup-only is used")
            index_batches(
                batch_stems=args.batches,
                model_key=args.model,
                recreate=args.recreate,
            )
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
