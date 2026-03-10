"""
src/core/indexer.py
~~~~~~~~~~~~~~~~~~~
Qdrant collection setup + embed-and-upsert pipeline.

Two independent entry points
-----------------------------
setup_collection(client, model_key, profile, recreate=False)
    Creates (or re-creates) the Qdrant collection with HNSW, quantisation,
    sparse vectors, payload indexes, and BM25 full-text index.

index_batches(batch_stems, model_key, recreate=False)
    High-level driver: sets up the collection, then for each batch stem
    loads chunks → embeds → upserts into Qdrant.

CLI
---
python -m src.core.indexer --model e5-base-v2 --batches path_to_batch_01 path_to_batch_02
python -m src.core.indexer --model bge-m3     --batches path_to_folder_containing_jsonls --recreate
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    PATHS,
    QDRANT_ACTIVE,
    QDRANT_CONNECTION,
    _QdrantProfile,
)
from src.core.builder import load_chunks, chunks_file_exists
from src.core.embedder import Embedder, BGEOutput

logger = logging.getLogger(__name__)


# Qdrant imports (lazy-checked at call time for cleaner errors)

def _qdrant_client():
    from qdrant_client import QdrantClient  # type: ignore
    conn = QDRANT_CONNECTION
    if conn.url:
        return QdrantClient(url=conn.url, api_key=conn.api_key)
    return QdrantClient(
        host=conn.host,
        port=conn.port,
        grpc_port=conn.grpc_port,
        prefer_grpc=conn.prefer_grpc,
    )


def _check_qdrant_alive(client) -> None:
    """
    Verify Qdrant is reachable.  Raises ``RuntimeError`` with a human-friendly
    message if the connection is refused or times out.
    """
    try:
        client.get_collections()
    except Exception as exc:
        conn = QDRANT_CONNECTION
        addr = conn.url or f"{conn.host}:{conn.port} (HTTP) / {conn.grpc_port} (gRPC)"
        logger.error(
            "Cannot reach Qdrant at %s.\n"
            "  Make sure Qdrant is running before indexing.\n"
            "  Quick start with Docker:\n"
            "    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant\n"
            "  or if on HPC: singularity run --bind $SCRATCH/qdrant_storage:/qdrant/storage qdrant.sif",
            addr,
        )
        raise RuntimeError(
            f"Qdrant is not reachable at {addr}. "
            "Start Qdrant and retry (see logged instructions above)."
        ) from exc


# Payload index type mapping 

_PAYLOAD_INDEX_TYPES: dict[str, str] = {
    # keyword indexes
    "paper_id_arxiv":   "keyword",
    "chunk_type":       "keyword",
    "section_title":    "keyword",
    "paper.categories": "keyword",
    "categories":       "keyword",
    # integer indexes
    "paper.year":       "integer",
    "year":             "integer",
}


# Collection UUID helpers

def _uid_to_uuid(chunk_uid: str) -> str:
    """Convert a 40-char SHA-1 hex string to a deterministic UUID5."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_uid))


# Collection setup 

def setup_collection(
    client,
    model_key: str = DEFAULT_EMBEDDING_MODEL,
    profile: _QdrantProfile = QDRANT_ACTIVE,
    *,
    recreate: bool = False,
) -> None:
    """
    Create the Qdrant collection described by *profile* for *model_key*.

    Idempotent by default (skips if collection already exists).
    Pass ``recreate=True`` to drop and rebuild.
    """
    from qdrant_client.models import (  # type: ignore
        Distance,
        HnswConfigDiff,
        PayloadSchemaType,
        QuantizationConfig,
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
        SparseIndexParams,
        SparseVectorParams,
        TextIndexParams,
        TokenizerType,
        VectorParams,
        VectorsConfig,
    )

    cfg             = EMBEDDING_MODELS[model_key]
    collection_name = profile.collection_name
    distance        = Distance[profile.distance.upper()]
    hnsw            = HnswConfigDiff(
        m            = profile.hnsw.m,
        ef_construct = profile.hnsw.ef_construct,
        full_scan_threshold = profile.hnsw.full_scan_threshold,
    )

    #  Drop if recreate 
    _check_qdrant_alive(client)
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        if recreate:
            logger.info("Dropping existing collection %r …", collection_name)
            client.delete_collection(collection_name)
        else:
            logger.info(
                "Collection %r already exists — skipping setup. "
                "Pass recreate=True to rebuild.",
                collection_name,
            )
            return

    # Quantisation config (HPC only) 
    quant_config: Optional[QuantizationConfig] = None
    if profile.quantize:
        quant_config = ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                always_ram=profile.quantize_always_ram,
            )
        )

    #  Dense vector params 
    dense_params = VectorParams(
        size     = cfg.dim,
        distance = distance,
        hnsw_config = hnsw,
        on_disk     = profile.vectors_on_disk,
        quantization_config = quant_config,
    )

    #  Build vectors_config: named when sparse is enabled 
    if profile.enable_sparse:
        vectors_config = VectorsConfig(
            root={
                "dense": dense_params,
            }
        )
        sparse_vectors_config = {
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=profile.vectors_on_disk)
            )
        }
    else:
        vectors_config        = dense_params          # single unnamed vector
        sparse_vectors_config = None

    #  Create collection 
    logger.info(
        "Creating collection %r  dim=%d  distance=%s  sparse=%s  quantize=%s …",
        collection_name, cfg.dim, profile.distance,
        profile.enable_sparse, profile.quantize,
    )
    create_kwargs: dict = dict(
        collection_name    = collection_name,
        vectors_config     = vectors_config,
        on_disk_payload    = profile.payload_on_disk,
        hnsw_config        = hnsw,
    )
    if sparse_vectors_config:
        create_kwargs["sparse_vectors_config"] = sparse_vectors_config

    client.create_collection(**create_kwargs)
    logger.info("Collection %r created.", collection_name)

    #  Payload indexes 
    for field in profile.payload_indexes:
        field_type_str = _PAYLOAD_INDEX_TYPES.get(field, "keyword")
        schema_type    = (
            PayloadSchemaType.INTEGER
            if field_type_str == "integer"
            else PayloadSchemaType.KEYWORD
        )
        logger.debug("Creating payload index: %s (%s)", field, field_type_str)
        client.create_payload_index(
            collection_name = collection_name,
            field_name      = field,
            field_schema    = schema_type,
        )
    logger.info("Payload indexes created: %s", list(profile.payload_indexes))

    #  Full-text (BM25) index on embed_text 
    client.create_payload_index(
        collection_name = collection_name,
        field_name      = profile.fulltext_field,
        field_schema    = TextIndexParams(
            type      = "text",
            tokenizer = TokenizerType.WORD,
            min_token_len  = 2,
            max_token_len  = 40,
            lowercase      = True,
        ),
    )
    logger.info("Full-text index created on field %r.", profile.fulltext_field)


#  Upsert helpers 

def _build_points_dense(
    chunks: list[dict],
    dense:  np.ndarray,
) -> list:
    """Build PointStruct list for dense-only models (unnamed vector)."""
    from qdrant_client.models import PointStruct  # type: ignore

    points = []
    for chunk, vec in zip(chunks, dense):
        points.append(
            PointStruct(
                id      = _uid_to_uuid(chunk["chunk_uid"]),
                vector  = vec.tolist(),
                payload = chunk,
            )
        )
    return points


def _build_points_bge(
    chunks:  list[dict],
    dense:   np.ndarray,
    sparse:  list[dict],
) -> list:
    """Build PointStruct list for BGE-M3 (named dense + sparse vectors)."""
    from qdrant_client.models import PointStruct, SparseVector  # type: ignore

    points = []
    for chunk, d_vec, s_dict in zip(chunks, dense, sparse):
        indices = list(s_dict.keys())
        values  = [s_dict[i] for i in indices]
        points.append(
            PointStruct(
                id      = _uid_to_uuid(chunk["chunk_uid"]),
                vector  = {
                    "dense":  d_vec.tolist(),
                    "sparse": SparseVector(indices=indices, values=values),
                },
                payload = chunk,
            )
        )
    return points


# index_batches 

def index_batches(
    batch_stems: list[str],
    model_key:   str = DEFAULT_EMBEDDING_MODEL,
    *,
    recreate: bool = False,
    profile:  _QdrantProfile = QDRANT_ACTIVE,
) -> None:
    """
    Full embed + upsert loop for a list of batch stems.

    Parameters
    ----------
    batch_stems:
        List of chunk file stems, e.g. ``["batch_01", "batch_02"]``.
        Pass ``["all"]`` to auto-discover every ``*_chunks.jsonl`` in
        ``PATHS.chunks``.
    model_key:
        Key into ``EMBEDDING_MODELS``.
    recreate:
        If True, drop and rebuild the Qdrant collection before indexing.
    profile:
        Qdrant profile to use (default: ``QDRANT_ACTIVE``).
    """
    # Resolve "all" shorthand
    if batch_stems == ["all"]:
        batch_stems = sorted(
            p.stem.replace("_chunks", "")
            for p in PATHS.chunks.glob("*_chunks.jsonl")
        )
        if not batch_stems:
            logger.warning("No *_chunks.jsonl files found in %s", PATHS.chunks)
            return
        logger.info("Resolved --batches all → %s", batch_stems)

    client    = _qdrant_client()
    _check_qdrant_alive(client)
    setup_collection(client, model_key, profile, recreate=recreate)
    embedder  = Embedder.from_model_key(model_key)
    is_bge    = profile.enable_sparse and EMBEDDING_MODELS[model_key].bge_produces_sparse

    for stem in batch_stems:
        if not chunks_file_exists(stem):
            logger.warning("Chunks file for %r not found — skipping.", stem)
            continue

        chunks = load_chunks(stem)
        if not chunks:
            logger.info("Batch %r has no chunks — skipping.", stem)
            continue

        logger.info("Batch %r — %d chunks to index …", stem, len(chunks))
        batch_size = profile.upsert_batch_size

        for i in tqdm(
            range(0, len(chunks), batch_size),
            desc=f"Upserting {stem}",
            unit="batch",
        ):
            sub_chunks = chunks[i : i + batch_size]
            texts      = [c["embed_text"] for c in sub_chunks]

            embed_result = embedder.embed_passages(texts)

            if is_bge:
                assert isinstance(embed_result, BGEOutput)
                points = _build_points_bge(sub_chunks, embed_result.dense, embed_result.sparse)
            else:
                assert isinstance(embed_result, np.ndarray)
                points = _build_points_dense(sub_chunks, embed_result)

            client.upsert(
                collection_name = profile.collection_name,
                points          = points,
                wait            = True,
            )

        logger.info("Batch %r — upserted %d points.", stem, len(chunks))

    info = client.get_collection(profile.collection_name)
    points_count = getattr(info, "points_count", None)
    indexed_vectors_count = getattr(info, "indexed_vectors_count", None)

    logger.info(
        "Collection %r stats: points=%s, indexed_vectors=%s",
        profile.collection_name,
        points_count,
        indexed_vectors_count,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

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
        "--batches", nargs="+", required=True,
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
        help="Only create the collection, do not embed or upsert",
    )
    args = parser.parse_args()

    try:
        if args.setup_only:
            c = _qdrant_client()
            setup_collection(c, args.model, recreate=args.recreate)
        else:
            index_batches(
                batch_stems = args.batches,
                model_key   = args.model,
                recreate    = args.recreate,
            )
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
