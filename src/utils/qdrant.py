
from __future__ import annotations
import uuid
import logging
from typing import Union
import numpy as np

from src.core.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    QDRANT_ACTIVE,
    QDRANT_CONNECTION,
    _QdrantProfile,
)
from src.core.embedder import BGEOutput

logger = logging.getLogger(__name__)





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
        m=profile.hnsw.m,
        ef_construct=profile.hnsw.ef_construct,
        full_scan_threshold=profile.hnsw.full_scan_threshold,
        on_disk=profile.vectors_on_disk,
    )

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
    quant_config = None
    if profile.quantize:
        quant_config = ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType[profile.quantize_scalar_type.upper()],
                always_ram=profile.quantize_always_ram,
            )
        )

    dense_params = VectorParams(
        size=cfg.dim,
        distance=distance,
        hnsw_config=hnsw,
        on_disk=profile.vectors_on_disk,
        quantization_config=quant_config,
    )

    if profile.enable_sparse:
        vectors_config = {"dense": dense_params}
        sparse_vectors_config = {
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=profile.vectors_on_disk)
            )
        }
    else:
        vectors_config = dense_params
        sparse_vectors_config = None

    logger.info(
        "Creating collection %r  dim=%d  distance=%s  sparse=%s  quantize=%s …",
        collection_name, cfg.dim, profile.distance,
        profile.enable_sparse, profile.quantize,
    )

    create_kwargs: dict = dict(
        collection_name=collection_name,
        vectors_config=vectors_config,
        on_disk_payload=profile.payload_on_disk,
        hnsw_config=hnsw,
    )
    if sparse_vectors_config:
        create_kwargs["sparse_vectors_config"] = sparse_vectors_config

    client.create_collection(**create_kwargs)
    logger.info("Collection %r created.", collection_name)

    # Payload indexes
    for field_name in profile.payload_indexes:
        field_type_str = profile.payload_index_types.get(field_name, "keyword")
        schema_type = (
            PayloadSchemaType.INTEGER
            if field_type_str == "integer"
            else PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )
    logger.info("Payload indexes created: %s", list(profile.payload_indexes))

    # Full-text (BM25) index on embed_text
    client.create_payload_index(
        collection_name=collection_name,
        field_name=profile.fulltext_field,
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType[profile.fulltext_tokenizer.upper()],
            min_token_len=profile.fulltext_min_token_len,
            max_token_len=profile.fulltext_max_token_len,
            lowercase=profile.fulltext_lowercase,
        ),
    )
    logger.info("Full-text index created on field %r.", profile.fulltext_field)



def build_points(
    chunks: list[dict],
    embed_result: Union[np.ndarray, BGEOutput],
    profile: _QdrantProfile = QDRANT_ACTIVE,
    model_key: str = DEFAULT_EMBEDDING_MODEL,
) -> list:
    """
    Convert chunk dicts + embedding output into a list of PointStructs ready
    for ``client.upsert()``.

    Handles both dense-only (E5, Qwen, Jina) and dense+sparse (BGE-M3).
    """
    is_bge = profile.enable_sparse and EMBEDDING_MODELS[model_key].bge_produces_sparse

    if is_bge:
        assert isinstance(embed_result, BGEOutput)
        return _build_points_bge(chunks, embed_result.dense, embed_result.sparse)
    else:
        # For dense-only storage, extract dense vectors if using BGE-M3 without sparse
        if isinstance(embed_result, BGEOutput):
            return _build_points_dense(chunks, embed_result.dense)
        assert isinstance(embed_result, np.ndarray)
        return _build_points_dense(chunks, embed_result)


def _build_points_dense(chunks: list[dict], dense: np.ndarray) -> list:
    from qdrant_client.models import PointStruct  # type: ignore

    return [
        PointStruct(
            id=uid_to_uuid(chunk["chunk_uid"]),
            vector=vec.tolist(),
            payload=chunk,
        )
        for chunk, vec in zip(chunks, dense)
    ]


def _build_points_bge(
    chunks: list[dict],
    dense: np.ndarray,
    sparse: list[dict],
) -> list:
    from qdrant_client.models import PointStruct, SparseVector  # type: ignore

    points = []
    for chunk, d_vec, s_dict in zip(chunks, dense, sparse):
        indices = list(s_dict.keys())
        values  = [s_dict[i] for i in indices]
        points.append(
            PointStruct(
                id=uid_to_uuid(chunk["chunk_uid"]),
                vector={
                    "dense":  d_vec.tolist(),
                    "sparse": SparseVector(indices=indices, values=values),
                },
                payload=chunk,
            )
        )
    return points



def get_collection_info(client, collection_name: str) -> dict:
    """Return a small stats dict for the named collection."""
    info = client.get_collection(collection_name)
    return {
        "collection_name":      collection_name,
        "points_count":         getattr(info, "points_count", None),
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        "status":               str(getattr(info, "status", None)),
    }


def uid_to_uuid(chunk_uid: str) -> str:
    """Convert a 40-char SHA-1 hex string to a deterministic UUID5."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_uid))


def qdrant_client():
    """Build a QdrantClient from QDRANT_CONNECTION config."""
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


def check_qdrant_alive(client) -> None:
    """Raise RuntimeError with startup hints if Qdrant is unreachable."""
    try:
        client.get_collections()
    except Exception as exc:
        conn = QDRANT_CONNECTION
        addr = conn.url or f"{conn.host}:{conn.port}"
        raise RuntimeError(
            f"Qdrant is not reachable at {addr}.\n"
            f"  Local:  docker run -p {conn.port}:{conn.port} "
            f"-p {conn.grpc_port}:{conn.grpc_port} qdrant/qdrant\n"
            f"  HPC:    singularity run --bind $SCRATCH/qdrant_storage:"
            f"/qdrant/storage qdrant.sif"
        ) from exc
