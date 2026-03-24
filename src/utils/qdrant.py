from __future__ import annotations

import logging
import uuid
from typing import Union

import numpy as np

try:
    from config.settings import (
        DEFAULT_EMBEDDING_MODEL,
        EMBEDDING_MODELS,
        QDRANT_ACTIVE,
        QDRANT_CONNECTION,
        _QdrantProfile,
    )
    from retrieval.embedder import BGEOutput
except ModuleNotFoundError:
    from src.config.settings import (
        DEFAULT_EMBEDDING_MODEL,
        EMBEDDING_MODELS,
        QDRANT_ACTIVE,
        QDRANT_CONNECTION,
        _QdrantProfile,
    )
    from src.retrieval.embedder import BGEOutput

logger = logging.getLogger(__name__)


def setup_collection(
    client,
    model_key: str = DEFAULT_EMBEDDING_MODEL,
    profile: _QdrantProfile = QDRANT_ACTIVE,
    *,
    recreate: bool = False,
) -> None:
    from qdrant_client.models import (  # type: ignore
        Distance,
        HnswConfigDiff,
        PayloadSchemaType,
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
        SparseIndexParams,
        SparseVectorParams,
        TextIndexParams,
        TokenizerType,
        VectorParams,
    )

    cfg = EMBEDDING_MODELS[model_key]
    collection_name = profile.collection_name
    distance = Distance[profile.distance.upper()]

    hnsw = HnswConfigDiff(
        m=profile.hnsw.m,
        ef_construct=profile.hnsw.ef_construct,
        full_scan_threshold=profile.hnsw.full_scan_threshold,
        on_disk=profile.vectors_on_disk,
    )

    existing = {collection.name for collection in client.get_collections().collections}
    if collection_name in existing:
        if recreate:
            logger.info("Dropping existing collection %r", collection_name)
            client.delete_collection(collection_name)
        else:
            logger.info("Collection %r already exists, skipping setup", collection_name)
            return

    quantization_config = None
    if profile.quantize:
        quantization_config = ScalarQuantization(
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
        quantization_config=quantization_config,
    )

    create_kwargs = {
        "collection_name": collection_name,
        "vectors_config": {profile.dense_vector_name: dense_params},
        "on_disk_payload": profile.payload_on_disk,
    }
    if cfg.bge_produces_sparse:
        create_kwargs["sparse_vectors_config"] = {
            profile.sparse_vector_name: SparseVectorParams(
                index=SparseIndexParams(on_disk=profile.vectors_on_disk)
            )
        }
    client.create_collection(**create_kwargs)
    logger.info("Collection %r created", collection_name)

    for field_name in profile.payload_indexes:
        field_type_str = profile.payload_index_types.get(field_name, "keyword")
        field_schema = (
            PayloadSchemaType.INTEGER
            if field_type_str == "integer"
            else PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )

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


def build_points(
    chunks: list[dict],
    embed_result: Union[np.ndarray, BGEOutput],
    profile: _QdrantProfile = QDRANT_ACTIVE,
    model_key: str = DEFAULT_EMBEDDING_MODEL,
) -> list:
    is_sparse_model = EMBEDDING_MODELS[model_key].bge_produces_sparse
    if is_sparse_model:
        assert isinstance(embed_result, BGEOutput)
        return _build_points_bge(chunks, embed_result.dense, embed_result.sparse, profile)

    if isinstance(embed_result, BGEOutput):
        return _build_points_dense(chunks, embed_result.dense, profile)

    assert isinstance(embed_result, np.ndarray)
    return _build_points_dense(chunks, embed_result, profile)


def embed_result_to_serializable(embed_result: Union[np.ndarray, BGEOutput]) -> list[dict]:
    if isinstance(embed_result, BGEOutput):
        dense = embed_result.dense.tolist()
        return [
            {
                "dense": dense_row,
                "sparse": {
                    "indices": list(sparse_row.keys()),
                    "values": list(sparse_row.values()),
                },
            }
            for dense_row, sparse_row in zip(dense, embed_result.sparse)
        ]

    return [{"dense": row} for row in embed_result.tolist()]


def build_points_from_shard_records(
    records: list[dict],
    profile: _QdrantProfile = QDRANT_ACTIVE,
) -> list:
    from qdrant_client.models import PointStruct, SparseVector  # type: ignore

    points = []
    for record in records:
        vectors = record["vectors"]
        payload = record["payload"]
        vector_payload = {profile.dense_vector_name: vectors["dense"]}

        sparse = vectors.get("sparse")
        if sparse and sparse.get("indices"):
            vector_payload[profile.sparse_vector_name] = SparseVector(
                indices=sparse["indices"],
                values=sparse["values"],
            )

        points.append(
            PointStruct(
                id=uid_to_uuid(record["chunk_uid"]),
                vector=vector_payload,
                payload=payload,
            )
        )
    return points


def _build_points_dense(chunks: list[dict], dense: np.ndarray, profile: _QdrantProfile) -> list:
    from qdrant_client.models import PointStruct  # type: ignore

    return [
        PointStruct(
            id=uid_to_uuid(chunk["chunk_uid"]),
            vector={profile.dense_vector_name: vec.tolist()},
            payload=chunk,
        )
        for chunk, vec in zip(chunks, dense)
    ]


def _build_points_bge(
    chunks: list[dict],
    dense: np.ndarray,
    sparse: list[dict],
    profile: _QdrantProfile,
) -> list:
    from qdrant_client.models import PointStruct, SparseVector  # type: ignore

    points = []
    for chunk, dense_vec, sparse_dict in zip(chunks, dense, sparse):
        indices = list(sparse_dict.keys())
        values = [sparse_dict[index] for index in indices]
        points.append(
            PointStruct(
                id=uid_to_uuid(chunk["chunk_uid"]),
                vector={
                    profile.dense_vector_name: dense_vec.tolist(),
                    profile.sparse_vector_name: SparseVector(indices=indices, values=values),
                },
                payload=chunk,
            )
        )
    return points


def get_collection_info(client, collection_name: str) -> dict:
    info = client.get_collection(collection_name)
    return {
        "collection_name": collection_name,
        "points_count": getattr(info, "points_count", None),
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        "status": str(getattr(info, "status", None)),
    }


def uid_to_uuid(chunk_uid: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_uid))


def qdrant_client():
    from qdrant_client import QdrantClient  # type: ignore

    connection = QDRANT_CONNECTION
    if connection.url:
        return QdrantClient(url=connection.url, api_key=connection.api_key)

    return QdrantClient(
        host=connection.host,
        port=connection.port,
        grpc_port=connection.grpc_port,
        prefer_grpc=connection.prefer_grpc,
    )


def check_qdrant_alive(client, *, profile: str | None = None) -> None:
    try:
        client.get_collections()
    except Exception as exc:
        connection = QDRANT_CONNECTION
        address = connection.url or f"{connection.host}:{connection.port}"
        runtime_profile = profile or QDRANT_ACTIVE.profile
        if runtime_profile == "local":
            hint = (
                f"docker run -d --name evigraph-qdrant "
                f"-p {connection.port}:6333 -p {connection.grpc_port}:6334 "
                f"-v <qdrant_storage>:/qdrant/storage "
                f"-v <qdrant_snapshots>:/qdrant/snapshots qdrant/qdrant"
            )
        else:
            hint = (
                "apptainer run --bind <qdrant_storage>:/qdrant/storage "
                "--bind <qdrant_snapshots>:/qdrant/snapshots <qdrant.sif>"
            )
        raise RuntimeError(
            f"Qdrant is not reachable at {address}.\n"
            f"  Expected startup command: {hint}"
        ) from exc


def create_collection_snapshot(client, collection_name: str) -> str:
    snapshot = client.create_snapshot(collection_name=collection_name)
    if isinstance(snapshot, dict):
        return snapshot.get("name") or snapshot.get("snapshot_name") or ""
    return getattr(snapshot, "name", "") or getattr(snapshot, "snapshot_name", "")
