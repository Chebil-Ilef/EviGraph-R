from __future__ import annotations
import logging
import os
import shutil
import subprocess
import uuid
from typing import Union
import numpy as np
import time
from config.settings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    PATHS,
    QDRANT_ACTIVE,
    QDRANT_CONNECTION,
    QDRANT_RUNTIME,
    _QdrantProfile,
)
from retrieval.embedder import BGEOutput
from qdrant_client import QdrantClient
from qdrant_client.models import ( 
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVectorParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
    PointStruct, 
    SparseVector 
)


logger = logging.getLogger(__name__)


# CLIENT & CONNECTION MANAGEMENT

def qdrant_client(timeout: int = 300):

    connection = QDRANT_CONNECTION
    if connection.url:
        return QdrantClient(url=connection.url, api_key=connection.api_key, timeout=timeout)

    return QdrantClient(
        host=connection.host,
        port=connection.port,
        grpc_port=connection.grpc_port,
        prefer_grpc=connection.prefer_grpc,
        timeout=timeout,
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
                "singularity run --bind <qdrant_storage>:/qdrant/storage "
                "--bind <qdrant_snapshots>:/qdrant/snapshots <qdrant.sif>"
            )
        raise RuntimeError(
            f"Qdrant is not reachable at {address}.\n"
            f"  Expected startup command: {hint}"
        ) from exc


def ensure_qdrant_runtime(profile: str, startup_timeout: int = 30) -> None:

    if profile == "local":
        _ensure_local_docker_qdrant()
    else:
        _ensure_hpc_singularity_instance()

    client = qdrant_client()
    deadline = time.monotonic() + startup_timeout
    last_exc: Exception | None = None
    attempt = 0
    while time.monotonic() < deadline:
        try:
            client.get_collections()
            if attempt > 0:
                logger.info("Qdrant reachable after %.0fs", time.monotonic() - (deadline - startup_timeout))
            return
        except Exception as exc:
            last_exc = exc
            attempt += 1
            wait = 2
            logger.info("Qdrant not ready yet (attempt %d), retrying in %ds…", attempt, wait)
            time.sleep(wait)

    check_qdrant_alive(client, profile=profile)


# DOCKER RUNTIME (LOCAL DEVELOPMENT)

def _ensure_local_docker_qdrant() -> None:

    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for local Qdrant runs but is not installed.")

    container_name = QDRANT_RUNTIME.local_container_name
    image = QDRANT_RUNTIME.local_image

    status = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode == 0 and status.stdout.strip() == "true":
        return

    if status.returncode == 0:
        logger.info("Starting existing Qdrant container %s", container_name)
        subprocess.run(["docker", "start", container_name], check=True)
        return

    logger.info("Creating local Qdrant container %s", container_name)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{QDRANT_CONNECTION.port}:6333",
            "-p",
            f"{QDRANT_CONNECTION.grpc_port}:6334",
            "-v",
            f"{PATHS.qdrant_storage.resolve()}:/qdrant/storage",
            "-v",
            f"{PATHS.qdrant_snapshots.resolve()}:/qdrant/snapshots",
            image,
        ],
        check=True,
    )

# SINGULARITY RUNTIME (HPC)

def _get_singularity_tool() -> str:

    if shutil.which("singularity"):
        logger.debug("Found Singularity container tool")
        return "singularity"
    raise RuntimeError(
        "HPC profile requires Singularity on PATH. "
        "See documentation/hpc.md for setup."
    )


def _parse_instance_list(output: str) -> set[str]:

    instances: set[str] = set()
    lines = output.strip().split("\n")
    if len(lines) < 2:  # Header only
        return instances
    
    for line in lines[1:]:
        parts = line.split()
        if parts:
            instances.add(parts[0])
    return instances


def _get_running_instances(tool: str) -> set[str]:

    result = subprocess.run(
        [tool, "instance", "list"],
        capture_output=True,
        text=True,
        check=True,
    )
    return _parse_instance_list(result.stdout)


def _instance_is_running(tool: str, name: str) -> bool:

    instances = _get_running_instances(tool)
    return name in instances


def _start_singularity_instance(
    tool: str,
    name: str,
    sif_path: str,
    storage_bind: str,
    snapshots_bind: str,
) -> None:

    logger.info(
        "Starting Singularity instance %r with binds: storage=%s snapshots=%s",
        name,
        storage_bind,
        snapshots_bind,
    )
    subprocess.run(
        [
            tool,
            "instance",
            "start",
            "--bind", f"{storage_bind}:/qdrant/storage",
            "--bind", f"{snapshots_bind}:/qdrant/snapshots",
            sif_path,
            name,
        ],
        check=True,
    )
    logger.info("Singularity instance %r started successfully", name)


def _start_qdrant_in_instance(tool: str, instance_name: str) -> None:

    logger.info("Launching Qdrant server inside Singularity instance %r", instance_name)
    subprocess.Popen(  # non-blocking — Qdrant runs in the background
        [tool, "exec", f"instance://{instance_name}", "/qdrant/qdrant"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _ensure_hpc_singularity_instance() -> None:
   
    try:
        tool = _get_singularity_tool()
    except RuntimeError as e:
        logger.error(str(e))
        raise

    required_env = {
        "SINGULARITY_CACHEDIR": os.getenv("SINGULARITY_CACHEDIR"),
        "SINGULARITY_TMPDIR": os.getenv("SINGULARITY_TMPDIR"),
    }
    missing = [key for key, value in required_env.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing HPC environment variables: " + ", ".join(missing)
            + ". See documentation/hpc.md for setup."
        )

    # Validate paths
    if not PATHS.qdrant_storage.exists():
        raise RuntimeError(f"Qdrant storage path does not exist: {PATHS.qdrant_storage}")
    if not PATHS.qdrant_snapshots.exists():
        raise RuntimeError(f"Qdrant snapshots path does not exist: {PATHS.qdrant_snapshots}")

    instance_name = QDRANT_RUNTIME.hpc_instance_name
    sif_path = QDRANT_RUNTIME.hpc_sif_path

    # Check if already running
    if _instance_is_running(tool, instance_name):
        logger.info("Singularity instance %r already running", instance_name)
        return

    # Get list of all instances (running and stopped)
    result = subprocess.run(
        [tool, "instance", "list"],
        capture_output=True,
        text=True,
        check=True,
    )
    all_instances = _parse_instance_list(result.stdout)

    if instance_name in all_instances:
        logger.info("Restarting stopped Singularity instance %r", instance_name)
        subprocess.run(
            [tool, "instance", "start", sif_path, instance_name],
            check=True,
        )
        logger.info("Singularity instance %r restarted successfully", instance_name)
        _start_qdrant_in_instance(tool, instance_name)
        return

    _start_singularity_instance(
        tool,
        instance_name,
        sif_path,
        str(PATHS.qdrant_storage.resolve()),
        str(PATHS.qdrant_snapshots.resolve()),
    )
    _start_qdrant_in_instance(tool, instance_name)


# COLLECTION MANAGEMENT

def setup_collection(
    client,
    model_key: str = DEFAULT_EMBEDDING_MODEL,
    profile: _QdrantProfile = QDRANT_ACTIVE,
    *,
    recreate: bool = False,
) -> None:

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
            type=TextIndexType.TEXT,
            tokenizer=TokenizerType[profile.fulltext_tokenizer.upper()],
            min_token_len=profile.fulltext_min_token_len,
            max_token_len=profile.fulltext_max_token_len,
            lowercase=profile.fulltext_lowercase,
        ),
    )


# POINT BUILDING & INSERTION

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


# UTILITIES & HELPERS

def uid_to_uuid(chunk_uid: str) -> str:

    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_uid))


def get_collection_info(client, collection_name: str) -> dict:

    info = client.get_collection(collection_name)
    return {
        "collection_name": collection_name,
        "points_count": getattr(info, "points_count", None),
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        "status": str(getattr(info, "status", None)),
    }


def disable_hnsw_indexing(client, collection_name: str) -> None:

    client.update_collection(
        collection_name=collection_name,
        hnsw_config=HnswConfigDiff(m=0),
    )


def enable_hnsw_indexing(client, collection_name: str, m: int) -> None:

    client.update_collection(
        collection_name=collection_name,
        hnsw_config=HnswConfigDiff(m=m),
    )


def create_collection_snapshot(client, collection_name: str) -> str:
   
    snapshot = client.create_snapshot(collection_name=collection_name)
    if isinstance(snapshot, dict):
        return snapshot.get("name") or snapshot.get("snapshot_name") or ""
    return getattr(snapshot, "name", "") or getattr(snapshot, "snapshot_name", "")
