from __future__ import annotations
import argparse
import json
import logging
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
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
    get_qdrant_profile,
    _QdrantProfile,
)
from indexing.utils.storage import write_json
from retrieval.embedder import BGEOutput
from qdrant_client import QdrantClient
from qdrant_client.models import ( 
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    WalConfigDiff,
    PointStruct,
    SparseVector
)


logger = logging.getLogger(__name__)

PREVIOUS_SUFFIX = "_previous"
POSTPROCESSED_SUFFIX = "_postprocessed"


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
                "singularity instance start --bind <qdrant_storage>:/qdrant/storage "
                "--bind <qdrant_snapshots>:/qdrant/snapshots <qdrant.sif> evigraph-qdrant && "
                "singularity exec instance://evigraph-qdrant /qdrant/qdrant"
            )
        raise RuntimeError(
            f"Qdrant is not reachable at {address}.\n"
            f"  Expected startup command: {hint}"
        ) from exc


def ensure_qdrant_runtime(profile: str, startup_timeout: int = 600) -> None:

    if profile == "local":
        _ensure_local_docker_qdrant()
    elif profile != "compose":
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
            "--userns",
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
    log_path = PATHS.root / "logs" / "qdrant_singularity.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a")
    subprocess.Popen(  # non-blocking — Qdrant runs in the background
        [tool, "exec", f"instance://{instance_name}", "/qdrant/qdrant"],
        stdout=log_fh,
        stderr=log_fh,
    )
    logger.info("Qdrant stdout/stderr → %s", log_path)


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
        _start_singularity_instance(
            tool,
            instance_name,
            sif_path,
            str(PATHS.qdrant_storage.resolve()),
            str(PATHS.qdrant_snapshots.resolve()),
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
        "optimizers_config": OptimizersConfigDiff(
            deleted_threshold=profile.optimizer.deleted_threshold,
            vacuum_min_vector_number=profile.optimizer.vacuum_min_vector_number,
            default_segment_number=profile.optimizer.default_segment_number,
            max_segment_size=profile.optimizer.max_segment_size,
            memmap_threshold=profile.optimizer.memmap_threshold,
            indexing_threshold=profile.optimizer.indexing_threshold,
            flush_interval_sec=profile.optimizer.flush_interval_sec,
            max_optimization_threads=profile.optimizer.max_optimization_threads,
        ),
        "wal_config": WalConfigDiff(
            wal_capacity_mb=profile.wal.wal_capacity_mb,
            wal_segments_ahead=profile.wal.wal_segments_ahead,
            wal_retain_closed=profile.wal.wal_retain_closed,
        ),
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



def ensure_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    profile: "_QdrantProfile | None" = None,
) -> None:

    if profile is None:
        profile = QDRANT_ACTIVE

    existing: set[str] = set()
    try:
        info = client.get_collection(collection_name)
        existing = set(
            (info.payload_schema or {}).keys()
        )
    except Exception as exc:
        logger.warning("ensure_payload_indexes: could not fetch collection info: %s", exc)

    for field_name in profile.payload_indexes:
        if field_name in existing:
            logger.debug("ensure_payload_indexes: %r already indexed, skipping", field_name)
            continue
        field_type_str = profile.payload_index_types.get(field_name, "keyword")
        field_schema = (
            PayloadSchemaType.INTEGER
            if field_type_str == "integer"
            else PayloadSchemaType.KEYWORD
        )
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
            logger.info("ensure_payload_indexes: created %r index on %r", field_name, collection_name)
        except Exception as exc:
            logger.warning("ensure_payload_indexes: failed to create %r index: %s", field_name, exc)


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
    # m=0 stops HNSW graph construction (the RAM-heavy part) while leaving the
    # segment merger free to run — this keeps segment count low during bulk
    # upload so startup RAM stays bounded.  Setting indexing_threshold=0 froze
    # the merger too, causing 85+ segments to accumulate and OOM on next start.
    client.update_collection(
        collection_name=collection_name,
        hnsw_config=HnswConfigDiff(m=0),
    )


def enable_hnsw_indexing(client, collection_name: str, m: int) -> None:

    client.update_collection(
        collection_name=collection_name,
        hnsw_config=HnswConfigDiff(m=m),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=10_000),
    )


def create_collection_snapshot(
    client, collection_name: str, max_retries: int = 3, base_timeout: int = 1800
) -> str:
   
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            # Create a new client with extended timeout for this operation
            snapshot_client = qdrant_client(timeout=base_timeout)
            snapshot = snapshot_client.create_snapshot(collection_name=collection_name)
            
            if isinstance(snapshot, dict):
                snapshot_name = snapshot.get("name") or snapshot.get("snapshot_name") or ""
            else:
                snapshot_name = getattr(snapshot, "name", "") or getattr(snapshot, "snapshot_name", "")
            
            if snapshot_name:
                logger.info("Snapshot created successfully: %s", snapshot_name)
                return snapshot_name
            else:
                raise ValueError("Snapshot created but name could not be extracted")
                
        except Exception as exc:
            last_exception = exc
            if attempt < max_retries:
                wait_time = 2 ** (attempt - 1)  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    "Snapshot creation attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt,
                    max_retries,
                    str(exc),
                    wait_time,
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    "Snapshot creation failed after %d attempts: %s",
                    max_retries,
                    str(exc),
                )
    
    raise RuntimeError(
        f"Failed to create snapshot for collection '{collection_name}' "
        f"after {max_retries} retries: {last_exception}"
    ) from last_exception


def path_with_suffix(path: Path, suffix: str) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return path.parent / f"{path.name}{suffix}"


def snapshot_name_with_suffix(snapshot_name: str, suffix: str) -> str:
    return path_with_suffix(Path(snapshot_name), suffix).name


def copy_path(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None

    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst


def move_path(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.move(str(src), str(dst)))


def load_snapshot_metadata(metadata_path: Path | None = None) -> dict[str, Any]:
    target = metadata_path or PATHS.snapshot_metadata
    if not target.exists():
        return {}
    return json.loads(target.read_text())


def backup_previous_qdrant_state(
    *,
    metadata_path: Path = PATHS.snapshot_metadata,
    progress_dir: Path = PATHS.progress,
) -> dict[str, str]:
    metadata_backup = progress_dir / "snapshot_previous.json"
    if metadata_backup.exists():
        existing = load_snapshot_metadata(metadata_backup)
        logger.info(
            "Skipping *_previous backup — snapshot_previous.json already exists (snapshot=%s, captured_at=%s). "
            "Delete it manually to force a new baseline.",
            existing.get("snapshot_name", "?"),
            existing.get("captured_at", "?"),
        )
        return {
            "snapshot_name": str(existing.get("snapshot_name") or ""),
            "metadata_backup": str(metadata_backup),
            "skipped": "already_exists",
        }

    metadata = load_snapshot_metadata(metadata_path)
    snapshot_name = str(metadata.get("snapshot_name") or "")
    written_backup = None
    if snapshot_name:
        metadata["snapshot_name"] = snapshot_name
        metadata["artifact_label"] = "previous"
        metadata["captured_at"] = _now_iso()
        write_json(metadata_backup, metadata)
        written_backup = metadata_backup

    result = {
        "snapshot_name": snapshot_name,
        "metadata_backup": str(written_backup) if written_backup else "",
    }
    logger.info("Backed up previous Qdrant state: %s", result)
    return result


def capture_postprocessed_qdrant_state(
    *,
    profile_name: str,
    snapshots_dir: Path = PATHS.qdrant_snapshots,
    progress_dir: Path = PATHS.progress,
    snapshot_creator: Callable[[str], str] | None = None,
    label: str = "imrad",
) -> dict[str, str]:
    profile = get_qdrant_profile(profile_name)
    snapshot_creator = snapshot_creator or _create_snapshot_for_profile
    snapshot_name = snapshot_creator(profile_name)
    collection_snapshot_dir = snapshots_dir / profile.collection_name
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{label}{POSTPROCESSED_SUFFIX}_{ts}"
    snapshot_path = move_path(
        collection_snapshot_dir / snapshot_name,
        collection_snapshot_dir / snapshot_name_with_suffix(snapshot_name, suffix),
    )
    if snapshot_path is None:
        raise FileNotFoundError(
            f"Created snapshot {snapshot_name!r} was not found in {collection_snapshot_dir}"
        )

    metadata = {
        "collection_name": profile.collection_name,
        "snapshot_name": snapshot_path.name if snapshot_path else "",
        "source_snapshot_name": snapshot_name,
        "snapshot_dir": str(collection_snapshot_dir),
        "artifact_label": "postprocessed",
        "created_at": _now_iso(),
    }
    metadata_path = progress_dir / "snapshot_postprocessed.json"
    write_json(metadata_path, metadata)

    result = {
        "snapshot_path": str(snapshot_path) if snapshot_path else "",
        "metadata_path": str(metadata_path),
    }
    logger.info("Captured postprocessed Qdrant state: %s", result)
    return result


def _create_snapshot_for_profile(profile_name: str) -> str:
    profile = get_qdrant_profile(profile_name)
    ensure_qdrant_runtime(profile.profile)
    client = qdrant_client()
    return create_collection_snapshot(client, profile.collection_name)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_qdrant_artifact_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-mode", choices=("backup-previous", "capture-postprocessed"))
    parser.add_argument("--profile", default="hpc")
    parser.add_argument("--label", default="imrad",
                        help="Postprocessing label embedded in the snapshot name (e.g. 'imrad').")
    return parser


def qdrant_artifact_main(argv: list[str] | None = None) -> dict[str, str] | None:
    parser = build_qdrant_artifact_arg_parser()
    args = parser.parse_args(argv)
    if not args.artifact_mode:
        return None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if args.artifact_mode == "backup-previous":
        return backup_previous_qdrant_state()
    return capture_postprocessed_qdrant_state(profile_name=args.profile, label=args.label)


if __name__ == "__main__":
    qdrant_artifact_main()
