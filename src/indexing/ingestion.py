from __future__ import annotations

import logging
from datetime import UTC, datetime
from config.settings import PATHS, get_qdrant_profile
from indexing.storage import append_jsonl, read_jsonl, shard_artifacts, write_json
from utils.qdrant import (
    build_points_from_shard_records,
    create_collection_snapshot,
    get_collection_info,
    qdrant_client,
    setup_collection,
)

logger = logging.getLogger(__name__)


def ingest_shards(
    *,
    shard_stems: list[str],
    model_key: str,
    profile_name: str,
    recreate_collection: bool,
    resume: bool,
    snapshot_interval: int = 100,
) -> None:
    """Ingest shards with periodic snapshots for HPC 7-day recovery.
    
    Args:
        shard_stems: List of shard identifiers to ingest
        model_key: Embedding model identifier
        profile_name: Qdrant profile name
        recreate_collection: Whether to recreate the collection
        resume: Whether to resume from checkpoint
        snapshot_interval: Create snapshot every N shards (default 100)
    """
    profile = get_qdrant_profile(profile_name)
    client = qdrant_client()
    setup_collection(
        client,
        model_key=model_key,
        profile=profile,
        recreate=recreate_collection,
    )

    ingested = _load_ingested_stems() if resume else set()
    ingestion_count = 0
    
    for stem in shard_stems:
        if stem in ingested:
            logger.info("Skipping already ingested shard %s", stem)
            continue

        artifacts = shard_artifacts(PATHS.shards, stem)
        if not artifacts.records_path.exists():
            logger.warning("Shard file missing for %s", stem)
            continue

        records = list(read_jsonl(artifacts.records_path))
        if not records:
            logger.info("Shard %s is empty, skipping", stem)
            continue

        logger.info("Ingesting shard %s with %d record(s)", stem, len(records))
        for index in range(0, len(records), profile.upsert_batch_size):
            batch = records[index:index + profile.upsert_batch_size]
            points = build_points_from_shard_records(batch, profile)
            client.upsert(
                collection_name=profile.collection_name,
                points=points,
                wait=True,
            )

        append_jsonl(
            PATHS.ingested_shards,
            {
                "stem": stem,
                "status": "INGESTED",
                "rows": len(records),
                "timestamp": _now_iso(),
            },
        )
        
        ingestion_count += 1
        
        # Create periodic snapshot for HPC recovery
        if ingestion_count % snapshot_interval == 0:
            snapshot_name = _create_periodic_snapshot(client, profile.collection_name, ingestion_count)
            logger.info("Periodic snapshot created after %d shards: %s", ingestion_count, snapshot_name)

    # Final snapshot after all ingestion
    snapshot_name = _create_periodic_snapshot(client, profile.collection_name, ingestion_count)
    stats = get_collection_info(client, profile.collection_name)
    logger.info("Ingestion complete: %s", stats)
    logger.info("Final snapshot: %s", snapshot_name)


def write_snapshot_metadata(profile_name: str) -> str:
    profile = get_qdrant_profile(profile_name)
    client = qdrant_client()
    snapshot_name = create_collection_snapshot(client, profile.collection_name)
    metadata = {
        "collection_name": profile.collection_name,
        "snapshot_name": snapshot_name,
        "snapshot_dir": str(PATHS.qdrant_snapshots),
        "created_at": _now_iso(),
    }
    write_json(PATHS.snapshot_metadata, metadata)
    return snapshot_name


def _create_periodic_snapshot(client, collection_name: str, shard_count: int) -> str:
    """Create snapshot and record metadata for periodic checkpoint.
    
    Args:
        client: Qdrant client instance
        collection_name: Qdrant collection name
        shard_count: Number of shards ingested so far
        
    Returns:
        Snapshot name (e.g., "2024-01-15T10-30-45-123456Z-shard-100")
    """
    snapshot_name = create_collection_snapshot(client, collection_name)
    
    # Append to snapshots metadata file to keep history
    append_jsonl(
        PATHS.qdrant_snapshots / "manifest.jsonl",
        {
            "snapshot_name": snapshot_name,
            "shard_count": shard_count,
            "collection_name": collection_name,
            "created_at": _now_iso(),
        },
    )
    
    return snapshot_name


def _load_ingested_stems() -> set[str]:
    return {
        row["stem"]
        for row in read_jsonl(PATHS.ingested_shards)
        if row.get("status") == "INGESTED" and row.get("stem")
    }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
