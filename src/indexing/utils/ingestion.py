from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from config.settings import PATHS, get_qdrant_profile
from indexing.utils.storage import append_jsonl, read_jsonl, shard_artifacts, write_json
from utils.qdrant import (
    build_points_from_shard_records,
    create_collection_snapshot,
    disable_hnsw_indexing,
    enable_hnsw_indexing,
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

    profile = get_qdrant_profile(profile_name)
    client = qdrant_client()
    setup_collection(
        client,
        model_key=model_key,
        profile=profile,
        recreate=recreate_collection,
    )

    ingested = _load_ingested_stems() if (resume and not recreate_collection) else set()
    ingestion_count = 0

    pending = [s for s in shard_stems if s not in ingested]
    if not pending:
        logger.info("All shards already ingested, nothing to do")
    else:
        logger.info("Disabling HNSW indexing for bulk upload (%d shards)", len(pending))
        disable_hnsw_indexing(client, profile.collection_name)

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
            batch = records[index : index + profile.upsert_batch_size]
            points = build_points_from_shard_records(batch, profile)
            client.upsert(
                collection_name=profile.collection_name,
                points=points,
                wait=False,
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

    if pending:
        logger.info("Re-enabling HNSW indexing (m=%d) — index build will proceed in background", profile.hnsw.m)
        enable_hnsw_indexing(client, profile.collection_name, profile.hnsw.m)
        settled_stats = _wait_for_collection_green(client, profile.collection_name)
    else:
        settled_stats = None

    snapshot_name = _create_periodic_snapshot(client, profile.collection_name, ingestion_count)
    stats = settled_stats or get_collection_info(client, profile.collection_name)
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

    snapshot_name = create_collection_snapshot(client, collection_name)
    
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
    return datetime.now(timezone.utc).isoformat()


def _wait_for_collection_green(
    client,
    collection_name: str,
    *,
    timeout_sec: int = 1800,
    poll_interval_sec: int = 5,
) -> dict | None:
    deadline = time.monotonic() + timeout_sec
    last_info: dict | None = None

    while time.monotonic() < deadline:
        info = get_collection_info(client, collection_name)
        if not isinstance(info, dict):
            logger.warning(
                "Collection status check returned %r, skipping optimizer wait",
                type(info).__name__,
            )
            return None

        last_info = info
        status = str(info.get("status") or "").lower()
        if status == "green":
            logger.info("Collection %s reached green status", collection_name)
            return info

        logger.info(
            "Waiting for collection %s to finish indexing: status=%s indexed_vectors=%s points=%s",
            collection_name,
            status or "unknown",
            info.get("indexed_vectors_count"),
            info.get("points_count"),
        )
        time.sleep(poll_interval_sec)

    logger.warning(
        "Collection %s did not reach green status within %ss; proceeding with latest stats: %s",
        collection_name,
        timeout_sec,
        last_info,
    )
    return last_info
