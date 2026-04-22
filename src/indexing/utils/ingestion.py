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
    client = qdrant_client(timeout=600)  # 10 minute timeout for bulk operations
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

        record_count = 0
        batch: list = []
        batch_count = 0
        for record in read_jsonl(artifacts.records_path):
            batch.append(record)
            record_count += 1
            if len(batch) >= profile.upsert_batch_size:
                points = build_points_from_shard_records(batch, profile)
                # Upsert with explicit wait every 5 batches to avoid buffer overflow
                wait_on_this_batch = (batch_count % 5 == 4)
                client.upsert(
                    collection_name=profile.collection_name,
                    points=points,
                    wait=wait_on_this_batch,
                )
                batch = []
                batch_count += 1
                # Small pause every 10 batches to give Qdrant time to process
                if batch_count % 10 == 0:
                    logger.debug("Pausing after %d batches to allow Qdrant processing", batch_count)
                    time.sleep(0.5)
        if not record_count:
            logger.info("Shard %s is empty, skipping", stem)
            continue
        if batch:
            points = build_points_from_shard_records(batch, profile)
            # Final batch always waits for completion
            client.upsert(
                collection_name=profile.collection_name,
                points=points,
                wait=True,
            )
        logger.info("Ingested shard %s with %d record(s)", stem, record_count)

        append_jsonl(
            PATHS.ingested_shards,
            {
                "stem": stem,
                "status": "INGESTED",
                "rows": record_count,
                "timestamp": _now_iso(),
            },
        )

        ingestion_count += 1

        # Create periodic snapshot for HPC recovery
        if snapshot_interval > 0 and ingestion_count % snapshot_interval == 0:
            try:
                snapshot_name = _create_periodic_snapshot(client, profile.collection_name, ingestion_count)
                logger.info("Periodic snapshot created after %d shards: %s", ingestion_count, snapshot_name)
            except Exception as exc:
                logger.error(
                    "Failed to create periodic snapshot after %d shards (continuing anyway): %s",
                    ingestion_count,
                    str(exc),
                )

    if pending:
        logger.info("Re-enabling HNSW indexing (m=%d) — index build will proceed in background", profile.hnsw.m)
        enable_hnsw_indexing(client, profile.collection_name, profile.hnsw.m)
        settled_stats = _wait_for_collection_green(client, profile.collection_name)
    else:
        settled_stats = None

    final_snapshot_name: str | None
    try:
        final_snapshot_name = _create_periodic_snapshot(client, profile.collection_name, ingestion_count)
        logger.info("Final snapshot: %s", final_snapshot_name)
    except Exception as exc:
        logger.error("Failed to create final snapshot (continuing anyway): %s", str(exc))
        final_snapshot_name = None

    stats = settled_stats or get_collection_info(client, profile.collection_name)


def write_snapshot_metadata(profile_name: str) -> str:
    profile = get_qdrant_profile(profile_name)
    client = qdrant_client()
    try:
        snapshot_name = create_collection_snapshot(client, profile.collection_name)
        metadata = {
            "collection_name": profile.collection_name,
            "snapshot_name": snapshot_name,
            "snapshot_dir": str(PATHS.qdrant_snapshots),
            "created_at": _now_iso(),
        }
        write_json(PATHS.snapshot_metadata, metadata)
        return snapshot_name
    except Exception as exc:
        logger.error("Failed to create snapshot metadata: %s", str(exc))
        raise


def _create_periodic_snapshot(client, collection_name: str, shard_count: int) -> str:
    prev_name = _last_periodic_snapshot_name()

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

    if prev_name:
        try:
            client.delete_snapshot(collection_name=collection_name, snapshot_name=prev_name)
            logger.info("Deleted previous periodic snapshot: %s", prev_name)
        except Exception as exc:
            logger.warning("Could not delete previous snapshot %s (continuing): %s", prev_name, exc)

    return snapshot_name


def _last_periodic_snapshot_name() -> str | None:
    manifest_path = PATHS.qdrant_snapshots / "manifest.jsonl"
    if not manifest_path.exists():
        return None
    last: dict | None = None
    for row in read_jsonl(manifest_path):
        if row.get("snapshot_name"):
            last = row
    return last["snapshot_name"] if last else None


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
