from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from evigraph.config.settings import PATHS, get_qdrant_profile
from evigraph.indexing.utils.storage import append_jsonl, read_jsonl, shard_artifacts, write_json
from qdrant_client.http.exceptions import UnexpectedResponse
from evigraph.utils.qdrant import (
    build_points_from_shard_records,
    create_collection_snapshot,
    disable_hnsw_indexing,
    enable_hnsw_indexing,
    get_collection_info,
    qdrant_client,
    setup_collection,
)

logger = logging.getLogger(__name__)

# Upsert configuration for resilience
UPSERT_RETRY_MAX_ATTEMPTS = 5
UPSERT_INITIAL_BACKOFF_SEC = 2.0
UPSERT_MAX_BACKOFF_SEC = 60.0
UPSERT_BACKOFF_MULTIPLIER = 2.0
INTER_BATCH_THROTTLE_SEC = 0.5  # Small delay between batches to reduce write pressure


def _env_int(name: str, default: int, *, min_value: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _env_float(name: str, default: float, *, min_value: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _upsert_with_retry(
    client,
    collection_name: str,
    points,
    wait: bool,
    batch_id: int = 0,
) -> None:

    backoff_sec = UPSERT_INITIAL_BACKOFF_SEC
    attempt = 0
    
    while attempt < UPSERT_RETRY_MAX_ATTEMPTS:
        try:
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=wait,
            )
            if attempt > 0:
                logger.info("Batch %d upserted successfully after %d attempt(s)", batch_id, attempt + 1)
            return
        except UnexpectedResponse as exc:
            # Handle 408 Request Timeout specifically
            if exc.status_code == 408:
                attempt += 1
                if attempt < UPSERT_RETRY_MAX_ATTEMPTS:
                    logger.warning(
                        "Batch %d: Qdrant timeout (408) on attempt %d/%d. "
                        "Qdrant likely under lock contention. Retrying in %.1fs…",
                        batch_id,
                        attempt,
                        UPSERT_RETRY_MAX_ATTEMPTS,
                        backoff_sec,
                    )
                    time.sleep(backoff_sec)
                    # Exponential backoff with cap
                    backoff_sec = min(
                        backoff_sec * UPSERT_BACKOFF_MULTIPLIER,
                        UPSERT_MAX_BACKOFF_SEC,
                    )
                else:
                    logger.error(
                        "Batch %d: Failed after %d retries due to Qdrant timeout (408). "
                        "Database appears saturated; consider increasing timeout or reducing write load.",
                        batch_id,
                        UPSERT_RETRY_MAX_ATTEMPTS,
                    )
                    raise
            else:
                # Re-raise non-timeout errors immediately
                raise
        except Exception as exc:
            # Non-timeout errors: fail fast
            logger.error("Batch %d upsert failed with non-timeout error: %s", batch_id, str(exc))
            raise


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
    upsert_batch_size = _env_int("EVI_UPSERT_BATCH_SIZE", profile.upsert_batch_size, min_value=1)
    wait_every_n_batches = _env_int("EVI_WAIT_EVERY_N_BATCHES", 2, min_value=0)
    inter_batch_throttle_sec = _env_float(
        "EVI_INGEST_THROTTLE_SEC",
        INTER_BATCH_THROTTLE_SEC,
        min_value=0.0,
    )
    logger.info(
        "Ingestion tuning: upsert_batch_size=%d wait_every_n_batches=%d "
        "inter_batch_throttle_sec=%.3f snapshot_interval=%d",
        upsert_batch_size,
        wait_every_n_batches,
        inter_batch_throttle_sec,
        snapshot_interval,
    )

    # Increased timeout from 600s (10min) to 1800s (30min) for highly concurrent operations
    # Individual upserts can take 10-60s during lock contention; 30min allows proper retry backoff
    client = qdrant_client(timeout=1800)
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
            if len(batch) >= upsert_batch_size:
                points = build_points_from_shard_records(batch, profile)
                wait_on_this_batch = (
                    wait_every_n_batches > 0
                    and (batch_count + 1) % wait_every_n_batches == 0
                )
                _upsert_with_retry(
                    client,
                    collection_name=profile.collection_name,
                    points=points,
                    wait=wait_on_this_batch,
                    batch_id=batch_count,
                )
                # Small throttle between batches to reduce concurrent write pressure
                if inter_batch_throttle_sec:
                    time.sleep(inter_batch_throttle_sec)
                batch = []
                batch_count += 1
        if not record_count:
            logger.info("Shard %s is empty, skipping", stem)
            continue
        if batch:
            points = build_points_from_shard_records(batch, profile)
            # Final batch always waits for completion
            _upsert_with_retry(
                client,
                collection_name=profile.collection_name,
                points=points,
                wait=True,
                batch_id=batch_count,
            )
            # Small throttle after final batch
            if inter_batch_throttle_sec:
                time.sleep(inter_batch_throttle_sec)
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


_SNAPSHOT_KEEP = 5


def _create_periodic_snapshot(client, collection_name: str, shard_count: int) -> str:
    recent = _recent_periodic_snapshot_names()

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

    # Evict oldest snapshots beyond the retention window
    for old_name in recent[:-(_SNAPSHOT_KEEP - 1)] if len(recent) >= _SNAPSHOT_KEEP else []:
        try:
            client.delete_snapshot(collection_name=collection_name, snapshot_name=old_name)
            logger.info("Deleted old periodic snapshot: %s", old_name)
        except Exception as exc:
            logger.warning("Could not delete old snapshot %s (continuing): %s", old_name, exc)

    return snapshot_name


def _recent_periodic_snapshot_names() -> list[str]:
    manifest_path = PATHS.qdrant_snapshots / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    return [row["snapshot_name"] for row in read_jsonl(manifest_path) if row.get("snapshot_name")]


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
