import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import aiohttp
from src.utils.qdrant import qdrant_client, check_qdrant_alive, ensure_qdrant_runtime
from src.config.settings import QDRANT_ACTIVE
from .resolve_title import resolve_bib_entry, USER_AGENT, api_stats as _api_stats
from qdrant_client import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_RESOLVE_CONCURRENCY = 16

_QDRANT_WRITE_BATCH = 2000
_RESOLVE_PROGRESS_EVERY = 1
_RESOLVE_PROGRESS_EARLY = 10


@dataclass
class CitationRecord:
    chunk_uid: str
    source_ref_id: str
    cite_index: int
    raw: Optional[str] = None


@dataclass
class ProcessingReport:
    total_chunks_scanned: int = 0
    citations_without_ids_before: int = 0
    citations_without_ids_after: int = 0
    citations_resolved: int = 0
    citations_failed: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def has_public_id(cite_span: dict) -> bool:
    return bool(
        cite_span.get("doi") or
        cite_span.get("openalex_id") or
        cite_span.get("arxiv_id")
    )


def resolve_citation(record: CitationRecord) -> Optional[dict]:

    import inspect

    if not record.raw:
        logger.warning("No raw citation text for ref_id=%s, cannot resolve", record.source_ref_id)
        return None

    maybe = resolve_bib_entry(None, raw=record.raw, ref_id=record.source_ref_id)  # type: ignore[arg-type]
    # resolve_bib_entry is async; tests may mock it with a plain dict instead.
    result = asyncio.run(maybe) if inspect.isawaitable(maybe) else maybe
    if result and result.get("id_source") != "unresolved":
        return {
            "doi":         result.get("doi", ""),
            "openalex_id": result.get("openalex_id", ""),
            "arxiv_id":    result.get("arxiv_id", ""),
        }
    return None


def scan_citations_needing_resolution(client, collection_name: str, batch_size: int = 100) -> list[CitationRecord]:
    logger.info("Scanning collection '%s' for citations without IDs...", collection_name)

    missing = []
    offset = None

    while True:
        try:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.error("Failed to scroll collection: %s", exc)
            break

        if not records:
            break

        for record in records:
            payload = record.payload or {}
            cite_spans = payload.get("spans", {}).get("cite_spans", [])

            for idx, cite in enumerate(cite_spans):
                if not has_public_id(cite):
                    missing.append(CitationRecord(
                        chunk_uid=payload.get("chunk_uid", ""),
                        source_ref_id=cite.get("source_ref_id", ""),
                        cite_index=idx,
                        raw=cite.get("raw"),
                    ))

        offset = next_offset
        if offset is None:
            break

    logger.info("Found %d citations without public IDs", len(missing))
    return missing


def _deduplicate(missing: list[CitationRecord]) -> tuple[
    dict[str, list[CitationRecord]],   # ref_id → records sharing that ref_id
    list[str],                          # ordered unique ref_ids to resolve
]:
    groups: dict[str, list[CitationRecord]] = defaultdict(list)
    order: list[str] = []
    for rec in missing:
        key = rec.source_ref_id
        if key not in groups:
            order.append(key)
        groups[key].append(rec)

    dupes = sum(len(v) - 1 for v in groups.values())
    logger.info(
        "Deduplicated %d citations → %d unique ref_ids (%d duplicates skipped)",
        len(missing), len(order), dupes,
    )
    return groups, order


async def _resolve_all(
    unique_ref_ids: list[str],
    groups: dict[str, list[CitationRecord]],
) -> dict[str, Optional[dict]]:
   
    results: dict[str, Optional[dict]] = {}
    sem = asyncio.Semaphore(_RESOLVE_CONCURRENCY)
    done = 0
    total = len(unique_ref_ids)

    async def _resolve_one(ref_id: str, session: aiohttp.ClientSession) -> None:
        nonlocal done
        rec = groups[ref_id][0]   # representative record; all share the same raw
        async with sem:
            try:
                result = await resolve_bib_entry(
                    session,
                    raw=rec.raw,
                    ref_id=rec.source_ref_id,
                )
                if result:
                    logger.info(
                        "Resolution result for %s: id_source=%s, doi=%s, arxiv=%s",
                        rec.source_ref_id,
                        result.get("id_source"),
                        result.get("doi", ""),
                        result.get("arxiv_id", ""),
                    )
                results[ref_id] = result
            except Exception as exc:
                logger.warning("Failed to resolve %s: %s", ref_id, exc)
                results[ref_id] = None
            finally:
                done += 1
                if (
                    done <= _RESOLVE_PROGRESS_EARLY
                    or done % _RESOLVE_PROGRESS_EVERY == 0
                    or done == total
                ):
                    logger.info(
                        "Progress: resolved %d / %d unique ref_ids (%.1f%%)",
                        done,
                        total,
                        done / total * 100 if total else 0.0,
                    )

    connector = aiohttp.TCPConnector(limit=0)   # semaphores in resolve_title handle the actual cap
    async with aiohttp.ClientSession(
        headers={"User-Agent": USER_AGENT},
        connector=connector,
    ) as session:
        await asyncio.gather(*[_resolve_one(ref_id, session) for ref_id in unique_ref_ids])

    return results


def _apply_results_and_write(
    client,
    collection_name: str,
    groups: dict[str, list[CitationRecord]],
    resolutions: dict[str, Optional[dict]],
    report: ProcessingReport,
    dry_run: bool,
) -> None:
   
    # Build: chunk_uid → list[(cite_index, resolved_ids)]
    chunk_updates: dict[str, list[tuple[int, dict]]] = defaultdict(list)

    for ref_id, resolved in resolutions.items():
        if resolved and resolved.get("id_source") != "unresolved":
            ids = {
                "doi":         resolved.get("doi", ""),
                "openalex_id": resolved.get("openalex_id", ""),
                "arxiv_id":    resolved.get("arxiv_id", ""),
            }
            for rec in groups[ref_id]:
                chunk_updates[rec.chunk_uid].append((rec.cite_index, ids))
        else:
            report.citations_failed += len(groups[ref_id])

    logger.info("Writing updates for %d chunks to Qdrant...", len(chunk_updates))

    chunk_uids = list(chunk_updates.keys())
    for batch_start in range(0, len(chunk_uids), _QDRANT_WRITE_BATCH):
        batch = chunk_uids[batch_start: batch_start + _QDRANT_WRITE_BATCH]

        if dry_run:
            for uid in batch:
                for cite_index, ids in chunk_updates[uid]:
                    logger.info("[DRY RUN] Would update %s:%d with %s", uid, cite_index, ids)
                    report.citations_resolved += 1
            continue

        # Fetch all chunks in this batch in one scroll call
        try:
            scroll_result, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="chunk_uid",
                        match=models.MatchAny(any=batch),
                    )]
                ),
                limit=len(batch),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.error("Batch scroll failed: %s", exc)
            for uid in batch:
                report.citations_failed += len(chunk_updates[uid])
                report.errors.append(f"scroll failed for batch containing {uid}")
            continue

        fetched = {r.payload.get("chunk_uid"): r for r in scroll_result if r.payload}

        for uid in batch:
            if uid not in fetched:
                logger.warning("Chunk %s not found in Qdrant", uid)
                report.citations_failed += len(chunk_updates[uid])
                report.errors.append(f"chunk not found: {uid}")
                continue

            record = fetched[uid]
            payload = record.payload
            cite_spans = payload.get("spans", {}).get("cite_spans", [])

            applied = 0
            for cite_index, ids in chunk_updates[uid]:
                if cite_index >= len(cite_spans):
                    logger.warning("Citation index %d out of bounds for chunk %s", cite_index, uid)
                    report.citations_failed += 1
                    report.errors.append(f"index OOB {uid}:{cite_index}")
                    continue
                cite_spans[cite_index].update(ids)
                applied += 1

            try:
                client.set_payload(
                    collection_name=collection_name,
                    payload={"spans": {"cite_spans": cite_spans}},
                    points=[record.id],
                )
                report.citations_resolved += applied
            except Exception as exc:
                logger.error("Failed to write chunk %s: %s", uid, exc)
                report.citations_failed += applied
                report.errors.append(f"write failed {uid}: {exc}")

        logger.info(
            "Flushed batch %d–%d / %d chunks",
            batch_start + 1, min(batch_start + _QDRANT_WRITE_BATCH, len(chunk_uids)), len(chunk_uids),
        )


def build_stats(
    missing_citations: list[CitationRecord],
    groups: dict[str, list[CitationRecord]],
    resolutions: dict[str, Optional[dict]],
    citations_without_ids_after: int,
    total_chunks_scanned: int,
    elapsed_seconds: float,
    dry_run: bool,
) -> dict:
 
    total_citations   = len(missing_citations)
    unique_refs       = len(groups)
    dedup_savings     = total_citations - unique_refs

    # Count resolutions by id_source
    source_counts: dict[str, int] = {"doi": 0, "arxiv": 0, "openalex": 0, "unresolved": 0}
    for result in resolutions.values():
        src = (result or {}).get("id_source", "unresolved")
        source_counts[src] = source_counts.get(src, 0) + 1

    # Expand unique-ref counts back to citation counts (one ref_id → N citations)
    citation_source_counts: dict[str, int] = {"doi": 0, "arxiv": 0, "openalex": 0, "unresolved": 0}
    for ref_id, result in resolutions.items():
        src = (result or {}).get("id_source", "unresolved")
        n   = len(groups[ref_id])
        citation_source_counts[src] = citation_source_counts.get(src, 0) + n

    citations_resolved_total = (
        citation_source_counts["doi"] +
        citation_source_counts["arxiv"] +
        citation_source_counts["openalex"]
    )
    unique_refs_resolved = source_counts["doi"] + source_counts["arxiv"] + source_counts["openalex"]

    return {
        "run": {
            "dry_run":         dry_run,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "total_chunks_scanned": total_chunks_scanned,
        },

        "citations": {
            "without_ids_before":  total_citations,
            "without_ids_after":   citations_without_ids_after,
            "resolved":            citations_resolved_total,
            "unresolved":          citation_source_counts["unresolved"],
            "resolution_rate_pct": round(citations_resolved_total / total_citations * 100, 2) if total_citations else 0.0,
        },

        "unique_refs": {
            "total":      unique_refs,
            "resolved":   unique_refs_resolved,
            "unresolved": source_counts["unresolved"],
            "dedup_savings": dedup_savings,
        },

        "resolution_by_id_source": {
            "citations": citation_source_counts,
            "unique_refs": source_counts,
        },

        "api_usage": {
            api: {
                "calls":        s["calls"],
                "matched":      s["matched"],
                "errors":       s["errors"],
                "rate_limited": s["rate_limited"],
                "hit_rate_pct": round(s["matched"] / s["calls"] * 100, 2) if s["calls"] else 0.0,
                "error_rate_pct": round(s["errors"] / s["calls"] * 100, 2) if s["calls"] else 0.0,
            }
            for api, s in _api_stats.items()
        },
    }


def count_remaining_missing(client, collection_name: str) -> int:
    remaining = scan_citations_needing_resolution(client, collection_name)
    return len(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Resolve IDs without writing updates to Qdrant.")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Process only the first N unique ref_ids (fast iteration/testing).")
    return parser.parse_args()


def main():
    args = parse_args()

    collection_name    = QDRANT_ACTIVE.collection_name
    start_time         = time.time()
    errors: list[str]  = []

    # state filled in as we go — also used by build_stats
    missing_citations: list[CitationRecord] = []
    groups:       dict[str, list[CitationRecord]] = {}
    resolutions:  dict[str, Optional[dict]]       = {}
    total_chunks_scanned        = 0
    citations_without_ids_after = 0

    try:
        logger.info("Ensuring Qdrant runtime (profile: %s)...", QDRANT_ACTIVE.profile)
        ensure_qdrant_runtime(QDRANT_ACTIVE.profile)

        client = qdrant_client()
        check_qdrant_alive(client, profile=QDRANT_ACTIVE.profile)
        logger.info("✓ Connected to Qdrant")

        missing_citations = scan_citations_needing_resolution(client, collection_name)
        total_chunks_scanned = client.get_collection(collection_name).points_count

        if not missing_citations:
            logger.info("No citations need resolution!")
        else:
            groups, unique_ref_ids = _deduplicate(missing_citations)

            if args.limit is not None:
                unique_ref_ids = unique_ref_ids[:args.limit]
                logger.info("--limit %d: truncating to %d unique ref_ids", args.limit, len(unique_ref_ids))

            logger.info("Resolving %d unique ref_ids concurrently (max %d at once)...",
                        len(unique_ref_ids), _RESOLVE_CONCURRENCY)
            resolutions = asyncio.run(_resolve_all(unique_ref_ids, groups))

            # Borrow a temporary ProcessingReport just for _apply_results_and_write
            # (the counts end up in build_stats, not here)
            _tmp = ProcessingReport()
            _apply_results_and_write(
                client, collection_name, groups, resolutions, _tmp, dry_run=args.dry_run
            )
            errors.extend(_tmp.errors)

            citations_without_ids_after = (
                len(missing_citations) if args.dry_run
                else count_remaining_missing(client, collection_name)
            )

    except Exception as exc:
        logger.error("Postprocessing failed: %s", exc)
        errors.append(str(exc))

    finally:
        elapsed = time.time() - start_time

        stats = build_stats(
            missing_citations=missing_citations,
            groups=groups,
            resolutions=resolutions,
            citations_without_ids_after=citations_without_ids_after,
            total_chunks_scanned=total_chunks_scanned,
            elapsed_seconds=elapsed,
            dry_run=args.dry_run,
        )
        if errors:
            stats["errors"] = errors

        report_path = Path("postprocessing_report.json")
        with open(report_path, "w") as f:
            json.dump(stats, f, indent=2)

        c  = stats["citations"]
        u  = stats["unique_refs"]
        rs = stats["resolution_by_id_source"]["citations"]

        logger.info("=" * 60)
        logger.info("POSTPROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info("Total chunks scanned:         %d", total_chunks_scanned)
        logger.info("Citations without IDs before: %d", c["without_ids_before"])
        logger.info("Citations without IDs after:  %d", c["without_ids_after"])
        logger.info("Citations resolved:           %d  (%.1f%%)", c["resolved"], c["resolution_rate_pct"])
        logger.info("  ├─ via DOI      (Crossref):  %d", rs["doi"])
        logger.info("  ├─ via arXiv:               %d", rs["arxiv"])
        logger.info("  └─ via OpenAlex:             %d", rs["openalex"])
        logger.info("Citations unresolved:         %d", c["unresolved"])
        logger.info("Unique refs resolved:         %d / %d  (dedup saved %d API calls)",
                    u["resolved"], u["total"], u["dedup_savings"])
        logger.info("")
        logger.info("API usage:")
        for api, s in stats["api_usage"].items():
            logger.info("  %-10s  calls=%-6d  matched=%-6d  hit_rate=%5.1f%%  errors=%-4d  rate_limited=%d",
                        api, s["calls"], s["matched"], s["hit_rate_pct"], s["errors"], s["rate_limited"])
        logger.info("Elapsed time:                 %.2f seconds", elapsed)

        if errors:
            logger.warning("Errors encountered: %d", len(errors))
            for err in errors[:5]:
                logger.warning("  - %s", err)

        logger.info("Report saved to: %s", report_path.absolute())
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
