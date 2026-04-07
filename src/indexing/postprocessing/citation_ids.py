import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.qdrant import qdrant_client, check_qdrant_alive, ensure_qdrant_runtime
from src.config.settings import QDRANT_ACTIVE
from .resolve_title import resolve_bib_entry
from qdrant_client import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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


def resolve_citation(record: CitationRecord) -> Optional[dict]:

    if not record.raw:
        logger.warning("No raw citation text for ref_id=%s, cannot resolve", record.source_ref_id)
        return None

    try:
        result = resolve_bib_entry(
            title="",
            raw=record.raw,
            ref_id=record.source_ref_id
        )

        if result:
            logger.info("Resolution result for %s: id_source=%s, doi=%s, arxiv=%s",
                       record.source_ref_id, result.get("id_source"), result.get("doi", ""), result.get("arxiv_id", ""))

        if result and result.get("id_source") != "unresolved":
            return {
                "doi": result.get("doi", ""),
                "openalex_id": result.get("openalex_id", ""),
                "arxiv_id": result.get("arxiv_id", ""),
            }
    except Exception as exc:
        logger.warning("Failed to resolve %s: %s", record.source_ref_id, exc)

    return None


def update_citation_in_qdrant(
    client,
    collection_name: str,
    chunk_uid: str,
    cite_index: int,
    resolved_ids: dict
) -> bool:

    try:
        # Fetch current payload
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="chunk_uid",
                    match=models.MatchValue(value=chunk_uid)
                )]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )[0]
        
        if not results:
            logger.warning("Chunk %s not found", chunk_uid)
            return False
        
        record = results[0]
        payload = record.payload
        cite_spans = payload.get("spans", {}).get("cite_spans", [])
        
        if cite_index >= len(cite_spans):
            logger.warning("Citation index %d out of bounds for chunk %s", cite_index, chunk_uid)
            return False
        
        # Update citation with resolved IDs
        cite_spans[cite_index].update(resolved_ids)
        
        # Write back to Qdrant
        client.set_payload(
            collection_name=collection_name,
            payload={"spans": {"cite_spans": cite_spans}},
            points=[record.id],
        )
        
        return True
    
    except Exception as exc:
        logger.error("Failed to update chunk %s: %s", chunk_uid, exc)
        return False


def process_citations(
    client,
    collection_name: str,
    missing_citations: list[CitationRecord],
    report: ProcessingReport,
    dry_run: bool = False,
) -> None:

    total = len(missing_citations)
    logger.info("Processing %d citations...", total)
    
    for i, record in enumerate(missing_citations, 1):
        if i % 10 == 0:
            logger.info("Progress: %d / %d", i, total)
        
        resolved = resolve_citation(record)
        
        if resolved and has_public_id(resolved):
            if dry_run:
                logger.info(
                    "[DRY RUN] Would update %s:%d with %s",
                    record.chunk_uid,
                    record.cite_index,
                    resolved,
                )
                success = True
            else:
                success = update_citation_in_qdrant(
                    client,
                    collection_name,
                    record.chunk_uid,
                    record.cite_index,
                    resolved
                )
            
            if success:
                report.citations_resolved += 1
            else:
                report.citations_failed += 1
                report.errors.append(f"Failed to update {record.chunk_uid}:{record.cite_index}")
        else:
            report.citations_failed += 1


def count_remaining_missing(client, collection_name: str) -> int:

    remaining = scan_citations_needing_resolution(client, collection_name)
    return len(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Resolve IDs without writing updates to Qdrant.")
    return parser.parse_args()


def main():
    args = parse_args()

    collection_name = QDRANT_ACTIVE.collection_name
    start_time = time.time()
    report = ProcessingReport()
    
    try:
        # Ensure Qdrant is running
        logger.info("Ensuring Qdrant runtime (profile: %s)...", QDRANT_ACTIVE.profile)
        ensure_qdrant_runtime(QDRANT_ACTIVE.profile)
        
        # Connect to Qdrant
        client = qdrant_client()
        check_qdrant_alive(client, profile=QDRANT_ACTIVE.profile)
        logger.info("✓ Connected to Qdrant")
        
        # Scan for missing citations
        missing_citations = scan_citations_needing_resolution(client, collection_name)
        report.citations_without_ids_before = len(missing_citations)
        
        # Count total chunks scanned
        collection_info = client.get_collection(collection_name)
        report.total_chunks_scanned = collection_info.points_count
        
        if not missing_citations:
            logger.info("No citations need resolution!")
        else:
            # Process citations
            process_citations(client, collection_name, missing_citations, report, dry_run=args.dry_run)

            if args.dry_run:
                report.citations_without_ids_after = report.citations_without_ids_before
            else:
                # Rescan to count remaining
                report.citations_without_ids_after = count_remaining_missing(client, collection_name)
        
    except Exception as exc:
        logger.error("Postprocessing failed: %s", exc)
        report.errors.append(str(exc))
    
    finally:
        report.elapsed_seconds = time.time() - start_time
        
        report_path = Path("postprocessing_report.json")
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        
        logger.info("=" * 60)
        logger.info("POSTPROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info("Total chunks scanned:       %d", report.total_chunks_scanned)
        logger.info("Citations without IDs before: %d", report.citations_without_ids_before)
        logger.info("Citations resolved:          %d", report.citations_resolved)
        logger.info("Citations failed:            %d", report.citations_failed)
        logger.info("Citations without IDs after:  %d", report.citations_without_ids_after)
        logger.info("Elapsed time:                %.2f seconds", report.elapsed_seconds)
        
        if report.errors:
            logger.warning("Errors encountered: %d", len(report.errors))
            for err in report.errors[:5]:
                logger.warning("  - %s", err)
        
        logger.info("Report saved to: %s", report_path.absolute())
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
