# Usage (standalone):
#     python qdrant_inspector.py [--collection unarxive_chunks] [--batch 1000]

# Usage (from SLURM script):
#     Add to inspect_qdrant_stats.sh or call directly in uv run python

from __future__ import annotations
import argparse
import logging
import sys
from collections import Counter
from datetime import datetime

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from config.settings import QDRANT_ACTIVE, PATHS
    from utils.qdrant import qdrant_client, check_qdrant_alive
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.info("Make sure to run from project root: uv run python src/utils/qdrant_inspector.py")
    sys.exit(1)


class QdrantInspector:

    def __init__(
        self,
        collection_name: str = "unarxive_chunks",
        batch_size: int = 5000,
        verbose: bool = False
    ):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.verbose = verbose

        self.client = qdrant_client(timeout=600)
        self.dense_vec_name = QDRANT_ACTIVE.dense_vector_name or "dense"
        self.sparse_vec_name = QDRANT_ACTIVE.sparse_vector_name or "sparse"

        self.collection_info = None
        self.first_point = None  # kept for sample payload display
        self.stats = {
            "total_processed": 0,
            # paper-level id coverage (per chunk, not deduplicated)
            "paper_has_arxiv_id": 0,
            "paper_has_doi": 0,
            "paper_has_both_ids": 0,
            "unique_papers": set(),
            "unique_papers_with_arxiv": set(),
            "unique_papers_with_doi": set(),
            # cited-span id coverage
            "cited_span_total": 0,
            "cited_span_has_doi": 0,
            "cited_span_has_arxiv": 0,
            "cited_span_has_openalex": 0,
            "cited_span_has_any": 0,
            # distributions
            "papers_by_year": Counter(),
            "papers_by_category": Counter(),
            "section_titles": Counter(),
            "chunk_types": Counter(),
        }

    def connect_and_verify(self) -> bool:
        try:
            check_qdrant_alive(self.client)
            logger.info("✓ Qdrant is alive and responding")
            return True
        except Exception as e:
            logger.error(f"✗ Qdrant connection failed: {e}")
            return False

    def fetch_collection_info(self) -> bool:
        try:
            self.collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"✓ Fetched collection info for '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to fetch collection info: {e}")
            return False

    def scroll_all_points(self) -> bool:
        if not self.collection_info:
            logger.error("Collection info not available")
            return False

        total_points = self.collection_info.points_count
        if total_points == 0:
            logger.error("Collection is empty")
            return False

        logger.info(f"Scrolling all {total_points:,} points in batches of {self.batch_size:,}…")

        offset = None
        processed = 0

        while True:
            try:
                batch, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=self.batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,  # dimensions come from collection config
                )
            except Exception as e:
                logger.error(f"✗ Scroll failed at offset {offset}: {e}")
                return False

            if not batch:
                break

            for point in batch:
                if self.first_point is None:
                    self.first_point = point
                self._analyze_point(point)

            processed += len(batch)
            if processed % 50_000 == 0 or next_offset is None:
                logger.info(f"  … processed {processed:,} / {total_points:,} points")

            if next_offset is None:
                break
            offset = next_offset

        self.stats["total_processed"] = processed
        logger.info(f"✓ Finished scrolling — {processed:,} points processed")
        return True

    def _analyze_point(self, point) -> None:
        payload = point.payload or {}

        paper_id = payload.get('paper_id_arxiv')
        doi = payload.get('paper_doi')
        chunk_type = payload.get('chunk_type', 'unknown')

        # paper id coverage
        has_arxiv = bool(paper_id)
        has_doi = bool(doi)

        if has_arxiv:
            self.stats["paper_has_arxiv_id"] += 1
        if has_doi:
            self.stats["paper_has_doi"] += 1
        if has_arxiv and has_doi:
            self.stats["paper_has_both_ids"] += 1

        # unique papers (use arxiv id as primary key, fall back to doi)
        pid = paper_id or doi or None
        if pid:
            self.stats["unique_papers"].add(pid)
        if has_arxiv:
            self.stats["unique_papers_with_arxiv"].add(paper_id)
        if has_doi:
            self.stats["unique_papers_with_doi"].add(doi)

        # cited-span coverage — each chunk carries cite_spans inline
        cite_spans = payload.get('spans', {}).get('cite_spans', [])
        for span in cite_spans:
            self.stats["cited_span_total"] += 1
            cs_doi = bool(span.get('doi'))
            cs_arxiv = bool(span.get('arxiv_id'))
            cs_openalex = bool(span.get('openalex_id'))
            if cs_doi:
                self.stats["cited_span_has_doi"] += 1
            if cs_arxiv:
                self.stats["cited_span_has_arxiv"] += 1
            if cs_openalex:
                self.stats["cited_span_has_openalex"] += 1
            if cs_doi or cs_arxiv or cs_openalex:
                self.stats["cited_span_has_any"] += 1

        # distributions
        year = payload.get('year')
        if year:
            self.stats["papers_by_year"][year] += 1

        categories = payload.get('categories', [])
        if isinstance(categories, list):
            for cat in categories:
                self.stats["papers_by_category"][cat] += 1

        section_title = payload.get('section_title', 'Unknown')
        self.stats["section_titles"][section_title] += 1
        self.stats["chunk_types"][chunk_type] += 1

    # ------------------------------------------------------------------ #
    # Print sections                                                       #
    # ------------------------------------------------------------------ #

    def print_collection_overview(self) -> None:
        print("\n" + "="*80)
        print("COLLECTION OVERVIEW")
        print("="*80)

        if not self.collection_info:
            print("Collection info not available")
            return

        info = self.collection_info
        print(f"\nCollection Name: {self.collection_name}")
        print(f"Status: {info.status}")
        print(f"Total Points: {info.points_count:,}")
        print(f"Indexed Vectors: {info.indexed_vectors_count:,}")

        if hasattr(info, 'config') and info.config and hasattr(info.config, 'vectors'):
            print(f"\nVector Configuration:")
            for vec_name, vec_config in info.config.vectors.items():
                print(f"  - {vec_name}:")
                print(f"    Size: {vec_config.size} dimensions")
                print(f"    Distance: {vec_config.distance}")
                if hasattr(vec_config, 'datatype'):
                    print(f"    Datatype: {vec_config.datatype}")

        if hasattr(info, 'payload_schema') and info.payload_schema:
            print(f"\nPayload Schema Fields ({len(info.payload_schema)}):")
            for field_name in sorted(info.payload_schema.keys())[:20]:
                field_type = info.payload_schema[field_name]
                print(f"  - {field_name}: {field_type}")
            if len(info.payload_schema) > 20:
                print(f"  … and {len(info.payload_schema) - 20} more")

    def print_vector_statistics(self) -> None:
        print("\n" + "="*80)
        print("VECTOR STATISTICS")
        print("="*80)

        info = self.collection_info
        if not (hasattr(info, 'config') and info.config and hasattr(info.config, 'vectors')):
            print("Vector config not available")
            return

        total_estimated = 0.0
        for vec_name, vec_config in info.config.vectors.items():
            dims = vec_config.size
            if dims:
                total_bytes = info.points_count * dims * 4
                total_gb = total_bytes / (1024**3)
                total_estimated += total_gb
                print(f"\nDense Vectors ({vec_name}):")
                print(f"  Dimensions: {dims}")
                print(f"  Bytes per vector (float32): {dims * 4:,}")
                print(f"  Estimated total storage: {total_gb:.2f} GB ({total_bytes:,} bytes)")

        if hasattr(info.config, 'sparse_vectors') and info.config.sparse_vectors:
            print(f"\nSparse vector configs found: {list(info.config.sparse_vectors.keys())}")
            print(f"  (Sparse storage estimate requires index inspection — see disk analysis below)")

        print(f"\n✓ Estimated dense vector storage: {total_estimated:.2f} GB")

    def print_paper_id_coverage(self) -> None:
        print("\n" + "="*80)
        print("PAPER ID COVERAGE (full collection)")
        print("="*80)

        n = self.stats["total_processed"]
        if n == 0:
            print("No points processed")
            return

        unique_papers = len(self.stats["unique_papers"])
        unique_with_arxiv = len(self.stats["unique_papers_with_arxiv"])
        unique_with_doi = len(self.stats["unique_papers_with_doi"])

        print(f"\nChunks processed: {n:,}")
        print(f"\nUnique papers (any id): {unique_papers:,}")
        print(f"  with arXiv ID:   {unique_with_arxiv:,}  ({100*unique_with_arxiv/unique_papers:.1f}% of unique papers)" if unique_papers else "")
        print(f"  with DOI:        {unique_with_doi:,}  ({100*unique_with_doi/unique_papers:.1f}% of unique papers)" if unique_papers else "")

        print(f"\nPer-chunk id presence (chunk may represent same paper multiple times):")
        print(f"  chunks with arXiv ID: {self.stats['paper_has_arxiv_id']:,}  ({100*self.stats['paper_has_arxiv_id']/n:.1f}%)")
        print(f"  chunks with DOI:      {self.stats['paper_has_doi']:,}  ({100*self.stats['paper_has_doi']/n:.1f}%)")
        print(f"  chunks with both:     {self.stats['paper_has_both_ids']:,}  ({100*self.stats['paper_has_both_ids']/n:.1f}%)")

    def print_cited_span_coverage(self) -> None:
        print("\n" + "="*80)
        print("CITED SPAN PUBLIC ID COVERAGE (full collection)")
        print("="*80)

        total = self.stats["cited_span_total"]
        if total == 0:
            print("\nNo cite_spans found in any chunk payload.")
            return

        def pct(n):
            return f"{n:,}  ({100*n/total:.1f}%)"

        print(f"\nTotal cite_spans across all chunks: {total:,}")
        print(f"  with DOI:           {pct(self.stats['cited_span_has_doi'])}")
        print(f"  with arXiv ID:      {pct(self.stats['cited_span_has_arxiv'])}")
        print(f"  with OpenAlex ID:   {pct(self.stats['cited_span_has_openalex'])}")
        print(f"  with any public ID: {pct(self.stats['cited_span_has_any'])}")
        missing = total - self.stats["cited_span_has_any"]
        print(f"  with NO public ID:  {missing:,}  ({100*missing/total:.1f}%)")

    def print_paper_statistics(self) -> None:
        print("\n" + "="*80)
        print("PAPER STATISTICS (full collection)")
        print("="*80)

        info = self.collection_info
        n = self.stats["total_processed"]
        unique_papers = len(self.stats["unique_papers"])

        print(f"\nUnique papers in collection: {unique_papers:,}")

        if n > 0 and unique_papers > 0:
            avg_chunks_per_paper = n / unique_papers
            print(f"Average chunks per paper: {avg_chunks_per_paper:.1f}")

        if self.stats["papers_by_year"]:
            print(f"\nChunks by year (top 10):")
            for year, count in self.stats["papers_by_year"].most_common(10):
                print(f"  {year}: {count:,}")

        if self.stats["papers_by_category"]:
            print(f"\nChunks by category (top 10):")
            for category, count in self.stats["papers_by_category"].most_common(10):
                print(f"  {category}: {count:,}")

    def print_chunk_statistics(self) -> None:
        print("\n" + "="*80)
        print("CHUNK STATISTICS (full collection)")
        print("="*80)

        n = self.stats["total_processed"]
        if n == 0:
            return

        if self.stats["chunk_types"]:
            print(f"\nChunk types:")
            for chunk_type, count in self.stats["chunk_types"].most_common():
                pct = (count / n) * 100
                print(f"  {chunk_type}: {count:,} ({pct:.1f}%)")

        if self.stats["section_titles"]:
            print(f"\nSection titles (top 15):")
            for section, count in self.stats["section_titles"].most_common(15):
                pct = (count / n) * 100
                print(f"  {section}: {count:,} ({pct:.1f}%)")

    def print_disk_storage(self) -> None:
        print("\n" + "="*80)
        print("DISK STORAGE ANALYSIS")
        print("="*80)

        storage_path = PATHS.qdrant_storage / "collections" / self.collection_name / "0" / "segments"

        if storage_path.exists():
            try:
                total_size = 0
                segment_count = 0
                segment_sizes = []

                for segment_dir in storage_path.iterdir():
                    if segment_dir.is_dir():
                        segment_count += 1
                        size = sum(f.stat().st_size for f in segment_dir.rglob('*') if f.is_file())
                        segment_sizes.append((segment_dir.name, size))
                        total_size += size

                total_gb = total_size / (1024**3)
                avg_segment_mb = (total_size / segment_count / (1024**2)) if segment_count > 0 else 0

                print(f"\nSegments on disk: {segment_count}")
                print(f"Total segment storage: {total_gb:.2f} GB ({total_size:,} bytes)")
                print(f"Average segment size: {avg_segment_mb:.1f} MB")

                if segment_sizes:
                    segment_sizes.sort(key=lambda x: x[1], reverse=True)
                    print(f"\nLargest segments (top 10):")
                    for seg_name, seg_size in segment_sizes[:10]:
                        seg_mb = seg_size / (1024**2)
                        print(f"  {seg_name}: {seg_mb:.1f} MB")
                    if len(segment_sizes) > 10:
                        print(f"  … and {len(segment_sizes) - 10} more segments")

            except Exception as e:
                logger.error(f"Failed to analyze disk storage: {e}")
        else:
            print(f"\nStorage path not found: {storage_path}")

    def print_snapshots(self) -> None:
        print("\n" + "="*80)
        print("SNAPSHOT INFORMATION")
        print("="*80)

        snapshots_path = PATHS.qdrant_snapshots

        if snapshots_path.exists():
            try:
                snapshots = list(snapshots_path.glob("*.tar")) + list(snapshots_path.glob("*.tar.gz"))

                if snapshots:
                    print(f"\nSnapshots found: {len(snapshots)}")
                    total_size = 0

                    for snap in sorted(snapshots, reverse=True)[:10]:
                        size_gb = snap.stat().st_size / (1024**3)
                        mtime = datetime.fromtimestamp(snap.stat().st_mtime)
                        print(f"  {snap.name}: {size_gb:.2f} GB (created: {mtime})")
                        total_size += snap.stat().st_size

                    if len(snapshots) > 10:
                        print(f"  … and {len(snapshots) - 10} more snapshots")

                    print(f"\nTotal snapshot storage: {total_size / (1024**3):.2f} GB")
                else:
                    print("No snapshots found")

            except Exception as e:
                logger.error(f"Failed to analyze snapshots: {e}")
        else:
            print(f"Snapshots directory not found: {snapshots_path}")

    def print_sample_payload(self) -> None:
        print("\n" + "="*80)
        print("SAMPLE PAYLOAD STRUCTURE")
        print("="*80)

        if not self.first_point:
            print("No point available")
            return

        point = self.first_point
        print(f"\nSample chunk payload (Point ID: {point.id}):")

        if point.payload:
            for key in sorted(point.payload.keys()):
                value = point.payload[key]

                if isinstance(value, str):
                    preview = value[:60] + ("…" if len(value) > 60 else "")
                    print(f"  - {key}: \"{preview}\"")
                elif isinstance(value, (int, float, bool)):
                    print(f"  - {key}: {value}")
                elif isinstance(value, list):
                    preview = str(value[:2])[1:-1]
                    print(f"  - {key}: [{len(value)} items] {preview}{'…' if len(value) > 2 else ''}")
                elif isinstance(value, dict):
                    print(f"  - {key}: {{dict with {len(value)} keys}}")
                else:
                    print(f"  - {key}: {type(value).__name__}")

    def print_summary(self) -> None:
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        info = self.collection_info
        n = self.stats["total_processed"]
        unique_papers = len(self.stats["unique_papers"])

        print(f"\n✓ Total points in collection: {info.points_count:,}")
        print(f"✓ Points fully scrolled:      {n:,}")
        print(f"✓ Unique papers:               {unique_papers:,}")

        if n > 0 and unique_papers > 0:
            print(f"✓ Avg chunks per paper:        {n/unique_papers:.1f}")

        if hasattr(info, 'config') and info.config and hasattr(info.config, 'vectors'):
            for vec_name, vec_config in info.config.vectors.items():
                if vec_config.size:
                    dense_gb = info.points_count * vec_config.size * 4 / (1024**3)
                    print(f"✓ Dense '{vec_name}': {vec_config.size}D × {info.points_count:,} pts = {dense_gb:.2f} GB")

        print(f"✓ Indexed vectors: {info.indexed_vectors_count:,}")

        # paper id coverage
        if unique_papers:
            aw = len(self.stats["unique_papers_with_arxiv"])
            dw = len(self.stats["unique_papers_with_doi"])
            print(f"✓ Papers with arXiv ID: {aw:,} ({100*aw/unique_papers:.1f}%)")
            print(f"✓ Papers with DOI:      {dw:,} ({100*dw/unique_papers:.1f}%)")

        # cited span coverage
        cs = self.stats["cited_span_total"]
        if cs:
            any_id = self.stats["cited_span_has_any"]
            print(f"✓ Cited spans with any public ID: {any_id:,} / {cs:,} ({100*any_id/cs:.1f}%)")

        storage_path = PATHS.qdrant_storage / "collections" / self.collection_name / "0" / "segments"
        if storage_path.exists():
            total_size = sum(f.stat().st_size for f in storage_path.rglob('*') if f.is_file())
            print(f"✓ Actual disk storage: {total_size / (1024**3):.2f} GB")

        print(f"\n✓ Inspection complete!")

    def run_full_inspection(self) -> bool:
        steps = [
            ("Connecting to Qdrant", self.connect_and_verify),
            ("Fetching collection info", self.fetch_collection_info),
            ("Scrolling all points", self.scroll_all_points),
        ]

        for step_name, step_func in steps:
            logger.info(f"→ {step_name}…")
            if not step_func():
                logger.error(f"✗ Failed at step: {step_name}")
                return False

        self.print_collection_overview()
        self.print_vector_statistics()
        self.print_paper_id_coverage()
        self.print_cited_span_coverage()
        self.print_paper_statistics()
        self.print_chunk_statistics()
        self.print_disk_storage()
        self.print_snapshots()
        self.print_sample_payload()
        self.print_summary()

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Qdrant collection inspector (full collection)"
    )
    parser.add_argument(
        "--collection",
        default="unarxive_chunks",
        help="Collection name (default: unarxive_chunks)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1000,
        help="Scroll batch size (default: 5000)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    inspector = QdrantInspector(
        collection_name=args.collection,
        batch_size=args.batch,
        verbose=args.verbose,
    )

    success = inspector.run_full_inspection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
