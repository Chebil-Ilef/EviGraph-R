#!/usr/bin/env python3
"""
Advanced Qdrant collection inspector – generates detailed statistics about:
- Vector counts and storage estimates
- Unique papers and chunk distributions
- Segment information
- Payload field analysis
- Sparse vs dense vector statistics
- Temporal analysis (if timestamps present)

Usage (standalone):
    python src/utils/qdrant_inspector.py [--collection unarxive_chunks] [--sample 5000]

Usage (from SLURM script):
    Add to inspect_qdrant_stats.sh or call directly in uv run python
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Any, Optional

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from config.settings import QDRANT_ACTIVE, PATHS
    from utils.qdrant import qdrant_client, check_qdrant_alive
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.info("Make sure to run from project root: uv run python src/utils/qdrant_inspector.py")
    sys.exit(1)


class QdrantInspector:
    """Comprehensive Qdrant collection inspector."""
    
    def __init__(
        self,
        collection_name: str = "unarxive_chunks",
        sample_size: int = 5000,
        verbose: bool = False
    ):
        self.collection_name = collection_name
        self.sample_size = sample_size
        self.verbose = verbose
        
        self.client = qdrant_client(timeout=300)
        self.dense_vec_name = QDRANT_ACTIVE.dense_vector_name or "dense"
        self.sparse_vec_name = QDRANT_ACTIVE.sparse_vector_name or "sparse"
        
        self.collection_info = None
        self.points_sample = []
        self.stats = {
            "dense_vectors": [],
            "sparse_vectors": [],
            "unique_papers": set(),
            "papers_by_year": Counter(),
            "papers_by_category": Counter(),
            "chunk_counts_by_paper": [],
            "section_titles": Counter(),
            "chunk_types": Counter(),
        }
    
    def connect_and_verify(self) -> bool:
        """Verify Qdrant is reachable."""
        try:
            check_qdrant_alive(self.client)
            logger.info("✓ Qdrant is alive and responding")
            return True
        except Exception as e:
            logger.error(f"✗ Qdrant connection failed: {e}")
            return False
    
    def fetch_collection_info(self) -> bool:
        """Fetch collection metadata."""
        try:
            self.collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"✓ Fetched collection info for '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to fetch collection info: {e}")
            return False
    
    def fetch_point_samples(self) -> bool:
        """Fetch sample points from collection."""
        if not self.collection_info:
            logger.error("Collection info not available")
            return False
        
        total_points = self.collection_info.points_count
        if total_points == 0:
            logger.error("Collection is empty")
            return False
        
        sample_limit = min(self.sample_size, total_points)
        logger.info(f"Fetching sample of {sample_limit:,} points from {total_points:,} total…")
        
        try:
            self.points_sample, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=sample_limit,
                with_payload=True,
                with_vectors=True,
            )
            logger.info(f"✓ Retrieved {len(self.points_sample):,} point samples")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to fetch points: {e}")
            return False
    
    def analyze_samples(self) -> None:
        """Analyze the point samples."""
        logger.info("Analyzing point samples…")
        
        for point in self.points_sample:
            payload = point.payload or {}
            vectors = point.vector or {}
            
            # Dense vector analysis
            if self.dense_vec_name in vectors:
                dense_vec = vectors[self.dense_vec_name]
                if isinstance(dense_vec, (list, np.ndarray)):
                    self.stats["dense_vectors"].append(len(dense_vec))
            
            # Sparse vector analysis
            if self.sparse_vec_name in vectors:
                sparse_vec = vectors[self.sparse_vec_name]
                if isinstance(sparse_vec, dict) and 'indices' in sparse_vec:
                    self.stats["sparse_vectors"].append(len(sparse_vec['indices']))
                elif hasattr(sparse_vec, 'indices'):
                    self.stats["sparse_vectors"].append(len(sparse_vec.indices))
            
            # Paper tracking
            paper_id = payload.get('paper_id_arxiv')
            if paper_id:
                self.stats["unique_papers"].add(paper_id)
            
            # Metadata tracking
            year = payload.get('year')
            if year:
                self.stats["papers_by_year"][year] += 1
            
            categories = payload.get('categories', [])
            if isinstance(categories, list):
                for cat in categories:
                    self.stats["papers_by_category"][cat] += 1
            
            section_title = payload.get('section_title', 'Unknown')
            self.stats["section_titles"][section_title] += 1
            
            chunk_type = payload.get('chunk_type', 'unknown')
            self.stats["chunk_types"][chunk_type] += 1
    
    def print_collection_overview(self) -> None:
        """Print collection overview."""
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
        
        # Vector configuration
        if hasattr(info, 'config') and info.config and hasattr(info.config, 'vectors'):
            print(f"\nVector Configuration:")
            for vec_name, vec_config in info.config.vectors.items():
                print(f"  - {vec_name}:")
                print(f"    Size: {vec_config.size} dimensions")
                print(f"    Distance: {vec_config.distance}")
                if hasattr(vec_config, 'datatype'):
                    print(f"    Datatype: {vec_config.datatype}")
        
        # Payload schema
        if hasattr(info, 'payload_schema') and info.payload_schema:
            print(f"\nPayload Schema Fields ({len(info.payload_schema)}):")
            for field_name in sorted(info.payload_schema.keys())[:15]:
                field_type = info.payload_schema[field_name]
                print(f"  - {field_name}: {field_type}")
            if len(info.payload_schema) > 15:
                print(f"  … and {len(info.payload_schema) - 15} more")
    
    def print_vector_statistics(self) -> None:
        """Print detailed vector statistics."""
        print("\n" + "="*80)
        print("VECTOR STATISTICS (from sample)")
        print("="*80)
        
        info = self.collection_info
        
        # Dense vectors
        if self.stats["dense_vectors"]:
            dims = self.stats["dense_vectors"][0]
            count = len(self.stats["dense_vectors"])
            print(f"\nDense Vectors ({self.dense_vec_name}):")
            print(f"  Sample count: {count:,}")
            print(f"  Dimensions: {dims}")
            print(f"  Bytes per vector (float32): {dims * 4:,}")
            
            # Estimate total storage
            total_bytes = info.points_count * dims * 4
            total_gb = total_bytes / (1024**3)
            print(f"  Estimated total storage: {total_gb:.2f} GB ({total_bytes:,} bytes)")
        
        # Sparse vectors
        if self.stats["sparse_vectors"]:
            indices_list = self.stats["sparse_vectors"]
            avg_indices = np.mean(indices_list)
            min_indices = min(indices_list)
            max_indices = max(indices_list)
            
            print(f"\nSparse Vectors ({self.sparse_vec_name}):")
            print(f"  Sample count: {len(indices_list):,}")
            print(f"  Avg indices per vector: {avg_indices:.1f}")
            print(f"  Min indices: {min_indices}")
            print(f"  Max indices: {max_indices}")
            print(f"  Std dev: {np.std(indices_list):.1f}")
            
            # Estimate sparse storage (index + value = 8 bytes per entry)
            avg_bytes_per_vector = avg_indices * 8
            total_sparse_bytes = info.points_count * avg_bytes_per_vector
            total_sparse_gb = total_sparse_bytes / (1024**3)
            print(f"  Estimated total storage: {total_sparse_gb:.2f} GB ({total_sparse_bytes:,} bytes)")
        
        # Combined estimate
        total_estimated = 0
        if self.stats["dense_vectors"]:
            total_estimated += info.points_count * self.stats["dense_vectors"][0] * 4 / (1024**3)
        if self.stats["sparse_vectors"]:
            avg_sparse = np.mean(self.stats["sparse_vectors"]) * 8
            total_estimated += info.points_count * avg_sparse / (1024**3)
        
        print(f"\n✓ Total estimated vector storage: {total_estimated:.2f} GB")
    
    def print_paper_statistics(self) -> None:
        """Print paper statistics."""
        print("\n" + "="*80)
        print("PAPER STATISTICS (from sample)")
        print("="*80)
        
        info = self.collection_info
        unique_papers_sample = len(self.stats["unique_papers"])
        
        print(f"\nUnique papers in sample: {unique_papers_sample:,}")
        
        if len(self.points_sample) > 0 and unique_papers_sample > 0:
            avg_chunks_per_paper = len(self.points_sample) / unique_papers_sample
            estimated_total_papers = info.points_count / avg_chunks_per_paper
            
            print(f"Average chunks per paper (sample): {avg_chunks_per_paper:.1f}")
            print(f"Estimated total unique papers: {int(estimated_total_papers):,}")
        
        # Year distribution
        if self.stats["papers_by_year"]:
            print(f"\nPapers by year (top 10):")
            for year, count in self.stats["papers_by_year"].most_common(10):
                print(f"  {year}: {count:,}")
        
        # Category distribution  
        if self.stats["papers_by_category"]:
            print(f"\nPapers by category (top 10):")
            for category, count in self.stats["papers_by_category"].most_common(10):
                print(f"  {category}: {count:,}")
    
    def print_chunk_statistics(self) -> None:
        """Print chunk-level statistics."""
        print("\n" + "="*80)
        print("CHUNK STATISTICS (from sample)")
        print("="*80)
        
        if self.stats["chunk_types"]:
            print(f"\nChunk types:")
            for chunk_type, count in self.stats["chunk_types"].most_common():
                pct = (count / len(self.points_sample)) * 100
                print(f"  {chunk_type}: {count:,} ({pct:.1f}%)")
        
        if self.stats["section_titles"]:
            print(f"\nSection titles (top 15):")
            for section, count in self.stats["section_titles"].most_common(15):
                pct = (count / len(self.points_sample)) * 100
                print(f"  {section}: {count:,} ({pct:.1f}%)")
    
    def print_disk_storage(self) -> None:
        """Print disk storage analysis."""
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
        """Print snapshot information."""
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
        """Print sample payload structure."""
        print("\n" + "="*80)
        print("SAMPLE PAYLOAD STRUCTURE")
        print("="*80)
        
        if not self.points_sample:
            print("No point samples available")
            return
        
        sample_point = self.points_sample[0]
        print(f"\nSample chunk payload (Point ID: {sample_point.id}):")
        
        if sample_point.payload:
            for key in sorted(sample_point.payload.keys()):
                value = sample_point.payload[key]
                
                if isinstance(value, str):
                    preview = value[:60] + ("…" if len(value) > 60 else "")
                    print(f"  - {key}: \"{preview}\"")
                elif isinstance(value, (int, float, bool)):
                    print(f"  - {key}: {value}")
                elif isinstance(value, list):
                    print(f"  - {key}: [{len(value)} items] {str(value[:2])[1:-1]}{'…' if len(value) > 2 else ''}")
                elif isinstance(value, dict):
                    print(f"  - {key}: {{dict with {len(value)} keys}}")
                else:
                    print(f"  - {key}: {type(value).__name__}")
    
    def print_summary(self) -> None:
        """Print summary statistics."""
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        info = self.collection_info
        
        print(f"\n✓ Total points in collection: {info.points_count:,}")
        print(f"✓ Unique papers (sample): {len(self.stats['unique_papers']):,}")
        
        if len(self.points_sample) > 0 and len(self.stats["unique_papers"]) > 0:
            avg_chunks_per_paper = len(self.points_sample) / len(self.stats["unique_papers"])
            estimated_papers = info.points_count / avg_chunks_per_paper
            print(f"✓ Estimated total unique papers: {int(estimated_papers):,}")
        
        if self.stats["dense_vectors"]:
            dims = self.stats["dense_vectors"][0]
            dense_gb = info.points_count * dims * 4 / (1024**3)
            print(f"✓ Dense vectors: {dims}D x {info.points_count:,} points = {dense_gb:.2f} GB")
        
        if self.stats["sparse_vectors"]:
            avg_indices = np.mean(self.stats["sparse_vectors"])
            sparse_gb = info.points_count * avg_indices * 8 / (1024**3)
            print(f"✓ Sparse vectors: ~{avg_indices:.0f} indices/vector x {info.points_count:,} points = {sparse_gb:.2f} GB")
        
        print(f"✓ Indexed vectors: {info.indexed_vectors_count:,}")
        
        storage_path = PATHS.qdrant_storage / "collections" / self.collection_name / "0" / "segments"
        if storage_path.exists():
            total_size = sum(f.stat().st_size for f in storage_path.rglob('*') if f.is_file())
            print(f"✓ Actual disk storage: {total_size / (1024**3):.2f} GB")
        
        print(f"\n✓ Inspection complete!")
    
    def run_full_inspection(self) -> bool:
        """Run complete inspection pipeline."""
        steps = [
            ("Connecting to Qdrant", self.connect_and_verify),
            ("Fetching collection info", self.fetch_collection_info),
            ("Fetching point samples", self.fetch_point_samples),
            ("Analyzing samples", self.analyze_samples),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"→ {step_name}…")
            if not step_func():
                logger.error(f"✗ Failed at step: {step_name}")
                return False
        
        # Print all reports
        self.print_collection_overview()
        self.print_vector_statistics()
        self.print_paper_statistics()
        self.print_chunk_statistics()
        self.print_disk_storage()
        self.print_snapshots()
        self.print_sample_payload()
        self.print_summary()
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Qdrant collection inspector"
    )
    parser.add_argument(
        "--collection",
        default="unarxive_chunks",
        help="Collection name (default: unarxive_chunks)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="Sample size for analysis (default: 5000)"
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
        sample_size=args.sample,
        verbose=args.verbose
    )
    
    success = inspector.run_full_inspection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
