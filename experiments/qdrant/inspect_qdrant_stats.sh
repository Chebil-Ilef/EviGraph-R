#!/bin/bash
#SBATCH --job-name=evigraph-inspect
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=188000M
#SBATCH --time=01:00:00
#SBATCH --output=/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R/logs/STATS_%x_%j.log

set -euo pipefail

REPO_DIR="/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R"
cd "$REPO_DIR"

if [[ ! -d "$REPO_DIR/src" ]]; then
  echo "ERROR: Cannot find project root (src/ directory not found)"
  echo "  Tried: $REPO_DIR"
  exit 1
fi

cd "$REPO_DIR"
echo "Project root: $REPO_DIR"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

export QDRANT_SIF_PATH="${QDRANT_SIF_PATH:-${REPO_DIR}/qdrant.sif}"
if [[ ! -f "$QDRANT_SIF_PATH" ]]; then
  echo "⚠ Qdrant SIF not found at: $QDRANT_SIF_PATH"
  echo "  (Normally it should already exist. Using default path for indexing job.)"
  export QDRANT_SIF_PATH="${REPO_DIR}/qdrant.sif"
fi

# Start Qdrant on THIS node (different from indexing job node)
# Both will use the same shared storage directory
echo "================================================"
echo "Starting Qdrant on this node…"
echo "================================================"

uv run python -c "
import logging
import os
from config.settings import get_qdrant_profile
from utils.qdrant import ensure_qdrant_runtime, qdrant_client

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

profile_name = os.getenv('INDEXING_PROFILE', 'hpc')
profile = get_qdrant_profile(profile_name)

try:
    ensure_qdrant_runtime(profile, startup_timeout=600)
    logger.info('✓ Qdrant started')
    
    client = qdrant_client(timeout=30)
    collections = client.get_collections()
    logger.info(f'✓ Collections: {[c.name for c in collections.collections]}')
except Exception as e:
    logger.error(f'✗ Failed: {e}')
    exit(1)
"

echo ""
echo "================================================"
echo "Querying Qdrant Collection Statistics"
echo "================================================"
echo ""

# Main inspection script
uv run python << 'INSPECTION_SCRIPT'
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

# Ensure imports work
sys.path.insert(0, '/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R/src')

from utils.qdrant import qdrant_client
from config.settings import QDRANT_ACTIVE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLLECTION_NAME = "unarxive_chunks"

try:
    client = qdrant_client(timeout=120)
    
    # Get collection info
    print("\n" + "="*80)
    print("COLLECTION OVERVIEW")
    print("="*80)
    
    info = client.get_collection(COLLECTION_NAME)
    
    print(f"\nCollection Name: {COLLECTION_NAME}")
    print(f"Status: {info.status}")
    print(f"Total Points: {info.points_count:,}")
    print(f"Indexed Vectors: {info.indexed_vectors_count:,}")
    
    # Get config info
    if hasattr(info, 'config') and info.config:
        config = info.config
        print(f"\nVector Configuration:")
        if hasattr(config, 'vectors') and config.vectors:
            for vec_name, vec_config in config.vectors.items():
                print(f"  - {vec_name}:")
                print(f"    Size: {vec_config.size} dimensions")
                print(f"    Distance: {vec_config.distance}")
                if hasattr(vec_config, 'datatype'):
                    print(f"    Datatype: {vec_config.datatype}")
    
    # Get storage info
    print(f"\nStorage Information:")
    if hasattr(info, 'data_dir'):
        print(f"  Data Directory: {info.data_dir}")
    
    # Check segments
    print(f"\n" + "="*80)
    print("SEGMENT INFORMATION")
    print("="*80)
    
    segments_info = client.get_collection(COLLECTION_NAME)
    if hasattr(segments_info, 'payload_schema') and segments_info.payload_schema:
        print(f"\nPayload Schema Fields:")
        for field_name in sorted(segments_info.payload_schema.keys()):
            field_type = segments_info.payload_schema[field_name]
            print(f"  - {field_name}: {field_type}")
    
    # Query points with limit to get sample data for statistics
    print(f"\n" + "="*80)
    print("SAMPLING POINTS FOR DETAILED STATISTICS")
    print("="*80)
    
    # Sample multiple batches throughout the collection
    sample_size = min(5000, info.points_count)
    print(f"\nSampling {sample_size:,} points (of {info.points_count:,} total)…")

    # Try with payload only first (vectors may be unavailable in mutable segments)
    try:
        points = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=sample_size,
            with_payload=True,
            with_vectors=True,
        )[0]
        print(f"✓ Retrieved {len(points):,} point samples (with vectors)")
    except Exception as scroll_err:
        logger.warning(f"Scroll with vectors failed ({scroll_err}), retrying payload-only…")
        points = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=sample_size,
            with_payload=True,
            with_vectors=False,
        )[0]
        print(f"✓ Retrieved {len(points):,} point samples (payload only)")

    print(f"✓ Retrieved {len(points):,} point samples")
    
    # Analyze samples
    print(f"\n" + "="*80)
    print("VECTOR STATISTICS (from sample)")
    print("="*80)
    
    dense_vector_counts = []
    sparse_vector_counts = []
    unique_papers = set()
    papers_by_batch = defaultdict(int)
    
    dense_vec_name = QDRANT_ACTIVE.dense_vector_name or "dense"
    sparse_vec_name = QDRANT_ACTIVE.sparse_vector_name or "sparse"
    
    for point in points:
        payload = point.payload or {}
        vectors = point.vector or {}
        
        # Track dense vectors
        if dense_vec_name in vectors:
            dense_vec = vectors[dense_vec_name]
            if isinstance(dense_vec, list):
                dense_vector_counts.append(len(dense_vec))
            elif hasattr(dense_vec, '__len__'):
                dense_vector_counts.append(len(dense_vec))
        
        # Track sparse vectors
        if sparse_vec_name in vectors:
            sparse_vec = vectors[sparse_vec_name]
            if isinstance(sparse_vec, dict) and 'indices' in sparse_vec:
                sparse_vector_counts.append(len(sparse_vec['indices']))
            elif hasattr(sparse_vec, 'indices'):
                sparse_vector_counts.append(len(sparse_vec.indices))
        
        # Track papers
        paper_id = payload.get('paper_id_arxiv')
        if paper_id:
            unique_papers.add(paper_id)
        
        # Track batch
        chunk_uid = payload.get('chunk_uid', '')
        if chunk_uid and '_' in chunk_uid:
            try:
                batch_part = payload.get('chunk_uid', '').split('-')[0] if '-' in payload.get('chunk_uid', '') else None
                if not batch_part and 'batch' in json.dumps(payload):
                    pass  # Can't extract from this format
            except:
                pass
    
    if dense_vector_counts:
        print(f"\nDense Vectors ({dense_vec_name}):")
        print(f"  Count in sample: {len(dense_vector_counts):,}")
        print(f"  Dimensions: {dense_vector_counts[0] if dense_vector_counts else 0}")
        print(f"  Bytes per vector (float32): {dense_vector_counts[0] * 4 if dense_vector_counts else 0:,} bytes")
        dense_total_bytes = info.points_count * dense_vector_counts[0] * 4 if dense_vector_counts else 0
        dense_total_gb = dense_total_bytes / (1024**3)
        print(f"  Estimated total dense storage: {dense_total_gb:.2f} GB")
    
    if sparse_vector_counts:
        print(f"\nSparse Vectors ({sparse_vec_name}):")
        print(f"  Count in sample: {len(sparse_vector_counts):,}")
        print(f"  Avg indices per vector: {sum(sparse_vector_counts) / len(sparse_vector_counts):.1f}")
        print(f"  Min indices: {min(sparse_vector_counts)}")
        print(f"  Max indices: {max(sparse_vector_counts)}")
        # Estimate: each sparse index ~8 bytes (uint32 index + float32 value)
        avg_sparse_bytes = (sum(sparse_vector_counts) / len(sparse_vector_counts)) * 8
        sparse_total_bytes = info.points_count * avg_sparse_bytes
        sparse_total_gb = sparse_total_bytes / (1024**3)
        print(f"  Estimated total sparse storage: {sparse_total_gb:.2f} GB")
    
    print(f"\n" + "="*80)
    print("PAPER STATISTICS (from sample)")
    print("="*80)
    
    print(f"\nUnique papers in sample: {len(unique_papers):,}")
    
    # Estimate total unique papers
    if info.points_count > 0 and len(points) > 0:
        avg_chunks_per_paper = len(points) / len(unique_papers) if unique_papers else 1
        estimated_total_papers = info.points_count / avg_chunks_per_paper
        print(f"Avg chunks per paper: {avg_chunks_per_paper:.1f}")
        print(f"Estimated total unique papers: {int(estimated_total_papers):,}")
    
    print(f"\n" + "="*80)
    print("PAYLOAD FIELD SAMPLE")
    print("="*80)
    
    # Show sample point payload structure
    if points:
        sample_point = points[0]
        print(f"\nSample chunk payload fields:")
        if sample_point.payload:
            for key in sorted(sample_point.payload.keys()):
                value = sample_point.payload[key]
                if isinstance(value, (str, int, float, bool)):
                    preview = str(value)[:60]
                    print(f"  - {key}: {preview}")
                elif isinstance(value, list):
                    print(f"  - {key}: [list with {len(value)} items]")
                elif isinstance(value, dict):
                    print(f"  - {key}: {{dict with {len(value)} keys}}")
                else:
                    print(f"  - {key}: {type(value).__name__}")
    
    print(f"\n" + "="*80)
    print("DISK STORAGE ANALYSIS")
    print("="*80)
    
    storage_path = Path("storage/collections/unarxive_chunks/0/segments")
    if storage_path.exists():
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
        print(f"\nSegments on disk: {segment_count}")
        print(f"Total segment storage: {total_gb:.2f} GB")
        
        if segment_sizes:
            segment_sizes.sort(key=lambda x: x[1], reverse=True)
            avg_segment_size = total_size / segment_count / (1024**2)
            print(f"Average segment size: {avg_segment_size:.1f} MB")
            print(f"\nLargest segments:")
            for seg_name, seg_size in segment_sizes[:10]:
                seg_mb = seg_size / (1024**2)
                print(f"  - {seg_name}: {seg_mb:.1f} MB")
    else:
        print(f"Storage path not found: {storage_path}")
    
    print(f"\n" + "="*80)
    print("CHECKPOINT SNAPSHOT INFO")
    print("="*80)
    
    snapshots_path = Path("snapshots")
    if snapshots_path.exists():
        snapshots = list(snapshots_path.glob("*.tar"))
        if snapshots:
            print(f"\nSnapshots found: {len(snapshots)}")
            total_snapshot_size = 0
            for snap in sorted(snapshots, reverse=True)[:5]:
                size_gb = snap.stat().st_size / (1024**3)
                print(f"  - {snap.name}: {size_gb:.2f} GB (created: {snap.stat().st_mtime})")
                total_snapshot_size += snap.stat().st_size
            print(f"Total snapshot storage: {total_snapshot_size / (1024**3):.2f} GB")
        else:
            print("No snapshots found")
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_estimated_gb = 0
    if dense_vector_counts and dense_vector_counts[0] > 0:
        total_estimated_gb += info.points_count * dense_vector_counts[0] * 4 / (1024**3)
    if sparse_vector_counts and sum(sparse_vector_counts) > 0:
        avg_sparse_bytes = (sum(sparse_vector_counts) / len(sparse_vector_counts)) * 8
        total_estimated_gb += info.points_count * avg_sparse_bytes / (1024**3)
    
    print(f"\n✓ Total points: {info.points_count:,}")
    print(f"✓ Unique papers (estimated): {int(estimated_total_papers) if info.points_count > 0 else 0:,}")
    print(f"✓ Dense vectors: {len(dense_vector_counts):,}D x {info.points_count:,} points")
    print(f"✓ Sparse vectors available: {len(sparse_vector_counts) > 0}")
    print(f"✓ Estimated vector storage: {total_estimated_gb:.2f} GB")
    if storage_path.exists():
        print(f"✓ Actual disk storage: {total_gb:.2f} GB")
    
    print("\n✓ Inspection complete!")

except Exception as e:
    import traceback
    logger.error(f"Error during inspection: {e}")
    traceback.print_exc()
    exit(1)

INSPECTION_SCRIPT

echo ""
echo "================================================"
echo "✓ Inspection complete"
echo "================================================"
