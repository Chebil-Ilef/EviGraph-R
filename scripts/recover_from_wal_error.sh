#!/bin/bash
# Recover from Qdrant WAL corruption without re-ingesting all batches
# 
# Usage: bash scripts/recover_from_wal_error.sh
# 
# What it does:
#   1. Stops Qdrant Singularity instance
#   2. Deletes corrupted WAL files (but keeps shards + progress log)
#   3. Shows what's in progress log
#   4. Prompts to resume ingestion (only missing batches)
#
# Result: Loss ~0 batches (vs 900 with WIPE_STORAGE)

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "======================================"
echo "Qdrant WAL Corruption Recovery"
echo "======================================"
echo ""

# Step 1: Stop Qdrant
echo "[1/4] Stopping Qdrant Singularity instance..."
if singularity instance list | grep -q evigraph-qdrant; then
  singularity instance stop evigraph-qdrant 2>/dev/null || true
  sleep 2
  echo "  ✓ Stopped"
else
  echo "  (Not running)"
fi

# Step 2: Delete corrupted WAL
echo "[2/4] Deleting corrupted WAL files..."
if [[ -d "storage/collections/unarxive_chunks/0/wal/" ]]; then
  rm -rf storage/collections/unarxive_chunks/0/wal/
  echo "  ✓ Deleted WAL directory"
fi

if [[ -f "storage/collections/unarxive_chunks/collection.json" ]]; then
  rm -f storage/collections/unarxive_chunks/collection.json
  echo "  ✓ Deleted collection metadata (will be recreated)"
fi

# Step 3: Show progress
echo "[3/4] Checking progress log..."
if [[ -f "_data/progress/ingested_shards.jsonl" ]]; then
  INGESTED_COUNT=$(cat _data/progress/ingested_shards.jsonl | wc -l)
  UNIQUE_SHARDS=$(cut -d'"' -f4 _data/progress/ingested_shards.jsonl 2>/dev/null | sort -u | wc -l)
  echo "  Total entries: $INGESTED_COUNT"
  echo "  Unique shards: $UNIQUE_SHARDS"
  echo ""
  echo "  Last 3 ingested:"
  tail -3 _data/progress/ingested_shards.jsonl | jq -r '.stem' 2>/dev/null | sed 's/^/    - /'
else
  echo "  ⚠ No progress log found (will start from scratch)"
fi

echo ""
echo "======================================"
echo "Recovery Options:"
echo "======================================"
echo ""
echo "A) Resume ingestion (skip already-ingested batches)"
echo "   Command: sbatch --export=ALL,TOTAL_TASKS=1,INGEST_ONLY=1,RESUME_INGEST=1 scripts/run_indexing_array_capella.sh"
echo ""
echo "B) Do nothing (manual recovery)"
echo "   - You can run the sbatch command yourself"
echo "   - Or delete more files if needed"
echo ""
read -p "Continue with auto-resume? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "[4/4] Submitting recovery job..."
  sbatch --export=ALL,TOTAL_TASKS=1,INGEST_ONLY=1,RESUME_INGEST=1 \
    scripts/run_indexing_array_capella.sh
  echo ""
  echo "  ✓ Job submitted"
  echo ""
  echo "Monitor with: squeue -u \$USER"
elif [[ $REPLY =~ ^[Nn]$ ]]; then
  echo "[4/4] Skipped auto-resume"
  echo ""
  echo "To resume manually later, run:"
  echo "  sbatch --export=ALL,TOTAL_TASKS=1,INGEST_ONLY=1,RESUME_INGEST=1 scripts/run_indexing_array_capella.sh"
else
  echo "Invalid response"
  exit 1
fi

echo ""
echo "======================================"
echo "What was recovered:"
echo "======================================"
echo "  ✓ Shard files (_data/shards/) - intact"
echo "  ✓ Progress log (_data/progress/) - intact"  
echo "  ✓ Reconstructed WAL (fresh empty)"
echo "  ✓ Only missing batches will re-ingest"
echo ""
echo "Expected loss: ~0 batches"
echo "Expected time: ~5-10 minutes (vs 30-40 min full re-ingest)"
echo ""
