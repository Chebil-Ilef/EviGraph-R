#!/bin/bash
#SBATCH --job-name=evigraph-inspect
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/inspect_%x_%j.log

# Usage:
#   sbatch run_qdrant_inspection.sh
#   sbatch run_qdrant_inspection.sh --collection unarxive_chunks --batch 2000

set -euo pipefail

REPO_DIR="/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R"
cd "$REPO_DIR"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

export QDRANT_SIF_PATH="${QDRANT_SIF_PATH:-${REPO_DIR}/qdrant.sif}"

echo "================================================"
echo "Starting Qdrant on this node…"
echo "================================================"

uv run python -c "
import logging, os
from config.settings import get_qdrant_profile
from utils.qdrant import ensure_qdrant_runtime, qdrant_client

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

profile_name = os.getenv('INDEXING_PROFILE', 'hpc')
profile = get_qdrant_profile(profile_name)

try:
    ensure_qdrant_runtime(profile, startup_timeout=3600)
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
echo "Running full collection inspection…"
echo "================================================"
echo ""

uv run python experiments/qdrant_indexed_collection/qdrant_inspector.py "$@"

echo ""
echo "================================================"
echo "✓ Inspection complete"
echo "================================================"
