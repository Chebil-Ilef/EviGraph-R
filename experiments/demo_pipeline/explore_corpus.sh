#!/bin/bash
#
# Corpus Explorer — samples chunks from Qdrant to identify good test queries
#
# Usage: ./experiments/demo_pipeline/explore_corpus.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_DIR

echo "════════════════════════════════════════════════════════════════"
echo "  EviGraph-R Corpus Explorer"
echo "════════════════════════════════════════════════════════════════"
echo "  Requesting interactive compute node..."
echo ""

srun -N 1 --partition=capella-interactive --gres=gpu:1 --mem=8G --time=0:30:00 bash -c "
set -e

cd '${REPO_DIR}'

export PYTHONPATH=\"\${PWD}/src:\${PYTHONPATH:-}\"
export SINGULARITY_CACHEDIR=\"\${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}\"
export SINGULARITY_TMPDIR=\"\${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}\"
mkdir -p \"\$SINGULARITY_CACHEDIR\" \"\$SINGULARITY_TMPDIR\"

export QDRANT_SIF_PATH=\"\${QDRANT_SIF_PATH:-\${HOME}/qdrant.sif}\"
if [[ ! -f \"\$QDRANT_SIF_PATH\" ]]; then
  singularity build \"\$QDRANT_SIF_PATH\" docker://qdrant/qdrant
fi

echo \"Running on: \$(hostname) | Job: \$SLURM_JOB_ID\"
echo \"\"

uv run python experiments/demo_pipeline/explore_corpus.py 2>&1
"
