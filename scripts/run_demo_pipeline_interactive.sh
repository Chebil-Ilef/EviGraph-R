#!/bin/bash
#
# Interactive Dev Pipeline Runner
#
# Run from login node - automatically requests interactive compute node
# Usage: ./scripts/run_demo_pipeline_interactive.sh 

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "════════════════════════════════════════════════════════════════"
echo "  EviGraph-R Dev Pipeline - Interactive Mode"
echo "════════════════════════════════════════════════════════════════"
echo "  Requesting interactive compute node..."
echo "  (1 GPU, 16GB RAM, 1 hour time limit)"
echo ""

srun -N 1 --partition=capella-interactive --gres=gpu:1 --mem=16G --time=1:00:00 bash -c "
set -e

cd ${REPO_DIR}

# Setup environment
export PYTHONPATH=\"\${PWD}/src:\${PYTHONPATH:-}\"
export SINGULARITY_CACHEDIR=\"\${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}\"
export SINGULARITY_TMPDIR=\"\${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}\"
mkdir -p \"\$SINGULARITY_CACHEDIR\" \"\$SINGULARITY_TMPDIR\"

echo \"════════════════════════════════════════════════════════════════\"
echo \"  Running on: \$(hostname) | Job: \$SLURM_JOB_ID\"
echo \"════════════════════════════════════════════════════════════════\"
echo \"\"

# Run pipeline (Qdrant auto-starts!)
uv run python examples/pipeline_demo.py

EXIT_CODE=\$?
echo \"\"
echo \"════════════════════════════════════════════════════════════════\"
echo \"  Exit code: \$EXIT_CODE\"
echo \"════════════════════════════════════════════════════════════════\"
exit \$EXIT_CODE
"
