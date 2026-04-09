#!/bin/bash
#
# Batch Quality Evaluation Runner
#
# Run from login node - automatically requests interactive compute node
# Usage: ./experiments/demo_pipeline/quality_evaluation/run_batch_eval.sh
#        ./experiments/demo_pipeline/quality_evaluation/run_batch_eval.sh --top-k 15
#        ./experiments/demo_pipeline/quality_evaluation/run_batch_eval.sh --queries-file my_queries.txt

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXTRA_ARGS="$*"

echo "════════════════════════════════════════════════════════════════"
echo "  EviGraph-R Batch Quality Evaluation"
echo "════════════════════════════════════════════════════════════════"
echo "  Requesting interactive compute node..."
echo "  (1 GPU, 32GB RAM, 2 hour time limit)"
echo ""

srun -N 1 --partition=capella-interactive --gres=gpu:1 --mem=32G --time=2:00:00 bash -c "
set -e

cd ${REPO_DIR}

# Setup environment
export PYTHONPATH=\"\${PWD}/src:\${PYTHONPATH:-}\"
export SINGULARITY_CACHEDIR=\"\${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}\"
export SINGULARITY_TMPDIR=\"\${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}\"
mkdir -p \"\$SINGULARITY_CACHEDIR\" \"\$SINGULARITY_TMPDIR\"

# Setup Qdrant Singularity image
export QDRANT_SIF_PATH=\"\${QDRANT_SIF_PATH:-\${HOME}/qdrant.sif}\"
if [[ ! -f \"\$QDRANT_SIF_PATH\" ]]; then
  echo \"Building Qdrant Singularity image from docker://qdrant/qdrant (takes ~2 min)…\"
  singularity build \"\$QDRANT_SIF_PATH\" docker://qdrant/qdrant
  echo \"✓ Qdrant image built: \$QDRANT_SIF_PATH\"
fi

echo \"════════════════════════════════════════════════════════════════\"
echo \"  Running on: \$(hostname) | Job: \$SLURM_JOB_ID\"
echo \"════════════════════════════════════════════════════════════════\"
echo \"\"

# Setup logging
mkdir -p logs
LOG_FILE=\"logs/batch_eval_\$(date +%Y%m%d_%H%M%S).log\"
echo \"Logging to: \$LOG_FILE\"
echo \"\"

# Run batch eval (Qdrant auto-starts via ensure_qdrant_runtime)
uv run python experiments/demo_pipeline/quality_evaluation/batch_eval.py ${EXTRA_ARGS} 2>&1 | tee \"\$LOG_FILE\"

EXIT_CODE=\$?
echo \"\"
echo \"════════════════════════════════════════════════════════════════\"
echo \"  Exit code: \$EXIT_CODE\"
echo \"  Results: experiments/demo_pipeline/quality_evaluation/_data/\"
echo \"════════════════════════════════════════════════════════════════\"
exit \$EXIT_CODE
"
