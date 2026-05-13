#!/bin/bash
#SBATCH --job-name=evigraph-genq
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R/logs/genq_%j.log

set -euo pipefail

REPO_DIR=/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R
cd "$REPO_DIR"

if [[ -f "${REPO_DIR}/.env" ]]; then
    set -a; source "${REPO_DIR}/.env"; set +a
fi

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export INDEXING_PROFILE=hpc

echo "Starting question generation at $(date)"

# Start Qdrant so _fetch_section_texts can pull body chunks
uv run python -c "
from utils.qdrant import ensure_qdrant_runtime
ensure_qdrant_runtime('hpc', startup_timeout=600)
print('Qdrant ready')
"

uv run python -m experiments.index_comparison.generate_questions \
    --papers experiments/index_comparison/results/sampled_papers.jsonl \
    --output experiments/index_comparison/results/questions.jsonl \
    --workers 4

echo "Done at $(date)"
echo "Questions: experiments/index_comparison/results/questions.jsonl"
