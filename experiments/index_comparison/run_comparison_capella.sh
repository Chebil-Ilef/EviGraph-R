#!/bin/bash
#SBATCH --job-name=evigraph-comparison
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --output=/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R/logs/comparison_%j.log

set -euo pipefail

REPO_DIR=/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R
cd "$REPO_DIR"

if [[ -f "${REPO_DIR}/.env" ]]; then
    set -a; source "${REPO_DIR}/.env"; set +a
fi

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export INDEXING_PROFILE=hpc
export RERANKER_ENABLED=false

echo "Starting index comparison at $(date)"
uv run python -m experiments.index_comparison.run_comparison \
    --questions experiments/index_comparison/results/questions.jsonl \
    --top-k 1 5 10 \
    --squai-alpha 0.5 \
    --output experiments/index_comparison/results/

echo "Done at $(date)"
echo "Results: experiments/index_comparison/results/comparison_table.md"
