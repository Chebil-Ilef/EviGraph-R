#!/bin/bash
#SBATCH --job-name=evigraph-indexing
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/indexing_%j.log

set -euo pipefail

REPO_DIR="/data/cat/ws/ilch217i-horse/EviGraph-R"
cd "$REPO_DIR"

mkdir -p logs

# Optional overrides:
#   PHASE=prepare-dataset|chunk|ingest|snapshot|run
#   SAMPLE_SIZE=1000
#   MODEL_KEY=e5-base-v2
#   DATASET_MODE=stream|mirror
#   INDEXING_DEVICE=cuda|cpu|mps
PHASE="${PHASE:-run}"
MODEL_KEY="${MODEL_KEY:-e5-base-v2}"
DATASET_MODE="${DATASET_MODE:-stream}"
RESUME_FLAG="${RESUME_FLAG:---resume}"

CMD=(
  uv run python -m src.indexing.indexing_pipeline
  --phase "$PHASE"
  --profile hpc
  --dataset-mode "$DATASET_MODE"
  --model "$MODEL_KEY"
)

if [[ -n "${SAMPLE_SIZE:-}" ]]; then
  CMD+=(--sample-size "$SAMPLE_SIZE")
fi

if [[ -n "$RESUME_FLAG" ]]; then
  CMD+=("$RESUME_FLAG")
fi

echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Command: ${CMD[*]}"

srun "${CMD[@]}"
