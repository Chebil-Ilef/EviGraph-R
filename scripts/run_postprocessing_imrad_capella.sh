#!/bin/bash
#SBATCH --job-name=evigraph-imrad-arr
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --time=00:15:00
#SBATCH --output=logs/imrad_post_%A_%a.log
#
# USAGE:
#   # Small test run (3 tasks, 9,000 points total)
#   sbatch --array=0-2 --export=ALL,TOTAL_TASKS=3,POINTS_PER_TASK=3000 \
#         scripts/run_postprocessing_imrad_capella.sh
#
#   # Full 2.3M run - option 1: many small tasks (767 tasks × 3K points, max 100 concurrent)
#   sbatch --array=0-766%100 --export=ALL,TOTAL_TASKS=767,POINTS_PER_TASK=3000 \
#         scripts/run_postprocessing_imrad_capella.sh
#
#   # Full 2.3M run - option 2: fewer medium tasks (230 tasks × 10K points, max 50 concurrent)
#   sbatch --array=0-229%50 --export=ALL,TOTAL_TASKS=230,POINTS_PER_TASK=10000 \
#         scripts/run_postprocessing_imrad_capella.sh
#
#   # Full 2.3M run - option 3: larger tasks (115 tasks × 20K points, max 30 concurrent)
#   sbatch --array=0-114%30 --export=ALL,TOTAL_TASKS=115,POINTS_PER_TASK=20000 \
#         scripts/run_postprocessing_imrad_capella.sh

set -euo pipefail

REPO_DIR="/data/cat/ws/ilch217i-horse/EviGraph-R"
cd "$REPO_DIR"

mkdir -p logs

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

# Required for Singularity-managed Qdrant runtime on HPC.
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

export QDRANT_SIF_PATH="${QDRANT_SIF_PATH:-${HOME}/qdrant.sif}"
if [[ ! -f "$QDRANT_SIF_PATH" ]]; then
  echo "Building Qdrant Singularity image from docker://qdrant/qdrant (takes ~2 min)…"
  singularity build "$QDRANT_SIF_PATH" docker://qdrant/qdrant
  echo "✓ Qdrant image built: $QDRANT_SIF_PATH"
fi

# Array job parameters
TOTAL_TASKS="${TOTAL_TASKS:-230}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
POINTS_PER_TASK="${POINTS_PER_TASK:-10000}"

# Calculate offset for this task
OFFSET=$((TASK_ID * POINTS_PER_TASK))

# Postprocessing parameters
COLLECTION_NAME="${COLLECTION_NAME:-unarxive_chunks}"
MODEL_ID="${MODEL_ID:-lostelf/section-classifier-imrad}"
DEVICE="${DEVICE:-auto}"
SCROLL_PAGE_SIZE="${SCROLL_PAGE_SIZE:-2048}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
QDRANT_UPDATE_BATCH_SIZE="${QDRANT_UPDATE_BATCH_SIZE:-512}"
DRY_RUN="${DRY_RUN:-0}"
REPORT_PATH="${REPORT_PATH:-${REPO_DIR}/_data/progress/imrad_postprocessing_report_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}.json}"

CMD=(
  uv run python -m indexing.postprocessing.imrad_postprocessing
  --profile hpc
  --collection-name "$COLLECTION_NAME"
  --model-id "$MODEL_ID"
  --device "$DEVICE"
  --scroll-page-size "$SCROLL_PAGE_SIZE"
  --inference-batch-size "$INFERENCE_BATCH_SIZE"
  --qdrant-update-batch-size "$QDRANT_UPDATE_BATCH_SIZE"
  --report-path "$REPORT_PATH"
  --offset "$OFFSET"
  --max-points "$POINTS_PER_TASK"
)

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

echo "Running on host: $(hostname)"
echo "Array task: ${TASK_ID}/${TOTAL_TASKS}"
echo "Processing points: ${OFFSET} to $((OFFSET + POINTS_PER_TASK - 1))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Command: ${CMD[*]}"

srun "${CMD[@]}"

echo "✓ Task ${TASK_ID}: processed ${POINTS_PER_TASK} points starting at offset ${OFFSET}"
