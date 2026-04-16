#!/bin/bash
#SBATCH --job-name=evigraph-imrad
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/imrad_post_%j.log
#
# IMRAD postprocessing script - processes points concurrently via internal batching.
# The script queries Qdrant concurrently, so no array jobs are needed.
#
# USAGE:
#   # Test run with 200K points
#   sbatch --export=ALL,DRY_RUN=1,MAX_POINTS=200000 scripts/run_postprocessing_imrad_capella.sh
#
#   # Full run (processes all points in collection)
#   sbatch --export=ALL scripts/run_postprocessing_imrad_capella.sh
#
# When DRY_RUN=0, the script also creates a fresh snapshot renamed with *_postprocessed

set -euo pipefail

REPO_DIR=$(pwd)
cd "$REPO_DIR"

mkdir -p logs

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

# Required for Singularity-managed Qdrant runtime on HPC.
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

export QDRANT_SIF_PATH="${QDRANT_SIF_PATH:-$(pwd)/qdrant.sif}"
if [[ ! -f "$QDRANT_SIF_PATH" ]]; then
  echo "Building Qdrant Singularity image from docker://qdrant/qdrant (takes ~2 min)..."
  singularity build "$QDRANT_SIF_PATH" docker://qdrant/qdrant
  echo "Qdrant image built: $QDRANT_SIF_PATH"
fi

# Postprocessing parameters
# COLLECTION_NAME: if set, overrides the default from config.settings.QDRANT_ACTIVE.collection_name
QDRANT_PROFILE="${QDRANT_PROFILE:-hpc}"
MODEL_ID="${MODEL_ID:-lostelf/section-classifier-imrad}"
DEVICE="${DEVICE:-auto}"
SCROLL_PAGE_SIZE="${SCROLL_PAGE_SIZE:-2048}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
QDRANT_UPDATE_BATCH_SIZE="${QDRANT_UPDATE_BATCH_SIZE:-512}"
MAX_POINTS="${MAX_POINTS:-}"
DRY_RUN="${DRY_RUN:-0}"
REPORT_PATH="${REPORT_PATH:-${REPO_DIR}/_data/progress/imrad_postprocessing_report_${SLURM_JOB_ID:-local}.json}"

CMD=(
  uv run python -m indexing.postprocessing.imrad_titles
  --profile "$QDRANT_PROFILE"
  --model-id "$MODEL_ID"
  --device "$DEVICE"
  --scroll-page-size "$SCROLL_PAGE_SIZE"
  --inference-batch-size "$INFERENCE_BATCH_SIZE"
  --qdrant-update-batch-size "$QDRANT_UPDATE_BATCH_SIZE"
  --report-path "$REPORT_PATH"
)

if [[ -n "${COLLECTION_NAME:-}" ]]; then
  CMD+=(--collection-name "$COLLECTION_NAME")
fi

if [[ -n "$MAX_POINTS" ]]; then
  CMD+=(--max-points "$MAX_POINTS")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

echo "Running on host: $(hostname)"
echo "QDRANT_PROFILE=${QDRANT_PROFILE}"
if [[ -n "$MAX_POINTS" ]]; then
  echo "Processing up to: ${MAX_POINTS} points"
else
  echo "Processing all points in collection"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Command: ${CMD[*]}"

if [[ "$DRY_RUN" != "1" ]]; then
  echo "Recording current Qdrant snapshot as *_previous baseline"
  srun uv run python -m utils.qdrant \
    --artifact-mode backup-previous \
    --profile "$QDRANT_PROFILE"
fi

srun "${CMD[@]}"

if [[ "$DRY_RUN" != "1" ]]; then
  echo "Capturing updated Qdrant snapshot as *_postprocessed"
  srun uv run python -m utils.qdrant \
    --artifact-mode capture-postprocessed \
    --profile "$QDRANT_PROFILE"
fi

echo "IMRAD postprocessing completed"
