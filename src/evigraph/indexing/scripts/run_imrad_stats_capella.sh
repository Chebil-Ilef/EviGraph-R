#!/bin/bash
#SBATCH --job-name=evigraph-imrad-stats
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/imrad_stats_%j.log
#
# Read-only IMRaD payload statistics for the active Qdrant collection.
#
# Examples:
#   # Quick sampled check:
#   sbatch --export=ALL,MAX_POINTS=100000 src/indexing/scripts/run_imrad_stats_capella.sh
#
#   # Full collection:
#   sbatch --export=ALL src/indexing/scripts/run_imrad_stats_capella.sh
#
#   # Include small text samples for top non-IMRaD titles:
#   sbatch --export=ALL,MAX_POINTS=1000000,INCLUDE_SAMPLES=1 src/indexing/scripts/run_imrad_stats_capella.sh

set -euo pipefail

REPO_DIR=$(pwd)
cd "$REPO_DIR"

mkdir -p logs _data/progress

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-${USER}}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}"
mkdir -p "$UV_CACHE_DIR" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

export QDRANT_SIF_PATH="${QDRANT_SIF_PATH:-$(pwd)/qdrant.sif}"
if [[ ! -f "$QDRANT_SIF_PATH" ]]; then
  echo "Building Qdrant Singularity image from docker://qdrant/qdrant (takes ~2 min)..."
  singularity build "$QDRANT_SIF_PATH" docker://qdrant/qdrant
  echo "Qdrant image built: $QDRANT_SIF_PATH"
fi

QDRANT_PROFILE="${QDRANT_PROFILE:-hpc}"
COLLECTION_NAME="${COLLECTION_NAME:-}"
SCROLL_PAGE_SIZE="${SCROLL_PAGE_SIZE:-4096}"
PAPER_BATCH_SIZE="${PAPER_BATCH_SIZE:-50000}"
MAX_POINTS="${MAX_POINTS:-}"
TOP_N="${TOP_N:-50}"
INCLUDE_SAMPLES="${INCLUDE_SAMPLES:-0}"
SAMPLE_CHARS="${SAMPLE_CHARS:-300}"
SAMPLE_PER_TITLE="${SAMPLE_PER_TITLE:-3}"
REPORT_PATH="${REPORT_PATH:-${REPO_DIR}/_data/progress/imrad_current_stats_${SLURM_JOB_ID:-local}.json}"

CMD=(
  uv run python -m indexing.postprocessing.imrad_stats
  --profile "$QDRANT_PROFILE"
  --scroll-page-size "$SCROLL_PAGE_SIZE"
  --paper-batch-size "$PAPER_BATCH_SIZE"
  --top-n "$TOP_N"
  --report-path "$REPORT_PATH"
)

if [[ -n "$COLLECTION_NAME" ]]; then
  CMD+=(--collection-name "$COLLECTION_NAME")
fi

if [[ -n "$MAX_POINTS" ]]; then
  CMD+=(--max-points "$MAX_POINTS")
fi

if [[ "$INCLUDE_SAMPLES" == "1" ]]; then
  CMD+=(
    --include-samples
    --sample-chars "$SAMPLE_CHARS"
    --sample-per-title "$SAMPLE_PER_TITLE"
  )
fi

echo "Running on host: $(hostname)"
echo "QDRANT_PROFILE=${QDRANT_PROFILE}"
echo "COLLECTION_NAME=${COLLECTION_NAME:-<active default>}"
echo "SCROLL_PAGE_SIZE=${SCROLL_PAGE_SIZE}  PAPER_BATCH_SIZE=${PAPER_BATCH_SIZE}"
echo "MAX_POINTS=${MAX_POINTS:-<full collection>}  TOP_N=${TOP_N}"
echo "INCLUDE_SAMPLES=${INCLUDE_SAMPLES}  SAMPLE_CHARS=${SAMPLE_CHARS}  SAMPLE_PER_TITLE=${SAMPLE_PER_TITLE}"
echo "REPORT_PATH=${REPORT_PATH}"
echo "Command: ${CMD[*]}"

# indexing.postprocessing.imrad_stats calls utils.qdrant.ensure_qdrant_runtime()
# before scrolling, so this job starts/reuses the configured Qdrant runtime.
srun "${CMD[@]}"

echo "IMRaD stats completed"
