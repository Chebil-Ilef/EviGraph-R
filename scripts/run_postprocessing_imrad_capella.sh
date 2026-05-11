#!/bin/bash
#SBATCH --job-name=evigraph-imrad
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=22:00:00
#SBATCH --output=logs/imrad_post_%j.log
#
# IMRaD postprocessing script.
#
# Phases:
#   1. Parallel async scroll  — SCROLL_WORKERS concurrent UUID-range shards
#   2. GPU inference           — heuristic sections skip the model entirely
#   3. Parallel async writes   — WRITE_WORKERS concurrent set_payload calls
#
# MODES
# ─────
#   TEST_MODE=1   Runs two back-to-back mini-runs on TEST_POINTS points:
#                   pass 1 — dry run  (validate labelling, no writes)
#                   pass 2 — real run (validate writes + emit extrapolation)
#                 No snapshot bookkeeping in test mode.
#
#   DRY_RUN=1     Single run, no writes (normal usage).
#
#   (default)     Full production run with snapshot bookkeeping.
#
# USAGE
# ─────
#   # Validate + time estimate on 1 000 pts:
#   sbatch --export=ALL,TEST_MODE=1 scripts/run_postprocessing_imrad_capella.sh
#
#   # Larger sample for tighter extrapolation:
#   sbatch --export=ALL,TEST_MODE=1,TEST_POINTS=50000 scripts/run_postprocessing_imrad_capella.sh
#
#   # Dry-run only (no writes, no snapshots):
#   sbatch --export=ALL,DRY_RUN=1 scripts/run_postprocessing_imrad_capella.sh
#
#   # Full production run:
#   sbatch --export=ALL scripts/run_postprocessing_imrad_capella.sh

set -euo pipefail

REPO_DIR=$(pwd)
cd "$REPO_DIR"

mkdir -p logs

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

export QDRANT_SIF_PATH="${QDRANT_SIF_PATH:-$(pwd)/qdrant.sif}"
if [[ ! -f "$QDRANT_SIF_PATH" ]]; then
  echo "Building Qdrant Singularity image from docker://qdrant/qdrant (takes ~2 min)..."
  singularity build "$QDRANT_SIF_PATH" docker://qdrant/qdrant
  echo "Qdrant image built: $QDRANT_SIF_PATH"
fi

# ── Parameters ────────────────────────────────────────────────────────────────
# COLLECTION_NAME: if set, overrides the default from config.settings.QDRANT_ACTIVE
QDRANT_PROFILE="${QDRANT_PROFILE:-hpc}"
MODEL_ID="${MODEL_ID:-lostelf/section-classifier-imrad}"
DEVICE="${DEVICE:-auto}"
SCROLL_PAGE_SIZE="${SCROLL_PAGE_SIZE:-2048}"
SCROLL_WORKERS="${SCROLL_WORKERS:-8}"
WRITE_WORKERS="${WRITE_WORKERS:-32}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-256}"
QDRANT_UPDATE_BATCH_SIZE="${QDRANT_UPDATE_BATCH_SIZE:-512}"
MAX_POINTS="${MAX_POINTS:-}"
DRY_RUN="${DRY_RUN:-0}"
TEST_MODE="${TEST_MODE:-0}"
TEST_POINTS="${TEST_POINTS:-1000}"
REPORT_PATH="${REPORT_PATH:-${REPO_DIR}/_data/progress/imrad_postprocessing_report_${SLURM_JOB_ID:-local}.json}"

# ── Common base args ───────────────────────────────────────────────────────────
BASE_CMD=(
  uv run python -m indexing.postprocessing.imrad_titles
  --profile "$QDRANT_PROFILE"
  --model-id "$MODEL_ID"
  --device "$DEVICE"
  --scroll-page-size "$SCROLL_PAGE_SIZE"
  --scroll-workers "$SCROLL_WORKERS"
  --write-workers "$WRITE_WORKERS"
  --inference-batch-size "$INFERENCE_BATCH_SIZE"
  --qdrant-update-batch-size "$QDRANT_UPDATE_BATCH_SIZE"
)
if [[ -n "${COLLECTION_NAME:-}" ]]; then
  BASE_CMD+=(--collection-name "$COLLECTION_NAME")
fi

echo "Running on host: $(hostname)"
echo "QDRANT_PROFILE=${QDRANT_PROFILE}"
echo "SCROLL_WORKERS=${SCROLL_WORKERS}  WRITE_WORKERS=${WRITE_WORKERS}"
echo "INFERENCE_BATCH_SIZE=${INFERENCE_BATCH_SIZE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# ── TEST MODE ──────────────────────────────────────────────────────────────────
if [[ "$TEST_MODE" == "1" ]]; then
  echo ""
  echo "══════════════════════════════════════════"
  echo "  TEST MODE — ${TEST_POINTS} points"
  echo "══════════════════════════════════════════"

  DRY_REPORT="${REPO_DIR}/_data/progress/imrad_test_dry_${SLURM_JOB_ID:-local}.json"
  REAL_REPORT="${REPO_DIR}/_data/progress/imrad_test_real_${SLURM_JOB_ID:-local}.json"

  echo ""
  echo "[1/2] DRY RUN (no writes) — validates labelling pipeline"
  srun "${BASE_CMD[@]}" \
    --max-points "$TEST_POINTS" \
    --dry-run \
    --report-path "$DRY_REPORT"
  echo "[1/2] DRY RUN done → $DRY_REPORT"

  echo ""
  echo "[2/2] REAL WRITE — validates async writes + emits extrapolation"
  srun "${BASE_CMD[@]}" \
    --max-points "$TEST_POINTS" \
    --report-path "$REAL_REPORT"
  echo "[2/2] REAL WRITE done → $REAL_REPORT"

  echo ""
  echo "══════════════════════════════════════════"
  echo "  TEST MODE complete. EXTRAPOLATION lines are above (grep for it)."
  echo "══════════════════════════════════════════"
  exit 0
fi

# ── NORMAL / DRY-RUN MODE ─────────────────────────────────────────────────────
CMD=("${BASE_CMD[@]}" --report-path "$REPORT_PATH")

if [[ -n "$MAX_POINTS" ]]; then
  CMD+=(--max-points "$MAX_POINTS")
  echo "Processing up to: ${MAX_POINTS} points (extrapolation will be logged)"
else
  echo "Processing ALL points in collection"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

echo "DRY_RUN=${DRY_RUN}"
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
