#!/bin/bash
#SBATCH --job-name=evigraph-resolve-ids
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/resolve_id_%j.log
#
# Citation ID postprocessing script - scans the Qdrant collection for
# citations missing public identifiers and resolves them via bibliographic APIs.
#
# USAGE:
#   # Default run
#   sbatch src/indexing/scripts/run_postprocessing_ids_capella.sh
#
#   # Override the Qdrant profile if needed
#   sbatch --export=ALL,QDRANT_PROFILE=hpc src/indexing/scripts/run_postprocessing_ids_capella.sh
#
#   # Dry run (resolve IDs without writing them back)
#   sbatch --export=ALL,DRY_RUN=1 src/indexing/scripts/run_postprocessing_ids_capella.sh
#
# When DRY_RUN=0, the script also:
#   1. Records the current Qdrant snapshot metadata in a *_previous artifact
#   2. Runs postprocessing updates
#   3. Creates a fresh snapshot renamed with *_postprocessed
#
#   # quick 10-ref test (dry-run implied for safety, but you can combine)
# sbatch --export=ALL,DRY_RUN=1,TEST_LIMIT=10 src/indexing/scripts/run_postprocessing_ids_capella.sh
#   # test with actual writes
# sbatch --export=ALL,TEST_LIMIT=10 src/indexing/scripts/run_postprocessing_ids_capella.sh

set -euo pipefail
REPO_DIR=$(pwd)
cd "$REPO_DIR"


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

# Optional override for environments that select the active profile from env.
QDRANT_PROFILE="${QDRANT_PROFILE:-hpc}"
export QDRANT_PROFILE
DRY_RUN="${DRY_RUN:-0}"
TEST_LIMIT="${TEST_LIMIT:-0}"

CMD=(
  uv run python -m src.indexing.postprocessing.citation_ids
)

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

if [[ "$TEST_LIMIT" -gt 0 ]]; then
  CMD+=(--limit "$TEST_LIMIT")
fi

echo "Running on host: $(hostname)"
echo "QDRANT_PROFILE=${QDRANT_PROFILE}"
echo "DRY_RUN=${DRY_RUN}"
echo "TEST_LIMIT=${TEST_LIMIT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Command: ${CMD[*]}"

if [[ "$DRY_RUN" != "1" ]]; then
  echo "Recording current Qdrant snapshot metadata as *_previous baseline"
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

echo "Citation ID postprocessing completed"
