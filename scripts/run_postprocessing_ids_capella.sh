#!/bin/bash
#SBATCH --job-name=evigraph-resolve-ids
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:04:00
#SBATCH --output=logs/resolve_id_%j.log
#
# Citation ID postprocessing script - scans the Qdrant collection for
# citations missing public identifiers and resolves them via bibliographic APIs.
#
# USAGE:
#   # Default run
#   sbatch scripts/run_postprocessing_ids_capella.sh
#
#   # Override the Qdrant profile if needed
#   sbatch --export=ALL,QDRANT_PROFILE=hpc scripts/run_postprocessing_ids_capella.sh
#
#   # Dry run (resolve IDs without writing them back)
#   sbatch --export=ALL,DRY_RUN=1 scripts/run_postprocessing_ids_capella.sh
#
# When DRY_RUN=0, the script also:
#   1. Copies the current Qdrant storage and latest snapshot metadata to *_previous
#   2. Runs postprocessing updates
#   3. Creates a fresh postprocessed snapshot + storage copy using *_postprocessed

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
  echo "Building Qdrant Singularity image from docker://qdrant/qdrant (takes ~2 min)..."
  singularity build "$QDRANT_SIF_PATH" docker://qdrant/qdrant
  echo "Qdrant image built: $QDRANT_SIF_PATH"
fi

# Optional override for environments that select the active profile from env.
QDRANT_PROFILE="${QDRANT_PROFILE:-hpc}"
export QDRANT_PROFILE
DRY_RUN="${DRY_RUN:-0}"

CMD=(
  uv run python -m src.indexing.postprocessing.citation_ids
)

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

echo "Running on host: $(hostname)"
echo "QDRANT_PROFILE=${QDRANT_PROFILE}"
echo "DRY_RUN=${DRY_RUN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Command: ${CMD[*]}"

if [[ "$DRY_RUN" != "1" ]]; then
  echo "Backing up current Qdrant state to *_previous artifacts"
  srun uv run python -m utils.qdrant \
    --artifact-mode backup-previous \
    --profile "$QDRANT_PROFILE"
fi

srun "${CMD[@]}"

if [[ "$DRY_RUN" != "1" ]]; then
  echo "Capturing updated Qdrant state to *_postprocessed artifacts"
  srun uv run python -m utils.qdrant \
    --artifact-mode capture-postprocessed \
    --profile "$QDRANT_PROFILE"
fi

echo "Citation ID postprocessing completed"
