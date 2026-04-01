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

CMD=(
  uv run python -m src.indexing.postprocessing.citation_ids
)

echo "Running on host: $(hostname)"
echo "QDRANT_PROFILE=${QDRANT_PROFILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Command: ${CMD[*]}"

srun "${CMD[@]}"

echo "Citation ID postprocessing completed"
