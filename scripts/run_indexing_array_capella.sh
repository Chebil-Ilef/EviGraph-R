#!/bin/bash
#SBATCH --job-name=evigraph-idx-arr
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/indexing_%A_%a.log
#
# Self-contained — no pre-flight jobs needed. Task 0 runs prepare-dataset
# automatically if batches are missing; other tasks wait up to 2 h.
#
#   # Small test run (3 tasks, 3 000 papers)
#   sbatch --array=0-2 --export=ALL,TOTAL_TASKS=3,SAMPLE_SIZE=3000 \
#         scripts/run_indexing_array_capella.sh
#
#   # Full 2.3 M run
#   sbatch --array=0-22 --export=ALL,TOTAL_TASKS=23 \
#         scripts/run_indexing_array_capella.sh

set -euo pipefail

REPO_DIR="/data/cat/ws/ilch217i-horse/EviGraph-R"
cd "$REPO_DIR"

mkdir -p logs

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

export QDRANT_SIF_PATH="${QDRANT_SIF_PATH:-${HOME}/qdrant.sif}"
if [[ ! -f "$QDRANT_SIF_PATH" ]]; then
  echo "Building Qdrant Singularity image from docker://qdrant/qdrant (takes ~2 min)…"
  singularity build "$QDRANT_SIF_PATH" docker://qdrant/qdrant
  echo "✓ Qdrant image built: $QDRANT_SIF_PATH"
fi

# ── Config ──────────────────────────────────────────────────────────────────
TOTAL_TASKS="${TOTAL_TASKS:-23}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_KEY="${MODEL_KEY:-bge-m3}"
RESUME_FLAG="${RESUME_FLAG:---resume}"
# phase=chunk: embed only (no Qdrant). Set PHASE=run to also ingest
# (only safe when a single job owns Qdrant).
PHASE="${PHASE:-chunk}"

BATCHES_DIR="${EVI_BATCHES_DIR:-${REPO_DIR}/_data/unarxive_batches}"
PREPARED_SENTINEL="${BATCHES_DIR}/.prepared"

# ── Prepare batches if missing (task 0 runs it; others wait) ─────────────────
if [[ ! -f "$PREPARED_SENTINEL" ]]; then
  if [[ "$TASK_ID" -eq 0 ]]; then
    echo "Task 0: batches not ready — running prepare-dataset…"
    PREPARE_CMD=(
      uv run python -m src.indexing.indexing_pipeline
      --phase prepare-dataset
      --profile hpc
      --dataset-mode stream
      --model "$MODEL_KEY"
    )
    if [[ -n "${SAMPLE_SIZE:-}" ]]; then
      PREPARE_CMD+=(--sample-size "$SAMPLE_SIZE")
    fi
    "${PREPARE_CMD[@]}"
    touch "$PREPARED_SENTINEL"
    echo "✓ Batches prepared: ${BATCHES_DIR}"
  else
    echo "Task ${TASK_ID}: waiting for task 0 to prepare batches…"
    WAIT_SECS=0
    MAX_WAIT=7200  # 2 hours
    while [[ ! -f "$PREPARED_SENTINEL" ]]; do
      sleep 30
      WAIT_SECS=$((WAIT_SECS + 30))
      if [[ $WAIT_SECS -ge $MAX_WAIT ]]; then
        echo "ERROR: timed out waiting for ${PREPARED_SENTINEL} after ${MAX_WAIT}s"
        exit 1
      fi
    done
    echo "✓ Batches ready (waited ${WAIT_SECS}s)"
  fi
fi

# ── Discover this task's slice of batch stems ────────────────────────────────
ALL_STEMS=($(uv run python -c "
import pathlib
d = pathlib.Path('${BATCHES_DIR}')
stems = sorted(p.stem for p in d.glob('*.jsonl'))
task_id = ${TASK_ID}
total   = ${TOTAL_TASKS}
print(' '.join(stems[task_id::total]))
" 2>/dev/null || true))

if [[ ${#ALL_STEMS[@]} -eq 0 ]]; then
  echo "ERROR: No batch files found in ${BATCHES_DIR} for task ${TASK_ID}."
  exit 1
fi

CMD=(
  uv run python -m src.indexing.indexing_pipeline
  --phase "$PHASE"
  --profile hpc
  --dataset-mode stream
  --model "$MODEL_KEY"
  --batches "${ALL_STEMS[@]}"
)

if [[ -n "${SAMPLE_SIZE:-}" ]]; then
  CMD+=(--sample-size "$SAMPLE_SIZE")
fi

if [[ -n "$RESUME_FLAG" ]]; then
  CMD+=("$RESUME_FLAG")
fi

echo "Running on host: $(hostname)"
echo "Array task: ${TASK_ID}/${TOTAL_TASKS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Stems for this task (${#ALL_STEMS[@]}): ${ALL_STEMS[*]}"
echo "Command: ${CMD[*]}"

srun "${CMD[@]}"
