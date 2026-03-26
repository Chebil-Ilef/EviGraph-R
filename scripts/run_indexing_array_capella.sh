#!/bin/bash
#SBATCH --job-name=evigraph-idx-arr
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
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

TOTAL_TASKS="${TOTAL_TASKS:-23}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_KEY="${MODEL_KEY:-bge-m3}"
RESUME_FLAG="${RESUME_FLAG:---resume}"
# phase=chunk: embed only (no Qdrant). Set PHASE=run to also ingest
# (only safe when a single job owns Qdrant).
PHASE="${PHASE:-chunk}"

BATCHES_DIR="${EVI_BATCHES_DIR:-${REPO_DIR}/_data/unarxive_batches}"
PREPARED_SENTINEL="${BATCHES_DIR}/.prepared"

# When SAMPLE_SIZE is set (test mode), shrink the batch size so that even if
# the actual paper count is much less than SAMPLE_SIZE, we still produce at
# least TOTAL_TASKS batch files.  A 5× safety factor handles data scarcity.
# Full runs (no SAMPLE_SIZE) keep the default EVI_DATASET_BATCH_SIZE=1000.
if [[ -n "${SAMPLE_SIZE:-}" ]]; then
  _bs=$(( SAMPLE_SIZE / TOTAL_TASKS / 5 ))
  export EVI_DATASET_BATCH_SIZE="$(( _bs < 1 ? 1 : _bs ))"
  echo "Task ${TASK_ID}: SAMPLE_SIZE=${SAMPLE_SIZE}, TOTAL_TASKS=${TOTAL_TASKS} → EVI_DATASET_BATCH_SIZE=${EVI_DATASET_BATCH_SIZE}"
fi

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

if [[ ! -d "$BATCHES_DIR" ]]; then
  echo "ERROR: Batches directory does not exist: ${BATCHES_DIR}"
  exit 1
fi

BATCH_FILES=("${BATCHES_DIR}"/*.jsonl)
if [[ ${#BATCH_FILES[@]} -eq 0 ]] || [[ ! -f "${BATCH_FILES[0]}" ]]; then
  echo "ERROR: No batch files (*.jsonl) found in ${BATCHES_DIR}"
  ls -la "$BATCHES_DIR" 2>/dev/null || echo "(directory listing failed)"
  exit 1
fi

# Distribute batches across tasks: task 0 gets stems 0,n,2n,... task 1 gets 1,n+1,2n+1,...
ALL_STEMS=()
for i in $(seq $TASK_ID $TOTAL_TASKS $((${#BATCH_FILES[@]} - 1))); do
  stem=$(basename "${BATCH_FILES[$i]}" .jsonl)
  ALL_STEMS+=("$stem")
done

echo "Task ${TASK_ID}: found ${#ALL_STEMS[@]} batch files to process (from ${#BATCH_FILES[@]} total)"
if [[ ${#ALL_STEMS[@]} -eq 0 ]]; then
  echo "  (This task has no work — exiting cleanly)"
  exit 0
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
