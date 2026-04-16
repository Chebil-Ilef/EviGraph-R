#!/bin/bash
#SBATCH --job-name=evigraph-idx-arr
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=logs/indexing_%A_%a.log

#   # Full 2.3 M run — two-step: chunk all tasks, then ingest after all succeed
#   ## STEP 1
#   R1=$(sbatch --parsable --array=0-199%50 \
#         --export=ALL,TOTAL_TASKS=200 \
#         scripts/run_indexing_array_capella.sh)
#   echo "Chunk job: $R1"
#
#   ## STEP 2
#   sbatch --dependency=afterok:$R1 \
#         --array=0-0 --export=ALL,TOTAL_TASKS=1,INGEST_ONLY=1 \
#         scripts/run_indexing_array_capella.sh
#
#   # Re-run ingest only (shards already on disk)
#   sbatch --array=0-0 --export=ALL,TOTAL_TASKS=1,INGEST_ONLY=1 \
#         scripts/run_indexing_array_capella.sh
#
#   # Clean start: delete ALL state, then run fresh
#   sbatch --array=0-2 --export=ALL,TOTAL_TASKS=3,SAMPLE_SIZE=3000,CLEAN_START=1 \
#         scripts/run_indexing_array_capella.sh

set -euo pipefail

 # **TODO**change according to the workspace
REPO_DIR="/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R"
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

TOTAL_TASKS="${TOTAL_TASKS:-200}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_KEY="${MODEL_KEY:-bge-m3}"
RESUME_FLAG="${RESUME_FLAG:---resume}"
# phase=chunk: embed only (no Qdrant). Set PHASE=run to also ingest
# (only safe when a single job owns Qdrant).
PHASE="${PHASE:-chunk}"
# Set INGEST_ONLY=1 to skip chunking entirely and go straight to ingest+snapshot.
# Requires shards to already exist on disk.  Only task 0 does meaningful work.
INGEST_ONLY="${INGEST_ONLY:-0}"
# Set CLEAN_START=1 to delete all state: manifests, progress logs, shards, Qdrant storage, snapshots
CLEAN_START="${CLEAN_START:-0}"

BATCHES_DIR="${EVI_BATCHES_DIR:-${REPO_DIR}/_data/unarxive_batches}"
PREPARED_SENTINEL="${BATCHES_DIR}/.prepared"

# CLEAN_START: Nuclear option — delete all indexing state
if [[ "$CLEAN_START" == "1" ]]; then
  if [[ "$TASK_ID" -eq 0 ]]; then
    echo "CLEAN_START=1 — Deleting all indexing state…"
    rm -rf "${REPO_DIR}/_data/manifests" && echo "  ✓ Deleted manifests"
    rm -rf "${REPO_DIR}/_data/progress" && echo "  ✓ Deleted progress logs"
    rm -rf "${REPO_DIR}/_data/shards" && echo "  ✓ Deleted shards"
    rm -rf "${REPO_DIR}/storage" && echo "  ✓ Deleted Qdrant storage"
    rm -rf "${REPO_DIR}/snapshots" && echo "  ✓ Deleted snapshots"
    rm -f "${BATCHES_DIR}/.prepared" "${BATCHES_DIR}/.chunk_done_"* && echo "  ✓ Deleted batch sentinels"
    echo "✓ Clean start complete — pipeline will re-prepare and re-ingest from scratch"
  else
    echo "Task ${TASK_ID}: waiting for task 0 clean start…"
    sleep 5
  fi
fi

if [[ "$INGEST_ONLY" == "1" ]]; then
  if [[ "$TASK_ID" -ne 0 ]]; then
    echo "Task ${TASK_ID}: INGEST_ONLY=1 — nothing to do for non-zero tasks, exiting"
    exit 0
  fi
  echo "Task 0: INGEST_ONLY=1 — skipping chunk phase, going straight to ingest"
  INGEST_CMD=(
    uv run python -m src.indexing.indexing_pipeline
    --phase ingest
    --profile hpc
    --model "$MODEL_KEY"
    ${RECREATE_COLLECTION:+--recreate-collection}
    ${RESUME_INGEST:+--resume}
  )
  echo "Task 0: ingesting all shards — ${INGEST_CMD[*]}"
  srun "${INGEST_CMD[@]}"
  SNAPSHOT_CMD=(
    uv run python -m src.indexing.indexing_pipeline
    --phase snapshot
    --profile hpc
    --model "$MODEL_KEY"
  )
  echo "Task 0: writing final snapshot — ${SNAPSHOT_CMD[*]}"
  srun "${SNAPSHOT_CMD[@]}"
  echo "✓ Ingest-only run complete"
  exit 0
fi

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

echo "Task ${TASK_ID}: chunk phase complete"
