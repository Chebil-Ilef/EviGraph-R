#!/bin/bash
#SBATCH --job-name=evigraph-idx-arr
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=170GB
#SBATCH --time=22:00:00
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
#         --array=0-0  --export=ALL,TOTAL_TASKS=1,INGEST_ONLY=1,RECREATE_COLLECTION=1 \
#         scripts/run_indexing_array_capella.sh
#   ## TO RESUME (storage healthy, job just died mid-ingest)
#  sbatch --array=0-0 \
#    --export=ALL,TOTAL_TASKS=1,INGEST_ONLY=1,RESUME_INGEST=1 \
#    scripts/run_indexing_array_capella.sh

set -euo pipefail

REPO_DIR=$(pwd)
cd "$REPO_DIR"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

export QDRANT_SIF_PATH="${QDRANT_SIF_PATH:-$(pwd)/qdrant.sif}"
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
# Set WIPE_STORAGE=1 to delete only Qdrant storage+snapshots before ingest (keeps shards+progress intact)
# Use this when storage is corrupt but you want to resume from the progress file
WIPE_STORAGE="${WIPE_STORAGE:-0}"
# Create periodic Qdrant snapshots every N ingested shard batches. Set to 0 to disable.
export EVI_SNAPSHOT_INTERVAL="${EVI_SNAPSHOT_INTERVAL:-50}"
# Ingest speed knobs. These are conservative-faster defaults for Capella.
export EVI_UPSERT_BATCH_SIZE="${EVI_UPSERT_BATCH_SIZE:-256}"
export EVI_INGEST_THROTTLE_SEC="${EVI_INGEST_THROTTLE_SEC:-0.1}"
export EVI_WAIT_EVERY_N_BATCHES="${EVI_WAIT_EVERY_N_BATCHES:-4}"

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

# WIPE_STORAGE: wipe only Qdrant storage+snapshots, keep shards+progress for resume
if [[ "$WIPE_STORAGE" == "1" ]]; then
  if [[ "$TASK_ID" -eq 0 ]]; then
    echo "WIPE_STORAGE=1 — Deleting Qdrant storage and snapshots (shards+progress intact)…"
    rm -rf "${REPO_DIR}/storage" && echo "  ✓ Deleted Qdrant storage"
    rm -rf "${REPO_DIR}/snapshots" && echo "  ✓ Deleted snapshots"
    echo "✓ Storage wiped — use RECREATE_COLLECTION=1 with RESUME_INGEST=1 to re-ingest from progress file"
  fi
fi

if [[ "$INGEST_ONLY" == "1" ]]; then
  if [[ "$TASK_ID" -ne 0 ]]; then
    echo "Task ${TASK_ID}: INGEST_ONLY=1 — nothing to do for non-zero tasks, exiting"
    exit 0
  fi
  echo "Task 0: INGEST_ONLY=1 — skipping chunk phase, going straight to ingest"
  echo "Task 0: periodic snapshot interval: every ${EVI_SNAPSHOT_INTERVAL} ingested shard batch(es)"
  echo "Task 0: ingest tuning: upsert_batch_size=${EVI_UPSERT_BATCH_SIZE}, throttle=${EVI_INGEST_THROTTLE_SEC}s, wait_every=${EVI_WAIT_EVERY_N_BATCHES}"
  INGEST_CMD=(
    uv run python -m src.indexing.indexing_pipeline
    --phase ingest
    --profile hpc
    --model "$MODEL_KEY"
    ${RECREATE_COLLECTION:+--recreate-collection}
    ${RESUME_INGEST:+--resume}
  )
  echo "Task 0: ingesting all shards — ${INGEST_CMD[*]}"
  _MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
  _STORAGE_DIR="${REPO_DIR}/storage"
  _ingest_monitor() {
    local tick=0
    while kill -0 "$1" 2>/dev/null; do
      sleep "$_MONITOR_INTERVAL"
      tick=$(( tick + _MONITOR_INTERVAL ))
      local mem_total mem_avail mem_used_gb mem_total_gb storage_sz="N/A"
      mem_total=$(awk '/MemTotal/{print $2}' /proc/meminfo)
      mem_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
      mem_used_gb=$(awk "BEGIN{printf \"%.1f\", ($mem_total - $mem_avail)/1048576}")
      mem_total_gb=$(awk "BEGIN{printf \"%.1f\", $mem_total/1048576}")
      [[ -d "$_STORAGE_DIR" ]] && storage_sz=$(du -sh "$_STORAGE_DIR" 2>/dev/null | cut -f1)
      echo "[MONITOR +${tick}s] RAM: ${mem_used_gb}/${mem_total_gb} GiB used | storage/: ${storage_sz}"
    done
  }
  _ingest_monitor $$ &
  _INGEST_MON_PID=$!
  srun "${INGEST_CMD[@]}"
  kill "$_INGEST_MON_PID" 2>/dev/null; wait "$_INGEST_MON_PID" 2>/dev/null
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

# Background resource monitor: logs RAM usage and storage folder size every 60s.
# Output goes to the same SLURM log (stdout).
_MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
_STORAGE_DIR="${REPO_DIR}/storage"
_monitor_loop() {
  local tick=0
  while kill -0 "$1" 2>/dev/null; do
    sleep "$_MONITOR_INTERVAL"
    tick=$(( tick + _MONITOR_INTERVAL ))
    # RAM: used = total - free - buffers/cache (MemAvailable proxy)
    local mem_total mem_avail mem_used_gb
    mem_total=$(awk '/MemTotal/{print $2}' /proc/meminfo)
    mem_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    mem_used_gb=$(awk "BEGIN{printf \"%.1f\", ($mem_total - $mem_avail)/1048576}")
    mem_total_gb=$(awk "BEGIN{printf \"%.1f\", $mem_total/1048576}")
    # Storage folder size (du -sh, suppress errors if dir missing)
    local storage_sz="N/A"
    if [[ -d "$_STORAGE_DIR" ]]; then
      storage_sz=$(du -sh "$_STORAGE_DIR" 2>/dev/null | cut -f1)
    fi
    echo "[MONITOR +${tick}s] RAM: ${mem_used_gb}/${mem_total_gb} GiB used | storage/: ${storage_sz}"
  done
}
_monitor_loop $$ &
_MONITOR_PID=$!

srun "${CMD[@]}"
_SRUN_EXIT=$?

kill "$_MONITOR_PID" 2>/dev/null
wait "$_MONITOR_PID" 2>/dev/null

echo "Task ${TASK_ID}: chunk phase complete (exit ${_SRUN_EXIT})"
exit $_SRUN_EXIT
