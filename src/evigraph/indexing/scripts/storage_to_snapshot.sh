#!/bin/bash
#SBATCH --job-name=qdrant-snapshot
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000M
#SBATCH --time=08:00:00
#SBATCH --output=/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R/logs/storage_to_snapshot_%j.log

# Creates a Qdrant snapshot from the current storage/ and saves it to
# snapshots/unarxive_chunks/ with a timestamp, then symlinks it as "latest.snapshot".

set -euo pipefail

REPO=/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R
COLLECTION=unarxive_chunks
PORT=6339
GRPC_PORT=6340

QDRANT_SIF=${REPO}/qdrant.sif
STORAGE=${REPO}/storage
SNAPSHOTS_DIR=${REPO}/snapshots/${COLLECTION}
LOG=${REPO}/logs/storage_to_snapshot_server_${SLURM_JOB_ID:-local}.log

mkdir -p "$SNAPSHOTS_DIR" "${REPO}/logs"

echo "Storage  : $STORAGE"
echo "Snapshots: $SNAPSHOTS_DIR"
echo "Node     : $(hostname)"
echo ""

# Start Qdrant bound to existing storage (read + snapshot path)
singularity exec \
  --bind "${STORAGE}:/qdrant/storage" \
  --bind "${SNAPSHOTS_DIR}:/qdrant/snapshots" \
  --env "QDRANT__SERVICE__HTTP_PORT=${PORT}" \
  --env "QDRANT__SERVICE__GRPC_PORT=${GRPC_PORT}" \
  --env "QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage" \
  --env "QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/snapshots" \
  "$QDRANT_SIF" /bin/sh -lc 'cd /qdrant && exec /qdrant/qdrant' > "$LOG" 2>&1 &
QDRANT_PID=$!

cleanup() {
  echo "Stopping Qdrant process..."
  kill "$QDRANT_PID" 2>/dev/null || true
  wait "$QDRANT_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for readiness
echo "Waiting for Qdrant on port ${PORT}..."
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1; then
    echo "Qdrant ready after $((i * 2))s"
    break
  fi
  if ! kill -0 "$QDRANT_PID" 2>/dev/null; then
    echo "ERROR: Qdrant process died. Last log lines:"
    tail -40 "$LOG" || true
    exit 1
  fi
  sleep 2
done

# Create snapshot
echo ""
echo "Creating snapshot..."
SNAP_RESP=$(curl -sS -X POST \
  "http://localhost:${PORT}/collections/${COLLECTION}/snapshots?wait=true" \
  -H 'Content-Type: application/json')
echo "Response: $SNAP_RESP"

SNAP_NAME=$(echo "$SNAP_RESP" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d['result']['name'])" 2>/dev/null || echo "")
if [[ -z "$SNAP_NAME" ]]; then
  echo "ERROR: snapshot creation failed or name could not be parsed."
  exit 1
fi

echo ""
echo "Snapshot created: ${SNAP_NAME}"
SNAP_PATH="${SNAPSHOTS_DIR}/${SNAP_NAME}"

# Wait for the file to appear on disk
for i in $(seq 1 60); do
  if [[ -f "$SNAP_PATH" ]]; then
    break
  fi
  sleep 5
done
if [[ ! -f "$SNAP_PATH" ]]; then
  echo "ERROR: snapshot file not found at ${SNAP_PATH}"
  exit 1
fi

# Symlink as latest.snapshot
LATEST_LINK="${SNAPSHOTS_DIR}/latest.snapshot"
ln -sf "$SNAP_NAME" "$LATEST_LINK"
echo "Symlinked: latest.snapshot → ${SNAP_NAME}"

echo ""
echo "=========================================="
echo " SNAPSHOT COMPLETE"
echo " File   : ${SNAP_PATH}"
echo " Latest : ${LATEST_LINK}"
echo " Size   : $(du -sh "$SNAP_PATH" | cut -f1)"
echo "=========================================="
