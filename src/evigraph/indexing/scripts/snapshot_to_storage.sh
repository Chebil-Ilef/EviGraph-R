#!/bin/bash
#SBATCH --job-name=qdrant-recover-storage
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000M
#SBATCH --time=02:00:00
#SBATCH --output=/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R/logs/snapshot_to_storage_%j.log

# Restores the latest snapshot into recovery-storage/ so it can be used
# directly by the indexing job as a drop-in replacement for storage/.
#
# After this job completes:
#   mv storage/ storage.broken/   # keep the corrupted one aside
#   mv recovery-storage/ storage/
# Then resubmit the indexing job with RESUME_INGEST=1.

set -euo pipefail

REPO=/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R
COLLECTION=unarxive_chunks
PORT=6337
GRPC_PORT=6338

QDRANT_SIF=${REPO}/qdrant.sif
SNAPSHOTS_DIR=${REPO}/snapshots
RECOVERY_STORAGE=${REPO}/recovery-storage
LOG=${REPO}/logs/snapshot_to_storage_server_${SLURM_JOB_ID:-local}.log

mkdir -p "$RECOVERY_STORAGE" "${REPO}/logs"

# Guard: refuse to overwrite an existing recovery-storage that already has data
if [[ -d "${RECOVERY_STORAGE}/collections" ]]; then
  echo "ERROR: ${RECOVERY_STORAGE}/collections already exists."
  echo "Remove or rename recovery-storage/ before running this script."
  exit 1
fi

# Pick the most recent snapshot automatically
SNAPSHOT_FILE=$(ls -t "${SNAPSHOTS_DIR}/${COLLECTION}/"*.snapshot 2>/dev/null | head -1)
if [[ -z "$SNAPSHOT_FILE" ]]; then
  echo "ERROR: no .snapshot file found in ${SNAPSHOTS_DIR}/${COLLECTION}/"
  exit 1
fi
SNAPSHOT_NAME=$(basename "$SNAPSHOT_FILE")
echo "Snapshot : $SNAPSHOT_NAME"
echo "Target   : $RECOVERY_STORAGE"
echo "Node     : $(hostname)"
echo ""

#   Start Qdrant bound to recovery-storage  
# No --userns: use the standard setuid Singularity flow so overlay works and
# bind-mounts to /qdrant/storage and /qdrant/snapshots are reliable.
singularity exec \
  --bind "${RECOVERY_STORAGE}:/qdrant/storage" \
  --bind "${SNAPSHOTS_DIR}/${COLLECTION}:/qdrant/snapshots" \
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

#   Wait for readiness            ─
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

#   Recover snapshot into recovery-storage  ─
echo ""
echo "Recovering snapshot — this writes into recovery-storage/..."
RECOVER_RESP=$(curl -sS -X PUT \
  "http://localhost:${PORT}/collections/${COLLECTION}/snapshots/recover?wait=true" \
  -H 'Content-Type: application/json' \
  -d "{
    \"location\": \"file:///qdrant/snapshots/${SNAPSHOT_NAME}\",
    \"priority\": \"snapshot\"
  }")
echo "Response: $RECOVER_RESP"

# Check success
OK=$(echo "$RECOVER_RESP" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d.get('result', False))" 2>/dev/null || echo "unknown")
if [[ "$OK" != "True" && "$OK" != "true" ]]; then
  echo "ERROR: snapshot recovery failed — recovery-storage/ is incomplete, not safe to use."
  exit 1
fi
if [[ ! -d "${RECOVERY_STORAGE}/collections/${COLLECTION}" ]]; then
  echo "ERROR: Qdrant reported success, but ${RECOVERY_STORAGE}/collections/${COLLECTION} was not created."
  echo "Check ${LOG}; storage path binding/configuration is not safe."
  exit 1
fi

# Wait for optimizer to finish flushing to disk c
echo ""
echo "Waiting for optimizer to settle (segments merge + flush)..."
for i in $(seq 1 120); do
  STATUS=$(curl -sS "http://localhost:${PORT}/collections/${COLLECTION}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result']['optimizer_status'])" 2>/dev/null || echo "unknown")
  COLL_STATUS=$(curl -sS "http://localhost:${PORT}/collections/${COLLECTION}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result']['status'])" 2>/dev/null || echo "unknown")
  echo "  [${i}] collection_status=${COLL_STATUS}  optimizer=${STATUS}"
  if [[ "$STATUS" == "ok" ]]; then
    echo "  Optimizer settled."
    break
  fi
  sleep 5
done

echo ""
POINTS=$(curl -sS "http://localhost:${PORT}/collections/${COLLECTION}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result']['points_count'])" 2>/dev/null || echo "?")
COLL_STATUS=$(curl -sS "http://localhost:${PORT}/collections/${COLLECTION}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result']['status'])" 2>/dev/null || echo "?")

echo "=========================================="
echo " RECOVERY COMPLETE"
echo " Points in recovery-storage : ${POINTS}"
echo " Collection status          : ${COLL_STATUS}"
echo " Snapshot used              : ${SNAPSHOT_NAME}"
echo "=========================================="

# Gracefully shut down so Qdrant flushes WAL before we stop
echo ""
echo "Flushing WAL before shutdown..."
curl -sS -X POST "http://localhost:${PORT}/collections/${COLLECTION}/snapshots" \
  -H 'Content-Type: application/json' >/dev/null 2>&1 || true
sleep 5

echo ""
echo "recovery-storage/ is ready. To use it as your main storage:"
echo ""
echo "  mv ${REPO}/storage ${REPO}/storage.broken"
echo "  mv ${REPO}/recovery-storage ${REPO}/storage"
echo ""
echo "Then resubmit the indexing job with RESUME_INGEST=1."
