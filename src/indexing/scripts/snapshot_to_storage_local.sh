#!/bin/bash
# Restores the latest snapshot into storage/ using Docker (local, no SLURM).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
COLLECTION=unarxive_chunks
PORT=6337
GRPC_PORT=6338

SNAPSHOTS_DIR=${REPO}/snapshots
RECOVERY_STORAGE=${REPO}/storage
LOG=${REPO}/logs/snapshot_to_storage_server_local.log

mkdir -p "$RECOVERY_STORAGE" "${REPO}/logs"

# Guard: refuse to overwrite an existing storage that already has data
if [[ -d "${RECOVERY_STORAGE}/collections" ]]; then
  echo "ERROR: ${RECOVERY_STORAGE}/collections already exists."
  echo "Remove or rename storage/ before running this script."
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
echo "Host     : $(hostname)"
echo ""

# Start Qdrant via Docker bound to storage
docker run -d --rm \
  --name qdrant-recovery \
  -p "${PORT}:6333" \
  -p "${GRPC_PORT}:6334" \
  -v "${RECOVERY_STORAGE}:/qdrant/storage" \
  -v "${SNAPSHOTS_DIR}/${COLLECTION}:/qdrant/snapshots" \
  -e "QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage" \
  -e "QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/snapshots" \
  qdrant/qdrant >"$LOG" 2>&1

CONTAINER_NAME=qdrant-recovery

cleanup() {
  echo "Stopping Qdrant container..."
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Wait for readiness
echo "Waiting for Qdrant on port ${PORT}..."
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1; then
    echo "Qdrant ready after $((i * 2))s"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "ERROR: Qdrant container stopped unexpectedly. Last log lines:"
    docker logs "$CONTAINER_NAME" 2>/dev/null | tail -40 || tail -40 "$LOG" || true
    exit 1
  fi
  sleep 2
done

# Recover snapshot into storage
echo ""
echo "Recovering snapshot — this writes into storage/..."
RECOVER_RESP=$(curl -sS -X PUT \
  "http://localhost:${PORT}/collections/${COLLECTION}/snapshots/recover?wait=true" \
  -H 'Content-Type: application/json' \
  -d "{
    \"location\": \"file:///qdrant/snapshots/${SNAPSHOT_NAME}\",
    \"priority\": \"snapshot\"
  }")
echo "Response: $RECOVER_RESP"

OK=$(echo "$RECOVER_RESP" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d.get('result', False))" 2>/dev/null || echo "unknown")
if [[ "$OK" != "True" && "$OK" != "true" ]]; then
  echo "ERROR: snapshot recovery failed — storage/ is incomplete, not safe to use."
  exit 1
fi
if [[ ! -d "${RECOVERY_STORAGE}/collections/${COLLECTION}" ]]; then
  echo "ERROR: Qdrant reported success, but ${RECOVERY_STORAGE}/collections/${COLLECTION} was not created."
  echo "Check ${LOG}; storage path binding may be misconfigured."
  exit 1
fi

# Wait for optimizer to finish flushing to disk
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
echo " Points in storage : ${POINTS}"
echo " Collection status          : ${COLL_STATUS}"
echo " Snapshot used              : ${SNAPSHOT_NAME}"
echo "=========================================="

# Gracefully flush WAL before container stops
echo ""
echo "Flushing WAL before shutdown..."
curl -sS -X POST "http://localhost:${PORT}/collections/${COLLECTION}/snapshots" \
  -H 'Content-Type: application/json' >/dev/null 2>&1 || true
sleep 5

echo ""
echo "storage/ is ready. To use it as your main storage:"
echo ""