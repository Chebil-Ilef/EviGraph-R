#!/bin/bash
#SBATCH --job-name=evigraph-backup-snapshot
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/backup_snapshot_%j.log

# Uploads the latest local snapshot (+ checksum) to S3.

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env.s3}"

if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r raw_line; do
    line="${raw_line#"${raw_line%%[![:space:]]*}"}"
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    value="${value#\"}" ; value="${value%\"}"
    value="${value#\'}" ; value="${value%\'}"
    [[ -v "$key" ]] || export "$key=$value"
  done < "$ENV_FILE"
fi

BUCKET="${BUCKET:-arxiv-latest-postprocessing}"
SNAPSHOT_PREFIX="${SNAPSHOT_PREFIX:-snapshots/unarxive_chunks}"
COLLECTION="${COLLECTION:-unarxive_chunks}"
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-${REPO_DIR}/snapshots}"
DRY_RUN="${DRY_RUN:-0}"

COLLECTION_DIR="${SNAPSHOTS_DIR}/${COLLECTION}/latest"
if [[ ! -d "$COLLECTION_DIR" ]]; then
  echo "ERROR: snapshot collection dir not found: $COLLECTION_DIR" >&2
  exit 1
fi

LATEST_SNAPSHOT=$(find "$COLLECTION_DIR" -maxdepth 1 -name "*.snapshot" \
  -printf '%T@ %p\n' | sort -rn | head -1 | awk '{print $2}')
if [[ -z "$LATEST_SNAPSHOT" ]]; then
  echo "ERROR: no .snapshot files found in $COLLECTION_DIR" >&2
  exit 1
fi

CHECKSUM_FILE="${LATEST_SNAPSHOT}.checksum"
SNAPSHOT_NAME=$(basename "$LATEST_SNAPSHOT")

echo "=== backup_snapshot_to_s3 ==="
echo "Host     : $(hostname)"
echo "Snapshot : $SNAPSHOT_NAME"
echo "Bucket   : s3://${BUCKET}/${SNAPSHOT_PREFIX}/"
echo "Dry run  : $DRY_RUN"
echo ""

export BUCKET SNAPSHOT_PREFIX DRY_RUN
uv run python - <<PYEOF
import os, time, threading
from pathlib import Path
import boto3

dry_run         = os.environ["DRY_RUN"] == "1"
bucket          = os.environ["BUCKET"]
snapshot_prefix = os.environ["SNAPSHOT_PREFIX"]
latest_snapshot = Path("${LATEST_SNAPSHOT}")
checksum_file   = Path("${CHECKSUM_FILE}")
snap_key        = f"{snapshot_prefix}/{latest_snapshot.name}"

endpoint_url = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3") or None
profile      = os.environ.get("AWS_PROFILE") or None

def make_client():
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3", endpoint_url=endpoint_url)

def fmt_bytes(n):
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024

class ProgressBar:
    def __init__(self, total, label):
        self.total = total; self.seen = 0; self.label = label
        self.start = time.time(); self._lock = threading.Lock()
    def __call__(self, chunk):
        with self._lock:
            self.seen += chunk
            pct     = self.seen / self.total * 100 if self.total else 0
            elapsed = time.time() - self.start
            speed   = self.seen / elapsed if elapsed > 0 else 0
            eta     = (self.total - self.seen) / speed if speed > 0 else 0
            print(f"\r  {self.label}: {fmt_bytes(self.seen)}/{fmt_bytes(self.total)}"
                  f"  {pct:.1f}%  {fmt_bytes(speed)}/s  ETA {eta/60:.1f}m", end="", flush=True)
    def done(self): print()

def upload_file(s3, local: Path, key: str):
    size = local.stat().st_size
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] uploading {local.name} ({fmt_bytes(size)}) -> s3://{bucket}/{key}")
    if dry_run:
        print("  [DRY-RUN] skipped"); return
    cb = ProgressBar(size, local.name)
    s3.upload_file(str(local), bucket, key, Callback=cb)
    cb.done()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] done: {local.name}")

s3 = None if dry_run else make_client()
upload_file(s3, latest_snapshot, snap_key)
if checksum_file.exists():
    upload_file(s3, checksum_file, f"{snap_key}.checksum")
PYEOF

echo ""
echo "=== Done ==="
