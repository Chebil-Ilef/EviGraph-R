#!/bin/bash
# Restore a snapshot OR storage from S3.
#
# Usage:
#   ./from_s3_restore_backup.sh --snapshot   [--collection NAME] [--out DIR]
#   ./from_s3_restore_backup.sh --storage    [--collection NAME] [--out DIR]
#   ./from_s3_restore_backup.sh --dry-run    --snapshot|--storage
# Example:
#   ./src/indexing/scripts/from_s3_restore_backup.sh --snapshot --out _data/restore
#   ./src/indexing/scripts/from_s3_restore_backup.sh --storage --out _data/restore 
# Reads AWS credentials from .env.s3 (same convention as backup script).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env.s3}"
BUCKET="${BUCKET:-arxiv-latest-postprocessing}"
COLLECTION="${COLLECTION:-unarxive_chunks}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/restored}"
DRY_RUN=0
MODE=""

usage() {
  echo "Usage: $0 --snapshot|--storage [--collection NAME] [--out DIR] [--dry-run]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --snapshot)   MODE=snapshot; shift ;;
    --storage)    MODE=storage;  shift ;;
    --collection) COLLECTION="$2"; shift 2 ;;
    --out)        OUT_DIR="$2";    shift 2 ;;
    --dry-run)    DRY_RUN=1;       shift ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

[[ -z "$MODE" ]] && { echo "ERROR: --snapshot or --storage is required"; usage; }

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
else
  echo "WARNING: $ENV_FILE not found — relying on environment credentials"
fi

if [[ "$MODE" == "snapshot" ]]; then
  S3_PREFIX="snapshots/${COLLECTION}"
else
  # full Qdrant storage tree: storage/.lock + storage/aliases/ + storage/collections/
  S3_PREFIX="storage"
fi

echo "=== from_s3_restore_backup ==="
echo "Host       : $(hostname)"
echo "Mode       : $MODE"
echo "Collection : $COLLECTION"
echo "Bucket     : s3://${BUCKET}/${S3_PREFIX}/"
echo "Output dir : $OUT_DIR"
echo "Dry run    : $DRY_RUN"
echo ""

mkdir -p "$OUT_DIR"

export BUCKET S3_PREFIX OUT_DIR DRY_RUN MODE COLLECTION
uv run python - <<'PYEOF'
import os, sys, time, threading
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

dry_run    = os.environ["DRY_RUN"] == "1"
bucket     = os.environ["BUCKET"]
prefix     = os.environ["S3_PREFIX"]
out_dir    = Path(os.environ["OUT_DIR"])
mode       = os.environ["MODE"]
collection = os.environ["COLLECTION"]

endpoint_url = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3") or None
profile      = os.environ.get("AWS_PROFILE") or None

def make_client():
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3", endpoint_url=endpoint_url)

def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
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
                  f"  {pct:.1f}%  {fmt_bytes(speed)}/s  ETA {eta/60:.1f}m",
                  end="", flush=True)
    def done(self): print()

s3 = make_client()

# list objects under prefix
paginator = s3.get_paginator("list_objects_v2")
objects = []
for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
    for obj in page.get("Contents", []):
        objects.append(obj)

if not objects:
    print(f"ERROR: no objects found at s3://{bucket}/{prefix}/", file=sys.stderr)
    sys.exit(1)

# for snapshots pick the latest .snapshot file; for storage download everything
if mode == "snapshot":
    snap_objects = [o for o in objects if o["Key"].endswith(".snapshot")]
    if not snap_objects:
        print(f"ERROR: no .snapshot files found under s3://{bucket}/{prefix}/", file=sys.stderr)
        sys.exit(1)
    latest = max(snap_objects, key=lambda o: o["LastModified"])
    checksum_key = latest["Key"] + ".checksum"
    to_download = [latest]
    try:
        s3.head_object(Bucket=bucket, Key=checksum_key)
        to_download.append({"Key": checksum_key, "Size": 0})
    except ClientError:
        pass
    # snapshot: flatten into out_dir
    def dest_for(key): return out_dir / Path(key).name
else:
    to_download = objects
    # storage: preserve subdirectory structure relative to the prefix
    def dest_for(key):
        rel = key[len(prefix):].lstrip("/")
        return out_dir / rel

total_size = sum(o.get("Size", 0) for o in to_download)
print(f"Found {len(to_download)} file(s) to download ({fmt_bytes(total_size)} total):\n")
for obj in to_download:
    key  = obj["Key"]
    size = obj.get("Size", 0)
    print(f"  s3://{bucket}/{key}  ({fmt_bytes(size)})")
print()

if dry_run:
    print("[DRY-RUN] no files downloaded.")
    sys.exit(0)

for i, obj in enumerate(to_download, 1):
    key  = obj["Key"]
    dest = dest_for(key)
    size = obj.get("Size", 0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{i}/{len(to_download)}] {key} ({fmt_bytes(size)}) -> {dest}")
    cb = ProgressBar(size, key) if size > 0 else None
    s3.download_file(bucket, key, str(dest), Callback=cb)
    if cb:
        cb.done()
    done_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{done_ts}] done")

print(f"\nAll files saved to: {out_dir}")
PYEOF


echo ""
echo "=== Done ==="
