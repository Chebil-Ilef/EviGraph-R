#!/bin/bash
#SBATCH --job-name=evigraph-backup-latest
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/backup_latest_%j.log

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env.s3}"

# Load .env.s3 (without overriding vars already set in the environment)
if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r raw_line; do
    line="${raw_line#"${raw_line%%[![:space:]]*}"}"   # ltrim
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    value="${value#\"}" ; value="${value%\"}"
    value="${value#\'}" ; value="${value%\'}"
    [[ -v "$key" ]] || export "$key=$value"
  done < "$ENV_FILE"
fi

BUCKET="${BUCKET:-arxiv-latest}"
SNAPSHOT_PREFIX="${SNAPSHOT_PREFIX:-snapshots/unarxive_chunks}"
STORAGE_PREFIX="${STORAGE_PREFIX:-storage}"
COLLECTION="${COLLECTION:-unarxive_chunks}"
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-${REPO_DIR}/snapshots}"
STORAGE_DIR="${STORAGE_DIR:-${REPO_DIR}/storage}"
DRY_RUN="${DRY_RUN:-0}"

# Find the latest local snapshot (by mtime)
COLLECTION_DIR="${SNAPSHOTS_DIR}/${COLLECTION}"
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

SNAPSHOT_NAME=$(basename "$LATEST_SNAPSHOT")
CHECKSUM_FILE="${LATEST_SNAPSHOT}.checksum"

echo "=== backup_latest_to_s3 ==="
echo "Host           : $(hostname)"
echo "Repo           : $REPO_DIR"
echo "Bucket         : s3://${BUCKET}"
echo "Snapshot       : $SNAPSHOT_NAME"
echo "Storage dir    : $STORAGE_DIR"
echo "Dry run        : $DRY_RUN"
echo ""

# Delegate all S3 work to Python (boto3 is already in the uv venv)
export BUCKET SNAPSHOT_PREFIX STORAGE_PREFIX COLLECTION SNAPSHOTS_DIR STORAGE_DIR DRY_RUN
uv run python - <<PYEOF
import os, sys, time, threading
from pathlib import Path
import boto3

dry_run        = os.environ["DRY_RUN"] == "1"
bucket         = os.environ["BUCKET"]
snapshot_prefix= os.environ["SNAPSHOT_PREFIX"]
storage_prefix = os.environ["STORAGE_PREFIX"]
storage_dir    = Path(os.environ["STORAGE_DIR"])
latest_snapshot= Path("${LATEST_SNAPSHOT}")
checksum_file  = Path("${CHECKSUM_FILE}")
snap_key       = f"{snapshot_prefix}/{latest_snapshot.name}"

endpoint_url   = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3") or None
profile        = os.environ.get("AWS_PROFILE") or None

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
        self.total = total
        self.seen  = 0
        self.label = label
        self.start = time.time()
        self._lock = threading.Lock()

    def __call__(self, chunk):
        with self._lock:
            self.seen += chunk
            pct     = self.seen / self.total * 100 if self.total else 0
            elapsed = time.time() - self.start
            speed   = self.seen / elapsed if elapsed > 0 else 0
            eta     = (self.total - self.seen) / speed if speed > 0 else 0
            print(
                f"\r  {self.label}: {fmt_bytes(self.seen)}/{fmt_bytes(self.total)}"
                f"  {pct:.1f}%  {fmt_bytes(speed)}/s  ETA {eta/60:.1f}m",
                end="", flush=True,
            )

    def done(self):
        print()  # newline after progress bar

def upload_file(s3, local: Path, key: str):
    size = local.stat().st_size
    ts   = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] uploading {local.name} ({fmt_bytes(size)}) -> s3://{bucket}/{key}")
    if dry_run:
        print(f"  [DRY-RUN] skipped")
        return
    cb = ProgressBar(size, local.name)
    s3.upload_file(str(local), bucket, key, Callback=cb)
    cb.done()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] done: {local.name}")

def object_exists(s3, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise

def sync_directory(s3, local_dir: Path, prefix: str):
    files = sorted(local_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] syncing {local_dir} ({len(files)} files, {fmt_bytes(total_size)}) -> s3://{bucket}/{prefix}/")
    if dry_run:
        for f in files:
            rel = f.relative_to(local_dir)
            print(f"  [DRY-RUN] {f} -> s3://{bucket}/{prefix}/{rel}")
        return
    uploaded = skipped = 0
    done_bytes = 0
    overall_start = time.time()
    for i, f in enumerate(files, 1):
        rel = f.relative_to(local_dir)
        key = f"{prefix}/{rel}"
        size = f.stat().st_size
        if object_exists(s3, key):
            skipped += 1
            done_bytes += size
        else:
            cb = ProgressBar(size, f"  [{i}/{len(files)}] {f.name}")
            s3.upload_file(str(f), bucket, key, Callback=cb)
            cb.done()
            uploaded += 1
            done_bytes += size
        elapsed = time.time() - overall_start
        overall_speed = done_bytes / elapsed if elapsed > 0 else 0
        remaining = total_size - done_bytes
        eta = remaining / overall_speed if overall_speed > 0 else 0
        print(
            f"  overall: {fmt_bytes(done_bytes)}/{fmt_bytes(total_size)}"
            f"  ({done_bytes/total_size*100:.1f}%)  {fmt_bytes(overall_speed)}/s"
            f"  ETA {eta/60:.1f}m  [{i}/{len(files)} files]",
            flush=True,
        )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] sync done: uploaded={uploaded}, skipped={skipped}")

s3 = None if dry_run else make_client()

print("--- Uploading snapshot ---")
upload_file(s3, latest_snapshot, snap_key)
if checksum_file.exists():
    upload_file(s3, checksum_file, f"{snap_key}.checksum")

print()
print("--- Syncing storage/ ---")
sync_directory(s3, storage_dir, storage_prefix)
PYEOF

echo ""
echo "=== Done ==="
