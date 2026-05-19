#!/bin/bash
#SBATCH --job-name=evigraph-backup-storage
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/backup_storage_%j.log

# Syncs storage/ to S3, skipping files already present.

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
STORAGE_PREFIX="${STORAGE_PREFIX:-storage}"
STORAGE_DIR="${STORAGE_DIR:-${REPO_DIR}/storage}"
DRY_RUN="${DRY_RUN:-0}"

echo "=== backup_storage_to_s3 ==="
echo "Host        : $(hostname)"
echo "Storage dir : $STORAGE_DIR"
echo "Bucket      : s3://${BUCKET}/${STORAGE_PREFIX}/"
echo "Dry run     : $DRY_RUN"
echo ""

export BUCKET STORAGE_PREFIX STORAGE_DIR DRY_RUN
uv run python - <<PYEOF
import os, time, threading
from pathlib import Path
import boto3

dry_run        = os.environ["DRY_RUN"] == "1"
bucket         = os.environ["BUCKET"]
storage_prefix = os.environ["STORAGE_PREFIX"]
storage_dir    = Path(os.environ["STORAGE_DIR"])

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

def object_exists(s3, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise

def sync_directory(s3, local_dir: Path, prefix: str):
    files = sorted(f for f in local_dir.rglob("*") if f.is_file())
    total_size = sum(f.stat().st_size for f in files)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] syncing {local_dir} "
          f"({len(files)} files, {fmt_bytes(total_size)}) -> s3://{bucket}/{prefix}/")
    if dry_run:
        for f in files:
            print(f"  [DRY-RUN] {f} -> s3://{bucket}/{prefix}/{f.relative_to(local_dir)}")
        return
    uploaded = skipped = 0
    done_bytes = 0
    overall_start = time.time()
    for i, f in enumerate(files, 1):
        key = f"{prefix}/{f.relative_to(local_dir)}"
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
        print(f"  overall: {fmt_bytes(done_bytes)}/{fmt_bytes(total_size)}"
              f"  ({done_bytes/total_size*100:.1f}%)  {fmt_bytes(overall_speed)}/s"
              f"  ETA {eta/60:.1f}m  [{i}/{len(files)} files]", flush=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] sync done: uploaded={uploaded}, skipped={skipped}")

s3 = None if dry_run else make_client()
sync_directory(s3, storage_dir, storage_prefix)
PYEOF

echo ""
echo "=== Done ==="
