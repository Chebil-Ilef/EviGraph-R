#!/bin/bash
#SBATCH --job-name=evigraph-push-hf
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=22:00:00
#SBATCH --output=logs/push_hf_%j.log

# Standalone script: push dense + sparse indexes from local shard files to HF.
# No Qdrant required. Reads directly from _data/shards/*.jsonl.
# Resumable: pass PUSH_KIND=dense|sparse|both (default: both).
# Env vars (loaded from .env if not already set):
#   HF_TOKEN, INDEXING_HF_USERNAME, INDEXING_HF_DENSE_DATASET, INDEXING_HF_SPARSE_DATASET
#   INDEXING_HF_EXPORT_SPLIT  (default: train)
#   EVI_SHARDS_DIR            (default: _data/shards)
#   EVI_HF_EXPORT_CACHE_DIR   (default: _data/dataset_index_cache)
#   PUSH_KIND                 (default: both)

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"

mkdir -p logs

ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"
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

export REPO_DIR
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export PUSH_KIND="${PUSH_KIND:-both}"

echo "=== push_indexes_to_hf ==="
echo "Host         : $(hostname)"
echo "Repo         : $REPO_DIR"
echo "Push kind    : $PUSH_KIND"
echo "Dense repo   : ${INDEXING_HF_USERNAME:-?}/${INDEXING_HF_DENSE_DATASET:-?}"
echo "Sparse repo  : ${INDEXING_HF_USERNAME:-?}/${INDEXING_HF_SPARSE_DATASET:-?}"
echo "Shards dir   : ${EVI_SHARDS_DIR:-${REPO_DIR}/_data/shards}"
echo ""

uv run python - <<'PYEOF'
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("push_hf")

#   settings   

REPO_DIR      = Path(os.environ["REPO_DIR"])
SHARDS_DIR    = Path(os.environ.get("EVI_SHARDS_DIR",        REPO_DIR / "_data" / "shards"))
CACHE_DIR     = Path(os.environ.get("EVI_HF_EXPORT_CACHE_DIR", REPO_DIR / "_data" / "dataset_index_cache"))
HF_TOKEN      = (os.environ.get("HF_TOKEN") or "").strip()
USERNAME      = (os.environ.get("INDEXING_HF_USERNAME") or "").strip()
DENSE_DATASET = (os.environ.get("INDEXING_HF_DENSE_DATASET") or "").strip()
SPARSE_DATASET= (os.environ.get("INDEXING_HF_SPARSE_DATASET") or "").strip()
SPLIT         = (os.environ.get("INDEXING_HF_EXPORT_SPLIT") or "train").strip()
PUSH_KIND     = (os.environ.get("PUSH_KIND") or "both").strip().lower()

missing = []
if not HF_TOKEN:      missing.append("HF_TOKEN")
if not USERNAME:      missing.append("INDEXING_HF_USERNAME")
if PUSH_KIND in ("dense",  "both") and not DENSE_DATASET:  missing.append("INDEXING_HF_DENSE_DATASET")
if PUSH_KIND in ("sparse", "both") and not SPARSE_DATASET: missing.append("INDEXING_HF_SPARSE_DATASET")
if missing:
    logger.error("Missing required env vars: %s", ", ".join(missing))
    sys.exit(1)

def repo_id(dataset_name: str) -> str:
    return dataset_name if "/" in dataset_name else f"{USERNAME}/{dataset_name}"

DENSE_REPO  = repo_id(DENSE_DATASET)  if DENSE_DATASET  else ""
SPARSE_REPO = repo_id(SPARSE_DATASET) if SPARSE_DATASET else ""

#   shard discovery  

shard_files = sorted(SHARDS_DIR.glob("*.jsonl"))
if not shard_files:
    logger.error("No .jsonl shard files found in %s", SHARDS_DIR)
    sys.exit(1)

logger.info("Found %d shard files in %s", len(shard_files), SHARDS_DIR)

#   row iterators 

import json

def iter_dense_rows():
    for path in shard_files:
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                vectors = rec.get("vectors") or {}
                dense   = vectors.get("dense")
                if not dense:
                    continue
                yield {
                    **(rec.get("payload") or {}),
                    "chunk_uid":    rec.get("chunk_uid"),
                    "dense_vector": dense,
                }

def iter_sparse_rows():
    for path in shard_files:
        with open(path) as fh:
            for line in fh:
                rec     = json.loads(line)
                vectors = rec.get("vectors") or {}
                sparse  = vectors.get("sparse") or {}
                indices = sparse.get("indices") or []
                values  = sparse.get("values")  or []
                if not indices or not values:
                    continue
                yield {
                    **(rec.get("payload") or {}),
                    "chunk_uid":      rec.get("chunk_uid"),
                    "sparse_indices": indices,
                    "sparse_values":  values,
                }

#   dataset card   ─

def build_card(repo: str, kind: str, shard_count: int) -> str:
    label  = "Dense" if kind == "dense" else "Sparse"
    tags   = "dense-embeddings" if kind == "dense" else "sparse-vectors"
    vcol   = "dense_vector" if kind == "dense" else "sparse_indices / sparse_values"
    return f"""---
pretty_name: EviGraph-R {label} Index
task_categories:
- feature-extraction
- text-retrieval
tags:
- evigraph-r
- scientific-literature
- qdrant
- {tags}
---

# EviGraph-R {label} Index

{kind.capitalize()} retrieval index for the unarXive 2024 corpus, generated by the EviGraph-R indexing pipeline using **BGE-M3**.

## Columns

- All chunk payload fields: `chunk_uid`, `chunk_type`, `chunk_index`, `total_chunks`, `section_title`, `embed_text`, `spans`, `paper_doi`, `paper_id_arxiv`, `title`, `authors`, `categories`, `year`, `language`, `discipline`
- Vector column(s): `{vcol}`

## Build info

- Repository: `{repo}`
- Split: `{SPLIT}`
- Shards: `{shard_count}`
- Embedding model: `BAAI/bge-m3`
"""

#   push helper    

import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationDelete

api = HfApi(token=HF_TOKEN)

# Rows streamed per parquet shard written to disk.
ROWS_PER_PARQUET_SHARD = int(os.environ.get("ROWS_PER_PARQUET_SHARD", "5_000_000"))
# Internal Arrow write-batch size — controls peak RAM (rows × ~3 KB ≈ 150 MB at 50k).
WRITE_BATCH_SIZE = int(os.environ.get("WRITE_BATCH_SIZE", "50_000"))
# Parallel upload workers (network I/O bound — threads are fine)
HF_UPLOAD_WORKERS = int(os.environ.get("HF_UPLOAD_WORKERS", "4"))

def _iter_chunks(iterable, n: int):
    """Yield successive n-sized lists from iterable without buffering the whole thing."""
    chunk: list[dict] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def push(repo: str, kind: str, rows_fn):
    logger.info("Creating/ensuring repo %s …", repo)
    api.create_repo(repo_id=repo, repo_type="dataset", private=False, exist_ok=True)

    parquet_dir = CACHE_DIR / kind / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Streaming %s rows → zstd parquet shards in %s (upload workers: %d) …",
        kind, parquet_dir, HF_UPLOAD_WORKERS,
    )

    shard_idx = 0
    total_rows = 0
    shard_rows = 0          # rows written into the current open shard
    writer: pq.ParquetWriter | None = None
    current_path: Path | None = None

    # (local_path, temp_hf_name) in order — populated as each shard is finalised
    shard_info: list[tuple[Path, str]] = []
    upload_futures: dict = {}  # future → (idx, temp_hf_name)

    def _close_shard() -> Path:
        nonlocal writer, current_path
        assert writer is not None and current_path is not None
        writer.close()
        writer = None
        logger.info("  closed shard %d: %d rows, %.1f MB",
                    shard_idx, shard_rows, current_path.stat().st_size / 1e6)
        path = current_path
        current_path = None
        return path

    def upload_shard(path: Path, dest: str, idx: int):
        logger.info("  uploading shard %d: %s (%.1f MB) …", idx, dest, path.stat().st_size / 1e6)
        api.upload_file(
            repo_id=repo,
            repo_type="dataset",
            path_in_repo=dest,
            path_or_fileobj=path,
            commit_message=f"Upload shard {idx} (temp)",
        )

    def drain_futures(pool: ThreadPoolExecutor, max_pending: int):
        while len(upload_futures) >= max_pending:
            done = next(as_completed(upload_futures))
            idx, name = upload_futures.pop(done)
            try:
                done.result()
            except Exception as exc:
                logger.error("  upload of shard %d (%s) failed: %s", idx, name, exc)
                raise

    with ThreadPoolExecutor(max_workers=HF_UPLOAD_WORKERS) as pool:
        for mini_batch in _iter_chunks(rows_fn(), WRITE_BATCH_SIZE):
            # Open a new writer if we don't have one yet (start of run or after shard roll).
            if writer is None:
                current_path = parquet_dir / f"{SPLIT}-{shard_idx:05d}.parquet"
                while current_path.exists():
                    # Shard was written in a previous (possibly interrupted) run — skip it.
                    logger.info("  shard %d already exists, skipping write", shard_idx)
                    existing_rows = pq.read_metadata(current_path).num_rows
                    total_rows += existing_rows
                    path = current_path
                    current_path = None
                    temp_name = f"data/{SPLIT}-{shard_idx:05d}.parquet"
                    shard_info.append((path, temp_name))
                    drain_futures(pool, HF_UPLOAD_WORKERS)
                    fut = pool.submit(upload_shard, path, temp_name, shard_idx)
                    upload_futures[fut] = (shard_idx, temp_name)
                    shard_idx += 1
                    shard_rows = 0
                    current_path = parquet_dir / f"{SPLIT}-{shard_idx:05d}.parquet"

                batch = pa.RecordBatch.from_pylist(mini_batch)
                writer = pq.ParquetWriter(current_path, batch.schema,
                                          compression="zstd", compression_level=3)
            else:
                batch = pa.RecordBatch.from_pylist(mini_batch)

            writer.write_batch(batch)
            shard_rows += len(mini_batch)
            total_rows += len(mini_batch)

            if shard_rows >= ROWS_PER_PARQUET_SHARD:
                path = _close_shard()
                temp_name = f"data/{SPLIT}-{shard_idx:05d}.parquet"
                shard_info.append((path, temp_name))
                drain_futures(pool, HF_UPLOAD_WORKERS)
                fut = pool.submit(upload_shard, path, temp_name, shard_idx)
                upload_futures[fut] = (shard_idx, temp_name)
                shard_idx += 1
                shard_rows = 0

        # Flush the final open shard (if any)
        if writer is not None:
            path = _close_shard()
            temp_name = f"data/{SPLIT}-{shard_idx:05d}.parquet"
            shard_info.append((path, temp_name))
            drain_futures(pool, HF_UPLOAD_WORKERS)
            fut = pool.submit(upload_shard, path, temp_name, shard_idx)
            upload_futures[fut] = (shard_idx, temp_name)

        # Wait for all remaining uploads
        for done in as_completed(upload_futures):
            idx, name = upload_futures[done]
            try:
                done.result()
            except Exception as exc:
                logger.error("  upload of shard %d (%s) failed: %s", idx, name, exc)
                raise

    num_shards = len(shard_info)
    logger.info("Written and uploaded %d parquet shards (%d total rows).", num_shards, total_rows)

    # Rename temp shards to HF naming convention in a single commit
    logger.info("Renaming shards to HF convention (of-%05d) in one commit …", num_shards)
    ops = []
    for i, (_, temp_name) in enumerate(shard_info):
        final_name = f"data/{SPLIT}-{i:05d}-of-{num_shards:05d}.parquet"
        if temp_name != final_name:
            ops.append(CommitOperationCopy(src_path_in_repo=temp_name, dest_path_in_repo=final_name))
            ops.append(CommitOperationDelete(path_in_repo=temp_name))
    if ops:
        api.create_commit(
            repo_id=repo,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Rename {num_shards} shards to HF naming convention",
        )

    logger.info("Uploading README …")
    api.upload_file(
        repo_id=repo,
        repo_type="dataset",
        path_in_repo="README.md",
        path_or_fileobj=build_card(repo, kind, len(shard_files)).encode(),
        commit_message="Update dataset card",
    )
    logger.info("Done: %s", repo)

#   main     ─

if PUSH_KIND in ("dense", "both"):
    push(DENSE_REPO, "dense", iter_dense_rows)

if PUSH_KIND in ("sparse", "both"):
    push(SPARSE_REPO, "sparse", iter_sparse_rows)

logger.info("All done.")
PYEOF

echo ""
echo "=== push complete ==="
