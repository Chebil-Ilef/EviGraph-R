# VM Setup: Restore EviGraph-R from S3 Snapshot

Follow these steps to get a fresh VM running with the Qdrant vector store restored from S3.

---

## Prerequisites

- Git, Docker, Python 3.10+, `uv`, `curl`, `aws` CLI (boto3 is installed via uv — no aws CLI needed)
- Docker daemon running (`docker info` should succeed)
- S3 credentials for the `arxiv-latest-postprocessing` bucket

---

## Step 1 — Clone the repository

```bash
git clone <repo-url> EviGraph-R
cd EviGraph-R
```

---

## Step 2 — Create `.env` (main config)

```bash
cp .env.example .env   # or copy the template below
```

Edit `.env` and fill in at minimum:

```ini
LLM_API_KEY=<your key>
LLM_API_BASE=https://llm.scads.ai/v1
HF_TOKEN=<your HuggingFace token>
```

Leave `QDRANT_URL` and `QDRANT_API_KEY` empty to use the local Docker Qdrant.

---

## Step 3 — Create `.env.s3` (S3 credentials)

Create a file named `.env.s3` in the repo root:

```ini
AWS_ACCESS_KEY_ID=<your access key>
AWS_SECRET_ACCESS_KEY=<your secret key>
AWS_DEFAULT_REGION=<region, e.g. eu-central-1>

# Optional: only needed if using a non-AWS S3-compatible endpoint
S3_ENDPOINT_URL=

# Optional overrides (defaults shown)
BUCKET=arxiv-latest-postprocessing
COLLECTION=unarxive_chunks
```

---

## Step 4 — Download the latest snapshot from S3

This pulls only the most recent `.snapshot` file into `snapshots/unarxive_chunks/`:

```bash
bash src/indexing/scripts/from_s3_restore_backup.sh \
  --snapshot \
  --out snapshots/unarxive_chunks
```

To preview what would be downloaded without actually downloading:

```bash
bash src/indexing/scripts/from_s3_restore_backup.sh --snapshot --dry-run
```

---

## Step 5 — Restore the snapshot into `storage/`

This starts a temporary Docker Qdrant container, loads the snapshot, waits for the optimizer to flush, then shuts down cleanly:

```bash
bash src/indexing/scripts/snapshot_to_storage_local.sh
```

The script writes directly into `storage/`. When it finishes you will see:

```
==========================================
 RECOVERY COMPLETE
 Points in storage : <N>
 Collection status : green
==========================================
```

---

## Step 6 — Verify

Start Qdrant normally and confirm the collection is healthy:

```bash
docker run -d --rm \
  --name qdrant \
  -p 6333:6333 \
  -v "$(pwd)/storage:/qdrant/storage" \
  qdrant/qdrant

curl -s http://localhost:6333/collections/unarxive_chunks | python3 -m json.tool
```

Expected: `"status": "green"` and `points_count` matching what the restore reported.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ERROR: storage/collections already exists` | The `storage/` dir already has data. Move it aside: `mv storage storage.old` |
| `no .snapshot files found` | Check your S3 credentials and bucket name in `.env.s3` |
| Qdrant container stops immediately | Run `docker logs qdrant-recovery` or check `logs/snapshot_to_storage_server_local.log` |
| `optimizer_status` never reaches `ok` | Wait longer; large collections can take several minutes to merge segments |
