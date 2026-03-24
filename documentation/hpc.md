# Scholarly Indexing Pipeline Architecture

## Overview

Goal: build a **semantic index for ~2.8M scholarly papers (~30M chunks)** using:

- **Qdrant** → vector database for embeddings

Dataset:

- `ines-besrour/unarxive_2024` (HuggingFace)

Pipeline runs on:

- **TU Dresden ZIH HPC**
- GPU cluster **Capella**
- optionally CPU cluster **Barnard**

The architecture uses a **two-phase pipeline** designed for HPC constraints.

---

# 1. Two-Phase Pipeline Architecture

## Phase A — Parallel Chunking + Embedding

Runs as **GPU job arrays** on Capella.

Input:

```
unarxive dataset JSON batches
```

Processing:

```
extract sections
↓
chunk text
↓
compute embeddings
↓
write shard artifacts
```

Output:

```
Parquet shards stored in horse workspace
```

---

## Phase B — Database Ingestion

Runs as **a sequential ingestion job**.

Input:

```
shard artifacts from Phase A
```

Processing:

```
read shard
↓
insert vectors into Qdrant
↓
checkpoint progress
```

Output:

```
Qdrant snapshot
```

These snapshots are then persisted:

```
cat → horse → /projects → optional S3
```

This separation ensures:

- **Phase A:** massively parallel GPU compute
- **Phase B:** controlled IO-heavy DB build

---

# 2. HPC Storage Constraints

Understanding the HPC storage hierarchy is essential.

| Location | Purpose |
| --- | --- |
| `/home` | small storage for code/scripts |
| `/projects` | persistent storage but **read-only on compute nodes** |
| `/data/horse` | long-lived scratch workspace |
| `/data/cat` | high-IO workspace for Capella |

---

## Workspace lifetimes

| Filesystem | Typical usage |
| --- | --- |
| horse | long running pipelines |
| cat | high IOPS scratch |
| walrus | mid-term storage (read-only on compute nodes) |

Important constraint:

```
cat workspace lifetime ≈ 30 days (extendable)
```

Therefore:

**Never rely on cat for permanent data.**

---

# 3. Workspace Strategy

## Phase A workspace

Filesystem:

```
/data/horse/ws/<project>
```

Reason:

- long-lived
- extendable
- safe checkpoint location

Structure:

```
horse workspace
│
├── shards/
├── manifest/
├── logs/
├── progress/
├── singularity_cache/
├── hf_cache/
```

---

## Phase B workspace

Filesystem:

```
/data/cat/ws/<project>
```

Reason:

- highest IO performance
- ideal for building vector DB

Structure:

```
cat workspace
│
├── qdrant_storage/
├── qdrant_snapshots/
├── temp/
└── logs/
```

---

# 4. Dataset Preparation

Dataset:

```
unarxive_2024
```

Size:

```
~2.8M papers
```

Chunk estimate:

```
~30M chunks
```

Dataset must be **pre-split into JSON batches**.

Example:

```
batch_0001.jsonl
batch_0002.jsonl
...
batch_2800.jsonl
```

Batch size:

```
1000 papers
```

Result:

```
2800 batches
```

This determines the **Slurm job array size**.

---

# 5. Phase A — Chunking + Embedding

## Slurm job array

Example:

```
sbatch --array=0-2799%284 embed_job.sh
```

The `%284` limits concurrency.

---

## Required Capella Slurm parameters

Example configuration:

```
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --array=0-2799%50
```

---

## Phase A workflow

Each job:

```
load JSON batch
↓
extract abstract + sections
↓
chunk text
↓
compute embeddings
↓
write Parquet shard
```

---

## Phase A outputs

Stored in horse workspace:

```
/data/horse/ws/<project>/shards/
```

Files:

```
vectors_shard_000123.parquet
citations_shard_000123.parquet
DONE_000123
```

---

## Vector shard schema

| field | description |
| --- | --- |
| point_id | deterministic ID |
| paper_id | article identifier |
| chunk_id | chunk index |
| text | chunk text |
| section | section name |
| embedding | vector |

---

## Citation shard schema

| field | description |
| --- | --- |
| paper_id | article ID |
| doi | DOI |
| ref_dois | referenced papers |

---

# 6. Deterministic Point IDs

The original design suggested:

```
point_id = hash(paper_id + chunk_id)
```

This is **incorrect** because Python `hash()` is randomized.

Correct deterministic approaches:

### Option A — UUID5

```
import uuid

point_id = str(uuid.uuid5(
    uuid.NAMESPACE_DNS,
    f"{paper_id}_{chunk_id}"
))
```

---

### Option B — xxhash → uint64

```
import xxhash

point_id = xxhash.xxh64(
    f"{paper_id}_{chunk_id}"
).intdigest()
```

Both are:

- deterministic
- cross-machine safe
- compatible with Qdrant

---

# 7. Phase A Checkpointing

Each shard produces a completion marker.

```
DONE_XXXX
```

Example:

```
DONE_0123
```

Manifest file:

```
manifest/shard_status.jsonl
```

Example entry:

```
{
  "shard_id": 123,
  "status": "DONE",
  "rows": 10432
}
```

Recovery:

```
if shard exists AND DONE marker exists → skip
otherwise → recompute shard
```

---

# 8. Phase B — Database Ingestion

Purpose:

```
convert shards → production databases
```

Targets:

```
Qdrant vector DB
```

---

## Capella GPU constraint

Capella **does not allow CPU-only jobs**.

Therefore Phase B must request a GPU even if unused.

Example:

```
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
```

Alternatively:

Phase B could run on **Barnard CPU cluster**.

---

# 9. Qdrant Collection Creation

Before ingestion:

```
create collection if not exists
```

Example:

```python
client.create_collection(
    collection_name="unarxiv_chunks",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)
```

Always check first:

```
if collection exists → skip creation
```

This ensures restartability.

---

# 10. Ingestion Workflow

Process shards sequentially:

```
read shard
↓
insert vectors into Qdrant
↓
update progress file
```

---

# 11. Phase B Checkpointing

Progress file:

```
progress/ingested_shards.jsonl
```

Example entry:

```
{
  "shard_id": 123,
  "status": "INGESTED"
}
```

Restart logic:

```
if shard already ingested → skip
else → ingest
```

Atomic write pattern:

```
write temp file
rename → final
```

---

# 12. Snapshot Strategy

After ingestion:

### Qdrant snapshot

API call:

```
POST /collections/{collection}/snapshots
```

Snapshot path inside container:

```
/qdrant/snapshots/
```

---

# 13. Singularity Container Setup

HPC compute nodes use **Singularity**, not Docker.

Convert Docker image:

```
singularity build qdrant.sif docker://qdrant/qdrant
```

---

## Important container rule

Database storage **must be mounted**, not inside container.

Example:

```
singularity exec \
  --bind $CAT_WS/qdrant_storage:/qdrant/storage \
  --bind $CAT_WS/qdrant_snapshots:/qdrant/snapshots \
  qdrant.sif \
  /qdrant
```

---

# 14. Singularity Cache Configuration

Avoid filling `/home` quota.

Set environment variables:

```
export SINGULARITY_CACHEDIR=/data/horse/ws/<project>/singularity_cache
export SINGULARITY_TMPDIR=/data/horse/ws/<project>/singularity_tmp
```

If using HuggingFace:

```
export HF_HOME=/data/horse/ws/<project>/hf_cache
```

---

# 15. Snapshot Storage Workflow

Snapshots initially stored in:

```
/data/cat/ws/<project>/qdrant_snapshots
```

Then transferred.

Correct transfer tool:

```
Datamover (dtcp / dtrsync)
```

Example:

```
dtcp -r /data/cat/ws/<project>/qdrant_snapshots \
        /data/horse/ws/<project>/snapshots
```

Next step:

```
copy to /projects
```

Note:

```
/projects is read-only on compute nodes
```

Transfer must be done from login or Datamover node.

---

# 16. Persistence Strategy

| Location | Purpose |
| --- | --- |
| cat | temporary DB build |
| horse | durable pipeline state |
| /projects | long-term storage |

Workflow:

```
cat
↓
horse
↓
projects
↓
(optional) S3
```

---

# 17. S3 Usage

S3 can be used as **cold storage backup**.

Good for:

```
qdrant_snapshot
```

Not good for:

```
live database usage
automatic backup by HPC
```

Recommended redundancy:

```
/projects
+ S3
```

---

# 18. Container Image Location

The `.sif` container image should live in:

```
/home
or
/projects
```

Reason:

- small (~200MB)
- permanent
- not subject to workspace expiration

---

# 19. Final Pipeline Summary

### Phase A

```
GPU job array (2800 tasks)
chunk + embed
write shards
horse workspace
```

---

### Phase B

```
single ingestion job
build DB on cat
checkpoint progress
create snapshots
```

---

### Persistence

```
cat
↓
horse
↓
projects
↓
SAVE INDEX TO HUGGINGFACE AS DATASET dense and hybrid
(optional S3 backup)
```

---
