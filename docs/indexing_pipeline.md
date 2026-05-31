# EviGraph-R Indexing Pipeline — Technical Documentation

---

## 1. Data Source — unarXive 2024

### 1.1 Corpus Description

The primary data source is **unarXive 2024** (`ines-besrour/unarxive_2024` on HuggingFace), a large-scale structured dataset derived from the arXiv open-access repository. unarXive provides full-text scientific papers in a parsed, structured JSON format — preserving section boundaries, inline citation markers, bibliography entries, figure/table references, and document metadata.

Each document in the corpus is a JSON object with the following top-level fields:

| Field | Description |
|---|---|
| `paper_id` | arXiv identifier (e.g., `2401.12345`) |
| `abstract` | Structured abstract object with `text` field |
| `sections` | Dictionary mapping section titles to section objects (`text`, inline markers) |
| `bib_entries` | Bibliography dictionary keyed by internal reference IDs |
| `ref_entries` | Figure and table reference dictionary with captions |
| `metadata` | Paper metadata: `title`, `authors_parsed`, `doi`, `categories`, `versions`, `update_date`, `language`, `discipline` |


### 1.2 arXiv Subject Coverage

The unarXive 2024 corpus covers all major arXiv subject areas, including physics, mathematics, computer science, quantitative biology, statistics, electrical engineering, and economics. Each paper carries one or more arXiv category codes (e.g., `cs.CL`, `math.AP`, `physics.hep-th`) stored as the `categories` payload field indexed for filtered retrieval.

### 1.3 Temporal Scope

Year metadata is extracted from the paper's `versions` list (first version creation date) with a fallback to `update_date`. Only years in the range 1900–2100 are accepted; malformed dates are stored as null and excluded from year-based filters.

---

## 2. Preprocessing Pipeline

Before any chunking or embedding, each raw paper passes through a normalization and cleaning stage implemented in `src/indexing/preprocessing/preprocessor.py`.

### 2.1 Section Filtering

Certain sections are structurally uninformative for evidence retrieval and are excluded from indexing entirely. The skip list is matched case-insensitively against normalized section titles:

- Acknowledgements / Acknowledgments
- References
- Bibliography
- Index
- Table of Contents
- List of Figures / Tables / Algorithms

Sections with empty or null titles are renamed to `"Body"`.

### 2.2 Citation Marker Processing

unarXive encodes inline citations as `{{cite:<ref_id>}}` placeholders within section text. The preprocessor:

1. Strips all `{{cite:…}}` markers from the raw text, producing clean prose.
2. Records the character position of each removed marker.
3. Resolves each position to a sentence span using a sentence boundary detector (handling common abbreviations: `et al.`, `e.g.`, `Fig.`, `Eq.`, `vs.`, etc., and decimal numerals like `3.14`).
4. Looks up the bibliographic entry for each `ref_id` and extracts `doi`, `openalex_id`, `arxiv_id`, and `bib_entry_raw` fields.

Each processed chunk therefore carries a `cite_spans` list — a list of citation objects with character-level sentence offsets relative to the chunk, enabling downstream citation tracing and hop retrieval.

### 2.3 Figure and Table Reference Cleaning

Figure and table inline markers (`{{figure:<uuid>}}`, `{{table:<uuid>}}`) are replaced with human-readable strings: `[Figure: <caption>]` or `[Table: <caption>]` when a caption exists, or removed entirely when none is available.

### 2.4 Metadata Normalization

Author lists are parsed from `authors_parsed` (structured name arrays) with a fallback to the raw `authors` string split on commas and semicolons. DOI strings are normalized by stripping URL prefixes (`https://doi.org/`, `http://dx.doi.org/`, etc.) to bare DOI form. Category strings are split on whitespace into lists.

### 2.5 Chunk UID Generation

Each chunk receives a deterministic unique identifier computed as:

```
chunk_uid = SHA-1( paper_id + "\x00" + section_label + "\x00" + chunk_index + "\x00" + text )
```

This ensures reproducible identifiers across re-runs and enables idempotent upserts into the vector database.

---

## 3. Text Chunking Strategy

Chunking is implemented in `src/indexing/utils/chunker.py` using exact token counting via the embedding model's own tokenizer (`BAAI/bge-m3` via HuggingFace `AutoTokenizer`).

### 3.1 Abstract Chunking

Abstracts are treated as a single logical unit. If the abstract token count is at or below the threshold (500 tokens), it is stored as a single chunk. When an abstract exceeds 500 tokens, a sliding window is applied with a 50-token overlap, always splitting at sentence boundaries.

| Parameter | Value |
|---|---|
| Max tokens before split | 500 |
| Overlap when split | 50 tokens |
| Chunk type label | `"abstract"` |

### 3.2 Section / Subsection Chunking

Body sections use a sliding window strategy to produce overlapping chunks that preserve local context across section splits:

| Parameter | Value |
|---|---|
| Max tokens (single-chunk threshold) | 900 |
| Window size | 900 tokens |
| Overlap between windows | 30 tokens |
| Chunk type label | `"subsection"` |

### 3.3 Sentence Boundary Respect

All splits are forced to occur at sentence boundaries. The sentence detector uses a regex `(?<=[.!?])\s+(?=[A-Z0-9\(\\])` and filters false positives from:
- Common scientific abbreviations (`et al.`, `Fig.`, `Eq.`, `i.e.`, `e.g.`, `vs.`, `approx.`, `no.`, `ref.`, `sec.`, `vol.`, etc.)
- Single lowercase letter initialisms (`e.`, `i.`)
- Decimal numbers (`3.14`, `0.001`)

### 3.4 Tokenizer

Token counting uses the `BAAI/bge-m3` tokenizer loaded via `AutoTokenizer.from_pretrained` with `add_special_tokens=False`. The tokenizer is cached in memory across all chunks within a pipeline run to avoid repeated I/O overhead. A hard cap of 8,192 tokens per sequence is enforced at inference time (BGE-M3's maximum context length).

---

## 4. Indexing Pipeline Architecture

The indexing pipeline is orchestrated by `src/indexing/indexing_pipeline.py` and executed on the **Capella HPC cluster** using SLURM array jobs. It is structured as a multi-phase pipeline with distinct, resumable stages:

```
Phase 1: prepare-dataset   →  Phase 2: chunk (embed)   →  Phase 3: ingest   →  Phase 4: snapshot
```

### 4.1 Phases

| Phase | CLI flag | Description |
|---|---|---|
| `prepare-dataset` | `--phase prepare-dataset` | Streams the raw HuggingFace dataset and splits it into 1,000-paper batch JSONL files |
| `chunk` | `--phase chunk` | Preprocesses, chunks, and embeds assigned batch shards; writes `.jsonl` shard records to disk |
| `ingest` | `--phase ingest` | Upserts all shard records into Qdrant; single-task serial phase |
| `snapshot` | `--phase snapshot` | Creates a named Qdrant collection snapshot for durability |
| `run` | `--phase run` | Combined chunk + ingest in a single task (used for small-scale runs) |

---

## 5. HPC Execution — SLURM Job Design

### 5.1 Two-Step Job Submission

The full pipeline is submitted as two dependent SLURM jobs (`src/indexing/scripts/run_indexing_array_capella.sh`):

**Step 1 — Parallel Chunking/Embedding (200 concurrent tasks):**
```bash
R1=$(sbatch --parsable \
    --array=0-199%50 \
    --export=ALL,TOTAL_TASKS=200 \
    src/indexing/scripts/run_indexing_array_capella.sh)
```

**Step 2 — Serial Ingestion (single task, after all chunk tasks succeed):**
```bash
sbatch --dependency=afterok:$R1 \
    --array=0-0 \
    --export=ALL,TOTAL_TASKS=1,INGEST_ONLY=1,RECREATE_COLLECTION=1 \
    src/indexing/scripts/run_indexing_array_capella.sh
```

### 5.2 SLURM Resource Allocation

Each array task is allocated:

| Resource | Value |
|---|---|
| Partition | `capella` |
| Nodes | 1 |
| CPUs per task | 8 |
| GPUs | 1 (`gres=gpu:1`) |
| Memory | 170 GB |
| Wall-time | 22 hours |
| Max concurrent tasks | 50 (from `--array=0-199%50`) |

### 5.3 Work Distribution Across Tasks

Batch files are distributed round-robin across tasks. Task `i` processes batches at indices `{i, i + TOTAL_TASKS, i + 2·TOTAL_TASKS, …}` from the sorted list of 2,215 batch files. With 200 tasks, each task handles approximately 11 batches (~11,000 papers).

### 5.4 Dataset Preparation Synchronization

Task 0 is responsible for calling the `prepare-dataset` phase, which streams the HuggingFace dataset and writes the 2,215 batch JSONL files. All other tasks wait for a `.prepared` sentinel file to appear at `_data/unarxive_batches/.prepared` before proceeding. The wait loop polls every 30 seconds with a 2-hour maximum timeout.

### 5.5 Environment Variables for Tuning

| Variable | Default | Role |
|---|---|---|
| `EVI_SNAPSHOT_INTERVAL` | 50 | Create Qdrant snapshot every N ingested shard batches |
| `EVI_UPSERT_BATCH_SIZE` | 256 | Points per Qdrant upsert API call |
| `EVI_INGEST_THROTTLE_SEC` | 0.1 | Sleep between upsert batches to reduce lock contention |
| `EVI_WAIT_EVERY_N_BATCHES` | 4 | Force Qdrant sync wait every N batches |
| `EVI_DATASET_BATCH_SIZE` | 1000 | Papers per batch JSONL file |
| `EVI_SHARD_BATCH_SIZE` | 128 | Papers per embedding inference batch |
| `TOTAL_TASKS` | 200 | Number of parallel worker tasks |
| `INGEST_ONLY` | 0 | Skip chunking, go straight to ingest |
| `CLEAN_START` | 0 | Wipe all state and restart from scratch |
| `WIPE_STORAGE` | 0 | Wipe only Qdrant storage (keep shards for resume) |

---

## 6. Embedding — BAAI/bge-m3

The embedding model used for the full production run is **BAAI/bge-m3**, a state-of-the-art multilingual embedding model capable of producing both dense and sparse (lexical) representations from the same forward pass — enabling true hybrid retrieval.

| Parameter | Value |
|---|---|
| HuggingFace model ID | `BAAI/bge-m3` |
| Dense vector dimension | 1024 |
| Sparse representation | Yes (SPLADE-style lexical weights) |
| Max sequence length | 8,192 tokens |
| Inference batch size | 512 sequences per GPU call |
| Data type | `float16` (2× throughput vs. float32 on CUDA) |
| Normalization | L2 normalization before upsert (required for Cosine distance) |

Each chunk is encoded once to produce:
- A **dense vector** (1024-dimensional float16) for semantic similarity search.
- A **sparse vector** (variable-length token weight map) for lexical/keyword matching.

---

## 7. Vector Database — Qdrant

All chunk vectors and metadata are stored in a self-hosted **Qdrant** instance deployed inside a Singularity container (`qdrant.sif`, built from `docker://qdrant/qdrant`) on the same HPC node as the ingestion task.

## Technical Specifications

### Embedding Model: BGE-M3

**Architecture:**
- **Base model:** BERT-like encoder (similar to RoBERTa)
- **Training:** 200M+ text pairs from BEIR, SQuAD, NQ, etc.
- **Matryoshka:** Supports 256, 512, 1024-dim projections
- **Sparse:** Learned sparse token weighting (similar to SPLADE)
- **Multilingual:** Supports 100+ languages

**For Academic Papers:**
- **Strength:** Good at capturing semantic similarity across domains
- **Limitation:** No domain-specific fine-tuning for arXiv papers
- **Alternative:** Could use **SciNCL** (domain-specific) or **E5** (general-purpose)

### Qdrant Vector Database Configuration

**Vector DB Choice: Qdrant**

Qdrant vs. alternatives:

| DB | Pros | Cons |
|----|------|------|
| **Qdrant** | Native sparse vectors, Rust (fast), snapshot support | No cloud managed service |
| **Pinecone** | Managed, easy scaling | Vendor lock-in, expensive at scale |
| **Weaviate** | Flexible, GraphQL API | Slower than Qdrant for large scale |
| **FAISS** | Ultra-fast for CPU, offline indexing | No persistence, limited features |
| **Milvus** | Open-source, scalable | Requires Kubernetes |

**Qdrant Configuration:**

```python
# Collection schema
{
    "name": "academic_papers",
    "vectors": {
        "dense": {
            "size": 1024,
            "distance": "Cosine"  # [0, 2] range for normalized vectors
        },
        "sparse": {
            "index": {"on_disk": True}  # ~10x compression
        }
    },
    "payload_schema": {
        "paper_id_arxiv": {"type": "text"},
        "chunk_index": {"type": "integer"},
        "section_title": {"type": "text"},
        "embed_text": {"type": "text"},
        "cite_spans": {"type": "object"},
        "imrad_section_title": {"type": "text"},  # Enriched later
        "imrad_label_source": {"type": "text"}    # Enriched later
    }
}

# HNSW index parameters
{
    "m": 64,            # max edges per node
    "ef_construct": 1024,  # build-time search width
    "ef": 256,          # query-time search width (tunable)
    "full_scan_threshold": 20000  # when to skip HNSW
}
```

**Memory Footprint:**

| Component | Size per 30M vectors |
|-----------|---------------------|
| Dense vectors (1,024 × float32) | 120 GB |
| HNSW graph index (m=64) | 60–80 GB |
| Sparse vectors (on-disk) | 30–40 GB |
| Payloads (metadata) | 10–15 GB |
| **Total** | ~250–300 GB |

**Query Performance:**

```
Dense vector search + reranking:
- Warm cache: ~10–50 ms (top-100)
- Cold cache: ~100–500 ms
- Throughput: 100–500 QPS per Qdrant instance
```

---



### 7.1 Collection Schema

| Parameter | Value |
|---|---|
| Collection name | `unarxive_chunks` |
| Distance metric | Cosine |
| Dense vector name | `dense` (1024 dimensions) |
| Sparse vector name | `sparse` (BGE-M3 SPLADE output) |
| Points at completion | **69,026,381** |


| Metric | Value |
|---|---|
| Total papers indexed | **2,214,380** |
| Total dataset batches | **2,215** (1,000 papers/batch, last batch: 380 papers) |
| Total chunks generated | **69,026,381** |
| Average chunks per paper | **~31.2 chunks/paper** |
| Average chunks per batch shard | **~31,163 chunks/shard** |
| Final Qdrant snapshot | `unarxive_chunks-6701363880429838-2026-05-08-16-39-07.snapshot` |

### 7.2 HNSW Index Parameters (HPC Profile)

| Parameter | Value |
|---|---|
| `m` (edges per node) | 32 |
| `ef_construct` (build-time candidate list) | 128 |
| `ef` (query-time default) | 64 |
| `full_scan_threshold` | 10,000 vectors (exact search below this) |

### 7.3 Storage Optimization (HPC Profile)

| Parameter | Value |
|---|---|
| `vectors_on_disk` | `True` — memory-mapped vector storage |
| `payload_on_disk` | `True` — metadata on disk |
| Scalar quantization | `int8` — ~4× RAM reduction |
| `quantize_always_ram` | `False` — quantized vectors not pinned in RAM |
| `memmap_threshold` | 10,000 vectors |
| `indexing_threshold` | 10,000 vectors |
| WAL capacity | 32 MB |
| WAL flush interval | 15 seconds |

### 7.4 Payload Indexes (Filtered Search)

The following 10 payload fields are indexed in Qdrant for O(1) filtered retrieval:

```
paper_id_arxiv, paper_doi, title, authors, chunk_type,
chunk_index, total_chunks, section_title, year, categories
```

### 7.5 Full-Text (BM25) Index

A server-side BM25 index is built over the `embed_text` field using the `Qdrant/bm25` model, enabling keyword-level retrieval without a separate search system:

| Parameter | Value |
|---|---|
| Field | `embed_text` |
| Tokenizer | `word` |
| Min token length | 2 |
| Max token length | 40 |
| Lowercase | `True` |

### 7.6 Chunk Payload Schema

Each Qdrant point stores the following payload fields alongside its vector(s):

| Field | Type | Description |
|---|---|---|
| `chunk_uid` | string | SHA-1 deterministic identifier |
| `paper_id_arxiv` | string | arXiv paper ID |
| `paper_doi` | string | Normalized DOI |
| `title` | string | Paper title |
| `authors` | list[string] | Author names |
| `categories` | list[string] | arXiv subject categories |
| `year` | integer | Publication year (nullable) |
| `chunk_type` | string | `"abstract"` or `"subsection"` |
| `chunk_index` | integer | Position of chunk within paper |
| `total_chunks` | integer | Total chunks for the paper |
| `section_title` | string | Original section heading |
| `embed_text` | string | The actual text used for embedding |
| `cite_spans` | list[object] | Citation spans with offsets, DOI, arXiv ID, OpenAlex ID |
| `imrad_label` | string | IMRaD category (post-processing, see §9) |

## Large-Scale Deployment on HPC

### Infrastructure: TU Dresden ZIH Capella GPU Cluster

**Compute Environment:**
| Resource | Specification |
|----------|----------------|
| Cluster | Capella (NVIDIA A100 GPUs) |
| Partition | `capella` |
| GPUs per node | 1–8 (variable) |
| CPU cores per node | 14–128 |
| Memory per node | 100–256 GB |
| Interconnect | InfiniBand (high-speed) |

**Storage Hierarchy:**

| Filesystem | Capacity | Lifetime | Purpose |
|-----------|----------|----------|---------|
| `/home` | Small | Permanent | Code, scripts |
| `/projects` | ~5 TB | Permanent | Archived data |
| `/data/horse` | ~100 TB | Long-term (~1 year) | Phase A shards |
| `/data/cat` | ~50 TB | Temporary (~30 days) | Phase B Qdrant |
| `/scratch` | Ultra-fast | Ephemeral (~24h) | Temp files |

**Strategy:** Separate workspaces optimize I/O patterns:
- **horse:** Ideal for Phase A (many writers, sequential reads)
- **cat:** Ideal for Phase B (high-throughput DB writes)


---

## 8. Fault Tolerance and Checkpointing

One of the central engineering challenges of running this pipeline at scale on an HPC cluster is fault tolerance. SLURM jobs can be preempted, nodes can fail, and the 22-hour wall-time limit means long-running tasks may be forcibly killed. The pipeline is designed to tolerate all of these scenarios through a multi-layer checkpointing strategy.

### 8.1 Checkpoint Artifacts

#### Shard-Level Checkpoints (Chunking Phase)
For each batch shard, after all chunks are computed and written:
- `_data/shards/<stem>.jsonl` — the chunk records (uid, vectors, payload)
- `_data/shards/<stem>.done` — a sentinel file marking the shard as complete

On resume (`--resume` flag), the pipeline checks for the `.done` sentinel before re-processing a shard, allowing it to skip already-completed work.

#### Ingestion Progress Log
- `_data/progress/ingested_shards.jsonl` — an append-only log where each line records a successfully ingested shard:
  ```json
  {"stem": "batch_0001", "status": "INGESTED", "rows": 49511, "timestamp": "2026-05-04T09:46:29+00:00"}
  ```
  On resume, the ingest phase reads this file and skips any shards already listed.

#### Manifest Files
- `_data/manifests/shard_status.jsonl` — shard status log used by chunk workers
- `_data/manifests/run_metadata.json` — records the pipeline configuration (phase, profile, model, resume mode, collection name)

### 8.2 Qdrant Snapshot Mechanism

The ingest task creates periodic Qdrant collection snapshots during ingestion, triggered every `EVI_SNAPSHOT_INTERVAL=50` shard batches (~1.55M chunks per snapshot interval). Snapshots are tracked in `_data/qdrant_snapshots/manifest.jsonl`; the five most recent are retained and older ones are automatically deleted.

A final snapshot is always created at the end of successful ingestion. The production snapshot is:

```
unarxive_chunks-6701363880429838-2026-05-08-16-39-07.snapshot
```

This snapshot can be used to restore the entire collection if the Qdrant storage directory becomes corrupted.

### 8.3 Recovery Procedures

| Failure Scenario | Recovery Procedure |
|---|---|
| Chunk task killed mid-shard | Re-submit with `--resume`; incomplete `.jsonl` shard (no `.done`) is re-processed |
| Ingest task killed mid-ingestion | Re-submit with `INGEST_ONLY=1` and `RESUME_INGEST=1`; ingested shards are skipped |
| Qdrant storage corrupted | Set `WIPE_STORAGE=1`, restore from latest snapshot, re-ingest with `RESUME_INGEST=1` |
| Full pipeline restart | Set `CLEAN_START=1`; deletes all manifests, shards, storage, and snapshots |

### 8.4 Atomic Writes

All shard JSONL files are written via a temp-file-then-rename pattern. The `.done` sentinel is only written after the final rename completes, guaranteeing that a partial write (from a killed process) cannot be mistaken for a complete shard. On startup, the pipeline scans for and removes any `.tmp` stale files left by interrupted writes.

### 8.5 Runtime Monitoring

A background shell monitor runs alongside both the chunk and ingest phases, logging memory usage and Qdrant storage directory size every 60 seconds to the SLURM log file:

```
[MONITOR +60s] RAM: 142.3/170.0 GiB used | storage/: 187G
[MONITOR +120s] RAM: 143.1/170.0 GiB used | storage/: 192G
```

---

## 9. Ingestion Details

### 9.1 Qdrant Upsert Configuration

| Parameter | Value |
|---|---|
| Upsert batch size | 256 points per API call (env: `EVI_UPSERT_BATCH_SIZE`) |
| Retry attempts | 5 max, exponential backoff: 2s → 60s |
| Retry trigger | HTTP 408 (timeout) or connection errors |
| Per-call timeout | 1,800 seconds (30 minutes) |
| Inter-batch throttle | 0.1 s sleep between batches (env: `EVI_INGEST_THROTTLE_SEC`) |
| Sync wait frequency | Every 4 batches (env: `EVI_WAIT_EVERY_N_BATCHES`) |

### 9.2 Scale

- **69,026,381 total vector points** inserted into Qdrant
- Each point carries: 1 dense vector (1024 × float16 = 2 KB), 1 sparse vector (variable), and ~10 payload fields
- The Qdrant storage directory (vectors + HNSW index + payload + BM25 index + WAL) reached several hundred GB on the HPC storage system

---

## 10. Singularity Container Deployment

Because Capella HPC nodes do not support Docker, Qdrant is deployed inside a **Singularity** container built from the official Qdrant Docker image:

```bash
singularity build qdrant.sif docker://qdrant/qdrant
```

The `.sif` image is built once and reused across all job submissions. The ingest task launches the Qdrant server inside Singularity and waits for the REST health endpoint to become available before starting upserts. This approach enables running a production-grade vector database on an HPC cluster without root privileges.

---

## 11. Pipeline Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      EviGraph-R INDEXING PIPELINE (HPC / Capella)               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────────────┐     ║
║  │  DATA SOURCE                                                            │     ║
║  │  unarXive 2024 (HuggingFace: ines-besrour/unarxive_2024)               │     ║
║  │  2,214,380 papers · Full text + sections + citations + metadata        │     ║
║  └──────────────────────────────┬──────────────────────────────────────────┘     ║
║                                 │                                                ║
║                                 ▼  PHASE 1: prepare-dataset (Task 0 only)       ║
║  ┌──────────────────────────────────────────────────────────────────────────┐    ║
║  │  DATASET PREPARATION                                                     │    ║
║  │  Stream dataset → split into 2,215 batch JSONL files (1,000 papers ea.) │    ║
║  │  Sentinel file .prepared written after completion                        │    ║
║  │  CHECKPOINT: _data/unarxive_batches/*.jsonl + .prepared sentinel        │    ║
║  └──────────────────────────────┬──────────────────────────────────────────┘    ║
║                                 │                                                ║
║                                 ▼  PHASE 2: chunk (200 SLURM array tasks)       ║
║  ┌──────────────────────────────────────────────────────────────────────────┐    ║
║  │  PARALLEL CHUNKING + EMBEDDING (200 tasks × 1 GPU × 170 GB RAM)         │    ║
║  │                                                                          │    ║
║  │  Task 0 → batches {0, 200, 400, …}       Round-robin assignment         │    ║
║  │  Task 1 → batches {1, 201, 401, …}       ~11 batches per task           │    ║
║  │  …                                       ~11,000 papers per task        │    ║
║  │  Task 199 → batches {199, 399, …}                                       │    ║
║  │                                                                          │    ║
║  │  Per batch:                                                              │    ║
║  │    1. Preprocessing: normalize, filter sections, clean citations         │    ║
║  │    2. Chunking: abstract (≤500 tok) + section sliding window (900 tok)  │    ║
║  │    3. Embedding: BAAI/bge-m3 fp16, batch_size=512 → dense + sparse      │    ║
║  │    4. Shard write: _data/shards/<stem>.jsonl + <stem>.done sentinel      │    ║
║  │                                                                          │    ║
║  │  CHECKPOINT: _data/shards/*.jsonl (shard records)                        │    ║
║  │              _data/shards/*.done  (completion sentinels)                  │    ║
║  │              _data/manifests/shard_status.jsonl                          │    ║
║  └──────────────────────────────┬──────────────────────────────────────────┘    ║
║                                 │  dependency=afterok (all 200 tasks complete)  ║
║                                 ▼  PHASE 3: ingest (1 SLURM task)               ║
║  ┌──────────────────────────────────────────────────────────────────────────┐    ║
║  │  SERIAL INGESTION INTO QDRANT (Task 0, single node)                      │    ║
║  │                                                                          │    ║
║  │  Qdrant running in Singularity container (qdrant.sif)                    │    ║
║  │  Collection: unarxive_chunks                                             │    ║
║  │  HNSW: m=32, ef_construct=128 · int8 scalar quantization                │    ║
║  │  Vectors on disk (mmap) + Payload on disk                                │    ║
║  │                                                                          │    ║
║  │  For each shard (2,215 total):                                           │    ║
║  │    1. Check progress log → skip if already ingested (RESUME support)     │    ║
║  │    2. Read shard JSONL → upsert in batches of 256 points                 │    ║
║  │    3. Retry on 408 timeout: 5 attempts, exp. backoff 2s→60s              │    ║
║  │    4. Append to _data/progress/ingested_shards.jsonl (checkpoint)        │    ║
║  │    5. Every 50 shards: create periodic Qdrant SNAPSHOT                   │    ║
║  │                                                                          │    ║
║  │  Total ingested: 69,026,381 points across 2,215 shards                   │    ║
║  │  Duration: ~4 days, 6 hours                                              │    ║
║  │                                                                          │    ║
║  │  CHECKPOINT: _data/progress/ingested_shards.jsonl (append-only log)      │    ║
║  │  SNAPSHOTS:  _data/qdrant_snapshots/ (periodic + final)                  │    ║
║  └──────────────────────────────┬──────────────────────────────────────────┘    ║
║                                 │                                                ║
║                                 ▼  PHASE 4: snapshot (final)                    ║
║  ┌──────────────────────────────────────────────────────────────────────────┐    ║
║  │  FINAL SNAPSHOT + EXPORT                                                  │    ║
║  │  Create named Qdrant snapshot for long-term durability                    │    ║
║  │  Snapshot: unarxive_chunks-6701363880429838-2026-05-08-16-39-07          │    ║
║  │  Optional: push shards to HuggingFace (hf_export.py)                     │    ║
║  └──────────────────────────────┬──────────────────────────────────────────┘    ║
║                                 │                                                ║
║                                 ▼  POST-PROCESSING (separate SLURM jobs)        ║
║  ┌──────────────────────────────────────────────────────────────────────────┐    ║
║  │  IMRaD LABELING + CITATION RESOLUTION                                    │    ║
║  │  (see §12 — Post-Processing Pipeline)                                    │    ║
║  └──────────────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

FAULT TOLERANCE LEGEND:
  ──── Normal flow
  CHECKPOINT  →  Persisted state enabling resume after failure
  SNAPSHOTS   →  Point-in-time Qdrant backups for storage-level recovery
```

---

## 12. Post-Processing Pipeline

After indexing, two post-processing pipelines enrich the Qdrant payload fields in-place.

### 12.1 IMRaD Section Labeling

**Script:** `src/indexing/postprocessing/imrad_titles.py`  
**SLURM script:** `src/indexing/scripts/run_postprocessing_imrad_capella.sh`

The IMRaD framework (Introduction, Methods, Results, Discussion) is the standard organizational structure for empirical scientific papers. The post-processing pipeline labels each indexed chunk with its IMRaD category by classifying the `section_title` payload field.

**Classification strategy:**

The classifier uses a two-tier approach:

1. **Heuristic pattern matching** (fast, zero-cost): Section titles are matched against curated regular expression lists for each IMRaD category. Examples:

   | Category | Example patterns |
   |---|---|
   | Introduction | `^intro(duction)?$`, `^background$`, `^related work$`, `^motivation$`, `^literature review$` |
   | Methods | `^methods?$`, `^materials? and methods?$`, `^methodology$`, `^proposed (method\|model\|framework)$` |
   | Results | `^results?$`, `^experiments?$`, `^evaluation$`, `^performance$`, `^findings$` |
   | Discussion | `^discussion$`, `^conclusion$`, `^limitations$`, `^future work$` |
   | Skip | `^acknowledg`, `^references$`, `^bibliography$` |

2. **Neural classifier fallback** (for ambiguous titles): A fine-tuned sequence classification model (`AutoModelForSequenceClassification`) is applied to section titles that do not match any heuristic pattern, using GPU inference in batches.

The labeling runs as a Qdrant scroll-and-update job: it scrolls the entire collection in pages of 4,096 points, classifies each unique `section_title`, and writes the `imrad_label` field back to the payload using batch point updates.

### 12.2 Citation ID Resolution

**Script:** `src/indexing/postprocessing/citation_ids.py`  
**SLURM script:** `src/indexing/scripts/run_postprocessing_ids_capella.sh`

For chunks where bibliography entries lack DOIs or arXiv IDs, the citation resolution post-processor attempts to resolve them via external APIs:

- **OpenAlex API** — queried by raw citation string or partial metadata
- **arXiv API** — queried by title fuzzy matching

Resolved IDs are written back to the `cite_spans` payload field of the affected chunks.

### 12.3 IMRaD Coverage Statistics

**Script:** `src/indexing/postprocessing/imrad_stats.py`

After labeling, a statistics collection job scrolls the entire Qdrant collection (page size: 4,096 points) and accumulates coverage metrics. Final statistics from the production run:

| Metric | Value |
|---|---|
| Total chunks scrolled | 69,026,381 |
| Chunks with IMRaD label | ~64,799,685 (**94.55%**) |
| Unlabeled chunks | ~935,681 (1.36%) |
| Skipped sections (ack., refs.) | ~2,805,995 (4.06%) |

**IMRaD label distribution (labeled chunks):**

| Label | Count | Share of labeled |
|---|---|---|
| Results | 30,332,989 | 43.9% |
| Introduction | 13,991,399 | 20.3% |
| Methods | 13,725,554 | 19.9% |
| Discussion | 7,230,836 | 10.5% |
| Other | ~3,518,907 | 5.1% |

The dominance of Results sections reflects the nature of arXiv papers, which tend to be results-heavy experimental reports. The near-complete labeling coverage (94.55%) demonstrates the effectiveness of the heuristic-first approach: the vast majority of section titles follow standard naming conventions that match the curated regex patterns.

---

## 13. Engineering Challenges and Lessons Learned

### 13.1 Scale

Running a pipeline at 69 million vector insertions is qualitatively different from small-scale experiments:
- **HNSW index build time** dominates toward the end of ingestion as the index grows. The `indexing_threshold=10,000` setting defers HNSW construction until enough vectors accumulate, amortizing the cost.
- **Qdrant WAL (Write-Ahead Log)** pressure required careful tuning of flush intervals and upsert throttling to prevent log accumulation from exceeding available disk space.
- **Memory-mapped storage** (`vectors_on_disk=True`) was essential: keeping 69M × 1024-dim float16 vectors in RAM would require ~140 GB; mmap reduces the resident set to only the hot working set.

### 13.2 Fault Tolerance Engineering

Over the ~5-week development period (April–May 2026), the pipeline was interrupted multiple times due to:
- SLURM preemptions (higher-priority jobs)
- Qdrant storage corruption after ungraceful node failures
- WAL overflow when ingestion was too aggressive without throttling

Each incident drove a new checkpoint or recovery mechanism: the `ingested_shards.jsonl` progress log, the `WIPE_STORAGE` recovery mode, the `.done` sentinel pattern, and the periodic snapshot system.

### 13.3 HPC Deployment Without Docker

Deploying Qdrant (a containerized Rust service) on an HPC cluster without root access required using Singularity. Building the `.sif` image from `docker://qdrant/qdrant` added complexity (cache management, `SINGULARITY_TMPDIR` configuration) but enabled running a production vector database in a fully reproducible, self-contained environment on the HPC compute nodes.

