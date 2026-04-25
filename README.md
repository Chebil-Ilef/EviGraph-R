# EviGraph-R

Multi-Agent Evidence Graph Reasoning for Scientific Question Answering

**Live Architecture Visualizations:** [https://chebil-ilef.github.io/evigraph-R-diags/](https://chebil-ilef.github.io/evigraph-R-diags/)

# QUICK START

**1. Install uv (Python package manager):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Install project dependencies:**
```bash
uv sync
```

**3. Run any script:**
```bash
uv run path/to/script.py
```


# INDEXING PIPELINE (on HPC cluster)

## Setup (one-time)

**1. Build Qdrant Singularity image:**
```bash
singularity build $(pwd)/qdrant.sif docker://qdrant/qdrant
```

**2. Create storage directories:**
```bash
mkdir -p $(pwd)/data/{qdrant_storage,qdrant_snapshots}
```

## Running the indexing pipeline

**Small test run (3,000 papers, 3 array tasks):**
```bash
sbatch --array=0-2 --export=ALL,TOTAL_TASKS=3,SAMPLE_SIZE=3000 \
  scripts/run_indexing_array_capella.sh
```

**Full production run (2.3M papers, 23 array tasks):**
```bash
sbatch --array=0-22 --export=ALL,TOTAL_TASKS=23 \
  scripts/run_indexing_array_capella.sh
```

**Re-ingest only (shards already on disk, single task):**
```bash
sbatch --array=0-0 --export=ALL,TOTAL_TASKS=1,INGEST_ONLY=1 \
  scripts/run_indexing_array_capella.sh
```

The script handles chunk→ingest→snapshot phases automatically. Task 0 coordinates and waits for all other tasks to complete the chunking phase before running ingestion.


# QUERYING QDRANT (after indexing completes)

## On HPC cluster

**1. Request an interactive job:**
```bash
srun --partition=capella --nodes=1 --pty --time=2:00:00 --mem=8G --gres=gpu:1 bash
```

**2. Set environment and start Qdrant instance:**
```bash
export SINGULARITY_CACHEDIR=/tmp/singularity_cache
export SINGULARITY_TMPDIR=/tmp/singularity_tmp
export QDRANT_SIF_PATH=$(pwd)/qdrant.sif

singularity instance start \
  --bind $(pwd)/storage:/qdrant/storage \
  --bind $(pwd)/snapshots:/qdrant/snapshots \
  $QDRANT_SIF_PATH evigraph-qdrant

singularity exec instance://evigraph-qdrant /qdrant/qdrant &
sleep 2
```

**3. Verify collection is loaded:**
```bash
curl -s http://localhost:6333/collections/unarxive_chunks | jq '.result | {points: .points_count, indexed_vectors: .indexed_vectors_count}'
```

You can also check the dashboard at:

```bash
http://localhost:6333/dashboard
```

If you are connecting from another machine, create an SSH tunnel first:

```bash
ssh -J username@login username@node.cluster -L 6333:localhost:6333
```

Then open the same dashboard URL locally in your browser:

```bash
http://localhost:6333/dashboard
```

**4. When done, stop the instance:**
```bash
singularity instance stop evigraph-qdrant
```

## Locally (with Docker)

**First, copy indexed data from HPC:**
```bash
rsync -av /data/cat/ws/ilch217i-qdrant-indexing/qdrant_storage/ ./qdrant_local_storage/
```

**Then run the container:**
```bash
docker run -d --name qdrant-local -p 6333:6333 \
  -v $(pwd)/qdrant_local_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

**Verify:**
```bash
curl -s http://localhost:6333/collections/unarxive_chunks | jq '.result | {points: .points_count, indexed_vectors: .indexed_vectors_count}'
```

**Cleanup:**
```bash
docker stop qdrant-local && docker rm qdrant-local
```

# DEVELOPMENT & ARCHITECTURE

## Project Structure

The EviGraph system orchestrates multi-agent workflows for evidence graph construction from scientific papers. Key components:

- **Agents** → Specialized task handlers (decomposer, retriever, ranker, graph builder)
- **Retriever** → Multi-modal hybrid search with dense + sparse embeddings (BGE-M3)
- **Workflow** → LangGraph state machine coordinating agent interactions
- **Schemas** → Pydantic models and type-safe interfaces
- **Indexing** → Pipeline for chunking, embedding, and Qdrant ingestion

## Where Things Go

| Component | Location |
|-----------|----------|
| LLM Prompts | [src/config/prompts.py](src/config/prompts.py) |
| Configuration | [src/config/settings.py](src/config/settings.py) |
| Data Models | [src/schemas/objects.py](src/schemas/objects.py) |
| Workflow State | [src/schemas/state.py](src/schemas/state.py) |
| Type Contracts | [src/schemas/interfaces.py](src/schemas/interfaces.py) |
| Agent Logic | [src/agents/](src/agents/) |
| Workflow Nodes | [src/workflow/nodes.py](src/workflow/nodes.py) |
| Graph Orchestration | [src/workflow/graph.py](src/workflow/graph.py) |

## Quality Checks

**Before committing:**
```bash
uv run pytest tests/              # Run test suite
uv run mypy src/                   # Type checking
```

### Agent 1 — Decomposer Pipeline
1. **Decompose** query → 1-5 focused sub-queries (single answerable aspects)
2. **Map** each sub-query → relevant IMRaD sections (Abstract, Methods, Results, etc.)
3. **Allocate** retrieval budget → weights sum to 1.0 (higher = more important)

Output: `list[SubQuery]` with `text`, `sections`, `budget_weight`


### HybridQueryRetriever
Handles multi-modal retrieval from Qdrant with support for two retrieval modes:

**Mode A — Dense + BM25** (standard embedding models: e5, jina, qwen, etc.)
- Combines dense embeddings with keyword-based BM25 search
- Uses Reciprocal Rank Fusion (RRF) to fuse results from both modalities
- Falls back to dense-only if BM25 sparse vectors unavailable

**Mode B — Dense + Sparse Embeddings** (BGE-M3 only)
- Leverages both dense and sparse embeddings produced by BGE-M3 model
- Parallel prefetch queries with RRF fusion for improved retrieval
- Higher precision for specialized domain queries

**Key Features:**
- **Section Filtering**: Optional `target_sections` filter to limit retrieval to specific IMRaD sections
- **Cross-Encoder Reranking**: Optional pass through cross-encoder (ms-marco-MiniLM-L-6-v2) for final ranking
- **Configurable Top-K**: Fetch additional candidates when reranking enabled before final cutoff
- **Model Flexibility**: Supports multiple embedding models via config; auto-detects retrieval strategy

**Output:** `ChunkResult` with `chunk_uid`, `paper_id`, `score`, `embed_text`, `section_title`, `chunk_type`, `chunk_index`, `total_chunks`, `cite_spans`

### Embedder
Model-agnostic wrapper supporting both SentenceTransformer and BGE-M3:
- Batch processing with configurable batch sizes
- L2 normalization for dense vectors
- Returns `BGEOutput` namedtuple for BGE-M3 (dense + sparse embeddings) 


### Agent 2 — Retriever & Ranker
**Key Features:**
- Multi-modal hybrid search (dense + sparse embeddings via BGE-M3)
- Optional section filtering (target IMRaD sections)
- Cross-encoder reranking for precision

**Input:** `list[SubQuery]` with text, sections, budget_weight  
**Output:** `list[ChunkResult]` with chunk_uid, paper_id, score, section_title, embed_text

### Agent 3 — Judge
**Key Features:**
- Multi-verifier routing: NPM (semantic) → NLI (entailment) → LLM judge (hard cases)
- Route decision based on claim type (atomic vs. inferential) and hop depth
- Full verdict metadata: verifier_used, evidence_trail, error_stage

**Input:** EvidenceGraph + query  
**Output:** `JudgementResult` with filtered_documents[], judged_relations[], verdict_details{}

### Agent 4 — Answer Generator
**Key Features:**
- Generates coherent multi-sentence answers from verified claims only
- Per-sentence citations with full metadata (chunk_id, score, verdict)
- Conflict detection and inline flagging

**Input:** EvidenceGraph, query, verified claims  
**Output:** `FinalAnswer` with sentences[], citations[], reasoning_summary

# Serving the API 

PYTHONPATH=src uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload 