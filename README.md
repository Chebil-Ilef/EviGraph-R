

**[https://chebil-ilef.github.io/evigraph-R-diags/](https://chebil-ilef.github.io/evigraph-R-diags/)**



1) install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh

2) uv sync

3) run any script you want with : 
uv run path/to/script.py


# FOR INDEXING PIPELINE

singularity build /home/USERNAME/qdrant.sif docker://qdrant/qdrant


then


mkdir -p /data/cat/ws/ilch217i-qdrant-indexing/qdrant_storage \
         /data/cat/ws/ilch217i-qdrant-indexing/qdrant_snapshots



srun --partition=capella --nodes=1 --gres=gpu:1 --cpus-per-task=8 \
     --mem=64G --time=01:00:00 --pty bash


export SINGULARITY_CACHEDIR=/tmp/singularity_cache
export SINGULARITY_TMPDIR=/tmp/singularity_tmp
export QDRANT_SIF_PATH=$HOME/qdrant.sif

singularity instance start \
  --bind _data/qdrant_storage:/qdrant/storage \
  --bind _data/qdrant_snapshots:/qdrant/snapshots \
  $QDRANT_SIF_PATH evigraph-qdrant

singularity exec instance://evigraph-qdrant /qdrant/qdrant &
sleep 2

curl -s http://localhost:6333/collections/unarxive_chunks | jq '.result.points_count'


SAMPLE_SIZE=10 sbatch scripts/run_indexing_capella.sh


sbatch --array=0-4 --export=ALL,TOTAL_TASKS=5,SAMPLE_SIZE=3000 scripts/run_indexing_array_capella.sh

# FOR RUNNING QDRANT AFTER INDEXING

1) srun -N 1 --pty --time=1:00:00 --mem=8G --gres=gpu:1 bash
2) 
singularity instance start \
  --bind /data/cat/ws/ilch217i-horse/EviGraph-R/_data/qdrant_storage:/qdrant/storage \
  --bind /data/cat/ws/ilch217i-horse/EviGraph-R/_data/qdrant_snapshots:/qdrant/snapshots \
  /data/cat/ws/ilch217i-horse/EviGraph-R/qdrant.sif \
  evigraph-qdrant
3) 
singularity exec instance://evigraph-qdrant /qdrant/qdrant &

4) verify
curl -s http://localhost:6333/collections/unarxive_chunks | python3 -m json.tool

# GUIDE TO UPDATING/ WRITING AGENTS / CODE

## Where Things Go
- **Prompts** → `src/config/prompts.py` (all LLM system/user prompts)
- **Configuration** → `src/config/settings.py` (models, timeouts, paths)
- **Schemas** → `src/schemas/objects.py` (Pydantic models, enums)
- **State** → `src/schemas/state.py` (workflow state definitions)
- **Interfaces** → `src/schemas/interfaces.py` (protocol definitions)
- **Agent Logic** → `src/agents/*.py` (agent implementations)
- **Workflow Nodes** → `src/workflow/nodes.py` (graph node functions)
- **Graph** → `src/workflow/graph.py` (workflow orchestration)

## Agent 1 — Decomposer Pipeline
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