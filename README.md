# EviGraph-R

EviGraph-R answers scientific questions by retrieving evidence from a large corpus of academic papers (unarXive 2024), building an evidence graph, verifying claims with NLI, and generating a grounded answer with citations.

---

## Architecture Overview

A query goes through a 4-stage multi-agent pipeline orchestrated by LangGraph:

```
User Query
    │
    ▼
[Agent 1 — Decomposer]
Breaks the query into focused sub-questions
    │
    ▼
[Hybrid Retriever]
Dense (BGE-M3) 
+ Sparse (BM25) search over Qdrant
    │
    ▼
[Agent 2 — Evidence Graph Builder]
Extracts claims from retrieved chunks, builds a knowledge graph
    │
    ▼
[Agent 3 — Judge]
Verifies each claim with DeBERTa NLI → Supported / Contradicted / Not-Supported / Inconclusive
    │
    ▼
[Agent 4 — Answer Generator]
Synthesises a final grounded answer with inline citations
    │
    ▼
JSON response  +  optional SSE stream
```

**Key technologies:**

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Workflow orchestration | LangGraph |
| LLM integration | DSPy (OpenAI-compatible) |
| Vector database | Qdrant |
| Embeddings | BGE-M3 |
| NLI verification | DeBERTa-v3-small-tasksource |
| Package manager | uv (Astral) |

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.11 |
| Docker + Docker Compose | any recent version |
| [uv](https://docs.astral.sh/uv/) | latest |
| An OpenAI-compatible LLM endpoint | — |

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd EviGraph-R

# 2. Install all Python dependencies (creates .venv automatically)
uv sync

# 3. Copy the environment template
cp .env.example .env
```

---

## Running the API

### Option A — Docker Compose (recommended)

This starts Qdrant and the API server together.

```bash
docker compose up -d
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Qdrant dashboard | http://localhost:6334/dashboard |

Follow logs:

```bash
docker compose logs -f api
```

Stop everything:

```bash
docker compose down
```

> **Model cache:** The `hub/` directory is volume-mounted into the container. Models are downloaded once and reused on every restart — no re-downloads on rebuild.

> **Live reload:** `src/` is also volume-mounted, so code changes take effect immediately without rebuilding the image.

---

### Option B — Local Development

**Step 1 — Start Qdrant**

```bash
docker run -d --name qdrant \
  -p 6334:6333 \
  -v "$(pwd)/storage:/qdrant/storage" \
  qdrant/qdrant:v1.13.6
```

**Step 2 — Set environment**

```bash
export PYTHONPATH=src
export $(grep -v '^#' .env | xargs)
```

**Step 3 — Start the API server**

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The server is ready when you see:

```
INFO:     Application startup complete.
```

---

## API Reference

Base URL: `http://localhost:8000`

### Health check

```
GET /health
```

Returns Qdrant status and collection metadata. Use this to confirm the server and database are reachable before sending queries.

---

### System configuration

```
GET /api/v1/config
```

Returns all active settings: embedding model, LLM models, retrieval parameters, NLI thresholds, etc.

---

### Submit a query (blocking)

```
POST /api/v1/query
Content-Type: application/json
```

**Request body:**

```json
{
  "query": "What is the effect of BERT pre-training on downstream NLP tasks?",
  "config": {
    "top_k": 15,
    "score_threshold": 1.0,
    "enable_hop": true,
    "embedding_model": "bge-m3",
    "target_sections": null
  }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | **required** | The scientific question to answer |
| `config.top_k` | int | `15` | Maximum number of chunks to retrieve |
| `config.score_threshold` | float | `1.0` | Minimum retrieval score (lower = stricter) |
| `config.enable_hop` | bool | `true` | Enable multi-hop sub-question retrieval |
| `config.embedding_model` | string | `"bge-m3"` | Embedding model: `bge-m3`, `e5`, `qwen3`, `jina` |
| `config.target_sections` | list\|null | `null` | Restrict to IMRaD sections (e.g. `["Methods", "Results"]`) |

**Response:**

```json
{
  "job_id": "uuid",
  "status": "completed",
  "query": "...",
  "answer": "Based on the evidence...",
  "sentences": [
    {
      "text": "BERT improves performance on GLUE by 7.7%.",
      "citations": ["arxiv:1810.04805"]
    }
  ],
  "graph": { "nodes": [...], "edges": [...] },
  "scorecard": {
    "Supported": 12,
    "Contradicted": 2,
    "Not-Supported": 1,
    "Inconclusive": 3
  },
  "errors": [],
  "elapsed_s": 14.3
}
```

---

### Submit a query (streaming SSE)

```
GET /api/v1/query/stream?q=<question>&top_k=15&enable_hop=true&embedding_model=bge-m3
```

Returns a stream of Server-Sent Events. Each event marks a completed pipeline stage:

| Event | Payload |
|---|---|
| `decomposed` | List of sub-questions generated |
| `retrieved` | Number of chunks retrieved |
| `graph_built` | Partial evidence graph |
| `judged` | Claim verdicts (Supported / Contradicted / Not-Supported / Inconclusive) |
| `completed` | Full `QueryResponse` JSON |
| `error` | Error message |

**Example (curl):**

```bash
curl -N "http://localhost:8000/api/v1/query/stream?q=What+causes+Alzheimer%27s+disease"
```

---

### Render an evidence graph

```
POST /api/v1/graph/render
Content-Type: application/json
```

Accepts an `EvidenceGraph` JSON object (returned by `/api/v1/query`) and returns a self-contained interactive HTML page using Cytoscape.js.

---

## Project Structure

```
EviGraph-R/
├── src/
│   ├── api/
│   │   ├── main.py                  # FastAPI app, CORS, lifespan
│   │   ├── runner.py                # WorkflowRunner — top-level orchestration
│   │   ├── schemas.py               # Request / response Pydantic models
│   │   └── routes/
│   │       ├── query.py             # POST /api/v1/query, GET /api/v1/query/stream
│   │       ├── graph.py             # POST /api/v1/graph/render
│   │       ├── health.py            # GET /health
│   │       └── config.py            # GET /api/v1/config
│   ├── agents/
│   │   ├── decomposer.py            # Agent 1 — query decomposition
│   │   ├── evidence_graph_builder.py # Agent 2 — graph construction
│   │   ├── judge.py                 # Agent 3 — claim verification
│   │   └── answer_generator.py      # Agent 4 — answer synthesis
│   ├── workflow/
│   │   ├── graph.py                 # LangGraph StateGraph definition
│   │   └── nodes.py                 # Node implementations
│   ├── retrieval/
│   │   ├── retriever.py             # HybridQueryRetriever (dense + sparse)
│   │   └── embedder.py              # Embedding model wrapper
│   ├── indexing/                    # Document indexing pipeline
│   ├── schemas/
│   │   ├── objects.py               # EvidenceGraph, RetrievedDocument, SubQuery…
│   │   ├── state.py                 # WorkflowState (shared pipeline state)
│   │   └── interfaces.py            # Abstract base classes
│   ├── config/
│   │   ├── settings.py              # All configuration dataclasses
│   │   └── prompts.py               # Agent system/user prompts
│   ├── utils/
│   │   ├── llm.py                   # DSPy-backed LLMClient
│   │   ├── qdrant.py                # Qdrant connection helpers
│   │   ├── nli.py                   # DeBERTa NLI wrapper
│   │   ├── scicite.py               # Citation intent classifier
│   │   └── graph.py                 # Graph construction helpers
│   └── visualization/
│       └── cytoscape_renderer.py    # Interactive HTML graph renderer
├── src/indexing/scripts/            # HPC / indexing shell scripts
├── tests/                           # Unit and integration tests
├── experiments/                     # Benchmarks and model evaluations
├── documentation/                   # Technical deep-dives
├── storage/                         # Qdrant local storage (Docker volume)
├── hub/                             # HuggingFace model cache (Docker volume)
├── logs/                            # debug_logs.txt
├── _data/                           # Raw data, chunks, shards, manifests
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

---

## Testing

```bash
# Run all fast unit tests
uv run pytest -m "not slow and not integration and not hpc"

# Include integration tests (requires a running Qdrant instance)
uv run pytest -m "not slow and not hpc"

# Run everything including slow model tests
uv run pytest
```
