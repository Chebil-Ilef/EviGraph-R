# EviGraph-R

EviGraph-R answers scientific questions by building an **evidence graph** over 2.8 million arXiv papers. It retrieves relevant chunks, constructs a graph of claims and citations, verifies each claim with an NLI judge, and synthesises a grounded answer with per-sentence citations.

---

## Getting started

There are two ways to use EviGraph-R:

| | **pip install** | **Clone & run** |
|---|---|---|
| **Use case** | Use the Python API or CLI in your own project | Self-host the full stack with your own data |
| **Qdrant** | Connects to the hosted VM (no download needed) | Runs locally via Docker |
| **Setup** | `pip install evigraph-r` | `git clone` + `docker compose up` |

---

## Quick start (pip)

```bash
pip install evigraph-r
```

```python
import asyncio
import evigraph

runner = evigraph.WorkflowRunner()
request = evigraph.QueryRequest(query="What is the effect of BERT pre-training on downstream NLP tasks?")
result = asyncio.run(runner.run_query(request))

print(result.answer)
# → "BERT pre-training improves GLUE score by 7.7% [arxiv:1810.04805] ..."
```

Or from the terminal:

```bash
evigraph query "What causes Alzheimer's disease?"
evigraph query "What causes Alzheimer's disease?" --json   # full JSON response
evigraph serve                                             # start FastAPI on :8000
```

### Requirements

- Python ≥ 3.11
- An OpenAI-compatible LLM endpoint (set `LLM_BASE_URL` and `LLM_API_KEY` in env)

### Configuration

The pip package connects to the **hosted Qdrant instance** on the EviGraph VM by default — no local database or data download needed.

To override (e.g. point to your own Qdrant):

```bash
evigraph query "..." --qdrant-url http://your-host:6333
# or
export QDRANT_URL=http://your-host:6333
```

---

## Self-hosted (Docker)

To run everything locally with your own data:

```bash
git clone <repo-url>
cd EviGraph-R
cp .env.example .env   # fill in LLM_BASE_URL, LLM_API_KEY, etc.
docker compose up -d
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Qdrant dashboard | http://localhost:6334/dashboard |

---

## Python API

```python
import asyncio
import evigraph

# Basic query
runner = evigraph.WorkflowRunner()
request = evigraph.QueryRequest(
    query="Does dropout improve generalisation in transformers?",
    config=evigraph.PipelineConfig(
        top_k=20,
        enable_hop=True,
        target_sections=["Results", "Discussion"],
    ),
)
result = asyncio.run(runner.run_query(request))

# Answer with citations
print(result.answer)

# Per-sentence breakdown
for sentence in result.sentences:
    print(sentence.text, "→", sentence.citations)

# Claim verdict scorecard
print(result.scorecard)
# → {"Supported": 12, "Contradicted": 2, "Inconclusive": 3}

# Evidence graph (nodes + edges)
graph: evigraph.EvidenceGraph = result.graph
```

### `PipelineConfig` options

| Parameter | Type | Default | Description |
|---|---|---|---|
| `top_k` | int | `15` | Chunks retrieved per sub-query |
| `score_threshold` | float | `0.0` | Minimum retrieval score |
| `enable_hop` | bool | `true` | Multi-hop sub-question retrieval |
| `embedding_model` | str | `"bge-m3"` | `bge-m3`, `e5`, `qwen3`, `jina` |
| `target_sections` | list\|None | `None` | Restrict to IMRaD sections e.g. `["Methods", "Results"]` |

---

## REST API

When running via Docker or `evigraph serve`:

### Health check
```
GET /health
```

### Submit a query
```
POST /api/v1/query
Content-Type: application/json

{
  "query": "What is the effect of BERT pre-training on downstream NLP tasks?",
  "config": { "top_k": 15, "enable_hop": true }
}
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "answer": "Based on the evidence...",
  "sentences": [{ "text": "...", "citations": ["arxiv:1810.04805"] }],
  "graph": { "nodes": [...], "edges": [...] },
  "scorecard": { "Supported": 12, "Contradicted": 2, "Inconclusive": 3 },
  "elapsed_s": 14.3
}
```

### Streaming (SSE)
```
GET /api/v1/query/stream?q=What+causes+Alzheimers
```

Emits events: `decomposed` → `retrieved` → `graph_built` → `judged` → `completed`

---

## How it works

A query passes through a 5-agent LangGraph pipeline:

```
Query
  │
  ▼
[1] Decomposer
    Breaks the query into focused sub-queries, each tagged with
    IMRaD section targets and a retrieval budget weight.
  │
  ▼
[2] Hybrid Retriever
    BGE-M3 dense + BM25 sparse retrieval, cross-encoder reranking,
    IMRaD section-aware score boosting.
  │
  ▼
[3] Evidence Graph Builder
    Builds a graph of PAPER → CHUNK → CLAIM → CONCEPT nodes.
    Expands citations via SciCite (METHOD / BACKGROUND / RESULT_COMPARISON).
  │
  ▼
[4] Judge
    Three-route verifier: NLI batch → escalate to LLM if neutral
    → direct LLM for cross-paper contradictions.
    Verdict per claim: Supported / Contradicted / Inconclusive.
  │
  ▼
[5] Answer Generator
    Synthesises answer from supported claims only.
    Each sentence carries per-chunk citations and a verdict tag.
```

**Stack:**

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Workflow | LangGraph |
| LLM | DSPy (OpenAI-compatible) |
| Vector DB | Qdrant |
| Embeddings | BGE-M3 |
| NLI | DeBERTa-v3-small-tasksource |

---

## Development

```bash
git clone <repo-url>
cd EviGraph-R
uv sync
cp .env.example .env

# Run tests
uv run pytest -m "not slow and not integration and not hpc"

# Type check
uv run mypy src/
```

---

## License

MIT
