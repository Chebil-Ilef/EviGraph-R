# Examples

Example scripts demonstrating EviGraph-R functionality.

## `pipeline_demo.py`

Interactive demo of the full EviGraph-R pipeline:
- Query decomposition
- Hybrid retrieval (dense + sparse)
- Evidence graph construction
- Visualization output

**Usage:**

From compute node:
```bash
uv run python examples/pipeline_demo.py
```

From login node (auto-requests compute node):
```bash
./scripts/run_demo_pipeline_interactive.sh 
```

**Arguments:**
```bash
--query "Your question"          # Custom query
--model-key bge-m3               # Embedding model (default: bge-m3)
--top-k 10                       # Number of chunks to retrieve
--no-graph-output                # Skip visualization generation
```

**Example:**
```bash
uv run python examples/pipeline_demo.py \
  --query "How does contrastive learning work?" \
  --top-k 15
```

**Output:**
- Console: Detailed step-by-step pipeline execution
- File: Interactive HTML graph visualization in `/tmp/evigraph_dev/`
