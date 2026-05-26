# Full 280-Golden Evaluation Guide

> All commands run from: `/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R`

---

## Step 1 — Generate 280 goldens

| | |
|---|---|
| **Objective** | Sample context groups from Qdrant + synthesize Q&A pairs |
| **Input** | Qdrant index (running via Singularity) |
| **Output** | `evaluation/_data/benchmark/goldens.jsonl` (~280 lines) |
| **Est. time** | 3–5h |

```bash
mkdir -p evaluation/_data/benchmark/results evaluation/_data/benchmark/eval logs

CAT1_TARGET=70 CAT2_TARGET=70 CAT3_TARGET=70 CAT4_TARGET=70 \
GOLDENS_PATH=evaluation/_data/benchmark/goldens.jsonl \
GROUPS_DIR=evaluation/_data/benchmark/groups \
RESULTS_DIR=evaluation/_data/benchmark/results \
EVAL_DIR=evaluation/_data/benchmark/eval \
sbatch --time=15:00:00 evaluation/run_full_evaluation_capella.sh --generate-only
```

> **Wait for this job to finish before starting Step 2.**

---

## Step 2A — Run EviGraph-R ablation variants

| | |
|---|---|
| **Objective** | Run all 11 variants (`full`, `A1.1`…`J3`). `full` fills the cache; `G1/G2/J1/J2/J3` reuse it (skip Qdrant) |
| **Input** | `goldens.jsonl` |
| **Output** | `results/full.jsonl`, `results/A1_1.jsonl`, … (11 files) |
| **Est. time** | 18–22h |

```bash
GOLDENS_PATH=evaluation/_data/benchmark/goldens.jsonl \
RESULTS_DIR=evaluation/_data/benchmark/results \
EVAL_DIR=evaluation/_data/benchmark/eval \
sbatch --time=23:00:00 evaluation/run_full_evaluation_capella.sh --ablation-only
```

---

## Step 2B.1 — Run SQuAI baseline *(submit alongside 2A)*

| | |
|---|---|
| **Objective** | Run `squai` — uses its own FAISS index, no Qdrant needed |
| **Input** | `goldens.jsonl` |
| **Output** | `results/squai.jsonl` |
| **Est. time** | 4–6h |

> Safe to run in parallel with 2A — SQuAI never touches Qdrant.

```bash
GOLDENS_PATH=evaluation/_data/benchmark/goldens.jsonl \
RESULTS_DIR=evaluation/_data/benchmark/results \
EVAL_DIR=evaluation/_data/benchmark/eval \
BASELINES="squai" \
sbatch --time=08:00:00 --mem=36G evaluation/run_full_evaluation_capella.sh --baselines-only
```

---

## Step 2B.2 — Run standard RAG baseline *(after 2A finishes)*

| | |
|---|---|
| **Objective** | Run `standard_rag` — requires Qdrant (100 GB RAM) |
| **Input** | `goldens.jsonl` |
| **Output** | `results/standard_rag.jsonl` |
| **Est. time** | 3–5h |

> **Only submit after 2A is done** — cannot share a node with another Qdrant job (170 GB limit).

```bash
GOLDENS_PATH=evaluation/_data/benchmark/goldens.jsonl \
RESULTS_DIR=evaluation/_data/benchmark/results \
EVAL_DIR=evaluation/_data/benchmark/eval \
BASELINES="standard_rag" \
sbatch --time=06:00:00 evaluation/run_full_evaluation_capella.sh --baselines-only
```

---

## Step 3 — Score + build final table

| | |
|---|---|
| **Objective** | DeepEval scoring of all 13 result files → aggregation table |
| **Input** | `results/*.jsonl` (13 files) |
| **Output** | `eval/ablation_table.md`, `eval/metric_summary.csv` |
| **Est. time** | 4–6h |

Runs automatically at the end of Steps 2A and 2B. If a job dies before scoring finishes, re-run manually:

```bash
uv run python -m evaluation.full_evaluation \
  --results_dir evaluation/_data/benchmark/results \
  --output_dir  evaluation/_data/benchmark/eval \
  --model meta-llama/Llama-3.3-70B-Instruct
```

---

## Resume safety — just resubmit the same command

Every step skips already-completed work:

| Step | Skip condition |
|------|---------------|
| 1 | `goldens.jsonl` already has ≥ N lines |
| 2A / 2B.1 / 2B.2 | result file already has ≥ N lines |
| 3 | `agg_*.json` + `scores_*.jsonl` already complete |

**Zero progress wasted.**

---

## Quick status check

```bash
wc -l evaluation/_data/benchmark/goldens.jsonl
wc -l evaluation/_data/benchmark/results/*.jsonl 2>/dev/null
ls  evaluation/_data/benchmark/eval/
```
