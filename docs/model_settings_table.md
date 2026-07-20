# Model & Settings Table

## 1. Pipeline LLM agents

All four pipeline LLM calls go through `evigraph.utils.llm.LLMClient`, a DSPy/LiteLLM wrapper
calling the group's **ScaDS.AI** OpenAI-compatible endpoint (`LLM_API_BASE`). Source of truth:
[`LLMConfig` / `AGENT_MODELS`](../src/evigraph/config/settings.py).

| Component | Model | Provider | Temperature | Decoding | Timeout | Max retries | Max tokens |
|---|---|---|---|---|---|---|---|
| Decomposer (Agent 1) | `meta-llama/Llama-3.1-8B-Instruct`¹ | ScaDS.AI (remote) | 0.0 | greedy | 60s | 2 | 4096 |
| Evidence Graph Builder (Agent 2, claim extraction + hop reasoning) | `meta-llama/Llama-3.3-70B-Instruct` | ScaDS.AI (remote) | 0.0 | greedy | 90s | 2 | 4096 |
| Judge — LLM tier (Agent 3, escalation + direct cross-paper) | `meta-llama/Llama-3.1-8B-Instruct`¹ | ScaDS.AI (remote) | 0.0 | greedy | 60s | 2 | 4096 |
| Answer Generator (Agent 4) | `meta-llama/Llama-3.3-70B-Instruct` | ScaDS.AI (remote) | 0.0 | greedy | 120s | 2 | 4096 |


**Decoding/reproducibility caveat:** temperature=0.0 gives greedy decoding, but no `seed`
parameter is passed anywhere in [`LLMClient`/`_get_lm`](../src/evigraph/utils/llm.py). Greedy
decoding is *near*-deterministic but not guaranteed bit-identical across runs on a shared,
batched serving backend we don't control (ScaDS.AI). Do not claim exact reproducibility of
individual LLM outputs — only that decoding is deterministic-by-configuration (temperature 0),
not verified deterministic-by-output.

**Prompts:** [`src/evigraph/config/prompts.py`](../src/evigraph/config/prompts.py).

## 2. Judge — NLI tier

Three-route verifier ([`src/evigraph/agents/judge.py`](../src/evigraph/agents/judge.py)): NLI batch → escalate to LLM if neutral
→ direct LLM for cross-paper contradictions. The NLI tier runs **locally on CPU**, not through
ScaDS.AI.

| Parameter | Value |
|---|---|
| Model | `sileod/deberta-v3-small-tasksource-nli` |
| Support threshold | 0.65 |
| Contradiction threshold | 0.70 |

## 3. Embedder

| Parameter | Value |
|---|---|
| Model | `BAAI/bge-m3` (dense + sparse, single forward pass) |
| Dense dim | 1,024 |
| Sparse | Yes, SPLADE-style lexical weights, native to BGE-M3 (not BM25 — see T3 audit) |
| Max sequence length | 8,192 tokens |
| Batch size | 512 (indexing) / 1 (query time) |
| Dtype | float16 |
| Normalization | L2 |

## 4. Reranker and citation classifier

| Component | Model |
|---|---|
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Citation-type classifier (SciCite: METHOD/BACKGROUND/RESULT_COMPARISON) | `lostelf/scibert_scivocab_uncased_scicite_finetuned` |
| IMRaD section classifier (indexing-time only, not query-time) | `lostelf/section-classifier-imrad` (DistilBERT-based, fine-tuned on `saier/unarXive_imrad_clf`; F1 0.776, accuracy 77.1%) |


## 5. Evaluation-only model (not part of the pipeline)

| Component | Model | Provider | Temperature | Decoding | Seed | Used for |
|---|---|---|---|---|---|---|
| DeepEval / G-Eval judge | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | ScaDS.AI (remote) | 0.0 | greedy | none | Scores Answer Relevancy, Contextual Relevancy, Faithfulness, Claim Coverage, Attribution Faithfulness for all 13 configurations in Table V.9 |


## 6. Hardware and compute budget

Two entirely separate compute environments were used for two separate jobs. 

### 6.1 Indexing (2.28M papers → 69M chunks) — Capella, ZIH TU Dresden

| Spec | Value |
|---|---|
| Cluster | Capella (ZIH, TU Dresden) — 3rd-ranked German supercomputer, 51st worldwide (TOP500), 5th on Green500 |
| GPUs per node | 4× NVIDIA H100, 94 GB HBM2e each |
| CPUs per node | 2× AMD Genoa, 32 cores each |
| RAM per node | 768 GB DDR5 |
| Job shape | 200 parallel SLURM array tasks, **1 GPU + 8 CPU cores + 170 GB RAM per task**, 24h wall-time limit per task |
| Work per task | ~11 batches (~11,000 papers) |
| Output | 69,026,381 chunks from 2,284,380 papers |
| Qdrant ingestion (overlaps embedding, not GPU-bound) | ~1.5 weeks wall-clock, 2,215 shards |
| Citation resolution (CPU/API-bound, not GPU) | ~43 hours, 40M citation references, OpenAlex + arXiv APIs |
| Section classifier training/inference | GPU batch inference, same Capella allocation |

**GPU-hours:** 200 tasks × 24h wall-time limit × 1 GPU = **4,800 GPU-hours upper bound**.
Actual usage was less than this — tasks finished well under the 24h cap — but we have not
calculated/estimated the real figure. A real number would need Capella's SLURM accounting
(`sacct`) for the actual job IDs, which we have not pulled.

### 6.2 Evaluation / ablation study (268-question benchmark, all 13 configurations)

| Spec | Value |
|---|---|
| Cluster/partition | Barnard (`--partition=barnard`), **CPU-only** |
| GPUs | **0** — script header explicitly states "CPU-only, no GPU" |
| CPUs | 14 cores |
| RAM | 100 GB |
| Wall-time limit | 7 hours per SLURM job |

**Local GPU-hours for the eval/ablation study: 0.** All local computation (BGE-M3 embedding
at query time, cross-encoder reranking, NLI verification) ran on CPU. The only GPU compute
involved was on the remote **ScaDS.AI** endpoint serving the 4 pipeline LLMs and the DeepEval
judge — that hardware is provider-managed, not ours to inventory, and should be reported as
such rather than estimated.
