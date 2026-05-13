# Index Comparison Experiment Strategy

## Objective

Compare retrieval quality between:
- **SQuAI (baseline):** FAISS Flat index with E5-large-v2 embeddings + BM25, fused via linear alpha interpolation.
- **EviGraph-R (ours):** Qdrant collection with BGE-M3 dense+sparse embeddings, fused via Reciprocal Rank Fusion (RRF).

Both systems index the same unarXive corpus (~2.28M papers). The goal is to produce concrete retrieval metrics for the paper.

---

## Why Clustered Sampling

With 2M+ papers, both systems can easily surface papers that are *more* relevant than the designated gold paper if questions are too generic. A purely random evaluation would therefore be misleading — both systems could achieve near-zero Hit@k not because they fail, but because they retrieve legitimately relevant papers that happen not to be the gold one.

**Solution:** Sample papers in topically coherent clusters. Within each cluster, all papers are plausible hard negatives for each other. Questions are generated from specific sections (not just abstracts) to ensure they are answerable only from the gold paper.

---

## Sampling Design

### Broad domains (3)
CS · Physics · Math

### Sub-clusters per domain (3 each = 9 clusters total)

| Domain  | Cluster A                        | Cluster B                          | Cluster C              |
|---------|----------------------------------|------------------------------------|------------------------|
| CS      | LLM & prompt engineering         | Databases & information retrieval  | AI for healthcare      |
| Physics | Quantum computing                | Astrophysics / cosmology           | Condensed matter       |
| Math    | Number theory                    | Differential geometry              | Combinatorics          |

### Scale
- 5 papers per cluster → **45 papers total**
- **3 questions per paper** (1 section-specific + 1 full-paper + 1 abstract) → **~135 questions total**

Papers are selected by keyword/category filtering from `corpus.jsonl`, not uniformly random, to guarantee hard negatives within clusters.

---

## Question Design

Each paper produces three question types:

| Type | Source | Purpose |
|------|--------|---------|
| `section` | Best body section (methods/results) from Qdrant | Primary eval signal — requires full text retrieval |
| `fullpaper` | Same body section, different angle | Tests contribution-level retrieval |
| `abstract` | Abstract text | Control — SQuAI's home turf; EviGraph-R should still win |

Section and fullpaper questions are generated from body text fetched from Qdrant (`chunk_type=subsection`), not from the abstract, so they can only be answered by retrieving the full paper.

---

## Retrieval Systems

### SQuAI (baseline)
- **Embedding:** `intfloat/e5-large-v2` (1024-dim), query prefix `"query: "`
- **Dense index:** FAISS Flat (exact search)
- **Sparse index:** BM25 (Lucene variant, k1=1.5, b=0.75, δ=0.5), CSC sparse matrix
- **Fusion:** Linear interpolation — `score = α × dense_score + (1−α) × bm25_score`, α configurable (default 0.5)
- **Index path:** `/data/horse/ws/s3811141-faiss/inbe405h-unarxive/`

### EviGraph-R (ours)
- **Embedding:** `BAAI/bge-m3` (1024-dim), produces both dense and sparse weights in a single forward pass
- **Dense+sparse index:** Qdrant collection `unarxive_chunks`
- **Fusion:** Reciprocal Rank Fusion — `score = Σ 1/(k + rank_i)`, k=60, prefetch top-100
- **Index path:** `/data/cat/ws/ilch217i-indexing-pipeline/EviGraph-R/storage/`

---

## Metrics

### Primary (computed at k = 1, 5, 10)

| Metric    | Definition                                              |
|-----------|---------------------------------------------------------|
| Hit@k     | Gold paper_id found in top-k results                   |
| MRR@k     | Mean Reciprocal Rank = mean(1/rank) if found, else 0   |
| NDCG@k    | Normalized DCG with tiered relevance (1.0 / 0.5 / 0.0)|
| MAP@k     | Mean Average Precision across k positions              |

### Supplementary

| Metric           | Definition                                              | System       |
|------------------|---------------------------------------------------------|--------------|
| Precision@k      | # gold papers in top-k / k                             | Both         |
| Avg rank of gold | Mean rank position when gold is found                  | Both         |
| Section Hit@k    | Retrieved chunk is from the correct paper section      | EviGraph-R only (SQuAI has no section granularity) |
| Latency p50 (ms) | Median query wall-clock time                           | Both         |
| Latency p95 (ms) | 95th percentile query wall-clock time                  | Both         |

### Breakdown dimensions
- Per domain (CS / Physics / Math)
- Per question source (section / fullpaper / abstract)

---

## Scripts

| Script                  | Purpose                                         |
|-------------------------|-------------------------------------------------|
| `sample_papers.py`      | Clustered keyword-based sampling from corpus    |
| `generate_questions.py` | LLM question generation (section + full-paper)  |
| `retrieve_squai.py`     | SQuAI FAISS + BM25 + alpha-fusion retriever     |
| `retrieve_evigraph.py`  | EviGraph-R Qdrant BGE-M3 retriever              |
| `run_comparison.py`     | Orchestrator: retrieve, compute metrics, report |

## Outputs (`results/`)
- `sampled_papers.jsonl` — 45 selected papers with cluster labels
- `questions.jsonl` — single-paper questions with `type`, `source`, `gold_paper_ids`, `gold_section`
- `raw_results.json` — per-question retrieval results and metrics for both systems
- `comparison_table.md` — final metrics tables ready for the paper

## Execution Steps

1) Sample 45 papers across 9 clusters (~3 min):
```
uv run python -m experiments.index_comparison.sample_papers
```

2) Generate questions via Llama-3.3-70B (~15 min):
```
sbatch experiments/index_comparison/generate_questions_capella.sh
```

3) Run retrieval comparison on both systems and compute all metrics:
```
sbatch experiments/index_comparison/run_comparison_capella.sh
```

4) Read the results:
```
cat experiments/index_comparison/results/comparison_table.md
```
