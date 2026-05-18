# Synthetic Benchmark for Embedding Models and Retrieval Modes

## Overview

This benchmark evaluates the performance of modern **embedding models** and **retrieval strategies** for scholarly document retrieval.

The goal is to measure how well different models retrieve relevant evidence from scientific papers under realistic question-answering conditions.

The benchmark focuses on two retrieval settings:

1. **Hybrid Retrieval**  
   Combination of **BM25 lexical search + dense embeddings**.

2. **Both Dense and Sparse Retrieval**  
   Evaluating the best hybrid configuration against **BGE-M3**, which supports both dense and sparse representations and is currently recommended for use with Qdrant.

### Evaluated Embedding Models

Within a practical compute budget, we evaluate:

- `jinaai/jina-embeddings-v5-text-nano-retrieval`
- `Qwen/Qwen3-Embedding-0.6B`
- `intfloat/e5-base-v2`

The top-performing hybrid configuration is then benchmarked against:

- **`bge-m3`** (dense + sparse retrieval model)

---

# Synthetic Dataset Construction

## Dataset Size

The benchmark consists of **300 synthetic questions** generated from a corpus of scientific papers.

The dataset is designed to evaluate retrieval under varying levels of reasoning complexity.

---

# Evidence Scope Distribution

The dataset includes three retrieval regimes.

| Evidence Scope | Description | Count |
|---|---|---|
| Single Paper, Single Chunk | Answer contained within one chunk of one paper | 120 |
| Single Paper, Multi Chunk | Answer requires combining multiple chunks from the same paper | 120 |
| Multi Paper, Multi Chunk | Answer requires combining evidence across multiple papers | 60 |

### Rationale

This distribution reflects realistic scholarly QA patterns:

- Most questions are answerable within a **single document**
- Some questions require **multiple sections of the same paper**
- Harder questions involve **cross-paper reasoning**

This balance provides:

- strong baseline retrieval evaluation
- realistic reasoning complexity
- meaningful stress tests for retrieval models

---

# Question Type Distribution

To ensure semantic diversity, questions are distributed across **eight categories**.

| Question Type | Count |
|---|---|
| Definition / Concept | 36 |
| Method / Procedure | 42 |
| Purpose / Motivation | 30 |
| Dataset / Experimental Setup | 36 |
| Result / Quantitative Finding | 48 |
| Comparison | 42 |
| Limitation / Challenge | 30 |
| Evidence Synthesis / Relation | 36 |

### Why multiple question types?

Different question types stress different retrieval behaviors:

- **Definition** questions test semantic matching
- **Method** questions test structural retrieval
- **Quantitative results** require numeric evidence retrieval
- **Comparison and synthesis** require multi-document reasoning

Balanced distribution prevents models from overfitting to a single query style.

---

# Dataset Schema

Each dataset entry contains structured metadata describing the question, answer, and supporting evidence.

## Example: Single-Chunk Question

```json
{
  "question_id": "q0001",
  "query": "Which dataset was used to evaluate the proposed method?",
  "question_type": "dataset_setup",
  "evidence_scope": "single_chunk",
  "retrieval_target": "exact_chunk",

  "gold_answer_strings": ["CIFAR-10", "CIFAR10"],

  "gold_paper_ids": ["paper_123"],
  "gold_section_titles": ["Experiments"],
  "gold_chunk_uids": ["paper_123::chunk_08"],

  "supporting_chunk_uids": ["paper_123::chunk_08"],
  "supporting_paper_ids": ["paper_123"],

  "answerable_from": {
    "min_papers": 1,
    "max_papers": 1,
    "min_chunks": 1,
    "max_chunks": 1
  },

  "difficulty": "easy",
  "is_multi_hop": false
}
````

---

## Example: Multi-Paper Question

```json
{
  "question_id": "q0217",
  "query": "Which paper reports better F1 score on dataset X, and by how much?",
  "question_type": "comparison",
  "evidence_scope": "multi_chunk_multi_paper",
  "retrieval_target": "supporting_paper_set",

  "gold_answer_strings": ["2.3 points", "2.3%"],

  "gold_paper_ids": ["paper_A", "paper_B"],
  "gold_section_titles": ["Results", "Evaluation"],
  "gold_chunk_uids": ["paper_A::chunk_11", "paper_B::chunk_09"],

  "supporting_chunk_uids": [
    "paper_A::chunk_11",
    "paper_A::chunk_12",
    "paper_B::chunk_09"
  ],
  "supporting_paper_ids": ["paper_A", "paper_B"],

  "answerable_from": {
    "min_papers": 2,
    "max_papers": 2,
    "min_chunks": 2,
    "max_chunks": 3
  },

  "difficulty": "hard",
  "is_multi_hop": true
}
```

---

# Synthetic Data Quality Rules

To ensure dataset reliability and prevent trivial evaluation cases, the following rules are enforced.

### Rule 1 — No lexical copying

Questions must not be direct paraphrases of sentences in the source chunk.

Example of disallowed pattern:

> Chunk: "The model achieves 84.2 F1 score."
> Question: "What F1 score does the model achieve?"

Instead, questions must require **semantic retrieval**.

---

### Rule 2 — No metadata questions

Questions about metadata such as:

* paper title
* authors
* publication year

are excluded because they do not test evidence retrieval.

---

### Rule 3 — Answers must be short and verifiable

Answers should be:

* entities
* numbers
* short phrases

This ensures evaluation metrics remain reliable.

---

# Metric Design

The benchmark evaluates retrieval quality using **three metric families**.

---

# A. Exact Retrieval Metrics

Used for **single-chunk questions**.

| Metric          | Description                                            |
| --------------- | ------------------------------------------------------ |
| ExactChunkHit@k | Whether the correct chunk appears in the top-k results |
| MRR_exact@k     | Reciprocal rank of the correct chunk                   |

These measure the model's ability to locate **precise evidence**.

---

# B. Support-Set Retrieval Metrics

Used for **multi-chunk or multi-paper questions**.

| Metric               | Description                                         |
| -------------------- | --------------------------------------------------- |
| SupportChunkRecall@k | Fraction of supporting chunks retrieved in top-k    |
| SupportPaperRecall@k | Fraction of supporting papers retrieved             |
| SupportSetHit@k      | Whether all required evidence units appear in top-k |

These metrics measure the system's ability to retrieve **complete evidence sets**.

---

# C. Answerability Metrics

These metrics measure whether retrieved results contain enough information to answer the question.

| Metric                | Description                                                                   |
| --------------------- | ----------------------------------------------------------------------------- |
| AnswerContain@k       | Whether retrieved text contains the correct answer                            |
| EvidenceSufficiency@k | Whether the minimal evidence set required to answer the question is retrieved |

Evidence sufficiency rules:

* **Single-chunk questions**: correct chunk OR answer present
* **Multi-chunk questions**: required chunks retrieved
* **Multi-paper questions**: minimum supporting paper set retrieved

---

# Benchmark Reporting Structure

Results should not be reported as a single aggregate score.

Instead, evaluation should be reported across multiple dimensions.

## Overall Performance

Metrics aggregated across all **300 questions**.

---

## Performance by Evidence Scope

* Single Chunk
* Multi Chunk (same paper)
* Multi Paper

---

## Performance by Question Type

Evaluation reported separately for each of the **8 question classes**.

---

## Performance by Difficulty

| Difficulty | Count |
| ---------- | ----- |
| Easy       | 90    |
| Medium     | 150   |
| Hard       | 60    |

---

# Final Dataset Composition

The final benchmark contains **300 questions**.

### Evidence Scope

| Scope                    | Count |
| ------------------------ | ----- |
| Single Chunk             | 120   |
| Multi Chunk (same paper) | 120   |
| Multi Paper              | 60    |

### Difficulty

| Level  | Count |
| ------ | ----- |
| Easy   | 90    |
| Medium | 150   |
| Hard   | 60    |

### Question Types

Balanced across the eight categories defined earlier.

---

# Benchmark Design Goals

This benchmark aims to produce a dataset that is:

* **Broad** – covering multiple reasoning patterns
* **Interpretable** – metrics diagnose specific retrieval failures
* **Stable** – enough samples for reliable evaluation
* **Realistic** – reflecting real scholarly question answering scenarios

The resulting benchmark allows rigorous comparison of **embedding models**, **dense retrieval**, **sparse retrieval**, and **hybrid retrieval systems** in scientific document search.