# SQuAI vs EviGraph-R — Corrected Retrieval Comparison (T1)

*N = 113 questions. NDCG removed — see t1_ndcg_audit_notes.md.*

| Metric | SQuAI (E5+BM25) | EviGraph-R (BGE-M3) |
|--------|-----------------|---------------------|
| Hit@1              |           0.204 |               0.504 |
| Hit@5              |           0.319 |               0.637 |
| Hit@10              |           0.345 |               0.646 |
| MRR@1              |           0.204 |               0.504 |
| MRR@5              |           0.247 |               0.560 |
| MRR@10              |           0.250 |               0.561 |
| Recall@1           |           0.204 |               0.504 |
| Recall@5           |           0.319 |               0.637 |
| Recall@10           |           0.345 |               0.646 |
| Avg Rank of Gold   |           2.205 |               1.411 |
| Section Hit@1      |               — |               0.377 |
| Section Hit@5      |               — |               0.481 |
| Section Hit@10      |               — |               0.494 |

### Breakdown by Domain (Single-paper)

**CS** (N=39)

| Metric | SQuAI | EviGraph-R |
|--------|-------|------------|
| Hit@1     | 0.256 | 0.641 |
| MRR@1     | 0.256 | 0.641 |
| Recall@1  | 0.256 | 0.641 |
| Hit@5     | 0.359 | 0.769 |
| MRR@5     | 0.291 | 0.693 |
| Recall@5  | 0.359 | 0.769 |
| Hit@10     | 0.385 | 0.795 |
| MRR@10     | 0.294 | 0.697 |
| Recall@10  | 0.385 | 0.795 |

**Physics** (N=40)

| Metric | SQuAI | EviGraph-R |
|--------|-------|------------|
| Hit@1     | 0.150 | 0.375 |
| MRR@1     | 0.150 | 0.375 |
| Recall@1  | 0.150 | 0.375 |
| Hit@5     | 0.250 | 0.475 |
| MRR@5     | 0.188 | 0.421 |
| Recall@5  | 0.250 | 0.475 |
| Hit@10     | 0.275 | 0.475 |
| MRR@10     | 0.191 | 0.421 |
| Recall@10  | 0.275 | 0.475 |

**Math** (N=34)

| Metric | SQuAI | EviGraph-R |
|--------|-------|------------|
| Hit@1     | 0.206 | 0.500 |
| MRR@1     | 0.206 | 0.500 |
| Recall@1  | 0.206 | 0.500 |
| Hit@5     | 0.353 | 0.676 |
| MRR@5     | 0.265 | 0.570 |
| Recall@5  | 0.353 | 0.676 |
| Hit@10     | 0.382 | 0.676 |
| MRR@10     | 0.268 | 0.570 |
| Recall@10  | 0.382 | 0.676 |


### Breakdown by Question Source

| Source | N | SQuAI Hit@5 | EviGraph Hit@5 | SQuAI MRR@10 | EviGraph MRR@10 |
|--------|---|-------------|----------------|--------------|-----------------|
| section    | 38 | 0.079 | 0.526 | 0.069 | 0.465 |
| fullpaper  | 39 | 0.179 | 0.564 | 0.162 | 0.500 |
| abstract   | 36 | 0.722 | 0.833 | 0.536 | 0.729 |

---
*Section Hit reported for EviGraph-R only — SQuAI indexes abstracts with no section granularity.*
