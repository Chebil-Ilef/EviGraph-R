# Retrieval Evaluation Report
**Batches**: batch_01 + batch_02
**Timestamp**: 2026-03-10 20:53:49
**Total Queries**: 20

## Timing Summary

- **Indexing Time**: 33.0m 7.2s
- **Total Evaluation Time**: 0.0m 11.6s

## Summary Metrics by Model

| Model | Eval Time (s) | Recall@1 | Recall@5 | Recall@10 | MRR@5 | NDCG@10 | PaperHit@5 | SectionHit@5 |
|---|---|---|---|---|---|---|---|---|
| **qwen3-0.6b** | 11.60 | 0.0000 | 0.6500 | 0.0000 | 0.4958 | 0.0000 | 1.0000 | 0.8500 |

## Detailed Results by Model

### qwen3-0.6b

**Evaluation Time**: 11.60s
**Queries Evaluated**: 20

#### Aggregate Metrics

- Recall@5: 0.6500
- MRR@5: 0.4958
- NDCG@5: 1.6928
- PaperHit@5: 1.0000
- SectionHit@5: 0.8500
- AnswerContain@5: 0.6500

#### Per-Query Breakdown

| Query ID | Recall@1 | Recall@5 | MRR@5 | NDCG@5 | Paper Hit@5 | Answer@5 |
|---|---|---|---|---|---|---|
| q001 | ✗ | ✓ | 0.2500 | 1.6896 | ✓ | ✓ |
| q002 | ✗ | ✓ | 0.3333 | 1.7242 | ✓ | ✓ |
| q003 | ✗ | ✓ | 0.5000 | 1.7897 | ✓ | ✓ |
| q004 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q005 | ✗ | ✓ | 0.5000 | 1.7897 | ✓ | ✓ |
| q006 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q007 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q008 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q009 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q010 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q011 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q012 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q013 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q014 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q015 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q016 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q017 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q018 | ✗ | ✓ | 1.0000 | 1.0000 | ✓ | ✓ |
| q019 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q020 | ✗ | ✓ | 0.3333 | 1.7242 | ✓ | ✓ |
