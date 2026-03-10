# Retrieval Evaluation Report
**Batches**: batch_01 + batch_02
**Timestamp**: 2026-03-10 22:11:55
**Total Queries**: 20

## Timing Summary

- **Indexing Time**: 9.0m 31.8s
- **Total Evaluation Time**: 0.0m 8.4s

## Summary Metrics by Model

| Model | Eval Time (s) | Recall@1 | Recall@5 | Recall@10 | MRR@5 | NDCG@10 | PaperHit@5 | SectionHit@5 |
|---|---|---|---|---|---|---|---|---|
| **jina-v3-nano** | 8.37 | 0.0000 | 0.7500 | 0.0000 | 0.5392 | 0.0000 | 1.0000 | 0.9500 |

## Detailed Results by Model

### jina-v3-nano

**Evaluation Time**: 8.37s
**Queries Evaluated**: 20

#### Aggregate Metrics

- Recall@5: 0.7500
- MRR@5: 0.5392
- NDCG@5: 1.7703
- PaperHit@5: 1.0000
- SectionHit@5: 0.9500
- AnswerContain@5: 0.7500

#### Per-Query Breakdown

| Query ID | Recall@1 | Recall@5 | MRR@5 | NDCG@5 | Paper Hit@5 | Answer@5 |
|---|---|---|---|---|---|---|
| q001 | ✗ | ✓ | 0.2000 | 1.6677 | ✓ | ✓ |
| q002 | ✗ | ✓ | 0.5000 | 1.7897 | ✓ | ✓ |
| q003 | ✗ | ✓ | 0.5000 | 1.7897 | ✓ | ✓ |
| q004 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q005 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q006 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q007 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q008 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q009 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q010 | ✗ | ✓ | 0.2500 | 1.6896 | ✓ | ✓ |
| q011 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q012 | ✗ | ✓ | 0.5000 | 1.7897 | ✓ | ✓ |
| q013 | ✗ | ✓ | 0.3333 | 1.7242 | ✓ | ✓ |
| q014 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q015 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q016 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q017 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q018 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q019 | ✗ | ✓ | 0.5000 | 1.7897 | ✓ | ✓ |
| q020 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
