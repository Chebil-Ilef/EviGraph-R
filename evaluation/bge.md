# Retrieval Evaluation Report
**Batches**: batch_01 + batch_02
**Timestamp**: 2026-03-11 00:07:02
**Total Queries**: 20

## Timing Summary

- **Indexing Time**: 61.0m 57.6s
- **Total Evaluation Time**: 0.0m 19.0s

## Summary Metrics by Model

| Model | Eval Time (s) | Recall@1 | Recall@5 | Recall@10 | MRR@5 | NDCG@10 | PaperHit@5 | SectionHit@5 |
|---|---|---|---|---|---|---|---|---|
| **bge-m3** | 18.98 | 0.0000 | 0.6500 | 0.0000 | 0.5017 | 0.0000 | 1.0000 | 0.9500 |

## Detailed Results by Model

### bge-m3

**Evaluation Time**: 18.98s
**Queries Evaluated**: 20

#### Aggregate Metrics

- Recall@5: 0.6500
- MRR@5: 0.5017
- NDCG@5: 1.7047
- PaperHit@5: 1.0000
- SectionHit@5: 0.9500
- AnswerContain@5: 0.6500

#### Per-Query Breakdown

| Query ID | Recall@1 | Recall@5 | MRR@5 | NDCG@5 | Paper Hit@5 | Answer@5 |
|---|---|---|---|---|---|---|
| q001 | ✗ | ✓ | 0.5000 | 1.7897 | ✓ | ✓ |
| q002 | ✗ | ✓ | 0.3333 | 1.5089 | ✓ | ✓ |
| q003 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q004 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q005 | ✗ | ✓ | 0.2000 | 1.6677 | ✓ | ✓ |
| q006 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q007 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q008 | ✗ | ✓ | 0.5000 | 1.7897 | ✓ | ✓ |
| q009 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q010 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q011 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q012 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q013 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q014 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q015 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q016 | ✗ | ✓ | 0.5000 | 1.7897 | ✓ | ✓ |
| q017 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
| q018 | ✗ | ✓ | 1.0000 | 1.4088 | ✓ | ✓ |
| q019 | ✗ | ✗ | 0.0000 | 1.4742 | ✓ | ✗ |
| q020 | ✗ | ✓ | 1.0000 | 1.9742 | ✓ | ✓ |
