src/indexing_pipeline.py

Single entry-point for the full arXiv indexing pipeline.

Stages
------
1. CHUNK   — raw JSONL batches → token-bounded chunk dicts (saved to PATHS.chunks)
2. INDEX   — chunk dicts → embed → upsert into Qdrant
              · Collection schema is created from _QdrantProfile (config)
              · Dense vectors:  any model in EMBEDDING_MODELS
              · Sparse vectors: server-side BM25 (always) + BGE-M3 optional

Everything that is not a runtime decision (sizes, names, flags) lives in config.

Usage
-----
# Full pipeline on two batches
python -m src.indexing_pipeline --batches batch_01 batch_02

# Chunk only (no Qdrant needed)
python -m src.indexing_pipeline --batches all --chunk-only

# Index already-chunked batches
python -m src.indexing_pipeline --batches all --index-only

# Drop and rebuild the collection, then index
python -m src.indexing_pipeline --batches all --recreate
