## Chunking Only

python -m src.core.builder --model e5-base-v2 --batches all
python -m src.core.builder --model e5-base-v2 --batches batch_01 batch_02


## Full Pipeline with Skip Options

# Skip chunking (use existing chunks)
python -m src.core.builder --model e5-base-v2 --batches all --pipeline --skip-chunk --recreate

# Chunk only, no indexing
python -m src.core.builder --model e5-base-v2 --batches all --skip-index

## Clean Modular API

from src.core.builder import chunk_batches, run_pipeline

# Chunk only
stems = chunk_batches(["batch_01", "batch_02"], model_key="e5-base-v2")

# Full pipeline
exit_code = run_pipeline(
    batch_paths=["all"],
    model_key="e5-base-v2",
    skip_chunk=False,
    skip_index=False,
    recreate_collection=True,
)