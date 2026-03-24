

**[https://chebil-ilef.github.io/evigraph-R-diags/](https://chebil-ilef.github.io/evigraph-R-diags/)**



1) install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh

2) uv sync

3) run any script you want with : 
uv run path/to/script.py


for Qdrant

singularity build /home/USERNAME/qdrant.sif docker://qdrant/qdrant


then


mkdir -p /data/cat/ws/ilch217i-qdrant-indexing/qdrant_storage \
         /data/cat/ws/ilch217i-qdrant-indexing/qdrant_snapshots


srun --partition=capella -L cat --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash

singularity exec \
  --bind /data/cat/ws/ilch217i-qdrant-indexing/qdrant_storage:/qdrant/storage \
  --bind /data/cat/ws/ilch217i-qdrant-indexing/qdrant_snapshots:/qdrant/snapshots \
  /home/ilch217i/qdrant.sif \
  /qdrant/qdrant


SAMPLE_SIZE=10 sbatch scripts/run_indexing_capella.sh
