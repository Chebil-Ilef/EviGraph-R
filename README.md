

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



srun --partition=capella --nodes=1 --gres=gpu:1 --cpus-per-task=8 \
     --mem=64G --time=01:00:00 --pty bash


export SINGULARITY_CACHEDIR=/tmp/singularity_cache
export SINGULARITY_TMPDIR=/tmp/singularity_tmp
export QDRANT_SIF_PATH=$HOME/qdrant.sif

singularity instance start \
  --bind _data/qdrant_storage:/qdrant/storage \
  --bind _data/qdrant_snapshots:/qdrant/snapshots \
  $QDRANT_SIF_PATH evigraph-qdrant

singularity exec instance://evigraph-qdrant /qdrant/qdrant &
sleep 2

curl -s http://localhost:6333/collections/unarxive_chunks | jq '.result.points_count'


SAMPLE_SIZE=10 sbatch scripts/run_indexing_capella.sh


sbatch --array=0-4 --export=ALL,TOTAL_TASKS=5,SAMPLE_SIZE=3000 scripts/run_indexing_array_capella.sh

