#!/bin/bash
#SBATCH --job-name=scicite_ft
#SBATCH --partition=capella                   
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                       
#SBATCH --cpus-per-task=8                   
#SBATCH --mem=32G
#SBATCH --time=01:00:00                       
#SBATCH --output=logs/scicite_%j.out
#SBATCH --error=logs/scicite_%j.err

set -euo pipefail
mkdir -p logs

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Partition  : $SLURM_JOB_PARTITION"
echo "Started    : $(date)"
echo "=========================================="


SCRATCH="${SCRATCH:-$HOME/}"
export HF_HOME="$SCRATCH/hf_cache"
export HPC_SCRATCH="$SCRATCH/hf_cache"
mkdir -p "$HF_HOME"

# Load .env file to get HF_TOKEN and other configs
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    HF_TOKEN_FILE="$HOME/.hf_token"
    if [[ -f "$HF_TOKEN_FILE" ]]; then
        export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
    else
        echo "WARNING: HF_TOKEN not set and $HF_TOKEN_FILE not found. Hub push will fail."
    fi
fi

OUTPUT_DIR="logs/scicite_output_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo "Launching fine-tuning …"
uv run experiments/scicite_classifier/finetune_scicite.py \
    --model_name   allenai/scibert_scivocab_uncased \
    --output_dir   "$OUTPUT_DIR" \
    --max_length   256 \
    --batch_size   32 \
    --num_epochs   8 \
    --lr           1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --cache_dir    "$HF_HOME" \
    --fp16 \
    --push_to_hub


RESULT_DIR="$HOME/_data/scicite_results/job_${SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"
cp -r "$OUTPUT_DIR/best_model" "$RESULT_DIR/"
cp -r "$OUTPUT_DIR/logs"       "$RESULT_DIR/" 2>/dev/null || true

echo "Results saved to $RESULT_DIR"
echo "Finished : $(date)"