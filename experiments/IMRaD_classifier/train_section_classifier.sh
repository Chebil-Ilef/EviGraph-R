#!/bin/bash
#SBATCH --job-name=train_imrad_clf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_imrad_clf_%j.log
#SBATCH --error=logs/train_imrad_clf_%j.err

# Train IMRAD section classifier on HPC
# Usage: sbatch experiments/IMRaD_classifier/train_section_classifier.sh

set -e

echo "=========================================="
echo "IMRAD Section Classifier Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo ""

export PYTHONUNBUFFERED=1
export HUGGINGFACE_HUB_CACHE="_data/.model_cache"
export HF_HOME="${HUGGINGFACE_HUB_CACHE}"

if [ -f ".env" ]; then
    export $(grep '^HF_TOKEN=' .env | xargs)
fi

mkdir -p logs
mkdir -p experiments/IMRaD_classifier/_data/models
mkdir -p "${HUGGINGFACE_HUB_CACHE}"

MODEL_ID="distilbert-base-uncased"
DATASET_ID="saier/unarXive_imrad_clf"
OUTPUT_DIR="experiments/IMRaD_classifier/_data/models/section_classifier"
HUB_ORG="lostelf"
HUB_REPO_ID="${HUB_ORG}/section-classifier-imrad"

echo "Configuration:"
echo "  Base Model: ${MODEL_ID}"
echo "  Dataset: ${DATASET_ID}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo "  Hub Repo: ${HUB_REPO_ID}"
echo "  Cache Dir: ${HUGGINGFACE_HUB_CACHE}"
echo ""

echo "Starting training on GPU..."
uv run python experiments/IMRaD_classifier/train_section_classifier.py \
  --dataset-id "${DATASET_ID}" \
  --model-id "${MODEL_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-epochs 3 \
  --per-device-batch-size 32 \
  --learning-rate 2e-5 \
  --warmup-steps 500 \
  --max-seq-length 512 \
  --eval-strategy steps \
  --eval-steps 100 \
  --seed 42 \
  --device cuda \
  --push-to-hub \
  --hub-repo-id "${HUB_REPO_ID}"

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
  echo ""
  echo "✓ Training completed successfully"
  echo "✓ Model saved to: ${OUTPUT_DIR}"
  echo "✓ Model pushed to Hub: ${HUB_REPO_ID}"
  echo ""
  ls -lah "${OUTPUT_DIR}"
else
  echo ""
  echo "✗ Training failed with exit code $TRAIN_EXIT"
  exit $TRAIN_EXIT
fi

echo ""
echo "=========================================="
echo "Job completed"
echo "=========================================="
