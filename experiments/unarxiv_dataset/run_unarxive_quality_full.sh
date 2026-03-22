#!/bin/bash
#SBATCH --job-name=unarxive-quality-full
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/quality_full_%j.log

set -euo pipefail

cd /data/cat/ws/ilch217i-horse/EviGraph-R

mkdir -p logs reports checkpoints

uv run experiments/unarxiv_dataset/analyze_dataset.py \
  --n 2338911 \
  --local-dir _data/unarxive_2024_full \
  --out reports/UNARXIVE_FULL_REPORT.md \
  --checkpoint checkpoints/unarxive_full.json \
  --checkpoint-every 10000 \
  --resume \
  --verbose