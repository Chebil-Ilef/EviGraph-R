#!/bin/bash
#SBATCH --job-name=evigraph-eval-sample
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/eval_sample_%j.log

set -euo pipefail

REPO_DIR="/data/cat/ws/ilch217i-horse/EviGraph-R"
cd "$REPO_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
  # Unset empty environment variables that cause issues
  [[ -z "$HF_TOKEN" ]]     && unset HF_TOKEN
  [[ -z "$LLM_API_KEY" ]]  && unset LLM_API_KEY
  [[ -z "$LLM_API_BASE" ]] && unset LLM_API_BASE
  [[ -z "$LLM_MODEL" ]]    && unset LLM_MODEL
fi

export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src:${PYTHONPATH:-}"

export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

export QDRANT_SIF_PATH="${QDRANT_SIF_PATH:-${HOME}/qdrant.sif}"

if [[ ! -f "$QDRANT_SIF_PATH" ]]; then
  echo "Qdrant Singularity image not found at $QDRANT_SIF_PATH"
  echo "Building it from docker://qdrant/qdrant (needs internet, ~1-2 min)..."
  singularity build "$QDRANT_SIF_PATH" docker://qdrant/qdrant
  echo "  ✓ Built $QDRANT_SIF_PATH"
fi

mkdir -p logs experiments/embedding/data reports

SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
export INDEXING_PROFILE="${INDEXING_PROFILE:-hpc}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="logs/eval_sample_${SAMPLE_SIZE}"

# Skip flags: set SKIP_INDEXING=1, SKIP_EXPORT=1, SKIP_QA=1 to bypass phases
SKIP_INDEXING="${SKIP_INDEXING:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_QA="${SKIP_QA:-0}"

CHUNKS_FILE="${CHUNKS_FILE:-experiments/embedding/data/chunks_sample_${SAMPLE_SIZE}.jsonl}"
QA_FILE="${QA_FILE:-experiments/embedding/data/synthetic_qa_${SAMPLE_SIZE}.jsonl}"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "EviGraph Evaluation on HPC + GPU"
echo "=========================================="
echo "Sample size: $SAMPLE_SIZE articles"
echo "Profile: $INDEXING_PROFILE (local=Docker, hpc=Singularity)"
echo "GPU Device: $DEVICE"
echo "Output: $OUTPUT_DIR"
[[ "$SKIP_INDEXING" == "1" ]] && echo "  [SKIP] Indexing (SKIP_INDEXING=1)"
[[ "$SKIP_EXPORT"   == "1" ]] && echo "  [SKIP] Export   (SKIP_EXPORT=1)"
[[ "$SKIP_QA"       == "1" ]] && echo "  [SKIP] QA gen   (SKIP_QA=1)"
echo ""

# Phase 1: Prepare sample dataset (chunk + embed using indexing pipeline)

INDEXING_TIME=0
if [[ "$SKIP_INDEXING" == "1" ]]; then
  echo "[1/5] Skipping indexing (SKIP_INDEXING=1)"
else
  echo "[1/5] Preparing sample dataset with indexing pipeline..."
  echo "  Chunking + embedding $SAMPLE_SIZE articles..."

  INDEXING_START=$(date +%s)

  uv run python src/indexing/indexing_pipeline.py \
    --phase run \
    --sample-size "$SAMPLE_SIZE" \
    --model e5-base-v2 \
    --profile "$INDEXING_PROFILE" \
    --recreate-collection \
    --resume \
    2>&1 | tee "$OUTPUT_DIR/indexing.log"

  INDEXING_END=$(date +%s)
  INDEXING_TIME=$((INDEXING_END - INDEXING_START))
  echo "  ✓ Indexing completed in $(($INDEXING_TIME / 60))m $(($INDEXING_TIME % 60))s"
fi
echo ""

# Phase 2: Export chunks from Qdrant

if [[ "$SKIP_EXPORT" == "1" ]]; then
  echo "[2/5] Skipping export (SKIP_EXPORT=1)"
  if [[ ! -f "$CHUNKS_FILE" ]]; then
    echo "  ERROR: CHUNKS_FILE not found: $CHUNKS_FILE" >&2
    exit 1
  fi
else
  echo "[2/5] Exporting chunks from Qdrant..."

  uv run python experiments/embedding/prepare_sample_dataset.py \
    --collection unarxive_chunks \
    --output "$CHUNKS_FILE" \
    2>&1 | tee "$OUTPUT_DIR/export_chunks.log"
fi

CHUNK_COUNT=$(wc -l < "$CHUNKS_FILE")
echo "  ✓ $CHUNK_COUNT chunks in $CHUNKS_FILE"
echo ""

# Phase 3: Generate gold Q&A dataset with AI

if [[ "$SKIP_QA" == "1" ]]; then
  echo "[3/5] Skipping QA generation (SKIP_QA=1)"
  if [[ ! -f "$QA_FILE" ]]; then
    echo "  ERROR: QA_FILE not found: $QA_FILE" >&2
    exit 1
  fi
else
  echo "[3/5] Generating AI gold Q&A dataset..."
  if [[ -n "${LLM_API_KEY:-}" ]]; then
    echo "  Using AI generation (model: ${LLM_MODEL:-gpt-4o-mini})"
  else
    echo "  WARNING: LLM_API_KEY not set — using heuristic fallback"
    echo "  Set LLM_API_KEY and LLM_API_BASE in .env for AI-quality gold labels."
  fi

  QA_SAMPLE_SIZE="${QA_SAMPLE_SIZE:-500}"

  uv run python experiments/embedding/generate_gold_qa.py \
    --chunks "$CHUNKS_FILE" \
    --output "$QA_FILE" \
    --sample-size "$QA_SAMPLE_SIZE" \
    --workers 2 \
    2>&1 | tee "$OUTPUT_DIR/generate_qa.log"
fi

QA_COUNT=$(wc -l < "$QA_FILE")
echo "  ✓ $QA_COUNT Q&A pairs in $QA_FILE"
echo ""

# Phase 4: Evaluate all models

echo "[4/5] Running evaluation on all embedding models..."
echo "  Testing models: e5-base-v2 e5-large-v2 qwen3-0.6b jina-v3-nano bge-m3"

EVAL_START=$(date +%s)

uv run python experiments/embedding/evaluate_models.py eval-all \
  --chunks "$CHUNKS_FILE" \
  --qa-file "$QA_FILE" \
  --models e5-base-v2 e5-large-v2 qwen3-0.6b jina-v3-nano bge-m3  \
  --top-k 1 5 10 \
  --output "$OUTPUT_DIR/eval_results.json" \
  --report "$OUTPUT_DIR/eval_report.md" \
  2>&1 | tee "$OUTPUT_DIR/evaluation.log"

EVAL_END=$(date +%s)
EVAL_TIME=$((EVAL_END - EVAL_START))

echo "  ✓ Evaluation completed in $(($EVAL_TIME / 60))m $(($EVAL_TIME % 60))s"
echo ""

# Phase 5: Calculate time estimates for full dataset

echo "[5/5] Calculating time estimates for full 2.8M dataset..."

uv run python experiments/embedding/calculate_estimates.py \
  --sample-size "$SAMPLE_SIZE" \
  --chunk-count "$CHUNK_COUNT" \
  --indexing-time "$INDEXING_TIME" \
  --evaluation-time "$EVAL_TIME" \
  --output "$OUTPUT_DIR/time_estimates.md" \
  2>&1 | tee "$OUTPUT_DIR/estimates.log"

echo ""
echo "=========================================="
echo "✓ Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - eval_results.json   (detailed metrics)"
echo "  - eval_report.md      (markdown report)"
echo "  - time_estimates.md   (full dataset projections)"
echo ""
echo "Summary timing:"
echo "  - Indexing (chunk+embed): $(($INDEXING_TIME / 60))m $(($INDEXING_TIME % 60))s"
echo "  - Evaluation (all models): $(($EVAL_TIME / 60))m $(($EVAL_TIME % 60))s"
echo "  - Total: $(((INDEXING_TIME + EVAL_TIME) / 60))m $(((INDEXING_TIME + EVAL_TIME) % 60))s"
echo ""
