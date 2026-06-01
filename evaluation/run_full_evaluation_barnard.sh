#!/bin/bash
#SBATCH --job-name=evigraph-eval
#SBATCH --partition=barnard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100G
#SBATCH --time=07:00:00
#SBATCH --output=logs/full_evaluation_%j.log
#
# Full EviGraph-R benchmark launcher for Barnard (CPU-only, no GPU).
#
# Modes:
#   --generate-only    Sample context groups and synthesize goldens only.
#   --ablation-only    Run EviGraph-R full + ablation variants, then evaluate.
#   --baselines-only   Run baseline systems, then evaluate.
#   --everything       Generate dataset, run ablations, run baselines, evaluate.
#
# If no mode flag is provided, --everything is used.
#
# Examples:
#   CAT1_TARGET=10 CAT2_TARGET=10 CAT3_TARGET=10 CAT4_TARGET=10   sbatch evaluation/run_full_evaluation_barnard.sh --generate-only
#   sbatch evaluation/run_full_evaluation_barnard.sh --generate-only
#   sbatch evaluation/run_full_evaluation_barnard.sh --ablation-only
#   BASELINES=squai sbatch --time=04:00:00 --mem=32G evaluation/run_full_evaluation_barnard.sh --baselines-only
#   sbatch evaluation/run_full_evaluation_barnard.sh --everything

set -euo pipefail
trap 'echo "[ERROR] line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

REPO_DIR=$(pwd)
cd "$REPO_DIR"

mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src:${PYTHONPATH:-}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-${USER}}"

# Export .env vars into the shell so all subprocesses (including synthesize_dataset)
# inherit LLM_API_BASE, LLM_API_KEY, and other settings without needing load_dotenv.
if [[ -f "${REPO_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_DIR}/.env"
  set +a
fi

export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/tmp/singularity_cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/singularity_tmp}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

export QDRANT_SIF_PATH="${QDRANT_SIF_PATH:-${REPO_DIR}/qdrant.sif}"
if [[ ! -f "$QDRANT_SIF_PATH" ]]; then
  echo "Building Qdrant Singularity image from docker://qdrant/qdrant..."
  singularity build "$QDRANT_SIF_PATH" docker://qdrant/qdrant
  echo "Qdrant image built: $QDRANT_SIF_PATH"
fi

QDRANT_PROFILE="${QDRANT_PROFILE:-hpc}"
export QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
export QDRANT_TIMEOUT="${QDRANT_TIMEOUT:-300}"
QDRANT_STARTUP_TIMEOUT="${QDRANT_STARTUP_TIMEOUT:-1800}"

MODEL="${MODEL:-${LLM_ANSWER_GENERATOR_MODEL:-meta-llama/Llama-3.3-70B-Instruct}}"
EVAL_JUDGE_MODEL="${EVAL_JUDGE_MODEL:-${LLM_EVAL_JUDGE_MODEL:-${MODEL}}}"
EVOLUTION="${EVOLUTION:-reasoning}"
SEED="${SEED:-42}"

RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${REPO_DIR}/evaluation/_data}"
GROUPS_DIR="${GROUPS_DIR:-${BENCHMARK_DIR}/groups}"
GOLDENS_PATH="${GOLDENS_PATH:-${BENCHMARK_DIR}/goldens.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-${BENCHMARK_DIR}/results}"
EVAL_DIR="${EVAL_DIR:-${BENCHMARK_DIR}/eval}"
REPORT_DIR="${REPORT_DIR:-${BENCHMARK_DIR}/reports/${RUN_ID}}"

CAT1_TARGET="${CAT1_TARGET:-200}"
CAT2_TARGET="${CAT2_TARGET:-150}"
CAT3_TARGET="${CAT3_TARGET:-200}"
CAT4_TARGET="${CAT4_TARGET:-200}"

# Cat 1/2: 30% script-level buffer to absorb synthesis quality filter discards.
# Cat 3/4: the samplers apply their own 2x MuSiQue buffer internally,
#          so the script passes the plain target (factor=1.0).
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-1.30}"

CAT1_SAMPLE=$(python3 -c "import math; print(math.ceil(${CAT1_TARGET} * ${OVERSAMPLE_FACTOR}))")
CAT2_SAMPLE=$(python3 -c "import math; print(math.ceil(${CAT2_TARGET} * ${OVERSAMPLE_FACTOR}))")
CAT3_SAMPLE="${CAT3_TARGET}"
CAT4_SAMPLE="${CAT4_TARGET}"

ABLATION_VARIANTS="${ABLATION_VARIANTS:-full A1.1 A1.2 R1 R2 R3 G1 G2 J1 J2 J3}"
BASELINES="${BASELINES:-standard_rag squai}"

# SQuAI venv — activated only when squai is in BASELINES
SQUAI_VENV="${REPO_DIR}/../SQuAI/env/bin/activate"

GENERATE_ONLY=0
ABLATION_ONLY=0
BASELINES_ONLY=0
EVERYTHING=0
EVERYTHING_NO_GENERATE=0
REGEN_CAT34=0
SKIP_SAMPLING=0

usage() {
  cat <<'USAGE'
Full EviGraph-R benchmark launcher for Barnard (CPU-only).

Modes:
  --generate-only          Sample context groups and synthesize goldens only.
  --regen-cat34            Resample + resynthesize cat3/cat4 only; merge with saved cat1/cat2 goldens.
  --skip-sampling          (modifier for --regen-cat34) Skip resampling — use existing groups files.
  --ablation-only          Run EviGraph-R full + ablation variants, then evaluate.
  --baselines-only         Run baseline systems, then evaluate.
  --everything             Generate dataset, run ablations, run baselines, evaluate.
  --everything-no-generate Run ablations + baselines on an existing GOLDENS_PATH (skip generation).

If no mode flag is provided, --everything is used.

Examples:
  sbatch evaluation/run_full_evaluation_barnard.sh --generate-only
  SAVED_GOLDENS_PATH=evaluation/_data/benchmark/goldens_copy_saved.jsonl \
    CAT3_TARGET=70 CAT4_TARGET=70 \
    sbatch evaluation/run_full_evaluation_barnard.sh --regen-cat34
  sbatch evaluation/run_full_evaluation_barnard.sh --ablation-only
  BASELINES=squai sbatch --time=04:00:00 --mem=32G evaluation/run_full_evaluation_barnard.sh --baselines-only
  sbatch evaluation/run_full_evaluation_barnard.sh --everything
  GOLDENS_PATH=evaluation/_data/test_goldens_15.jsonl sbatch evaluation/run_full_evaluation_barnard.sh --everything-no-generate

Useful overrides:
  BENCHMARK_DIR=/path/to/benchmark
  MODEL=meta-llama/Llama-3.3-70B-Instruct
  CAT1_TARGET=50 CAT2_TARGET=50 CAT3_TARGET=50 CAT4_TARGET=50
  SAVED_GOLDENS_PATH=/path/to/goldens_copy_saved.jsonl   (used by --regen-cat34)
  ABLATION_VARIANTS="full G2 J1"
  BASELINES="standard_rag"
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generate-only)
      GENERATE_ONLY=1
      ;;
    --ablation-only)
      ABLATION_ONLY=1
      ;;
    --baselines-only)
      BASELINES_ONLY=1
      ;;
    --everything)
      EVERYTHING=1
      ;;
    --everything-no-generate)
      EVERYTHING_NO_GENERATE=1
      ;;
    --regen-cat34)
      REGEN_CAT34=1
      ;;
    --skip-sampling)
      SKIP_SAMPLING=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$GENERATE_ONLY$ABLATION_ONLY$BASELINES_ONLY$EVERYTHING$EVERYTHING_NO_GENERATE$REGEN_CAT34" == "000000" ]]; then
  EVERYTHING=1
fi

if (( GENERATE_ONLY + ABLATION_ONLY + BASELINES_ONLY + EVERYTHING + EVERYTHING_NO_GENERATE + REGEN_CAT34 > 1 )); then
  echo "Choose exactly one mode flag." >&2
  usage >&2
  exit 2
fi

mkdir -p "$GROUPS_DIR" "$RESULTS_DIR" "$EVAL_DIR" "$REPORT_DIR"

echo "══════════════════════════════════════════"
echo "  EviGraph-R Barnard Evaluation"
echo "══════════════════════════════════════════"
echo "Host: $(hostname)"
echo "Run ID: ${RUN_ID}"
echo "QDRANT_PROFILE=${QDRANT_PROFILE}"
echo "BENCHMARK_DIR=${BENCHMARK_DIR}"
echo "MODEL=${MODEL}"
echo "Mode: generate_only=${GENERATE_ONLY} ablation_only=${ABLATION_ONLY} baselines_only=${BASELINES_ONLY} everything=${EVERYTHING} everything_no_generate=${EVERYTHING_NO_GENERATE}"
echo "Targets (golden): cat1=${CAT1_TARGET} cat2=${CAT2_TARGET} cat3=${CAT3_TARGET} cat4=${CAT4_TARGET}"
echo "Sample (groups):  cat1=${CAT1_SAMPLE} cat2=${CAT2_SAMPLE} cat3=${CAT3_SAMPLE} cat4=${CAT4_SAMPLE}  (factor=${OVERSAMPLE_FACTOR})"
echo "Start: $(date -Is)"

run_step() {
  local label="$1"
  shift
  echo ""
  echo "[$(date -Is)] ${label}"
  echo "Command: $*"
  srun "$@"
}


line_count() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -l < "$path" | tr -d ' '
  else
    echo 0
  fi
}

write_report() {
  local status="$1"
  local report_json="${REPORT_DIR}/full_evaluation_report.json"
  local report_md="${REPORT_DIR}/full_evaluation_report.md"

  uv run python -c "
import json, os, sys
sys.path.insert(0, '${REPO_DIR}')
sys.path.insert(0, '${REPO_DIR}/src')
from dotenv import load_dotenv
load_dotenv('${REPO_DIR}/.env')
from pathlib import Path
from evigraph.config.settings import LLM, DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS

emb_cfg = EMBEDDING_MODELS.get(DEFAULT_EMBEDDING_MODEL)
groups_dir = Path('${GROUPS_DIR}')
results_dir = Path('${RESULTS_DIR}')
eval_dir = Path('${EVAL_DIR}')
report = {
    'status': '${status}',
    'run_id': '${RUN_ID}',
    'benchmark_dir': '${BENCHMARK_DIR}',
    'groups_dir': str(groups_dir),
    'goldens_path': '${GOLDENS_PATH}',
    'results_dir': str(results_dir),
    'eval_dir': str(eval_dir),
    'qdrant_profile': '${QDRANT_PROFILE}',
    'models': {
        'decomposer':      LLM.decomposer_model,
        'retriever_embed': emb_cfg.hf_model_id if emb_cfg else DEFAULT_EMBEDDING_MODEL,
        'retriever_mode':  'dense+sparse (BGE-M3)' if (emb_cfg and emb_cfg.bge_produces_sparse) else 'dense+bm25',
        'evidence_graph':  LLM.evidence_graph_builder_model,
        'judge':           LLM.judge_model,
        'answer_generator': LLM.answer_generator_model,
        'eval_judge':      '${EVAL_JUDGE_MODEL}',
    },
    'targets': {
        'cat1': int('${CAT1_TARGET}'),
        'cat2': int('${CAT2_TARGET}'),
        'cat3': int('${CAT3_TARGET}'),
        'cat4': int('${CAT4_TARGET}'),
    },
    'group_counts': {p.stem: sum(1 for _ in p.open()) for p in sorted(groups_dir.glob('cat*.jsonl'))},
    'golden_count': sum(1 for _ in Path('${GOLDENS_PATH}').open()) if Path('${GOLDENS_PATH}').exists() else 0,
    'result_counts': {p.stem: sum(1 for _ in p.open()) for p in sorted(results_dir.glob('*.jsonl'))},
    'eval_files': sorted(p.name for p in eval_dir.glob('*')),
}
Path('${report_json}').write_text(json.dumps(report, indent=2) + '\\n')
m = report['models']
lines = [
    '# EviGraph-R Evaluation Report',
    '',
    f'- Status: {report[\"status\"]}',
    f'- Run ID: {report[\"run_id\"]}',
    f'- Benchmark dir: {report[\"benchmark_dir\"]}',
    f'- Goldens: {report[\"golden_count\"]}',
    '',
    '## Models',
    f'- Decomposer LLM:       {m[\"decomposer\"]}',
    f'- Retriever embedder:   {m[\"retriever_embed\"]}  ({m[\"retriever_mode\"]})',
    f'- Evidence graph LLM:   {m[\"evidence_graph\"]}',
    f'- Judge LLM:            {m[\"judge\"]}',
    f'- Answer generator LLM: {m[\"answer_generator\"]}',
    f'- Eval judge LLM:       {m[\"eval_judge\"]}',
    '',
    '## Context Groups',
    *[f'- {k}: {v}' for k, v in report['group_counts'].items()],
    '',
    '## Results',
    *[f'- {k}: {v}' for k, v in report['result_counts'].items()],
    '',
    '## Eval Files',
    *[f'- {name}' for name in report['eval_files']],
]
Path('${report_md}').write_text('\\n'.join(lines) + '\\n')
print('${report_json}')
print('${report_md}')
"
}

generate_dataset() {
  # All sampling and synthesis run in a single srun so Qdrant (started by
  # ensure_qdrant_runtime) stays alive in the same task slot for every step.
  run_step "Generate dataset (sample + synthesize)" \
    uv run python -c "
import sys, os
sys.path.insert(0, '${REPO_DIR}')
sys.path.insert(0, '${REPO_DIR}/src')
from evigraph.utils.qdrant import ensure_qdrant_runtime
ensure_qdrant_runtime('${QDRANT_PROFILE}', startup_timeout=${QDRANT_STARTUP_TIMEOUT})

import subprocess, shlex
def run(cmd):
    print('>>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

run(['uv', 'run', 'python', '-m', 'evaluation.samplers.cat1_single_paper',
     '--output', '${GROUPS_DIR}/cat1.jsonl', '--target', '${CAT1_SAMPLE}', '--seed', '${SEED}'])
run(['uv', 'run', 'python', '-m', 'evaluation.samplers.cat2_cross_section',
     '--output', '${GROUPS_DIR}/cat2.jsonl', '--target', '${CAT2_SAMPLE}', '--seed', '${SEED}'])
run(['uv', 'run', 'python', '-m', 'evaluation.samplers.cat3_citation',
     '--output', '${GROUPS_DIR}/cat3.jsonl', '--target', '${CAT3_SAMPLE}', '--seed', '${SEED}'])
run(['uv', 'run', 'python', '-m', 'evaluation.samplers.cat4_thematic',
     '--output', '${GROUPS_DIR}/cat4.jsonl', '--target', '${CAT4_SAMPLE}', '--seed', '${SEED}'])
run(['uv', 'run', 'python', '-m', 'evaluation.utils.synthesize_dataset',
     '--groups_dir', '${GROUPS_DIR}', '--output', '${GOLDENS_PATH}',
     '--model', '${MODEL}', '--evolution', '${EVOLUTION}',
     '--cat1_target', '${CAT1_TARGET}',
     '--cat2_target', '${CAT2_TARGET}',
     '--cat3_target', '${CAT3_TARGET}',
     '--cat4_target', '${CAT4_TARGET}'])
"

  echo "Generated $(line_count "$GOLDENS_PATH") goldens at $GOLDENS_PATH"
}

regen_cat34_dataset() {
  local saved="${SAVED_GOLDENS_PATH:-${BENCHMARK_DIR}/benchmark/goldens_copy_saved.jsonl}"
  local tmp_groups="${BENCHMARK_DIR}/groups_34_only"
  local tmp_goldens="${BENCHMARK_DIR}/goldens_cat34_new.jsonl"

  if [[ ! -f "$saved" ]]; then
    echo "ERROR: SAVED_GOLDENS_PATH not found: $saved" >&2
    echo "Set SAVED_GOLDENS_PATH to the file containing the good cat1/cat2 goldens." >&2
    exit 1
  fi

  # Step 1 — resample cat3 and cat4 groups (Qdrant required, skippable)
  if [[ "${SKIP_SAMPLING}" == "1" ]]; then
    echo "Skipping sampling — using existing groups: $(line_count ${GROUPS_DIR}/cat3.jsonl) cat3, $(line_count ${GROUPS_DIR}/cat4.jsonl) cat4"
  else
    run_step "Resample cat3 + cat4 groups" \
      uv run python -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
sys.path.insert(0, '${REPO_DIR}/src')
from evigraph.utils.qdrant import ensure_qdrant_runtime
ensure_qdrant_runtime('${QDRANT_PROFILE}', startup_timeout=${QDRANT_STARTUP_TIMEOUT})

import subprocess
def run(cmd):
    print('>>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

run(['uv', 'run', 'python', '-m', 'evaluation.samplers.cat3_citation',
     '--output', '${GROUPS_DIR}/cat3.jsonl', '--target', '${CAT3_TARGET}', '--seed', '${SEED}'])
run(['uv', 'run', 'python', '-m', 'evaluation.samplers.cat4_thematic',
     '--output', '${GROUPS_DIR}/cat4.jsonl', '--target', '${CAT4_TARGET}', '--seed', '${SEED}'])
"
    echo "Groups: $(line_count ${GROUPS_DIR}/cat3.jsonl) cat3, $(line_count ${GROUPS_DIR}/cat4.jsonl) cat4"
  fi

  # Step 2 — synthesize cat3+cat4 in isolation (no cat1/cat2 groups present)
  mkdir -p "$tmp_groups"
  cp "${GROUPS_DIR}/cat3.jsonl" "${tmp_groups}/cat3.jsonl"
  cp "${GROUPS_DIR}/cat4.jsonl" "${tmp_groups}/cat4.jsonl"

  run_step "Synthesize cat3 + cat4 goldens" \
    uv run python -m evaluation.utils.synthesize_dataset \
      --groups_dir "${tmp_groups}" \
      --output "${tmp_goldens}" \
      --model "${MODEL}" \
      --evolution "${EVOLUTION}" \
      --cat3_target "${CAT3_TARGET}" \
      --cat4_target "${CAT4_TARGET}"

  echo "Synthesized: $(line_count $tmp_goldens) cat3+cat4 goldens"

  # Step 3 — merge saved cat1+cat2 with new cat3+cat4
  run_step "Merge cat1+cat2 (saved) with new cat3+cat4" \
    uv run python -c "
import json
from pathlib import Path
from collections import Counter

records = []
with open('${saved}') as f:
    for line in f:
        r = json.loads(line)
        if r['category'] in (1, 2):
            records.append(r)

with open('${tmp_goldens}') as f:
    for line in f:
        records.append(json.loads(line))

for i, r in enumerate(records):
    r['golden_id'] = f'g_{i:04d}'

Path('${GOLDENS_PATH}').parent.mkdir(parents=True, exist_ok=True)
with open('${GOLDENS_PATH}', 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')

cats = Counter(r['category'] for r in records)
print(f'Written {len(records)} goldens to ${GOLDENS_PATH}')
print('By category:', dict(sorted(cats.items())))
"

  echo "Final goldens: $(line_count $GOLDENS_PATH) at $GOLDENS_PATH"
}

require_goldens() {
  if [[ ! -s "$GOLDENS_PATH" ]]; then
    echo "Goldens file not found or empty: $GOLDENS_PATH" >&2
    echo "Run --generate-only first, or use --everything." >&2
    exit 1
  fi
}

run_ablation_variants() {
  require_goldens
  # All variants run in a single srun so Qdrant stays alive and models load only once.
  run_step "Run ablation variants + evaluate" \
    uv run python -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
sys.path.insert(0, '${REPO_DIR}/src')
from evigraph.utils.qdrant import ensure_qdrant_runtime
ensure_qdrant_runtime('${QDRANT_PROFILE}', startup_timeout=${QDRANT_STARTUP_TIMEOUT})

import subprocess
def run(cmd):
    print('>>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

# Single invocation: models (embedder, cross-encoder) load once and are shared across all variants.
run(['uv', 'run', 'python', '-m', 'evaluation.evigraph_runner',
     '--goldens', '${GOLDENS_PATH}',
     '--variants'] + '${ABLATION_VARIANTS}'.split() +
    ['--output_dir', '${RESULTS_DIR}'])

run(['uv', 'run', 'python', '-m', 'evaluation.full_evaluation',
     '--results_dir', '${RESULTS_DIR}', '--output_dir', '${EVAL_DIR}', '--model', '${EVAL_JUDGE_MODEL}'])
"
}

run_baselines() {
  require_goldens
  # Only start Qdrant if standard_rag is among the requested baselines.
  # squai uses its own FAISS index and never touches Qdrant.
  local needs_qdrant=0
  for b in $BASELINES; do
    [[ "$b" == "standard_rag" ]] && needs_qdrant=1
  done

  # squai needs SQuAI's venv (plyvel, faiss, sentence-transformers) + PyTorch modules.
  local needs_squai=0
  for b in $BASELINES; do
    [[ "$b" == "squai" ]] && needs_squai=1
  done

  if [[ "${needs_squai}" == "1" ]]; then
    module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2
    # shellcheck disable=SC1090
    source "${SQUAI_VENV}"
  fi

  run_step "Run baselines + evaluate" \
    uv run python -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
sys.path.insert(0, '${REPO_DIR}/src')

needs_qdrant = ${needs_qdrant}
if needs_qdrant:
    from evigraph.utils.qdrant import ensure_qdrant_runtime
    ensure_qdrant_runtime('${QDRANT_PROFILE}', startup_timeout=${QDRANT_STARTUP_TIMEOUT})

import subprocess
def run(cmd):
    print('>>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

for baseline in '${BASELINES}'.split():
    run(['uv', 'run', 'python', '-m', 'evaluation.baselines_runner',
         '--goldens', '${GOLDENS_PATH}', '--baseline', baseline,
         '--output', '${RESULTS_DIR}/' + baseline + '.jsonl'])

run(['uv', 'run', 'python', '-m', 'evaluation.full_evaluation',
     '--results_dir', '${RESULTS_DIR}', '--output_dir', '${EVAL_DIR}', '--model', '${EVAL_JUDGE_MODEL}'])
"
}

if [[ "$GENERATE_ONLY" == "1" ]]; then
  generate_dataset
elif [[ "$REGEN_CAT34" == "1" ]]; then
  regen_cat34_dataset
elif [[ "$ABLATION_ONLY" == "1" ]]; then
  run_ablation_variants
elif [[ "$BASELINES_ONLY" == "1" ]]; then
  run_baselines
elif [[ "$EVERYTHING_NO_GENERATE" == "1" ]]; then
  run_ablation_variants
  run_baselines
else
  generate_dataset
  run_ablation_variants
  run_baselines
fi

run_step "Build final evaluation table from available aggregations" \
  uv run python -m evaluation.full_evaluation \
    --output_dir "$EVAL_DIR" \
    --table_only

write_report "completed"

echo ""
echo "══════════════════════════════════════════"
echo "  EviGraph-R evaluation completed"
echo "══════════════════════════════════════════"
echo "End: $(date -Is)"
echo "Reports: ${REPORT_DIR}"
