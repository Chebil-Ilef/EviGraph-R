#!/bin/bash
#SBATCH --job-name=evigraph-eval
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/full_evaluation_%j.log
#
# Full EviGraph-R benchmark launcher for Capella.
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
#   sbatch evaluation/run_full_evaluation_capella.sh --generate-only
#   sbatch evaluation/run_full_evaluation_capella.sh --ablation-only
#   sbatch evaluation/run_full_evaluation_capella.sh --baselines-only
#   sbatch evaluation/run_full_evaluation_capella.sh --everything

set -euo pipefail
trap 'echo "[ERROR] line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

REPO_DIR=$(pwd)
cd "$REPO_DIR"

mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/src:${PYTHONPATH:-}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-${USER}}"

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
QDRANT_STARTUP_TIMEOUT="${QDRANT_STARTUP_TIMEOUT:-1800}"

MODEL="${MODEL:-${LLM_ANSWER_GENERATOR_MODEL:-meta-llama/Llama-3.3-70B-Instruct}}"
EVOLUTION="${EVOLUTION:-reasoning}"
SEED="${SEED:-42}"

RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${REPO_DIR}/_data/benchmark}"
GROUPS_DIR="${GROUPS_DIR:-${BENCHMARK_DIR}/groups}"
GOLDENS_PATH="${GOLDENS_PATH:-${BENCHMARK_DIR}/goldens.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-${BENCHMARK_DIR}/results}"
EVAL_DIR="${EVAL_DIR:-${BENCHMARK_DIR}/eval}"
REPORT_DIR="${REPORT_DIR:-${BENCHMARK_DIR}/reports/${RUN_ID}}"

CAT1_TARGET="${CAT1_TARGET:-200}"
CAT2_TARGET="${CAT2_TARGET:-150}"
CAT3_TARGET="${CAT3_TARGET:-200}"
CAT4_TARGET="${CAT4_TARGET:-200}"

ABLATION_VARIANTS="${ABLATION_VARIANTS:-full A1.1 A1.2 R1 R2 R3 G1 G2 J1 J2 J3}"
BASELINES="${BASELINES:-standard_rag}"

GENERATE_ONLY=0
ABLATION_ONLY=0
BASELINES_ONLY=0
EVERYTHING=0

usage() {
  cat <<'USAGE'
Full EviGraph-R benchmark launcher for Capella.

Modes:
  --generate-only    Sample context groups and synthesize goldens only.
  --ablation-only    Run EviGraph-R full + ablation variants, then evaluate.
  --baselines-only   Run baseline systems, then evaluate.
  --everything       Generate dataset, run ablations, run baselines, evaluate.

If no mode flag is provided, --everything is used.

Examples:
  sbatch evaluation/run_full_evaluation_capella.sh --generate-only
  sbatch evaluation/run_full_evaluation_capella.sh --ablation-only
  sbatch evaluation/run_full_evaluation_capella.sh --baselines-only
  sbatch evaluation/run_full_evaluation_capella.sh --everything

Useful overrides:
  BENCHMARK_DIR=/path/to/benchmark
  MODEL=meta-llama/Llama-3.3-70B-Instruct
  CAT1_TARGET=50 CAT2_TARGET=50 CAT3_TARGET=50 CAT4_TARGET=50
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

if [[ "$GENERATE_ONLY$ABLATION_ONLY$BASELINES_ONLY$EVERYTHING" == "0000" ]]; then
  EVERYTHING=1
fi

if (( GENERATE_ONLY + ABLATION_ONLY + BASELINES_ONLY + EVERYTHING > 1 )); then
  echo "Choose exactly one mode flag." >&2
  usage >&2
  exit 2
fi

mkdir -p "$GROUPS_DIR" "$RESULTS_DIR" "$EVAL_DIR" "$REPORT_DIR"

echo "══════════════════════════════════════════"
echo "  EviGraph-R Capella Evaluation"
echo "══════════════════════════════════════════"
echo "Host: $(hostname)"
echo "Run ID: ${RUN_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "QDRANT_PROFILE=${QDRANT_PROFILE}"
echo "BENCHMARK_DIR=${BENCHMARK_DIR}"
echo "MODEL=${MODEL}"
echo "Mode: generate_only=${GENERATE_ONLY} ablation_only=${ABLATION_ONLY} baselines_only=${BASELINES_ONLY} everything=${EVERYTHING}"
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
import json
from pathlib import Path
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
    'model': '${MODEL}',
    'qdrant_profile': '${QDRANT_PROFILE}',
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
lines = [
    '# EviGraph-R Evaluation Report',
    '',
    f'- Status: {report[\"status\"]}',
    f'- Run ID: {report[\"run_id\"]}',
    f'- Benchmark dir: {report[\"benchmark_dir\"]}',
    f'- Goldens: {report[\"golden_count\"]}',
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
from utils.qdrant import ensure_qdrant_runtime
ensure_qdrant_runtime('${QDRANT_PROFILE}', startup_timeout=${QDRANT_STARTUP_TIMEOUT})

import subprocess, shlex
def run(cmd):
    print('>>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

run(['uv', 'run', 'python', '-m', 'evaluation.samplers.cat1_single_paper',
     '--output', '${GROUPS_DIR}/cat1.jsonl', '--target', '${CAT1_TARGET}', '--seed', '${SEED}'])
run(['uv', 'run', 'python', '-m', 'evaluation.samplers.cat2_cross_section',
     '--output', '${GROUPS_DIR}/cat2.jsonl', '--target', '${CAT2_TARGET}', '--seed', '${SEED}'])
run(['uv', 'run', 'python', '-m', 'evaluation.samplers.cat3_citation',
     '--output', '${GROUPS_DIR}/cat3.jsonl', '--target', '${CAT3_TARGET}', '--seed', '${SEED}'])
run(['uv', 'run', 'python', '-m', 'evaluation.samplers.cat4_thematic',
     '--output', '${GROUPS_DIR}/cat4.jsonl', '--target', '${CAT4_TARGET}', '--seed', '${SEED}'])
run(['uv', 'run', 'python', '-m', 'evaluation.utils.synthesize_dataset',
     '--groups_dir', '${GROUPS_DIR}', '--output', '${GOLDENS_PATH}',
     '--model', '${MODEL}', '--evolution', '${EVOLUTION}'])
"

  echo "Generated $(line_count "$GOLDENS_PATH") goldens at $GOLDENS_PATH"
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
  # All variants run in a single srun so Qdrant stays alive across all of them.
  run_step "Run ablation variants + evaluate" \
    uv run python -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
sys.path.insert(0, '${REPO_DIR}/src')
from utils.qdrant import ensure_qdrant_runtime
ensure_qdrant_runtime('${QDRANT_PROFILE}', startup_timeout=${QDRANT_STARTUP_TIMEOUT})

import subprocess
def run(cmd):
    print('>>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

for variant in '${ABLATION_VARIANTS}'.split():
    out = '${RESULTS_DIR}/' + variant.replace('.', '_') + '.jsonl'
    run(['uv', 'run', 'python', '-m', 'evaluation.evigraph_runner',
         '--goldens', '${GOLDENS_PATH}', '--variant', variant, '--output', out])

run(['uv', 'run', 'python', '-m', 'evaluation.full_evaluation',
     '--results_dir', '${RESULTS_DIR}', '--output_dir', '${EVAL_DIR}', '--model', '${MODEL}'])
"
}

run_baselines() {
  require_goldens
  # Same pattern: single srun keeps Qdrant alive across all baselines.
  run_step "Run baselines + evaluate" \
    uv run python -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
sys.path.insert(0, '${REPO_DIR}/src')
from utils.qdrant import ensure_qdrant_runtime
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
     '--results_dir', '${RESULTS_DIR}', '--output_dir', '${EVAL_DIR}', '--model', '${MODEL}'])
"
}

if [[ "$GENERATE_ONLY" == "1" ]]; then
  generate_dataset
elif [[ "$ABLATION_ONLY" == "1" ]]; then
  run_ablation_variants
elif [[ "$BASELINES_ONLY" == "1" ]]; then
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
