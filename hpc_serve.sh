#!/bin/bash
# serve_hpc.sh — HPC replacement for docker-compose up
#
# Runs qdrant + EviGraph-R API as Singularity containers.
# Works both interactively and as a SLURM job.
#
# Usage:
#   ./hpc_serve.sh                         # submit SLURM job (default 24 h)
#   ./hpc_serve.sh --interactive            # foreground, no SLURM
#   ./hpc_serve.sh --time 48:00:00          # custom wall time
#   ./hpc_serve.sh --port 8000              # API port (default 8000)
#   ./hpc_serve.sh --qdrant-port 6333       # Qdrant port (default 6333)
#   ./hpc_serve.sh --rebuild                # force rebuild of api.sif
#
# The script mirrors docker-compose.yml exactly:
#   qdrant/qdrant:v1.13.6  →  qdrant.sif      (pulled on first run)
#   Dockerfile (build: .)  →  python.sif       (python:3.11-slim pulled on first run)
#                              + repo bind-mounted at runtime (no SIF build needed)

set -euo pipefail

API_PORT=8000
QDRANT_PORT=6333
WALLTIME=24:00:00
INTERACTIVE=false
REBUILD=false
PARTITION="${SLURM_PARTITION:-capella}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${REPO_ROOT}/.hpc_run"
QDRANT_SIF="${REPO_ROOT}/qdrant.sif"
PYTHON_SIF="${REPO_ROOT}/python311.sif"

die()  { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[serve_hpc] $*"; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)          API_PORT="$2";    shift 2 ;;
        --port=*)        API_PORT="${1#*=}"; shift ;;
        --qdrant-port)   QDRANT_PORT="$2"; shift 2 ;;
        --qdrant-port=*) QDRANT_PORT="${1#*=}"; shift ;;
        --time)          WALLTIME="$2";    shift 2 ;;
        --time=*)        WALLTIME="${1#*=}"; shift ;;
        --partition)     PARTITION="$2";   shift 2 ;;
        --partition=*)   PARTITION="${1#*=}"; shift ;;
        --interactive)   INTERACTIVE=true; shift ;;
        --rebuild)       REBUILD=true;     shift ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# 1. Qdrant SIF 
if [[ ! -f "${QDRANT_SIF}" ]]; then
    info "Pulling qdrant/qdrant:v1.13.6 → ${QDRANT_SIF} ..."
    singularity pull "${QDRANT_SIF}" docker://qdrant/qdrant:v1.13.6
    info "✓ qdrant.sif ready"
fi

# 2. Python SIF — pull python:3.11-slim once; repo is bind-mounted at runtime (no build needed)
if [[ "${REBUILD}" == true ]] || [[ ! -f "${PYTHON_SIF}" ]]; then
    info "Pulling python:3.11-slim → ${PYTHON_SIF} ..."
    singularity pull "${PYTHON_SIF}" docker://python:3.11-slim
    info "✓ python311.sif ready"
fi

# 3. Runtime directories 
mkdir -p "${RUN_DIR}/logs"

# 4. Core launch functions 

run_qdrant() {
    local storage_dir="${EVI_QDRANT_STORAGE_DIR:-${REPO_ROOT}/storage}"
    local snapshots_dir="${EVI_QDRANT_SNAPSHOTS_DIR:-${REPO_ROOT}/snapshots}"
    mkdir -p "${storage_dir}" "${snapshots_dir}"

    QDRANT_PORT="${QDRANT_PORT}" singularity exec \
        --bind "${storage_dir}:/qdrant/storage" \
        --bind "${snapshots_dir}:/qdrant/snapshots" \
        --writable-tmpfs \
        "${QDRANT_SIF}" \
        /bin/sh -c "QDRANT__SERVICE__HTTP_PORT=${QDRANT_PORT} /qdrant/qdrant"
}

run_api() {
    local hub_dir="${HF_HOME:-${REPO_ROOT}/hub}"
    mkdir -p "${hub_dir}"

    # Bind the whole repo into /app so the existing .venv and src are available.
    # This is the equivalent of `docker build .` + `docker run` but without needing
    # fakeroot or any SIF build step.
    singularity exec \
        --bind "${REPO_ROOT}:/app" \
        --bind "${hub_dir}:/app/hub" \
        --env "INDEXING_PROFILE=hpc" \
        --env "QDRANT_URL=http://localhost:${QDRANT_PORT}" \
        --env "PYTHONPATH=/app/src" \
        --env "HF_HOME=/app/hub" \
        --writable-tmpfs \
        "${PYTHON_SIF}" \
        /bin/sh -c "
            cd /app
            echo '[api] waiting for qdrant on port ${QDRANT_PORT}...'
            for i in \$(seq 1 30); do
                curl -sf http://localhost:${QDRANT_PORT}/healthz >/dev/null 2>&1 && break
                sleep 2
            done
            curl -sf http://localhost:${QDRANT_PORT}/healthz >/dev/null 2>&1 \
                || { echo '[api] qdrant never became healthy — aborting'; exit 1; }
            echo '[api] qdrant healthy, starting API on port ${API_PORT}'
            /app/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT} --workers 1
        "
}

# 5. Launch

# Load .env so run_qdrant / run_api pick up EVI_QDRANT_STORAGE_DIR etc.
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi

if [[ "${INTERACTIVE}" == true ]]; then
    info "Starting qdrant on port ${QDRANT_PORT} ..."
    run_qdrant >> "${RUN_DIR}/logs/qdrant.log" 2>&1 &
    QDRANT_PID=$!
    trap "kill ${QDRANT_PID} 2>/dev/null; exit" INT TERM EXIT

    info "Starting API on port ${API_PORT} ..."
    echo ""
    echo "  API URL:    http://$(hostname):${API_PORT}"
    echo "  SSH tunnel: ssh -L ${API_PORT}:$(hostname):${API_PORT} <hpc-login>"
    echo "  Qdrant log: ${RUN_DIR}/logs/qdrant.log"
    echo ""
    run_api  # foreground — Ctrl+C stops everything via trap

else
    command -v sbatch &>/dev/null || die "sbatch not found — use --interactive on non-SLURM systems"
    info "Submitting SLURM job (partition: ${PARTITION}, wall time: ${WALLTIME}) ..."

    # Export variables that the wrapped functions need
    export QDRANT_SIF PYTHON_SIF REPO_ROOT RUN_DIR QDRANT_PORT API_PORT
    export EVI_QDRANT_STORAGE_DIR="${EVI_QDRANT_STORAGE_DIR:-${REPO_ROOT}/storage}"
    export EVI_QDRANT_SNAPSHOTS_DIR="${EVI_QDRANT_SNAPSHOTS_DIR:-${REPO_ROOT}/snapshots}"
    export HF_HOME="${HF_HOME:-${REPO_ROOT}/hub}"

    JOB_ID=$(sbatch \
        --job-name=evigraph-serve \
        --partition="${PARTITION}" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --gpus=1 \
        --time="${WALLTIME}" \
        --output="${RUN_DIR}/logs/slurm-%j.out" \
        --error="${RUN_DIR}/logs/slurm-%j.err" \
        --wrap="
            set -euo pipefail
            $(declare -f run_qdrant)
            $(declare -f run_api)

            # Source .env inside the job
            if [[ -f '${REPO_ROOT}/.env' ]]; then
                set -a; source '${REPO_ROOT}/.env'; set +a
            fi

            echo '[slurm] starting qdrant...'
            run_qdrant >> '${RUN_DIR}/logs/qdrant.log' 2>&1 &
            QDRANT_PID=\$!
            trap 'kill \$QDRANT_PID 2>/dev/null' EXIT

            echo '[slurm] starting api...'
            run_api >> '${RUN_DIR}/logs/api.log' 2>&1
        " \
        | grep -oP '\d+')

    echo ""
    echo "  Job ID:    ${JOB_ID}"
    echo "  Logs:      ${RUN_DIR}/logs/slurm-${JOB_ID}.{out,err}"
    echo "             ${RUN_DIR}/logs/qdrant.log"
    echo "             ${RUN_DIR}/logs/api.log"
    echo "  Monitor:   squeue -j ${JOB_ID}"
    echo "  Cancel:    scancel ${JOB_ID}"
    echo ""
    echo "  Waiting for job to start..."

    STATE=""
    NODE=""
    for _ in $(seq 1 30); do
        STATE=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || echo "")
        NODE=$(squeue  -j "${JOB_ID}" -h -o "%N" 2>/dev/null || echo "")
        [[ "${STATE}" == "RUNNING" && -n "${NODE}" ]] && break
        sleep 2
    done

    if [[ "${STATE}" == "RUNNING" ]]; then
        echo ""
        echo "  ┌──────────────────────────────────────────────────────────────┐"
        echo "  │  API URL:    http://${NODE}:${API_PORT}                      "
        echo "  │  SSH tunnel: ssh -L ${API_PORT}:${NODE}:${API_PORT} $(whoami)@$(hostname --fqdn 2>/dev/null || hostname)"
        echo "  └──────────────────────────────────────────────────────────────┘"
    else
        echo "  Job still queued — once running check: squeue -j ${JOB_ID}"
        echo "  Then SSH tunnel: ssh -L ${API_PORT}:<node>:${API_PORT} $(whoami)@$(hostname)"
    fi
fi
