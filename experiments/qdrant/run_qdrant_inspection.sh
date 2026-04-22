#!/bin/bash
#
# Quick wrapper to run Qdrant inspection with proper environment setup
# Usage: bash scripts/run_qdrant_inspection.sh [--sample 5000] [--collection unarxive_chunks]
#

# Find project root (go up 3 levels: experiments/qdrant -> EviGraph-R)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$REPO_DIR/../.." && pwd)"
cd "$REPO_DIR"

echo "Project root: $REPO_DIR"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

# Pass all arguments to the Python inspector
uv run python experiments/qdrant/qdrant_inspector.py "$@"
