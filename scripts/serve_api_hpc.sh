set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT=8000
BG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --bg)   BG=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Load .env if present
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

export PYTHONPATH="$REPO_ROOT/src"
export INDEXING_PROFILE="${INDEXING_PROFILE:-hpc}"

echo "╔══════════════════════════════════════════╗"
echo "║        EviGraph-R API — HPC startup      ║"
echo "╚══════════════════════════════════════════╝"
echo "  profile : $INDEXING_PROFILE"
echo "  port    : $PORT"
echo "  python  : $(uv run python --version 2>&1)"
echo ""

CMD="uv run uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1"

if [[ $BG -eq 1 ]]; then
    LOG="$REPO_ROOT/logs/api.log"
    mkdir -p "$(dirname "$LOG")"
    echo "  mode    : background → $LOG"
    echo ""
    nohup $CMD >> "$LOG" 2>&1 &
    echo "API started (PID $!). Tail logs with:"
    echo "  tail -f $LOG"
else
    echo "  mode    : foreground (Ctrl+C to stop)"
    echo ""
    exec $CMD
fi
