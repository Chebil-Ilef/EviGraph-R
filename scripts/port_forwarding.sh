#!/bin/bash

set -e

GRAPH_DIR="${1:-_data/graphs}"
PORT="${2:-8888}"

echo "════════════════════════════════════════════════════════════════"
echo "  GRAPH VISUALIZATION SERVER"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Serving from: $GRAPH_DIR"
echo "🔌 Port: $PORT"
echo ""

if [ ! -d "$GRAPH_DIR" ]; then
    echo "❌ Error: Directory $GRAPH_DIR does not exist"
    exit 1
fi

cd "$GRAPH_DIR"

# Count HTML files
HTML_COUNT=$(find . -maxdepth 1 -name "*.html" | wc -l)
echo "📊 Found $HTML_COUNT HTML file(s)"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  NEXT STEPS:"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "1. Keep this terminal open (server is running)"
echo ""
echo "2. On YOUR LAPTOP, open a NEW terminal and run:"
echo ""
echo "   ssh -L $PORT:localhost:$PORT login1.capella.hpc.tu-dresden.de"
echo ""
echo "3. On YOUR LAPTOP, open a browser and go to:"
echo ""
echo "   http://localhost:$PORT"
echo ""
echo "4. Click any .html file to view the interactive graph"
echo ""
echo "5. Press Ctrl+C here to stop the server when done"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "🚀 Starting server..."
echo ""

python3 -m http.server "$PORT"
