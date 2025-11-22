#!/bin/bash
# Quick local test for Z3 autoformalization

set -euo pipefail

echo "========================================="
echo "Testing Z3 Autoformalization Locally"
echo "========================================="

# Check if API key is set
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "Error: GEMINI_API_KEY not set"
    echo "Please run: export GEMINI_API_KEY='your_key'"
    exit 1
fi

echo "API Key: ${GEMINI_API_KEY:0:20}..."
echo ""

# Test with just 2 examples
echo "Testing autoformalization agent with 2 examples..."

# Run with conda if available, otherwise direct python
if command -v conda &> /dev/null; then
    conda run -n folio-bench python main.py launch --max 2 --autoform-only
else
    python main.py launch --max 2 --autoform-only
fi

echo ""
echo "========================================="
echo "Local test complete!"
echo "========================================="

