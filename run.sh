#!/usr/bin/env bash
# Agent launcher script
# Can be called by AgentBeats controller (run_ctrl) or directly

set -euo pipefail

# Read environment variables
# PORT is set by Cloud Run, AGENT_PORT may be set by controller
HOST=${HOST:-0.0.0.0}
PORT=${AGENT_PORT:-${PORT:-8080}}
ROLE=${AGENT_ROLE:-green}

echo "====================================="
echo "Starting ${ROLE} agent"
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "PUBLIC_URL: ${PUBLIC_URL:-not set}"
echo "AGENT_URL: ${AGENT_URL:-not set}"
echo "====================================="
echo "ALL ENVIRONMENT VARIABLES (for debugging):"
env | grep -i "agent\|url\|port\|host\|cagent" | sort || true
echo "====================================="

# Launch the appropriate agent based on AGENT_ROLE
case "${ROLE}" in
  green)
    echo "Launching Green Agent (FOLIO Evaluator)..."
    exec python -c "import os,sys; sys.path.insert(0,'.'); from src.green_agent.agent import start_green_agent; start_green_agent(host='${HOST}', port=${PORT})"
    ;;
  baseline)
    echo "Launching Baseline White Agent..."
    echo "Environment check:"
    echo "  AGENT_ROLE=${ROLE}"
    echo "  HOST=${HOST}"
    echo "  PORT=${PORT}"
    echo "  GEMINI_API_KEY=${GEMINI_API_KEY:+set}"
    exec python -c "import os,sys; sys.path.insert(0,'.'); from src.white_agent_baseline.agent import start_baseline_white_agent; start_baseline_white_agent(host='${HOST}', port=${PORT})"
    ;;
  autoform)
    echo "Launching Z3 Autoformalization White Agent..."
    exec python -c "import os,sys; sys.path.insert(0,'.'); from src.white_agent_autoform.agent import start_autoform_white_agent; start_autoform_white_agent(host='${HOST}', port=${PORT})"
    ;;
  *)
    echo "ERROR: Unknown AGENT_ROLE='${ROLE}'" >&2
    echo "Valid roles: green, baseline, autoform" >&2
    exit 1
    ;;
esac

