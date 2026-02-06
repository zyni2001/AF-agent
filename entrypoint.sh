#!/bin/bash
# Smart entrypoint that handles two modes:
#
# 1. Docker-compose mode (GitHub Actions):
#    Receives --host, --port, --card-url args from generate_compose.py.
#    Runs the agent DIRECTLY via run.sh so /.well-known/agent-card.json
#    is served at the root (required by docker-compose health checks).
#
# 2. Cloud Run mode (no args):
#    Runs agentbeats run_ctrl which manages the agent lifecycle and
#    proxies requests via /to_agent/<id>/...

if [ $# -gt 0 ]; then
    # Docker-compose mode: parse args, run agent directly
    while [[ $# -gt 0 ]]; do
        case $1 in
            --host)
                export HOST="$2"
                shift 2
                ;;
            --port)
                export AGENT_PORT="$2"
                export PORT="$2"
                shift 2
                ;;
            --card-url)
                # e.g. http://baseline-agent:9009 — used as the agent card URL
                export AGENT_URL="$2"
                export PUBLIC_URL="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    echo "=== Docker-compose mode ==="
    echo "HOST=$HOST PORT=$PORT AGENT_PORT=$AGENT_PORT"
    echo "AGENT_URL=$AGENT_URL"
    echo "==========================="
    exec ./run.sh
else
    # Cloud Run mode: use agentbeats controller
    exec agentbeats run_ctrl
fi
