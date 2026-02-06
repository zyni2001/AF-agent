#!/bin/bash
# Entrypoint script that translates docker-compose command args
# (--host, --port, --card-url) into environment variables for
# agentbeats run_ctrl, which reads HOST, PORT, CLOUDRUN_HOST from env.

while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      export HOST="$2"
      shift 2
      ;;
    --port)
      export PORT="$2"
      shift 2
      ;;
    --card-url)
      # Extract host:port from URL like http://green-agent:9009
      CARD_URL="$2"
      HOST_PORT="${CARD_URL#*://}"
      export CLOUDRUN_HOST="$HOST_PORT"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

exec agentbeats run_ctrl

