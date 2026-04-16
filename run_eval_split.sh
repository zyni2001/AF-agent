#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <refined|original> [log_file]"
  exit 1
fi

SPLIT="$1"
LOG_FILE="${2:-experiment_logs/${SPLIT}_full.log}"
METRICS_FILE="experiment_logs/${SPLIT}_metrics.json"
CHECKPOINT_DIR="experiment_logs/${SPLIT}_checkpoints"

case "$SPLIT" in
  refined|cleaned)
    export FOLIO_DATASET_VARIANT="refined"
    ;;
  original|dev)
    export FOLIO_DATASET_VARIANT="original"
    ;;
  *)
    echo "Unsupported split: $SPLIT"
    echo "Use one of: refined, original"
    exit 1
    ;;
esac

mkdir -p "$(dirname "$LOG_FILE")"

VERTEX_ENV_FILE="${HOME}/.config/vertex-ai/env.zsh"
if [ -f "$VERTEX_ENV_FILE" ]; then
  # Load local Vertex AI defaults if available.
  # shellcheck disable=SC1090
  source "$VERTEX_ENV_FILE"
fi

export FOLIO_VERBOSE_LOGGING="${FOLIO_VERBOSE_LOGGING:-0}"
export FOLIO_METRICS_PATH="${FOLIO_METRICS_PATH:-$METRICS_FILE}"
export FOLIO_CHECKPOINT_DIR="${FOLIO_CHECKPOINT_DIR:-$CHECKPOINT_DIR}"
export FOLIO_MAX_CONCURRENT="${FOLIO_MAX_CONCURRENT:-4}"
export FOLIO_A2A_TIMEOUT_SECONDS="${FOLIO_A2A_TIMEOUT_SECONDS:-300}"
export PYTHONUNBUFFERED=1

echo "Running full evaluation on split: $FOLIO_DATASET_VARIANT"
echo "Log file: $LOG_FILE"
echo "Metrics file: $FOLIO_METRICS_PATH"
echo "Checkpoint dir: $FOLIO_CHECKPOINT_DIR"
echo "Max concurrent requests: $FOLIO_MAX_CONCURRENT"
echo "A2A timeout seconds: $FOLIO_A2A_TIMEOUT_SECONDS"

.venv/bin/python main.py full 2>&1 | tee "$LOG_FILE"
