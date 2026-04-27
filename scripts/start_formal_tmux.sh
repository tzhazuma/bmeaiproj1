#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG_PATH="${BMEAI_CONFIG:-config/formal_train.yaml}"
SESSION_NAME="${TMUX_SESSION_NAME:-brats-formal}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
RUN_TASK1="${RUN_TASK1:-1}"

OUTPUT_DIR="$($PYTHON_BIN -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_PATH')); print(cfg['output']['dir'])")"
OUTPUT_DIR="${OUTPUT_DIR#./}"
OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

LOG_PATH="$OUTPUT_DIR/logs/formal_pipeline.log"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "BMEAI_CONFIG='$CONFIG_PATH' PYTHON_BIN='$PYTHON_BIN' RUN_TASK1='$RUN_TASK1' bash scripts/run_formal_pipeline.sh"
echo "Started tmux session: $SESSION_NAME"
echo "Log file: $LOG_PATH"
echo "Attach with: tmux attach -t $SESSION_NAME"
