#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG_PATH="${BMEAI_CONFIG:-config/formal_train.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

OUTPUT_DIR="$($PYTHON_BIN -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_PATH')); print(cfg['output']['dir'])")"
OUTPUT_DIR="${OUTPUT_DIR#./}"
OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"

METADATA_PATH="$OUTPUT_DIR/run_metadata.json"

if [[ ! -f "$METADATA_PATH" ]]; then
  echo "No run metadata found at $METADATA_PATH"
  exit 0
fi

$PYTHON_BIN -c "import json; data=json.load(open('$METADATA_PATH')); print(json.dumps(data, indent=2))"
