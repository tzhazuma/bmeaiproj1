#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG_PATH="${BMEAI_CONFIG:-config/formal_train.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
RUN_TASK1="${RUN_TASK1:-1}"

OUTPUT_DIR="$($PYTHON_BIN -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_PATH')); print(cfg['output']['dir'])")"
OUTPUT_DIR="${OUTPUT_DIR#./}"
OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/report_assets" "$OUTPUT_DIR/logs"

LOG_PATH="$OUTPUT_DIR/logs/formal_pipeline.log"
RUN_METADATA_PATH="$OUTPUT_DIR/run_metadata.json"
exec > >(tee -a "$LOG_PATH") 2>&1

START_TS="$(date -Iseconds)"
START_UNIX="$(date +%s)"
CURRENT_STAGE="not_started"
LAST_COMPLETED_STAGE="none"

TASK1_DURATION=""
TASK2_TRAIN_DURATION=""
TASK2_EVAL_DURATION=""
TASK3_TRAIN_DURATION=""
TASK3_EVAL_DURATION=""
ASSETS_DURATION=""

write_metadata() {
  local status="$1"
  local end_ts
  local end_unix
  end_ts="$(date -Iseconds)"
  end_unix="$(date +%s)"
  RUN_STATUS="$status" \
  START_TS="$START_TS" \
  END_TS="$end_ts" \
  START_UNIX="$START_UNIX" \
  END_UNIX="$end_unix" \
  CURRENT_STAGE="$CURRENT_STAGE" \
  LAST_COMPLETED_STAGE="$LAST_COMPLETED_STAGE" \
  LOG_PATH="$LOG_PATH" \
  CONFIG_PATH="$CONFIG_PATH" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  TASK1_DURATION="$TASK1_DURATION" \
  TASK2_TRAIN_DURATION="$TASK2_TRAIN_DURATION" \
  TASK2_EVAL_DURATION="$TASK2_EVAL_DURATION" \
  TASK3_TRAIN_DURATION="$TASK3_TRAIN_DURATION" \
  TASK3_EVAL_DURATION="$TASK3_EVAL_DURATION" \
  ASSETS_DURATION="$ASSETS_DURATION" \
  RUN_METADATA_PATH="$RUN_METADATA_PATH" \
  $PYTHON_BIN -c "import json, os; durations = {}; keys = ['task1','task2_train','task2_eval','task3_train','task3_eval','assets']; env_keys = ['TASK1_DURATION','TASK2_TRAIN_DURATION','TASK2_EVAL_DURATION','TASK3_TRAIN_DURATION','TASK3_EVAL_DURATION','ASSETS_DURATION'];
for key, env_key in zip(keys, env_keys):
    value = os.environ.get(env_key, '')
    durations[key] = int(value) if value else None
data = {
    'status': os.environ['RUN_STATUS'],
    'config_path': os.environ['CONFIG_PATH'],
    'output_dir': os.environ['OUTPUT_DIR'],
    'log_path': os.environ['LOG_PATH'],
    'started_at': os.environ['START_TS'],
    'ended_at': os.environ['END_TS'],
    'current_stage': os.environ['CURRENT_STAGE'],
    'last_completed_stage': os.environ['LAST_COMPLETED_STAGE'],
    'total_duration_seconds': int(os.environ['END_UNIX']) - int(os.environ['START_UNIX']),
    'stage_durations_seconds': durations,
}
with open(os.environ['RUN_METADATA_PATH'], 'w') as f:
    json.dump(data, f, indent=2)"
}

finalize() {
  local exit_code="$1"
  if [[ "$exit_code" -eq 0 ]]; then
    write_metadata "success"
  else
    write_metadata "failed"
  fi

  BMEAI_CONFIG="$CONFIG_PATH" "$PYTHON_BIN" tasks/generate_report_assets.py || true
  BMEAI_CONFIG="$CONFIG_PATH" "$PYTHON_BIN" tasks/build_report.py || true
}

trap 'exit_code=$?; finalize "$exit_code"; exit "$exit_code"' EXIT

run_stage() {
  local stage_name="$1"
  shift
  local stage_start
  local stage_end
  CURRENT_STAGE="$stage_name"
  echo "[$(date -Iseconds)] Starting $stage_name"
  stage_start="$(date +%s)"
  "$@"
  stage_end="$(date +%s)"
  LAST_COMPLETED_STAGE="$stage_name"
  CURRENT_STAGE="idle"
  local duration=$((stage_end - stage_start))
  case "$stage_name" in
    task1) TASK1_DURATION="$duration" ;;
    task2_train) TASK2_TRAIN_DURATION="$duration" ;;
    task2_eval) TASK2_EVAL_DURATION="$duration" ;;
    task3_train) TASK3_TRAIN_DURATION="$duration" ;;
    task3_eval) TASK3_EVAL_DURATION="$duration" ;;
    assets) ASSETS_DURATION="$duration" ;;
  esac
  echo "[$(date -Iseconds)] Finished $stage_name in ${duration}s"
}

if [[ "$RUN_TASK1" == "1" ]]; then
  run_stage task1 env BMEAI_CONFIG="$CONFIG_PATH" "$PYTHON_BIN" tasks/task1_simulation.py
fi
run_stage task2_train env BMEAI_CONFIG="$CONFIG_PATH" "$PYTHON_BIN" tasks/task2_train.py
run_stage task2_eval env BMEAI_CONFIG="$CONFIG_PATH" "$PYTHON_BIN" tasks/task2_eval.py
run_stage task3_train env BMEAI_CONFIG="$CONFIG_PATH" "$PYTHON_BIN" tasks/task3_train.py
run_stage task3_eval env BMEAI_CONFIG="$CONFIG_PATH" "$PYTHON_BIN" tasks/task3_eval.py
run_stage assets env BMEAI_CONFIG="$CONFIG_PATH" "$PYTHON_BIN" tasks/generate_report_assets.py

echo "[$(date -Iseconds)] Formal pipeline finished successfully"
