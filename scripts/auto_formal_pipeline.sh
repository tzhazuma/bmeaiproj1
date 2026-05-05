#!/bin/bash
# Auto-run formal training pipeline end-to-end
set -e

cd /home/tzh03/bmeaiproj1
export BMEAI_CONFIG=config/formal_train.yaml
export PYTHON=/home/tzh03/myvenv/bin/python
export WRITE_MAIN_REPORT=1

LOGDIR=outputs_formal/logs
mkdir -p "$LOGDIR"

echo "[$(date)] === Starting Formal Training Pipeline ==="

echo "[$(date)] Task 2 training..."
$PYTHON tasks/task2_train.py > "$LOGDIR/task2_train.log" 2>&1
echo "[$(date)] Task 2 training done. Starting eval..."

$PYTHON tasks/task2_eval.py >> "$LOGDIR/task2_train.log" 2>&1
echo "[$(date)] Task 2 eval done. Starting Task 3 training..."

$PYTHON tasks/task3_train.py > "$LOGDIR/task3_train.log" 2>&1
echo "[$(date)] Task 3 training done. Starting Task 3 eval..."

$PYTHON tasks/task3_eval.py >> "$LOGDIR/task3_train.log" 2>&1
echo "[$(date)] Task 3 eval done. Generating report assets..."

$PYTHON tasks/generate_report_assets.py >> "$LOGDIR/task3_train.log" 2>&1
echo "[$(date)] Report assets done. Building final REPORT.md..."

$PYTHON tasks/build_report.py >> "$LOGDIR/report.log" 2>&1
echo "[$(date)] All formal pipeline stages completed!"
