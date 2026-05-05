#!/bin/bash
# Auto-run remaining medium test pipeline after Task 2 training completes
set -e

cd /home/tzh03/bmeaiproj1
export BMEAI_CONFIG=config/medium_test.yaml
export PYTHON=/home/tzh03/myvenv/bin/python

echo "[$(date)] Waiting for Task 2 training to complete..."
while pgrep -f "task2_train.py" > /dev/null; do
    sleep 30
done
echo "[$(date)] Task 2 training done. Starting eval..."

$PYTHON tasks/task2_eval.py >> outputs_medium/logs/task2_train.log 2>&1
echo "[$(date)] Task 2 eval done. Starting Task 3 training..."

$PYTHON tasks/task3_train.py > outputs_medium/logs/task3_train.log 2>&1
echo "[$(date)] Task 3 training done. Starting Task 3 eval..."

$PYTHON tasks/task3_eval.py >> outputs_medium/logs/task3_train.log 2>&1
echo "[$(date)] Task 3 eval done. Generating report assets..."

$PYTHON tasks/generate_report_assets.py >> outputs_medium/logs/task3_train.log 2>&1
echo "[$(date)] All done!"

# Update REPORT.md with medium results
$PYTHON tasks/build_report.py >> outputs_medium/logs/report.log 2>&1 || true
echo "[$(date)] Report updated."
