# Run Guide

## Environment

Create and use the project virtual environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Config Presets

### Smoke Test

- Config: `config/smoke_test.yaml`
- Purpose: verify end-to-end execution on a very small subset
- Expected runtime: minutes

### Medium Validation

- Config: `config/medium_test.yaml`
- Purpose: check whether the models actually learn and beat the aliased baseline
- Scope: 96 patients, 24 central slices per patient

### Formal Training

- Config: `config/formal_train.yaml`
- Purpose: final all-patient experiment under current hardware limits
- Scope: all patients, 8 central slices per patient
- Reason for central-slice restriction: the full BraTS slice set is too large for practical turnaround on the current 8GB GPU, while all-patient central slices still provide a representative formal experiment

## Commands

### Task 1

```bash
.venv/bin/python tasks/task1_simulation.py
```

### Medium Validation

```bash
BMEAI_CONFIG=config/medium_test.yaml .venv/bin/python tasks/task2_train.py
BMEAI_CONFIG=config/medium_test.yaml .venv/bin/python tasks/task2_eval.py
BMEAI_CONFIG=config/medium_test.yaml .venv/bin/python tasks/task3_train.py
BMEAI_CONFIG=config/medium_test.yaml .venv/bin/python tasks/task3_eval.py
BMEAI_CONFIG=config/medium_test.yaml .venv/bin/python tasks/generate_report_assets.py
```

### Formal Training

```bash
BMEAI_CONFIG=config/formal_train.yaml bash scripts/run_formal_pipeline.sh
```

## Background Training With tmux

Example:

```bash
bash scripts/start_formal_tmux.sh
```

The formal pipeline script automatically runs Task 1, Task 2 train/eval, Task 3 train/eval, regenerates report assets, updates `REPORT.md`, and writes `run_metadata.json`.

Only the formal config updates the root `REPORT.md` automatically. Other configs write `report_snapshot.md` under their own `report_assets/` directory.

Use the status helper while the run is in progress or after completion:

```bash
bash scripts/check_formal_status.sh
```

## Recommendations

1. Use smoke test only for debugging, not for conclusions.
2. Use medium validation to confirm that `PSNR after > PSNR before` and `SSIM after > SSIM before`.
3. Do not start Task 3 long training before Task 2 is clearly working.
4. Keep `resume: true` so interrupted long runs can restart from the latest checkpoint.
5. Prefer all-patient central-slice training over a small number of full-volume patients when GPU time is limited.
6. Fill `config/report_metadata.json` before the final PDF export so the generated report contains the correct names and division of labor.

## Expected Outputs

- `task1/`: sampling mask and visualization
- `task2/`: checkpoint, final weights, loss curve, evaluation summary, reconstruction figure
- `task3/`: checkpoint, final weights, loss curve, evaluation summary, best/worst reconstruction figures, worst-case JSON
- `report_assets/`: split chart, metric comparison chart, summary tables, worst-case table
