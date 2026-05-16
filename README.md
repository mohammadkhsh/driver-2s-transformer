# StopGo Transformer

This repository contains the code and processed data for predicting human driver stop/go behavior at yellow onset and generating the resulting longitudinal acceleration trajectory.

The workflow has two stages.

1. **Stage 1** predicts the driver decision, either `stop` or `go`, from yellow-onset variables.
2. **Stage 2** uses an autoregressive Transformer to generate the signed acceleration trajectory. The rollout includes kinematic integration, terminal gating, actuator limits, and a jerk limiter so the generated trajectory stays physically usable.

The repository intentionally excludes raw logs, personal information, exploratory analysis scripts, Neural ODE baselines, and mixture-of-primitives baselines. The released dataset is refined and run-level. Heart-rate data are included only as whole-run summaries, not as time-series signals.

## Repository Layout

```text
.
├── data/
│   ├── refined_run_level_dataset.csv
│   └── processed_trajectory_cache/
│       ├── trajectory_arrays.npz
│       ├── trajectory_meta.csv
│       └── trajectory_cache_info.json
├── decision_mlp_stage1_ablation.py
├── trajectory_transformer_ar.py
├── trajectory_transformer_ar_ablation.py
├── trajectory_shared_utils.py
├── requirements.txt
└── results/
```

## Environment

Python 3.11 is recommended. CUDA is used automatically by PyTorch when available.

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPU support, install the CUDA build of PyTorch if your current PyTorch installation is CPU-only.

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check the active device with:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
```

## Data

The main public file is:

```text
data/refined_run_level_dataset.csv
```

Important columns include:

- `speed_at_yellow_kmh`
- `distance_threshold_m`
- `tti_s`
- `a_req_mps2`
- `go_decision`
- `stop_decision`
- `comfort_rating`
- `max_decel_abs_mps2`
- `hr_delta_bpm`
- `hr_delta_pct`

Stage 2 uses:

```text
data/processed_trajectory_cache/
```

This cache contains fixed-length acceleration, speed, and distance trajectories. It is included so the Transformer can be trained and validated without the raw per-sample logs.

## Stage 1 Stop/Go Decision Model

Stage 1 trains an MLP classifier for the stop/go decision. The default split is 60 percent training and 40 percent validation.

Run a random stratified split:

```bash
python decision_mlp_stage1_ablation.py \
  --split-mode run_stratified \
  --train-ratio 0.60 \
  --epochs 100 \
  --seed 42 \
  --out-dir results/stage1_run_stratified
```

Run a participant-disjoint split:

```bash
python decision_mlp_stage1_ablation.py \
  --split-mode participant_disjoint \
  --train-ratio 0.60 \
  --epochs 100 \
  --seed 42 \
  --out-dir results/stage1_participant_disjoint
```

Useful parameters:

- `--split-mode` chooses `run_stratified` or `participant_disjoint`.
- `--train-ratio` sets the training fraction.
- `--epochs` sets the Stage 1 training epochs.
- `--seed` fixes the random seed.
- `--force-cpu` disables CUDA.

## Stage 2 Autoregressive Transformer

Stage 2 trains the full pipeline. Stage 1 is trained first, and its predicted decision label and confidence are passed into the Transformer. Validation uses the deployed cascade behavior, meaning the trajectory model receives the predicted decision, not the ground-truth decision.

```bash
python trajectory_transformer_ar.py \
  --seed 42 \
  --train-ratio 0.60 \
  --decision-epochs 80 \
  --epochs 140 \
  --out-dir results/trajectory_transformer_ar
```

Useful parameters:

- `--decision-epochs` controls Stage 1 training inside the full pipeline.
- `--epochs` controls Stage 2 Transformer training.
- `--train-ratio` controls the 60/40 split.
- `--out-dir` controls where models, metrics, predictions, plots, and logs are written.
- `--force-cpu` runs without CUDA.

The main outputs are written to the selected output directory:

- `decision_metrics.json`
- `transformer_metrics.json`
- `validation_predictions.csv`
- `validation_trajectory_arrays.npz`
- `decision_mlp_state_dict.pt`
- `trajectory_transformer_state_dict.pt`
- `run_progress.log`
- `report.txt`

`run_progress.log` is updated during training, so it can be opened in an editor while the model runs.

## Ablation Runs

The ablation script evaluates selected input sets for Stage 2. It supports both random stratified and participant-disjoint validation.

```bash
python trajectory_transformer_ar_ablation.py \
  --seed 42 \
  --epochs 140 \
  --decision-epochs 80 \
  --split-modes run_stratified,participant_disjoint \
  --combo-preset curated_user \
  --out-dir results/trajectory_transformer_ar_ablation
```

For a quick smoke test:

```bash
python trajectory_transformer_ar_ablation.py \
  --seed 42 \
  --epochs 5 \
  --decision-epochs 5 \
  --max-combos 1 \
  --split-modes run_stratified \
  --out-dir results/smoke_test
```

## Updating the GitHub Repository

After changing files locally:

```bash
git status
git add README.md data/ decision_mlp_stage1_ablation.py trajectory_transformer_ar.py trajectory_transformer_ar_ablation.py trajectory_shared_utils.py requirements.txt .gitignore
git commit -m "Update public StopGo Transformer release"
git push
```

If you renamed the remote repository, first check the remote URL:

```bash
git remote -v
```

Then update it if needed:

```bash
git remote set-url origin https://github.com/mohammadkhsh/stopgo-transformer.git
git push -u origin main
```
