# Human Driver Stopping Behaviour Modeling

This repository contains the code and sanitized run-level dataset for modeling human driver stop/go decisions and braking trajectories at yellow-light onset. The main pipeline combines run-level decision analysis, comfort and heart-rate interpretation, and a kinematic-constrained autoregressive Transformer for signed acceleration-trajectory generation.

Raw participant logs, high-frequency GNSS/acceleration traces, raw heart-rate time series, generated figures, trained weights, and paper files are intentionally excluded.

## What Is Included

- A sanitized refined run-level dataset with 392 decision runs.
- Run-level heart-rate summaries only, including minimum HR, maximum HR, absolute HR elevation, and percentage HR elevation over the whole run.
- Stop/go decision analysis scripts.
- Comfort and heart-rate analysis scripts.
- Dataset/statistics figure generators.
- The two-stage decision plus autoregressive Transformer trajectory pipeline.
- Ablation and visualization scripts for the Transformer pipeline.

## Repository Structure

```text
.
├── data/
│   ├── refined_run_level_dataset.csv     # Sanitized run-level dataset
│   └── README.md                         # Dataset field notes
├── results/README.md                     # Placeholder for generated outputs
├── analysis_decision.py                  # Stop/go decision modeling and decision-surface figures
├── analysis_hr_comfort.py                # Comfort, HR-delta, and braking-severity analysis
├── generate_data_collection_section.py   # Dataset/statistics figures
├── paper_decision_figures.py             # AME, diagnostics, and decision-surface figures
├── decision_mlp_stage1_ablation.py       # Stage-1 decision ablations
├── trajectory_shared_utils.py            # Shared extraction, preprocessing, and Stage-1 utilities
├── trajectory_transformer_ar.py          # Main two-stage AR Transformer trajectory pipeline
├── trajectory_transformer_ar_ablation.py # Curated Transformer input-modality ablations
├── plot_transformer_correct12_examples.py
├── plot_transformer_misclassified_examples.py
├── plot_transformer_go_cross_fail_examples.py
├── run_stop_aligned_reruns.py
└── requirements.txt
```

## Environment Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch wheel matching your system. For CUDA 12.1, use:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

Verify CUDA availability:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
```

## Dataset

The public dataset is located at:

```text
data/refined_run_level_dataset.csv
```

Each row corresponds to one decision run. The table includes yellow-onset kinematics, stop/go labels, demographic fields, comfort rating where valid, peak braking summaries, and whole-run HR summaries. Raw time-series HR values are not included.

Important fields include:

- `distance_threshold_m`
- `speed_at_yellow_kmh`
- `tti_s`
- `a_req_mps2`
- `go_decision`
- `stop_decision`
- `comfort_rating`
- `max_decel_abs_mps2`
- `hr_delta_bpm`
- `hr_delta_pct`

Comfort ratings are valid only for stop-valid runs.

## Main Workflow

Generate data-collection statistics:

```bash
python generate_data_collection_section.py
```

Analyze stop/go decision structure:

```bash
python analysis_decision.py
python paper_decision_figures.py
```

Analyze comfort and run-level HR elevation:

```bash
python analysis_hr_comfort.py
```

Train the two-stage Transformer trajectory model:

```bash
python trajectory_transformer_ar.py
```

Run curated Transformer ablations:

```bash
python trajectory_transformer_ar_ablation.py
```

Plot representative Transformer rollouts:

```bash
python plot_transformer_correct12_examples.py
```

## Model Summary

Stage 1 predicts the binary stop/go decision at yellow onset. Stage 2 uses this predicted decision and confidence, together with kinematic state and context, to autoregressively generate signed longitudinal acceleration. The raw Transformer output is passed through terminal gating, jerk limiting, actuator clamping, and discrete kinematic integration so that generated trajectories remain physically plausible.

The default Transformer uses an 11-dimensional token, a 96-dimensional embedding, four attention heads, and a per-head dimension of 24. The post-processing pipeline enforces bounded jerk and stop-aware terminal behavior.

## Reproducibility Notes

- The public dataset is run-level and sanitized.
- Raw logs are not required for using the released dataset, but full trajectory retraining from raw time series requires the original private logs.
- The scripts write generated outputs to `results/`.
- Participant-disjoint validation and input-modality ablation are supported.

## License

No license is assigned yet. Add a `LICENSE` file before making the repository public if reuse terms should be explicit.

## Citation

If this code or dataset is used in a publication, cite the associated paper once available.
