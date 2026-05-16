# Data files

This directory contains the public processed data used by the released prediction pipeline.

## `refined_run_level_dataset.csv`

One row corresponds to one decision run. The file includes the yellow-onset kinematic variables, stop/go labels, participant age/sex metadata, comfort rating when valid, and whole-run heart-rate summaries. It does not include raw GNSS fixes, raw heart-rate time series, or personally identifying information.

Heart-rate fields are stored only as run-level summaries:

- `hr_min_bpm`
- `hr_max_bpm`
- `hr_delta_bpm = hr_max_bpm - hr_min_bpm`
- `hr_delta_pct = 100 * hr_delta_bpm / hr_min_bpm`

Comfort ratings are valid only for stop-valid runs.

## `processed_trajectory_cache/`

Preprocessed fixed-length trajectories used by Stage 2. This cache contains acceleration, speed, and distance sequences aligned from yellow onset to the detected event, plus sanitized trajectory metadata. It lets the Transformer training scripts run without publishing the raw per-sample logs.
