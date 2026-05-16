# Refined Run-Level Dataset

`refined_run_level_dataset.csv` contains one row per decision run used in the stop/go analysis (`n=392`). It is a sanitized run-level table derived from the raw experiment logs.

The dataset includes kinematic variables at yellow onset, stop/go labels, demographic group variables, comfort ratings where applicable, braking-severity summaries, and run-level heart-rate summaries only. Raw GNSS traces, raw acceleration time series, raw heart-rate time series, original filenames, and participant names are not included.

Heart-rate fields are aggregated over the whole run:

- `hr_min_bpm`
- `hr_max_bpm`
- `hr_delta_bpm = hr_max_bpm - hr_min_bpm`
- `hr_delta_pct = 100 * (hr_max_bpm - hr_min_bpm) / hr_min_bpm`

Comfort ratings are valid only for stop-valid runs.
