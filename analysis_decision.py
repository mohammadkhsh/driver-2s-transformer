from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from patsy import build_design_matrices
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


STOP_DISTANCE_MIN_M = -2.0
STOP_DISTANCE_MAX_M = 5.0
STOP_SPEED_MAX_KMH = 0.2


def to_float(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == "":
        return np.nan
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_year(value: Any) -> float:
    text = str(value).strip()
    match = re.search(r"(19|20)\d{2}", text)
    if not match:
        return np.nan
    return float(match.group(0))


def normalize_text(value: Any, default: str = "unknown") -> str:
    text = str(value).strip().lower()
    return text if text else default


def parse_filename_timestamp(file_name: str) -> pd.Timestamp:
    match = re.search(r"(20\d{6}-\d{6})", file_name)
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group(1), format="%Y%m%d-%H%M%S", errors="coerce")


def parse_participant_group(participant_id: str, driver_name: str, file_name: str) -> str:
    candidates = [participant_id, driver_name, file_name]
    for candidate in candidates:
        match = re.search(r"\d+", str(candidate))
        if match:
            return match.group(0)
    return str(participant_id).strip() or Path(file_name).stem


def parse_speed_distance_from_filename(file_name: str) -> tuple[float, float]:
    match = re.search(
        r"_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)_20\d{6}-\d{6}\.csv$",
        file_name,
    )
    if not match:
        return np.nan, np.nan
    return float(match.group(2)), float(match.group(1))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def first_level_crossing(x: np.ndarray, y: np.ndarray, level: float = 0.5) -> float:
    if y.size < 2:
        return np.nan
    if np.all(y < level) or np.all(y > level):
        return np.nan

    exact = np.where(np.isclose(y, level, atol=1e-10))[0]
    if exact.size > 0:
        return float(x[int(exact[0])])

    diff = y - level
    cross_idx = np.where(diff[:-1] * diff[1:] < 0)[0]
    if cross_idx.size == 0:
        return np.nan

    i = int(cross_idx[0])
    x1, x2 = float(x[i]), float(x[i + 1])
    y1, y2 = float(y[i]), float(y[i + 1])
    if y2 == y1:
        return float(0.5 * (x1 + x2))
    return float(x1 + (level - y1) * (x2 - x1) / (y2 - y1))


def estimate_yellow_onset_state(fix: pd.DataFrame, distance_threshold_m: float) -> dict[str, Any]:
    result = {
        "speed_kmh": np.nan,
        "time_value": np.nan,
        "time_source": "unknown",
        "method": "missing",
        "alpha": np.nan,
        "row0": -1,
        "row1": -1,
        "d0": np.nan,
        "d1": np.nan,
        "s0": np.nan,
        "s1": np.nan,
        "distance_abs_error_m": np.nan,
    }
    if fix.empty or not np.isfinite(distance_threshold_m):
        return result

    d = fix["light_distance_m"].to_numpy(dtype=float)
    s = fix["speed_kmh"].to_numpy(dtype=float)

    t_source = "row_index"
    if "t_wall" in fix.columns and np.isfinite(fix["t_wall"].to_numpy(dtype=float)).sum() >= 2:
        t = fix["t_wall"].to_numpy(dtype=float)
        t_source = "t_wall"
    elif "ticks_utc" in fix.columns and np.isfinite(fix["ticks_utc"].to_numpy(dtype=float)).sum() >= 2:
        t = fix["ticks_utc"].to_numpy(dtype=float)
        t_source = "ticks_utc"
    else:
        t = np.arange(len(fix), dtype=float)
    result["time_source"] = t_source

    for i in range(len(fix) - 1):
        d0, d1 = d[i], d[i + 1]
        s0, s1 = s[i], s[i + 1]
        t0, t1 = t[i], t[i + 1]
        if not (np.isfinite(d0) and np.isfinite(d1) and np.isfinite(s0) and np.isfinite(s1) and np.isfinite(t0) and np.isfinite(t1)):
            continue

        if np.isclose(d0, distance_threshold_m, atol=1e-10):
            result.update(
                {
                    "speed_kmh": float(s0),
                    "time_value": float(t0),
                    "method": "exact_sample",
                    "alpha": 0.0,
                    "row0": int(i),
                    "row1": int(i),
                    "d0": float(d0),
                    "d1": float(d0),
                    "s0": float(s0),
                    "s1": float(s0),
                    "distance_abs_error_m": 0.0,
                }
            )
            return result

        if np.isclose(d1, distance_threshold_m, atol=1e-10):
            result.update(
                {
                    "speed_kmh": float(s1),
                    "time_value": float(t1),
                    "method": "exact_sample",
                    "alpha": 1.0,
                    "row0": int(i + 1),
                    "row1": int(i + 1),
                    "d0": float(d1),
                    "d1": float(d1),
                    "s0": float(s1),
                    "s1": float(s1),
                    "distance_abs_error_m": 0.0,
                }
            )
            return result

        if (d0 - distance_threshold_m) * (d1 - distance_threshold_m) < 0:
            if d1 == d0:
                alpha = 0.5
            else:
                alpha = float((distance_threshold_m - d0) / (d1 - d0))
            alpha = float(np.clip(alpha, 0.0, 1.0))
            t_cross = float(t0 + alpha * (t1 - t0))
            s_cross = float(s0 + alpha * (s1 - s0))
            result.update(
                {
                    "speed_kmh": s_cross,
                    "time_value": t_cross,
                    "method": "linear_interpolation",
                    "alpha": alpha,
                    "row0": int(i),
                    "row1": int(i + 1),
                    "d0": float(d0),
                    "d1": float(d1),
                    "s0": float(s0),
                    "s1": float(s1),
                    "distance_abs_error_m": 0.0,
                }
            )
            return result

    finite = np.isfinite(d) & np.isfinite(s) & np.isfinite(t)
    if finite.any():
        idxs = np.where(finite)[0]
        nearest = int(idxs[np.argmin(np.abs(d[idxs] - distance_threshold_m))])
        result.update(
            {
                "speed_kmh": float(s[nearest]),
                "time_value": float(t[nearest]),
                "method": "nearest_sample_fallback",
                "alpha": np.nan,
                "row0": nearest,
                "row1": nearest,
                "d0": float(d[nearest]),
                "d1": float(d[nearest]),
                "s0": float(s[nearest]),
                "s1": float(s[nearest]),
                "distance_abs_error_m": float(abs(d[nearest] - distance_threshold_m)),
            }
        )
    return result


def parse_one_log(file_path: Path) -> tuple[dict[str, Any] | None, pd.DataFrame | None, str]:
    try:
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    except Exception as exc:
        return None, None, f"read_error:{exc}"

    required = {"type", "speed_kmh", "light_distance_m"}
    if not required.issubset(set(df.columns)):
        return None, None, "missing_required_columns"

    meta_rows = df[df["type"].str.lower() == "meta"]
    if meta_rows.empty:
        return None, None, "missing_meta"
    meta = meta_rows.iloc[0]

    fix = df[df["type"].str.lower() == "fix"].copy()
    if fix.empty:
        return None, None, "missing_fix_rows"

    fix["speed_kmh"] = pd.to_numeric(fix["speed_kmh"], errors="coerce")
    fix["light_distance_m"] = pd.to_numeric(fix["light_distance_m"], errors="coerce")
    fix["t_wall"] = pd.to_numeric(fix.get("t_wall", np.nan), errors="coerce")
    fix["ticks_utc"] = pd.to_numeric(fix.get("ticks_utc", np.nan), errors="coerce")
    fix = fix.dropna(subset=["speed_kmh", "light_distance_m"])
    if fix.empty:
        return None, None, "missing_valid_fix_rows"
    if fix["t_wall"].notna().sum() >= 2:
        fix = fix.sort_values("t_wall").reset_index(drop=True)
    elif fix["ticks_utc"].notna().sum() >= 2:
        fix = fix.sort_values("ticks_utc").reset_index(drop=True)
    else:
        fix = fix.reset_index(drop=True)

    survey_rows = df[df["type"].str.lower() == "survey"]
    comfort_rating = np.nan
    if not survey_rows.empty:
        survey_last = survey_rows.iloc[-1]
        comfort_rating = to_float(survey_last.get("comfort_rating", np.nan))
        if np.isnan(comfort_rating):
            comfort_rating = to_float(survey_last.get("hr_bpm", np.nan))

    participant_id = str(meta.get("participant_id", "")).strip()
    driver_name = str(meta.get("driver_name", "")).strip()
    participant_group = parse_participant_group(participant_id, driver_name, file_path.name)

    experiment_kind = normalize_text(meta.get("experiment_kind", ""), default="")
    forced_stop = experiment_kind == "anyway_stop" or "anyway-stop" in file_path.name.lower()

    instructed_speed_kmh = to_float(meta.get("initial_speed", np.nan))
    distance_threshold_m = to_float(meta.get("distance_threshold_m", np.nan))
    file_speed_kmh, file_distance_m = parse_speed_distance_from_filename(file_path.name)
    if np.isnan(instructed_speed_kmh):
        instructed_speed_kmh = file_speed_kmh
    if np.isnan(distance_threshold_m):
        distance_threshold_m = file_distance_m

    onset_state = estimate_yellow_onset_state(fix, distance_threshold_m)
    initial_speed_kmh = float(onset_state["speed_kmh"]) if np.isfinite(onset_state["speed_kmh"]) else np.nan
    speed_delta_from_instruction_kmh = (
        initial_speed_kmh - instructed_speed_kmh
        if np.isfinite(initial_speed_kmh) and np.isfinite(instructed_speed_kmh)
        else np.nan
    )

    speed_mps = initial_speed_kmh / 3.6 if initial_speed_kmh > 0 else np.nan
    if speed_mps > 0 and distance_threshold_m > 0:
        tti_s = distance_threshold_m / speed_mps
        a_req_mps2 = speed_mps**2 / (2.0 * distance_threshold_m)
    else:
        tti_s = np.nan
        a_req_mps2 = np.nan

    file_timestamp = parse_filename_timestamp(file_path.name)
    run_year = float(file_timestamp.year) if not pd.isna(file_timestamp) else np.nan
    license_issue_year = parse_year(meta.get("license_issue_date", np.nan))
    experience_years = (
        run_year - license_issue_year
        if not np.isnan(run_year) and not np.isnan(license_issue_year)
        else np.nan
    )
    if not np.isnan(experience_years) and experience_years < 0:
        experience_years = np.nan

    window_mask = (fix["light_distance_m"] >= STOP_DISTANCE_MIN_M) & (
        fix["light_distance_m"] <= STOP_DISTANCE_MAX_M
    )
    stop_event_mask = window_mask & (fix["speed_kmh"].abs() <= STOP_SPEED_MAX_KMH)
    stop_detected = bool(stop_event_mask.any())
    go_decision = int(not stop_detected)
    stop_decision = int(stop_detected)

    stop_distance_at_event = np.nan
    if stop_detected:
        first_idx = stop_event_mask[stop_event_mask].index[0]
        stop_distance_at_event = float(fix.loc[first_idx, "light_distance_m"])

    sex_text = normalize_text(meta.get("driver_sex", ""), default="unknown")
    sex_female = 1 if "female" in sex_text else 0

    record = {
        "source_file": file_path.name,
        "file_timestamp": file_timestamp,
        "participant_id_raw": participant_id,
        "participant_group": participant_group,
        "experiment_num": str(meta.get("experiment_num", "")).strip(),
        "experiment_kind": experiment_kind,
        "forced_stop_instruction": int(forced_stop),
        "driver_name": driver_name,
        "driver_age": to_float(meta.get("driver_age", np.nan)),
        "driver_sex": sex_text,
        "sex_female": sex_female,
        "license_issue_year": license_issue_year,
        "experience_years": experience_years,
        "day_night": normalize_text(meta.get("day_night", ""), default="unknown"),
        "weather": normalize_text(meta.get("weather", ""), default="unknown"),
        "instructed_speed_kmh": instructed_speed_kmh,
        "initial_speed_kmh": initial_speed_kmh,
        "speed_at_yellow_kmh": initial_speed_kmh,
        "speed_delta_from_instruction_kmh": speed_delta_from_instruction_kmh,
        "distance_threshold_m": distance_threshold_m,
        "yellow_onset_time_value": onset_state["time_value"],
        "yellow_onset_time_source": onset_state["time_source"],
        "yellow_onset_method": onset_state["method"],
        "yellow_onset_interp_alpha": onset_state["alpha"],
        "yellow_onset_bracket_row0": onset_state["row0"],
        "yellow_onset_bracket_row1": onset_state["row1"],
        "yellow_onset_bracket_d0_m": onset_state["d0"],
        "yellow_onset_bracket_d1_m": onset_state["d1"],
        "yellow_onset_bracket_speed0_kmh": onset_state["s0"],
        "yellow_onset_bracket_speed1_kmh": onset_state["s1"],
        "yellow_onset_distance_abs_error_m": onset_state["distance_abs_error_m"],
        "tti_s": tti_s,
        "a_req_mps2": a_req_mps2,
        "comfort_rating": comfort_rating,
        "go_decision": go_decision,
        "stop_decision": stop_decision,
        "stop_detected_in_window": int(stop_detected),
        "stop_distance_event_m": stop_distance_at_event,
        "n_fix_rows": int(len(fix)),
        "min_speed_kmh_full_run": float(fix["speed_kmh"].min()),
        "min_light_distance_m": float(fix["light_distance_m"].min()),
        "max_light_distance_m": float(fix["light_distance_m"].max()),
    }

    fix_export = fix[["t_wall", "ticks_utc", "speed_kmh", "light_distance_m"]].copy()
    fix_export["source_file"] = file_path.name
    fix_export["participant_group"] = participant_group
    fix_export["distance_threshold_m"] = distance_threshold_m
    fix_export["relative_distance_to_threshold_m"] = fix_export["light_distance_m"] - distance_threshold_m
    fix_export["go_decision"] = go_decision
    fix_export["stop_decision"] = stop_decision
    fix_export["in_stop_window"] = window_mask.astype(int).values
    fix_export["stop_event"] = stop_event_mask.astype(int).values
    fix_export["forced_stop_instruction"] = int(forced_stop)
    fix_export["yellow_onset_bracket_sample"] = 0
    row0 = int(onset_state["row0"]) if onset_state["row0"] is not None else -1
    row1 = int(onset_state["row1"]) if onset_state["row1"] is not None else -1
    if 0 <= row0 < len(fix_export):
        fix_export.loc[row0, "yellow_onset_bracket_sample"] = 1
    if 0 <= row1 < len(fix_export):
        fix_export.loc[row1, "yellow_onset_bracket_sample"] = 1
    return record, fix_export, "ok"


def deduplicate_runs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, pd.DataFrame()

    main_mask = df["experiment_num"].astype(str).str.strip() != ""
    main_df = df[main_mask].copy()
    other_df = df[~main_mask].copy()

    if main_df.empty:
        dedup_df = df.copy()
        dedup_df = dedup_df.sort_values(["source_file"]).reset_index(drop=True)
        return dedup_df, pd.DataFrame()

    main_df = main_df.sort_values(
        ["participant_group", "experiment_num", "experiment_kind", "file_timestamp", "n_fix_rows"]
    )
    keep_idx = main_df.groupby(["participant_group", "experiment_num", "experiment_kind"]).tail(1).index
    duplicates = main_df.loc[~main_df.index.isin(keep_idx)].copy()
    dedup_main = main_df.loc[keep_idx].copy()

    dedup_df = pd.concat([dedup_main, other_df], ignore_index=True)
    dedup_df = dedup_df.sort_values(["file_timestamp", "source_file"]).reset_index(drop=True)
    duplicates = duplicates.sort_values(["file_timestamp", "source_file"]).reset_index(drop=True)
    return dedup_df, duplicates


def add_standardized_columns(df: pd.DataFrame, continuous_cols: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = df.copy()
    scalers: dict[str, dict[str, float]] = {}
    for col in continuous_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        median = float(out[col].median()) if not out[col].dropna().empty else 0.0
        out[col] = out[col].fillna(median)
        mean = float(out[col].mean())
        std = float(out[col].std(ddof=0))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        out[f"z_{col}"] = (out[col] - mean) / std
        scalers[col] = {"mean": mean, "std": std}
    return out, scalers


def _import_pyro_stack() -> tuple[Any, Any, Any, Any, Any, Any]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    local_site = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"
    if local_site.exists():
        site_str = str(local_site)
        if site_str not in sys.path:
            sys.path.append(site_str)
    try:
        import torch
        import pyro
        import pyro.distributions as pyro_dist
        from pyro.infer import MCMC, NUTS
        from pyro.ops.stats import effective_sample_size, split_gelman_rubin
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"Pyro stack is unavailable. Could not import torch/pyro from local environment: {exc}"
        ) from exc
    return torch, pyro, pyro_dist, MCMC, NUTS, (effective_sample_size, split_gelman_rubin)


def _tail_ess_5_95(samples: Any, effective_sample_size_fn: Any, torch_module: Any) -> np.ndarray:
    arr = np.asarray(samples.detach().cpu(), dtype=float)
    if arr.ndim < 2:
        arr = arr.reshape(1, -1)
    n_chains, n_samples = arr.shape[0], arr.shape[1]
    rest = int(np.prod(arr.shape[2:])) if arr.ndim > 2 else 1
    flat = arr.reshape(n_chains, n_samples, rest)
    out = np.full(rest, np.nan, dtype=float)

    for j in range(rest):
        x = flat[:, :, j]
        q05 = float(np.quantile(x, 0.05))
        q95 = float(np.quantile(x, 0.95))
        low_indicator = (x <= q05).astype(np.float32)
        high_indicator = (x >= q95).astype(np.float32)
        ess_low = float(
            effective_sample_size_fn(torch_module.tensor(low_indicator), chain_dim=0, sample_dim=1)
        )
        ess_high = float(
            effective_sample_size_fn(torch_module.tensor(high_indicator), chain_dim=0, sample_dim=1)
        )
        out[j] = min(ess_low, ess_high)

    if arr.ndim > 2:
        return out.reshape(arr.shape[2:])
    return out


def run_mcmc_diagnostics_and_ppc(
    model_exog: np.ndarray,
    y_obs: np.ndarray,
    participant_codes: np.ndarray,
    fe_names: list[str],
    n_participants: int,
    n_chains: int,
    warmup_steps: int,
    num_samples: int,
    target_accept: float,
    max_tree_depth: int,
    seed: int,
    diagnostics_csv: Path,
    ppc_csv: Path,
    parameter_summary_csv: Path,
    out_dashboard_plot: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any]]:
    (
        torch,
        pyro,
        pyro_dist,
        MCMC,
        NUTS,
        (effective_sample_size, split_gelman_rubin),
    ) = _import_pyro_stack()

    x_t = torch.tensor(np.asarray(model_exog, dtype=np.float32))
    y_t = torch.tensor(np.asarray(y_obs, dtype=np.float32))
    p_t = torch.tensor(np.asarray(participant_codes, dtype=np.int64))

    def bayes_model(x: Any, y: Any, p_idx: Any, n_p: int) -> None:
        beta = pyro.sample("beta", pyro_dist.Normal(0.0, 2.0).expand([x.shape[1]]).to_event(1))
        sigma_u = pyro.sample("sigma_u", pyro_dist.HalfNormal(1.0))
        with pyro.plate("participants", n_p):
            u = pyro.sample("u", pyro_dist.Normal(0.0, sigma_u))
        logits = (x @ beta) + u[p_idx]
        with pyro.plate("obs", x.shape[0]):
            pyro.sample("y_obs", pyro_dist.Bernoulli(logits=logits), obs=y)

    beta_chain_list = []
    sigma_chain_list = []
    u_chain_list = []
    for chain_idx in range(n_chains):
        pyro.clear_param_store()
        pyro.set_rng_seed(seed + chain_idx * 137)
        kernel = NUTS(
            bayes_model,
            target_accept_prob=float(target_accept),
            max_tree_depth=int(max_tree_depth),
        )
        mcmc = MCMC(
            kernel,
            warmup_steps=int(warmup_steps),
            num_samples=int(num_samples),
            num_chains=1,
            disable_progbar=True,
        )
        mcmc.run(x_t, y_t, p_t, int(n_participants))
        samples = mcmc.get_samples(group_by_chain=False)
        beta_chain_list.append(samples["beta"].detach().cpu())
        sigma_chain_list.append(samples["sigma_u"].detach().cpu().reshape(-1))
        u_chain_list.append(samples["u"].detach().cpu())

    beta_chains = torch.stack(beta_chain_list, dim=0)  # (C, S, K)
    sigma_chains = torch.stack(sigma_chain_list, dim=0)  # (C, S)
    u_chains = torch.stack(u_chain_list, dim=0)  # (C, S, P)

    beta_rhat = np.asarray(split_gelman_rubin(beta_chains, chain_dim=0, sample_dim=1), dtype=float)
    beta_ess_bulk = np.asarray(effective_sample_size(beta_chains, chain_dim=0, sample_dim=1), dtype=float)
    beta_ess_tail = _tail_ess_5_95(beta_chains, effective_sample_size, torch)

    sigma_rhat = float(split_gelman_rubin(sigma_chains, chain_dim=0, sample_dim=1))
    sigma_ess_bulk = float(effective_sample_size(sigma_chains, chain_dim=0, sample_dim=1))
    sigma_ess_tail = float(_tail_ess_5_95(sigma_chains, effective_sample_size, torch).reshape(-1)[0])

    diagnostics_rows = []
    for j, name in enumerate(fe_names):
        diagnostics_rows.append(
            {
                "term": name,
                "group": "fixed_effect",
                "r_hat": float(beta_rhat[j]),
                "ess_bulk": float(beta_ess_bulk[j]),
                "ess_tail_5_95": float(beta_ess_tail[j]),
                "r_hat_ok_1_01": int(beta_rhat[j] <= 1.01),
                "ess_bulk_ok_400": int(beta_ess_bulk[j] >= 400.0),
                "ess_tail_ok_400": int(beta_ess_tail[j] >= 400.0),
            }
        )
    diagnostics_rows.append(
        {
            "term": "sigma_u",
            "group": "random_effect_scale",
            "r_hat": sigma_rhat,
            "ess_bulk": sigma_ess_bulk,
            "ess_tail_5_95": sigma_ess_tail,
            "r_hat_ok_1_01": int(sigma_rhat <= 1.01),
            "ess_bulk_ok_400": int(sigma_ess_bulk >= 400.0),
            "ess_tail_ok_400": int(sigma_ess_tail >= 400.0),
        }
    )
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    beta_flat = np.asarray(beta_chains.reshape(-1, beta_chains.shape[-1]), dtype=float)
    parameter_rows = []
    for j, name in enumerate(fe_names):
        vals = beta_flat[:, j]
        p_gt = float(np.mean(vals > 0))
        p_lt = float(np.mean(vals < 0))
        pd_val = max(p_gt, p_lt)
        direction = "positive" if p_gt >= p_lt else "negative"
        q025, q50, q975 = np.quantile(vals, [0.025, 0.5, 0.975])
        parameter_rows.append(
            {
                "term": name,
                "posterior_mean": float(np.mean(vals)),
                "posterior_median": float(q50),
                "posterior_sd": float(np.std(vals, ddof=0)),
                "cri_2_5": float(q025),
                "cri_97_5": float(q975),
                "pd": float(pd_val),
                "direction": direction,
                "p_beta_gt_0": p_gt,
                "p_beta_lt_0": p_lt,
                "odds_ratio_mean": float(np.exp(np.mean(vals))),
                "odds_ratio_cri_2_5": float(np.exp(q025)),
                "odds_ratio_cri_97_5": float(np.exp(q975)),
            }
        )
    parameter_df = pd.DataFrame(parameter_rows)
    parameter_df.to_csv(parameter_summary_csv, index=False)

    rng = np.random.default_rng(seed + 2027)
    u_flat = np.asarray(u_chains.reshape(-1, u_chains.shape[-1]), dtype=float)
    eta_draw = beta_flat @ np.asarray(model_exog, dtype=float).T + u_flat[:, participant_codes]
    p_draw = sigmoid(eta_draw)
    y_rep = rng.binomial(1, np.clip(p_draw, 1e-6, 1.0 - 1e-6))
    go_count_rep = y_rep.sum(axis=1)

    obs_count = int(np.asarray(y_obs, dtype=int).sum())
    obs_rate = float(np.asarray(y_obs, dtype=float).mean())
    pred_rate_draw = p_draw.mean(axis=1)
    low_tail = float(np.mean(go_count_rep <= obs_count))
    high_tail = float(np.mean(go_count_rep >= obs_count))
    ppc_two_sided_p = float(min(1.0, 2.0 * min(low_tail, high_tail)))
    brier_mcmc = float(np.mean((p_draw.mean(axis=0) - np.asarray(y_obs, dtype=float)) ** 2))

    ppc_df = pd.DataFrame(
        [
            {
                "observed_go_count": obs_count,
                "observed_go_rate": obs_rate,
                "pp_go_rate_mean": float(np.mean(pred_rate_draw)),
                "pp_go_rate_cri_2_5": float(np.quantile(pred_rate_draw, 0.025)),
                "pp_go_rate_cri_97_5": float(np.quantile(pred_rate_draw, 0.975)),
                "pp_go_count_mean": float(np.mean(go_count_rep)),
                "pp_go_count_cri_2_5": float(np.quantile(go_count_rep, 0.025)),
                "pp_go_count_cri_97_5": float(np.quantile(go_count_rep, 0.975)),
                "ppc_two_sided_p": ppc_two_sided_p,
                "brier_mcmc_posterior_mean": brier_mcmc,
            }
        ]
    )
    ppc_df.to_csv(ppc_csv, index=False)

    diag_plot_df = diagnostics_df.copy().sort_values("r_hat", ascending=True).reset_index(drop=True)
    y_pos = np.arange(diag_plot_df.shape[0])

    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.6), constrained_layout=True)

    axes[0].scatter(diag_plot_df["r_hat"], y_pos, s=52, color="#2563eb", zorder=3)
    axes[0].axvline(1.01, color="#b91c1c", linestyle="--", linewidth=1.4, label="R-hat = 1.01")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(diag_plot_df["term"])
    axes[0].set_xlabel("Split R-hat")
    axes[0].set_title("(a) MCMC Convergence")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=9, frameon=True)

    axes[1].scatter(
        diag_plot_df["ess_bulk"],
        y_pos + 0.10,
        s=46,
        marker="o",
        color="#0f766e",
        label="Bulk ESS",
        zorder=3,
    )
    axes[1].scatter(
        diag_plot_df["ess_tail_5_95"],
        y_pos - 0.10,
        s=46,
        marker="D",
        color="#7c3aed",
        label="Tail ESS (5/95%)",
        zorder=3,
    )
    axes[1].axvline(400.0, color="#b91c1c", linestyle="--", linewidth=1.4, label="ESS = 400")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Effective sample size")
    axes[1].set_title("(b) Sampling Efficiency")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=9, frameon=True)

    bins = np.arange(int(go_count_rep.min()), int(go_count_rep.max()) + 2)
    axes[2].hist(go_count_rep, bins=bins, density=True, color="#0f766e", alpha=0.78, edgecolor="white")
    axes[2].axvline(obs_count, color="#b91c1c", linestyle="-", linewidth=2.0, label=f"Observed = {obs_count}")
    axes[2].axvline(
        float(np.mean(go_count_rep)),
        color="#111827",
        linestyle="--",
        linewidth=1.5,
        label=f"Posterior predictive mean = {np.mean(go_count_rep):.1f}",
    )
    axes[2].set_xlabel("Go decisions per replicated dataset")
    axes[2].set_ylabel("Density")
    axes[2].set_title("(c) Posterior Predictive Check")
    axes[2].legend(loc="upper right", fontsize=9, frameon=True)
    axes[2].text(
        0.03,
        0.97,
        f"Observed go-rate = {100.0 * obs_rate:.1f}%\nPPC two-sided p = {ppc_two_sided_p:.3f}",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.86, edgecolor="#9ca3af"),
    )

    fig.suptitle("Bayesian Diagnostic Dashboard (MCMC + PPC)", fontsize=16, fontweight="bold")
    fig.savefig(out_dashboard_plot, dpi=240)
    try:
        fig.savefig(out_dashboard_plot.with_suffix(".pdf"), dpi=240)
    except Exception:
        pass
    plt.close(fig)

    ppc_draws = {
        "go_count_rep": go_count_rep.astype(int),
        "pred_rate_draw": pred_rate_draw.astype(float),
        "observed_go_count": int(obs_count),
        "observed_go_rate": float(obs_rate),
    }
    return diagnostics_df, ppc_df, parameter_df, beta_flat, ppc_draws


def plot_ame_with_validators_composite(
    ame_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    ppc_df: pd.DataFrame,
    ppc_draws: dict[str, Any],
    out_plot: Path,
) -> None:
    if ame_df.empty or diag_df.empty or ppc_df.empty:
        return

    ame_plot = ame_df.sort_values("contribution_mean", ascending=True).reset_index(drop=True)
    y = np.arange(ame_plot.shape[0])
    contrib = 100.0 * ame_plot["contribution_mean"].to_numpy()
    c_low = 100.0 * ame_plot["contribution_ci_2_5"].to_numpy()
    c_high = 100.0 * ame_plot["contribution_ci_97_5"].to_numpy()
    xerr = [contrib - c_low, c_high - contrib]
    ame_pp = 100.0 * ame_plot["ame_mean"].to_numpy()
    ame_lo = 100.0 * ame_plot["ame_ci_2_5"].to_numpy()
    ame_hi = 100.0 * ame_plot["ame_ci_97_5"].to_numpy()
    colors = np.where(ame_pp >= 0.0, "#0f766e", "#b91c1c")

    diag_plot = diag_df.copy().sort_values("r_hat", ascending=True).reset_index(drop=True)
    yv = np.arange(diag_plot.shape[0])
    terms = diag_plot["term"].astype(str).tolist()

    fig = plt.figure(figsize=(17.0, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_ame = fig.add_subplot(gs[:, 0])
    ax_diag = fig.add_subplot(gs[0, 1])
    ax_ppc = fig.add_subplot(gs[1, 1])

    # Left panel (50%): AME contribution.
    short_labels = {
        "Required decel $a_{req}$": r"$a_{req}$",
        "Distance threshold $d_{th}$": r"$D_{th}$",
        "Gender (female vs male)": "Gender",
        "TTI": "TTI",
        "Speed at yellow onset": r"$V_0$",
        "Driver age": "Age",
    }
    ytick_labels = [short_labels.get(lbl, lbl) for lbl in ame_plot["label"].tolist()]

    ax_ame.barh(y, contrib, color=colors, alpha=0.94, edgecolor="white", linewidth=1.1, height=0.78, zorder=2)
    ax_ame.errorbar(
        contrib,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#111827",
        elinewidth=2.1,
        capsize=4.0,
        zorder=3,
    )
    ax_ame.set_yticks(y)
    ax_ame.set_yticklabels(ytick_labels, fontsize=12.5)
    ax_ame.set_xlabel("Relative contribution to stop/go decision (%)", fontsize=13)
    ax_ame.set_title("(a) AME-Based Contribution Factors", fontsize=15, fontweight="bold")
    ax_ame.grid(axis="x", alpha=0.22, linewidth=1.1)
    xmax = max(float(np.nanmax(c_high)), 1.0)
    ax_ame.set_xlim(0.0, xmax * 1.34)
    for i in range(ame_plot.shape[0]):
        txt = f"AME={ame_pp[i]:+0.1f} pp [{ame_lo[i]:+0.1f},{ame_hi[i]:+0.1f}]"
        ax_ame.text(
            float(c_high[i]) + 0.9,
            y[i],
            txt,
            ha="left",
            va="center",
            fontsize=10.2,
            color="#111827",
        )
    ax_ame.legend(
        handles=[
            Patch(facecolor="#0f766e", edgecolor="white", label="Positive AME"),
            Patch(facecolor="#b91c1c", edgecolor="white", label="Negative AME"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=10.8,
    )

    # Right-top validator: R-hat + ESS (bigger/bold markers).
    ax_diag.scatter(
        diag_plot["ess_bulk"],
        yv + 0.12,
        s=96,
        marker="o",
        color="#0f766e",
        edgecolor="white",
        linewidth=0.8,
        label="Bulk ESS",
        zorder=4,
    )
    ax_diag.scatter(
        diag_plot["ess_tail_5_95"],
        yv - 0.12,
        s=96,
        marker="D",
        color="#7c3aed",
        edgecolor="white",
        linewidth=0.8,
        label="Tail ESS (5/95%)",
        zorder=4,
    )
    ax_diag.axvline(400.0, color="#111827", linestyle="--", linewidth=2.0, zorder=1)
    ax_diag.set_yticks(yv)
    ax_diag.set_yticklabels(terms, fontsize=10.0)
    ax_diag.set_xlabel("Effective Sample Size (ESS)", fontsize=12.5)
    ax_diag.set_title("(b) Convergence Validators (R-hat and ESS)", fontsize=14.5, fontweight="bold")
    ax_diag.grid(axis="x", alpha=0.22, linewidth=1.0)

    ax_rhat = ax_diag.twiny()
    ax_rhat.scatter(
        diag_plot["r_hat"],
        yv,
        s=90,
        marker="s",
        color="#b91c1c",
        edgecolor="white",
        linewidth=0.8,
        label="Split R-hat",
        zorder=5,
    )
    ax_rhat.axvline(1.01, color="#b91c1c", linestyle="--", linewidth=2.0)
    rhat_min = float(diag_plot["r_hat"].min())
    rhat_max = float(diag_plot["r_hat"].max())
    pad = max(0.002, 0.25 * (rhat_max - rhat_min))
    ax_rhat.set_xlim(rhat_min - pad, max(1.011, rhat_max + pad))
    ax_rhat.set_xlabel("Split R-hat", fontsize=12.5)

    diag_txt = (
        f"max R-hat = {diag_plot['r_hat'].max():.4f}\n"
        f"min bulk ESS = {diag_plot['ess_bulk'].min():.1f}\n"
        f"min tail ESS = {diag_plot['ess_tail_5_95'].min():.1f}"
    )
    ax_diag.text(
        0.02,
        0.98,
        diag_txt,
        transform=ax_diag.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        bbox=dict(boxstyle="round,pad=0.26", facecolor="white", alpha=0.9, edgecolor="#9ca3af"),
    )
    h1, l1 = ax_diag.get_legend_handles_labels()
    h2, l2 = ax_rhat.get_legend_handles_labels()
    ax_diag.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        bbox_to_anchor=(0.998, 0.998),
        fontsize=9.8,
        frameon=True,
    )

    # Right-bottom validator: PPC with bold bars.
    rep = np.asarray(ppc_draws.get("go_count_rep", []), dtype=float)
    if rep.size > 0:
        bins = np.arange(int(rep.min()), int(rep.max()) + 2)
        ax_ppc.hist(rep, bins=bins, density=True, color="#2563eb", alpha=0.86, edgecolor="white", linewidth=0.8)
    ppc_row = ppc_df.iloc[0]
    obs = int(ppc_row["observed_go_count"])
    mean_rep = float(ppc_row["pp_go_count_mean"])
    lo = float(ppc_row["pp_go_count_cri_2_5"])
    hi = float(ppc_row["pp_go_count_cri_97_5"])
    ax_ppc.axvspan(lo, hi, color="#0f766e", alpha=0.16, zorder=0, label="95% PPC interval")
    ax_ppc.axvline(obs, color="#b91c1c", linestyle="-", linewidth=2.6, label=f"Observed = {obs}")
    ax_ppc.axvline(mean_rep, color="#111827", linestyle="--", linewidth=2.0, label=f"PPC mean = {mean_rep:.1f}")
    ax_ppc.set_title("(c) Posterior Predictive Check (Go Counts)", fontsize=14.5, fontweight="bold")
    ax_ppc.set_xlabel("Go decisions per replicated dataset", fontsize=12.5)
    ax_ppc.set_ylabel("Density", fontsize=12.5)
    ax_ppc.grid(axis="y", alpha=0.22, linewidth=1.0)
    ax_ppc.legend(loc="upper left", fontsize=10.0, frameon=True)
    ax_ppc.text(
        0.98,
        0.96,
        (
            f"Observed go-rate = {100.0 * float(ppc_row['observed_go_rate']):.1f}%\n"
            f"PPC two-sided p = {float(ppc_row['ppc_two_sided_p']):.3f}"
        ),
        transform=ax_ppc.transAxes,
        ha="right",
        va="top",
        fontsize=10.4,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#9ca3af"),
    )

    fig.suptitle("Stop/Go Factor Impact with Bayesian Validators", fontsize=17, fontweight="bold")
    fig.savefig(out_plot, dpi=250)
    try:
        fig.savefig(out_plot.with_suffix(".pdf"), dpi=250)
    except Exception:
        pass
    plt.close(fig)



def draw_posterior(
    model: BinomialBayesMixedGLM,
    result: Any,
    n_draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k_fe = model.k_fep
    k_vcp = model.k_vcp

    params = np.asarray(result.params)
    if params.size >= (k_fe + k_vcp):
        try:
            cov_df = result.cov_params()
            cov = np.asarray(cov_df, dtype=float)
            cov = 0.5 * (cov + cov.T)
            eigvals = np.linalg.eigvalsh(cov)
            min_eig = float(eigvals.min())
            if min_eig <= 0:
                cov += np.eye(cov.shape[0]) * (abs(min_eig) + 1e-6)
            draws = rng.multivariate_normal(params, cov, size=n_draws)
            fe_draws = draws[:, :k_fe]
            vcp_draws = draws[:, k_fe : k_fe + k_vcp]
            return fe_draws, vcp_draws
        except Exception:
            pass

    fe_mean = np.asarray(result.fe_mean)
    fe_sd = np.maximum(np.asarray(result.fe_sd), 1e-6)
    fe_draws = rng.normal(loc=fe_mean, scale=fe_sd, size=(n_draws, fe_mean.size))

    vcp_mean = np.asarray(result.vcp_mean)
    vcp_sd = np.maximum(np.asarray(result.vcp_sd), 1e-6)
    vcp_draws = rng.normal(loc=vcp_mean, scale=vcp_sd, size=(n_draws, vcp_mean.size))
    return fe_draws, vcp_draws


def summarize_fixed_effects(
    fe_names: list[str],
    fe_draws: np.ndarray,
    output_csv: Path,
) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(fe_names):
        values = fe_draws[:, idx]
        ci_low, ci_high = np.quantile(values, [0.025, 0.975])
        p_gt = float(np.mean(values > 0))
        p_lt = float(np.mean(values < 0))
        row = {
            "term": name,
            "posterior_mean": float(np.mean(values)),
            "posterior_sd": float(np.std(values, ddof=0)),
            "ci_2_5": float(ci_low),
            "ci_50": float(np.quantile(values, 0.5)),
            "ci_97_5": float(ci_high),
            "p_beta_gt_0": p_gt,
            "p_beta_lt_0": p_lt,
            "pd": float(max(p_gt, p_lt)),
            "direction": "positive" if p_gt >= p_lt else "negative",
            "odds_ratio_mean": float(np.exp(np.mean(values))),
            "odds_ratio_ci_2_5": float(np.exp(ci_low)),
            "odds_ratio_ci_97_5": float(np.exp(ci_high)),
        }
        rows.append(row)

    effect_df = pd.DataFrame(rows)
    effect_df.to_csv(output_csv, index=False)
    return effect_df


def plot_feature_importance(
    fe_names: list[str],
    fe_draws: np.ndarray,
    out_path: Path,
) -> pd.DataFrame:
    mask = [name != "Intercept" for name in fe_names]
    sel_names = [name for name in fe_names if name != "Intercept"]
    if not sel_names:
        return pd.DataFrame()

    beta = fe_draws[:, mask]
    abs_beta = np.abs(beta)
    denom = abs_beta.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    importance_draws = abs_beta / denom

    importance_rows = []
    for j, name in enumerate(sel_names):
        values = importance_draws[:, j]
        importance_rows.append(
            {
                "term": name,
                "importance_mean": float(np.mean(values)),
                "importance_ci_2_5": float(np.quantile(values, 0.025)),
                "importance_ci_97_5": float(np.quantile(values, 0.975)),
            }
        )
    imp_df = pd.DataFrame(importance_rows).sort_values("importance_mean", ascending=True)

    plt.figure(figsize=(10, 7))
    x = imp_df["importance_mean"].to_numpy()
    y = np.arange(len(imp_df))
    xerr_left = x - imp_df["importance_ci_2_5"].to_numpy()
    xerr_right = imp_df["importance_ci_97_5"].to_numpy() - x
    plt.barh(y, x, color="#2b6cb0", alpha=0.9)
    plt.errorbar(x, y, xerr=[xerr_left, xerr_right], fmt="none", ecolor="black", capsize=3)
    plt.yticks(y, imp_df["term"])
    plt.xlabel("Normalized |beta| importance")
    plt.title("Posterior Feature Importance (Normalized Absolute Coefficients)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return imp_df


def summarize_and_plot_factor_effects(
    fe_names: list[str],
    fe_draws: np.ndarray,
    model_exog: np.ndarray,
    model_df: pd.DataFrame,
    scalers: dict[str, dict[str, float]],
    out_csv: Path,
    out_age_csv: Path,
    out_plot: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "Intercept" not in fe_names or model_exog.ndim != 2:
        empty = pd.DataFrame()
        empty.to_csv(out_csv, index=False)
        empty.to_csv(out_age_csv, index=False)
        return empty, empty

    k_fe = len(fe_names)
    if fe_draws.shape[1] != k_fe or model_exog.shape[1] != k_fe:
        empty = pd.DataFrame()
        empty.to_csv(out_csv, index=False)
        empty.to_csv(out_age_csv, index=False)
        return empty, empty

    # Interpretable factors used for AME contribution analysis.
    factor_specs = [
        {"term": "z_initial_speed_kmh", "label": "Speed at yellow onset", "kind": "continuous"},
        {"term": "z_distance_threshold_m", "label": "Distance threshold $d_{th}$", "kind": "continuous"},
        {"term": "z_tti_s", "label": "TTI", "kind": "continuous"},
        {"term": "z_a_req_mps2", "label": "Required decel $a_{req}$", "kind": "continuous"},
        {"term": "z_driver_age", "label": "Driver age", "kind": "continuous"},
        {"term": "sex_female", "label": "Gender (female vs male)", "kind": "binary"},
    ]

    eta_base = fe_draws @ model_exog.T  # (n_draws, n_obs)

    rows: list[dict[str, Any]] = []
    ame_draws_cols: list[np.ndarray] = []

    for spec in factor_specs:
        term = spec["term"]
        if term not in fe_names:
            continue
        idx = fe_names.index(term)
        beta = fe_draws[:, idx]  # (n_draws,)

        if spec["kind"] == "continuous":
            # AME for +1 SD change (since predictors are z-standardized).
            eta_hi = eta_base + beta[:, None]
            ame_draw = (sigmoid(eta_hi) - sigmoid(eta_base)).mean(axis=1)
            contrast_label = "+1 SD increase"
        else:
            # Binary AME: female vs male, averaged over observed covariates.
            xj = model_exog[:, idx].astype(float)
            eta_without = eta_base - beta[:, None] * xj[None, :]
            p1 = sigmoid(eta_without + beta[:, None])
            p0 = sigmoid(eta_without)
            ame_draw = (p1 - p0).mean(axis=1)
            contrast_label = "female vs male"

        p_gt = float(np.mean(ame_draw > 0))
        p_lt = float(np.mean(ame_draw < 0))

        rows.append(
            {
                "term": term,
                "label": spec["label"],
                "contrast": contrast_label,
                "ame_mean": float(np.mean(ame_draw)),
                "ame_ci_2_5": float(np.quantile(ame_draw, 0.025)),
                "ame_ci_97_5": float(np.quantile(ame_draw, 0.975)),
                "p_ame_gt_0": p_gt,
                "p_ame_lt_0": p_lt,
                "pd_ame": float(max(p_gt, p_lt)),
                "ame_direction": "positive" if p_gt >= p_lt else "negative",
            }
        )
        ame_draws_cols.append(ame_draw)

    effect_df = pd.DataFrame(rows)
    if effect_df.empty:
        effect_df.to_csv(out_csv, index=False)
        pd.DataFrame().to_csv(out_age_csv, index=False)
        return effect_df, pd.DataFrame()

    ame_mat = np.column_stack(ame_draws_cols)  # (n_draws, n_factors)
    abs_ame = np.abs(ame_mat)
    denom = abs_ame.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    contribution = abs_ame / denom

    for j in range(effect_df.shape[0]):
        vals = contribution[:, j]
        effect_df.loc[j, "contribution_mean"] = float(np.mean(vals))
        effect_df.loc[j, "contribution_ci_2_5"] = float(np.quantile(vals, 0.025))
        effect_df.loc[j, "contribution_ci_97_5"] = float(np.quantile(vals, 0.975))

    effect_df["abs_ame_mean"] = effect_df["ame_mean"].abs()
    effect_df = effect_df.sort_values("contribution_mean", ascending=False).reset_index(drop=True)
    effect_df.to_csv(out_csv, index=False)

    # Age-group model-implied P(go), averaged over observed covariates.
    age_df = pd.DataFrame()
    if "z_driver_age" in fe_names and "driver_age" in model_df.columns and "driver_age" in scalers:
        age_idx = fe_names.index("z_driver_age")
        beta_age = fe_draws[:, age_idx]
        x_age = model_exog[:, age_idx].astype(float)
        eta_without_age = eta_base - beta_age[:, None] * x_age[None, :]

        age_groups = [("18-25", 18.0, 25.0), ("25-32", 25.0, 32.0), ("32+", 32.0, np.inf)]
        age_rows = []
        age_draws: dict[str, np.ndarray] = {}
        age_mean = float(scalers["driver_age"]["mean"])
        age_std = float(scalers["driver_age"]["std"])

        for label, lo, hi in age_groups:
            if np.isfinite(hi):
                sub = model_df[(model_df["driver_age"] >= lo) & (model_df["driver_age"] < hi)]
            else:
                sub = model_df[model_df["driver_age"] >= lo]
            if sub.empty:
                continue
            rep_age = float(sub["driver_age"].median())
            z_age = (rep_age - age_mean) / age_std
            p_draw = sigmoid(eta_without_age + beta_age[:, None] * z_age).mean(axis=1)
            age_draws[label] = p_draw
            age_rows.append(
                {
                    "age_group": label,
                    "n_runs": int(len(sub)),
                    "representative_age": rep_age,
                    "p_go_mean": float(np.mean(p_draw)),
                    "p_go_ci_2_5": float(np.quantile(p_draw, 0.025)),
                    "p_go_ci_97_5": float(np.quantile(p_draw, 0.975)),
                }
            )

        if age_rows:
            age_df = pd.DataFrame(age_rows)
            if "18-25" in age_draws:
                base = age_draws["18-25"]
                for i, row in age_df.iterrows():
                    grp = str(row["age_group"])
                    if grp == "18-25" or grp not in age_draws:
                        age_df.loc[i, "delta_vs_18_25_mean"] = 0.0
                        age_df.loc[i, "delta_vs_18_25_ci_2_5"] = 0.0
                        age_df.loc[i, "delta_vs_18_25_ci_97_5"] = 0.0
                    else:
                        diff = age_draws[grp] - base
                        age_df.loc[i, "delta_vs_18_25_mean"] = float(np.mean(diff))
                        age_df.loc[i, "delta_vs_18_25_ci_2_5"] = float(np.quantile(diff, 0.025))
                        age_df.loc[i, "delta_vs_18_25_ci_97_5"] = float(np.quantile(diff, 0.975))
        age_df.to_csv(out_age_csv, index=False)
    else:
        age_df.to_csv(out_age_csv, index=False)

    # Single advanced graph: AME-based contribution bars with uncertainty and AME annotation.
    plot_df = effect_df.sort_values("contribution_mean", ascending=True).reset_index(drop=True)
    y = np.arange(plot_df.shape[0])

    contrib_pct = 100.0 * plot_df["contribution_mean"].to_numpy()
    c_low = 100.0 * plot_df["contribution_ci_2_5"].to_numpy()
    c_high = 100.0 * plot_df["contribution_ci_97_5"].to_numpy()
    xerr_left = contrib_pct - c_low
    xerr_right = c_high - contrib_pct

    ame_pp = 100.0 * plot_df["ame_mean"].to_numpy()
    ame_low = 100.0 * plot_df["ame_ci_2_5"].to_numpy()
    ame_high = 100.0 * plot_df["ame_ci_97_5"].to_numpy()
    colors = np.where(ame_pp >= 0.0, "#0f766e", "#b91c1c")

    fig, ax = plt.subplots(figsize=(12.8, 7.6), constrained_layout=True)
    bars = ax.barh(y, contrib_pct, color=colors, alpha=0.92, edgecolor="white", linewidth=0.8, zorder=2)
    ax.errorbar(
        contrib_pct,
        y,
        xerr=[xerr_left, xerr_right],
        fmt="none",
        ecolor="#111827",
        elinewidth=1.5,
        capsize=3,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Relative contribution to stop/go decision (%)")
    ax.set_title("AME-Based Contribution of Decision Factors")
    ax.grid(axis="x", alpha=0.25)

    xmax = max(float(np.nanmax(c_high)), 1.0)
    ax.set_xlim(0.0, xmax * 1.28)
    for i in range(plot_df.shape[0]):
        txt = f"AME={ame_pp[i]:+0.1f} pp [{ame_low[i]:+0.1f}, {ame_high[i]:+0.1f}]"
        ax.text(
            float(c_high[i]) + 0.6,
            y[i],
            txt,
            ha="left",
            va="center",
            fontsize=9.0,
            color="#111827",
        )

    legend_items = [
        Patch(facecolor="#0f766e", edgecolor="white", label="Positive AME on P(go)"),
        Patch(facecolor="#b91c1c", edgecolor="white", label="Negative AME on P(go)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=True)

    fig.savefig(out_plot, dpi=250)
    try:
        fig.savefig(out_plot.with_suffix(".pdf"), dpi=250)
    except Exception:
        pass
    plt.close(fig)
    return effect_df, age_df


def plot_surface_outputs(
    model: BinomialBayesMixedGLM,
    fe_draws: np.ndarray,
    vcp_draws: np.ndarray,
    model_df: pd.DataFrame,
    scalers: dict[str, dict[str, float]],
    grid_size: int,
    participant_mc_samples: int,
    seed: int,
    export_dir: Path,
    out_decision: Path,
    out_uncertainty: Path,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + 11)

    speed_grid = np.linspace(
        float(model_df["initial_speed_kmh"].min()),
        float(model_df["initial_speed_kmh"].max()),
        grid_size,
    )
    tti_grid = np.linspace(
        float(model_df["tti_s"].min()),
        float(model_df["tti_s"].max()),
        grid_size,
    )
    sp_mesh, tti_mesh = np.meshgrid(speed_grid, tti_grid)
    speed_mps_mesh = sp_mesh / 3.6
    dist_mesh = tti_mesh * speed_mps_mesh
    dist_mesh = np.maximum(dist_mesh, 0.5)
    a_req_mesh = (speed_mps_mesh**2) / (2.0 * dist_mesh)

    ref_weather = str(model_df["weather"].mode().iloc[0])
    ref_day_night = str(model_df["day_night"].mode().iloc[0])

    grid_df = pd.DataFrame(
        {
            "initial_speed_kmh": sp_mesh.ravel(),
            "distance_threshold_m": dist_mesh.ravel(),
            "tti_s": tti_mesh.ravel(),
            "a_req_mps2": a_req_mesh.ravel(),
            "driver_age": float(model_df["driver_age"].median()),
            "sex_female": float(model_df["sex_female"].mean()),
            "weather": ref_weather,
            "day_night": ref_day_night,
            "participant_group": str(model_df["participant_group"].mode().iloc[0]),
        }
    )

    for raw_col in [
        "initial_speed_kmh",
        "distance_threshold_m",
        "tti_s",
        "a_req_mps2",
        "driver_age",
    ]:
        mean = scalers[raw_col]["mean"]
        std = scalers[raw_col]["std"]
        grid_df[f"z_{raw_col}"] = (grid_df[raw_col] - mean) / std

    exog_grid = build_design_matrices([model.data.design_info], grid_df, return_type="dataframe")[0]
    exog_grid_np = exog_grid.to_numpy()

    n_draws = fe_draws.shape[0]
    n_grid_points = exog_grid_np.shape[0]
    posterior_prob = np.empty((n_draws, n_grid_points), dtype=np.float32)
    participant_disagreement = np.empty((n_draws, n_grid_points), dtype=np.float32)

    for d in range(n_draws):
        eta = exog_grid_np @ fe_draws[d]
        sigma_u = math.exp(float(vcp_draws[d, 0])) if vcp_draws.shape[1] > 0 else 0.0
        random_u = rng.normal(0.0, sigma_u, size=participant_mc_samples)
        p_mat = sigmoid(eta[:, None] + random_u[None, :])
        posterior_prob[d] = p_mat.mean(axis=1)
        participant_disagreement[d] = p_mat.var(axis=1)

    p_mean = posterior_prob.mean(axis=0).reshape(grid_size, grid_size)
    p_low = np.quantile(posterior_prob, 0.025, axis=0).reshape(grid_size, grid_size)
    p_high = np.quantile(posterior_prob, 0.975, axis=0).reshape(grid_size, grid_size)
    p_post_sd = posterior_prob.std(axis=0).reshape(grid_size, grid_size)
    p_disagreement = participant_disagreement.mean(axis=0).reshape(grid_size, grid_size)

    hesitation_mask = ((p_mean >= 0.4) & (p_mean <= 0.6)).astype(float)

    export_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        export_dir / "surface_arrays.npz",
        speed_grid=speed_grid,
        tti_grid=tti_grid,
        distance_mesh=dist_mesh,
        a_req_mesh=a_req_mesh,
        p_go_mean=p_mean,
        p_go_ci_low=p_low,
        p_go_ci_high=p_high,
        p_go_post_sd=p_post_sd,
        p_go_disagreement_var=p_disagreement,
        hesitation_mask=hesitation_mask,
    )

    grid_export = pd.DataFrame(
        {
            "speed_kmh": sp_mesh.ravel(),
            "tti_s": tti_mesh.ravel(),
            "distance_threshold_m": dist_mesh.ravel(),
            "a_req_mps2": a_req_mesh.ravel(),
            "p_go_mean": p_mean.ravel(),
            "p_go_ci_low": p_low.ravel(),
            "p_go_ci_high": p_high.ravel(),
            "p_go_post_sd": p_post_sd.ravel(),
            "p_go_disagreement_var": p_disagreement.ravel(),
            "hesitation_region": hesitation_mask.ravel().astype(int),
        }
    )
    grid_export.to_csv(export_dir / "surface_grid_long.csv", index=False)

    posterior_prob_3d = posterior_prob.reshape(n_draws, grid_size, grid_size)
    boundary_draws = np.full((n_draws, grid_size), np.nan, dtype=np.float32)
    for d in range(n_draws):
        for j in range(grid_size):
            boundary_draws[d, j] = first_level_crossing(tti_grid, posterior_prob_3d[d, :, j], level=0.5)

    valid_frac = np.mean(np.isfinite(boundary_draws), axis=0)
    boundary_summary = pd.DataFrame(
        {
            "speed_kmh": speed_grid,
            "tti_p50_median": np.nanmedian(boundary_draws, axis=0),
            "tti_p50_ci_low": np.nanquantile(boundary_draws, 0.025, axis=0),
            "tti_p50_ci_high": np.nanquantile(boundary_draws, 0.975, axis=0),
            "valid_draw_fraction": valid_frac,
        }
    )
    boundary_summary.to_csv(export_dir / "decision_boundary_summary.csv", index=False)

    draw_ids = np.repeat(np.arange(n_draws), grid_size)
    speeds_rep = np.tile(speed_grid, n_draws)
    boundary_long = pd.DataFrame(
        {
            "draw_id": draw_ids,
            "speed_kmh": speeds_rep,
            "tti_at_p50": boundary_draws.reshape(-1),
        }
    )
    boundary_long = boundary_long[np.isfinite(boundary_long["tti_at_p50"])].reset_index(drop=True)
    boundary_long.to_csv(export_dir / "decision_boundary_draws.csv", index=False)

    observed_export = model_df[
        [
            "source_file",
            "participant_group",
            "driver_age",
            "initial_speed_kmh",
            "distance_threshold_m",
            "tti_s",
            "a_req_mps2",
            "go_decision",
            "stop_decision",
            "weather",
            "day_night",
        ]
    ].copy()
    observed_export.to_csv(export_dir / "observed_decision_samples.csv", index=False)

    metadata = {
        "grid_size": int(grid_size),
        "n_draws": int(n_draws),
        "participant_mc_samples_per_draw": int(participant_mc_samples),
        "decision_probability_name": "p_go",
        "hesitation_band": [0.4, 0.6],
        "boundary_probability_level": 0.5,
        "reference_weather": ref_weather,
        "reference_day_night": ref_day_night,
    }
    with open(export_dir / "plot_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    fig, ax = plt.subplots(figsize=(11, 8))
    levels = np.linspace(0.0, 1.0, 21)
    contour = ax.contourf(speed_grid, tti_grid, p_mean, levels=levels, cmap="RdYlGn", alpha=0.95)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Posterior mean P(go)")
    cs = ax.contour(
        speed_grid,
        tti_grid,
        p_mean,
        levels=[0.25, 0.4, 0.5, 0.6, 0.75],
        colors="black",
        linewidths=[1.0, 1.2, 1.6, 1.2, 1.0],
    )
    ax.clabel(cs, inline=True, fontsize=9, fmt="%.2f")
    ax.contourf(
        speed_grid,
        tti_grid,
        hesitation_mask,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////"],
    )
    ax.scatter(
        model_df["initial_speed_kmh"],
        model_df["tti_s"],
        c=model_df["go_decision"],
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        s=18,
        alpha=0.35,
        edgecolors="none",
        label="Observed runs (0=stop, 1=go)",
    )
    ax.set_xlabel("Speed at yellow onset (km/h)")
    ax.set_ylabel("TTI at yellow onset (s)")
    ax.set_title("Decision Surface with Hesitation Region (hatched: 0.4 <= P(go) <= 0.6)")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_decision, dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    c1 = axes[0].contourf(speed_grid, tti_grid, p_post_sd, levels=20, cmap="Blues")
    cb1 = fig.colorbar(c1, ax=axes[0])
    cb1.set_label("Posterior SD of P(go)")
    axes[0].contour(speed_grid, tti_grid, p_mean, levels=[0.5], colors="black", linewidths=1.2)
    axes[0].set_title("Epistemic Uncertainty")
    axes[0].set_xlabel("Speed (km/h)")
    axes[0].set_ylabel("TTI (s)")

    c2 = axes[1].contourf(speed_grid, tti_grid, p_disagreement, levels=20, cmap="magma")
    cb2 = fig.colorbar(c2, ax=axes[1])
    cb2.set_label("Between-participant variance in decision probability")
    axes[1].contour(speed_grid, tti_grid, p_mean, levels=[0.5], colors="white", linewidths=1.2)
    high_q = float(np.quantile(p_disagreement, 0.85))
    axes[1].contour(
        speed_grid,
        tti_grid,
        p_disagreement,
        levels=[high_q],
        colors="cyan",
        linewidths=1.6,
    )
    axes[1].set_title("Participant Disagreement Map")
    axes[1].set_xlabel("Speed (km/h)")
    axes[1].set_ylabel("TTI (s)")
    plt.savefig(out_uncertainty, dpi=240)
    plt.close(fig)

    max_dis_idx = int(np.argmax(p_disagreement))
    max_row, max_col = np.unravel_index(max_dis_idx, p_disagreement.shape)
    summary = {
        "hesitation_area_fraction": float(np.mean((p_mean >= 0.4) & (p_mean <= 0.6))),
        "max_disagreement_value": float(p_disagreement[max_row, max_col]),
        "max_disagreement_speed_kmh": float(speed_grid[max_col]),
        "max_disagreement_tti_s": float(tti_grid[max_row]),
        "mean_credible_band_width": float(np.mean(p_high - p_low)),
    }
    return summary


def participant_disagreement_regression(
    model: BinomialBayesMixedGLM,
    result: Any,
    model_df: pd.DataFrame,
    out_csv: Path,
    out_summary_txt: Path,
) -> pd.DataFrame:
    vc_names = list(model.vc_names)
    vc_mean = np.asarray(result.vc_mean)
    vc_sd = np.asarray(result.vc_sd)
    re_df = pd.DataFrame({"vc_name": vc_names, "u_mean": vc_mean, "u_sd": vc_sd})
    re_df["participant_group"] = (
        re_df["vc_name"].astype(str).str.extract(r"\[(.*?)\]")[0].fillna(re_df["vc_name"])
    )

    participant_cov = (
        model_df.groupby("participant_group", as_index=False)
        .agg(
            driver_age=("driver_age", "median"),
            sex_female=("sex_female", "median"),
            n_runs=("go_decision", "size"),
            go_rate=("go_decision", "mean"),
        )
        .copy()
    )
    merged = participant_cov.merge(re_df, on="participant_group", how="inner")

    if merged.shape[0] >= 6:
        X_raw = merged[["driver_age", "sex_female"]].copy()
        X_raw = X_raw.fillna(X_raw.median(numeric_only=True))
        for col in X_raw.columns:
            std = float(X_raw[col].std(ddof=0))
            if not np.isfinite(std) or std == 0:
                std = 1.0
            X_raw[col] = (X_raw[col] - float(X_raw[col].mean())) / std
        X = sm.add_constant(X_raw)
        weights = 1.0 / np.clip(merged["u_sd"].to_numpy() ** 2, 1e-6, None)
        wls = sm.WLS(merged["u_mean"], X, weights=weights).fit()
        with open(out_summary_txt, "w", encoding="utf-8") as f:
            f.write(wls.summary().as_text())

        coef_df = pd.DataFrame(
            {
                "term": wls.params.index,
                "coef": wls.params.values,
                "std_err": wls.bse.values,
                "p_value": wls.pvalues.values,
            }
        )
        coef_df.to_csv(out_csv, index=False)
    else:
        with open(out_summary_txt, "w", encoding="utf-8") as f:
            f.write("Not enough participants for participant-level disagreement regression.\n")
        coef_df = pd.DataFrame(columns=["term", "coef", "std_err", "p_value"])
        coef_df.to_csv(out_csv, index=False)

    merged.to_csv(out_csv.with_name("participant_random_effects.csv"), index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decision modeling pipeline for stop/go behavior with Bayesian mixed logistic regression."
    )
    parser.add_argument("--log-dir", type=str, default="data/logs")
    parser.add_argument("--clean-dir", type=str, default="data/clean_data")
    parser.add_argument("--results-dir", type=str, default="results/decision_model")
    parser.add_argument("--posterior-draws", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=70)
    parser.add_argument("--participant-mc-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-mcmc-checklist",
        action="store_true",
        help=(
            "Run full Bayesian peer-review diagnostics using sequential Pyro NUTS chains "
            "(R-hat, ESS, posterior predictive checks, MCMC parameter table, and AME table/plot)."
        ),
    )
    parser.add_argument("--mcmc-chains", type=int, default=4)
    parser.add_argument("--mcmc-warmup", type=int, default=350)
    parser.add_argument("--mcmc-samples", type=int, default=450)
    parser.add_argument("--mcmc-target-accept", type=float, default=0.87)
    parser.add_argument("--mcmc-max-tree-depth", type=int, default=8)
    parser.add_argument(
        "--export-fix-level",
        action="store_true",
        help="Also export per-sample fix rows with labels to clean-dir.",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    log_dir = Path(args.log_dir)
    clean_dir = Path(args.clean_dir)
    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plot_data_dir = results_dir / "plot_data"
    clean_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(log_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {log_dir}")

    run_rows: list[dict[str, Any]] = []
    fix_rows: list[pd.DataFrame] = []
    excluded_rows: list[dict[str, str]] = []

    for file_path in files:
        run_record, fix_export, status = parse_one_log(file_path)
        if status != "ok" or run_record is None:
            excluded_rows.append({"source_file": file_path.name, "reason": status})
            continue
        run_rows.append(run_record)
        if args.export_fix_level and fix_export is not None:
            fix_rows.append(fix_export)

    all_runs = pd.DataFrame(run_rows)
    excluded_df = pd.DataFrame(excluded_rows)
    if all_runs.empty:
        raise RuntimeError("No valid runs were parsed.")

    dedup_runs, duplicates_removed = deduplicate_runs(all_runs)

    valid_onset = (
        dedup_runs["initial_speed_kmh"].gt(0)
        & dedup_runs["distance_threshold_m"].gt(0)
        & dedup_runs["tti_s"].gt(0)
        & dedup_runs["a_req_mps2"].gt(0)
    )
    not_forced = dedup_runs["forced_stop_instruction"].eq(0)
    decision_runs = dedup_runs[valid_onset & not_forced].copy().reset_index(drop=True)

    all_runs.to_csv(clean_dir / "all_runs_parsed.csv", index=False)
    dedup_runs.to_csv(clean_dir / "all_runs_deduplicated.csv", index=False)
    decision_runs.to_csv(clean_dir / "decision_runs.csv", index=False)
    excluded_df.to_csv(clean_dir / "excluded_files.csv", index=False)
    duplicates_removed.to_csv(clean_dir / "duplicate_runs_removed.csv", index=False)

    if args.export_fix_level:
        fix_level_df = pd.concat(fix_rows, ignore_index=True) if fix_rows else pd.DataFrame()
        fix_level_df.to_csv(clean_dir / "fix_rows_labeled.csv", index=False)

    if decision_runs.empty:
        raise RuntimeError("No decision runs remain after filtering.")

    model_df = decision_runs.copy()
    model_df["weather"] = model_df["weather"].astype(str).replace("", "unknown")
    model_df["day_night"] = model_df["day_night"].astype(str).replace("", "unknown")

    continuous_cols = [
        "initial_speed_kmh",
        "distance_threshold_m",
        "tti_s",
        "a_req_mps2",
        "driver_age",
    ]
    model_df, scalers = add_standardized_columns(model_df, continuous_cols)

    formula = (
        "go_decision ~ z_initial_speed_kmh + z_distance_threshold_m + z_tti_s + z_a_req_mps2 + "
        "z_driver_age + sex_female + "
        "C(weather) + C(day_night)"
    )
    vc_formulas = {"participant_re": "0 + C(participant_group)"}

    model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas, model_df, vcp_p=0.5, fe_p=2)

    fit_mode = "map"
    try:
        result = model.fit_map(method="BFGS", minim_opts={"maxiter": 2000})
        sigma_re = float(np.exp(np.asarray(result.vcp_mean)[0])) if model.k_vcp > 0 else 0.0
        if (not np.isfinite(sigma_re)) or sigma_re < 0.05:
            fit_mode = "vb_fallback_after_map_shrinkage"
            result = model.fit_vb()
    except Exception:
        fit_mode = "vb_after_map_failure"
        result = model.fit_vb()

    fe_draws, vcp_draws = draw_posterior(model, result, n_draws=args.posterior_draws, seed=args.seed)

    effect_df = summarize_fixed_effects(
        fe_names=list(model.fep_names),
        fe_draws=fe_draws,
        output_csv=results_dir / "fixed_effect_posterior_summary.csv",
    )
    importance_df = plot_feature_importance(
        fe_names=list(model.fep_names),
        fe_draws=fe_draws,
        out_path=plots_dir / "feature_importance.png",
    )
    importance_df.to_csv(results_dir / "feature_importance.csv", index=False)
    factor_effect_df, age_group_effect_df = summarize_and_plot_factor_effects(
        fe_names=list(model.fep_names),
        fe_draws=fe_draws,
        model_exog=np.asarray(model.exog, dtype=float),
        model_df=model_df,
        scalers=scalers,
        out_csv=results_dir / "factor_effect_summary.csv",
        out_age_csv=results_dir / "age_group_effect_summary.csv",
        out_plot=plots_dir / "factor_effects_advanced.png",
    )

    mcmc_diag_df = pd.DataFrame()
    mcmc_ppc_df = pd.DataFrame()
    mcmc_param_df = pd.DataFrame()
    mcmc_factor_effect_df = pd.DataFrame()
    mcmc_age_group_effect_df = pd.DataFrame()
    mcmc_ppc_draws: dict[str, Any] = {}
    mcmc_error = ""

    if args.run_mcmc_checklist:
        try:
            mcmc_diag_df, mcmc_ppc_df, mcmc_param_df, mcmc_beta_draws, mcmc_ppc_draws = run_mcmc_diagnostics_and_ppc(
                model_exog=np.asarray(model.exog, dtype=float),
                y_obs=model_df["go_decision"].to_numpy(dtype=int),
                participant_codes=pd.Categorical(model_df["participant_group"]).codes.astype(int),
                fe_names=list(model.fep_names),
                n_participants=int(model_df["participant_group"].nunique()),
                n_chains=int(args.mcmc_chains),
                warmup_steps=int(args.mcmc_warmup),
                num_samples=int(args.mcmc_samples),
                target_accept=float(args.mcmc_target_accept),
                max_tree_depth=int(args.mcmc_max_tree_depth),
                seed=int(args.seed),
                diagnostics_csv=results_dir / "mcmc_chain_diagnostics.csv",
                ppc_csv=results_dir / "mcmc_ppc_summary.csv",
                parameter_summary_csv=results_dir / "mcmc_fixed_effect_summary.csv",
                out_dashboard_plot=plots_dir / "mcmc_diagnostic_dashboard.png",
            )
            mcmc_factor_effect_df, mcmc_age_group_effect_df = summarize_and_plot_factor_effects(
                fe_names=list(model.fep_names),
                fe_draws=mcmc_beta_draws,
                model_exog=np.asarray(model.exog, dtype=float),
                model_df=model_df,
                scalers=scalers,
                out_csv=results_dir / "factor_effect_summary_mcmc.csv",
                out_age_csv=results_dir / "age_group_effect_summary_mcmc.csv",
                out_plot=plots_dir / "factor_effects_advanced_mcmc.png",
            )
            plot_ame_with_validators_composite(
                ame_df=mcmc_factor_effect_df,
                diag_df=mcmc_diag_df,
                ppc_df=mcmc_ppc_df,
                ppc_draws=mcmc_ppc_draws,
                out_plot=plots_dir / "ame_validators_composite.png",
            )
        except Exception as exc:
            mcmc_error = str(exc)

    surface_summary = plot_surface_outputs(
        model=model,
        fe_draws=fe_draws,
        vcp_draws=vcp_draws,
        model_df=model_df,
        scalers=scalers,
        grid_size=args.grid_size,
        participant_mc_samples=args.participant_mc_samples,
        seed=args.seed,
        export_dir=plot_data_dir,
        out_decision=plots_dir / "decision_hesitation_surface.png",
        out_uncertainty=plots_dir / "uncertainty_disagreement_surface.png",
    )

    participant_re = participant_disagreement_regression(
        model=model,
        result=result,
        model_df=model_df,
        out_csv=results_dir / "participant_disagreement_regression.csv",
        out_summary_txt=results_dir / "participant_disagreement_regression.txt",
    )

    re_map = (
        participant_re.set_index("participant_group")["u_mean"].to_dict()
        if not participant_re.empty and "u_mean" in participant_re.columns
        else {}
    )
    row_u = model_df["participant_group"].map(re_map).fillna(0.0).to_numpy()
    eta_train = model.exog @ np.asarray(result.fe_mean) + row_u
    p_train = sigmoid(eta_train)
    y_train = model_df["go_decision"].to_numpy()

    accuracy = float(np.mean((p_train >= 0.5).astype(int) == y_train))
    brier = float(np.mean((p_train - y_train) ** 2))
    logloss = float(
        -np.mean(y_train * np.log(np.clip(p_train, 1e-9, 1.0)) + (1 - y_train) * np.log(np.clip(1 - p_train, 1e-9, 1.0)))
    )

    report_lines = [
        "Decision Modeling Report",
        "=======================",
        f"Parsed files: {len(files)}",
        f"Valid parsed runs: {len(all_runs)}",
        f"Runs after deduplication: {len(dedup_runs)}",
        f"Decision-analysis runs (forced-stop excluded): {len(decision_runs)}",
        f"Unique participants in decision model: {model_df['participant_group'].nunique()}",
        "",
        "Label definition:",
        f"  stop = speed_kmh <= {STOP_SPEED_MAX_KMH} within light_distance_m in [{STOP_DISTANCE_MIN_M}, {STOP_DISTANCE_MAX_M}]",
        "  go label = 1, stop label = 0",
        "Speed at yellow onset:",
        "  initial_speed_kmh is computed from linear interpolation at light_distance_m == distance_threshold_m",
        "  using two consecutive fix samples bracketing the threshold (not instructed speed).",
        "",
        f"Model fitting mode: {fit_mode}",
        f"Train accuracy (threshold=0.5): {accuracy:.4f}",
        f"Brier score: {brier:.4f}",
        f"Log-loss: {logloss:.4f}",
        "",
        "Core model equations:",
        "  logit(P(go_i)) = X_i * beta + u_participant[i],  u_j ~ Normal(0, sigma_u)",
        "  AME_j = E_X[ sigmoid(eta(X + Delta_j)) - sigmoid(eta(X)) ]",
        "  pd_j = max(P(theta_j > 0), P(theta_j < 0))",
        "",
        "Uncertainty / hesitation summary:",
        f"  Hesitation area fraction (0.4 <= P(go) <= 0.6): {surface_summary['hesitation_area_fraction']:.4f}",
        f"  Max disagreement variance: {surface_summary['max_disagreement_value']:.4f}",
        (
            "  Max disagreement location (speed km/h, TTI s): "
            f"({surface_summary['max_disagreement_speed_kmh']:.2f}, {surface_summary['max_disagreement_tti_s']:.2f})"
        ),
        f"  Mean 95% credible-band width on decision surface: {surface_summary['mean_credible_band_width']:.4f}",
        f"  Plot data export directory: {plot_data_dir}",
        "",
        "Bayesian checklist diagnostics (MCMC):",
    ]
    if args.run_mcmc_checklist:
        if mcmc_error:
            report_lines.append(f"  MCMC diagnostics run failed: {mcmc_error}")
        elif mcmc_diag_df.empty:
            report_lines.append("  MCMC diagnostics unavailable.")
        else:
            rhat_max = float(mcmc_diag_df["r_hat"].max())
            ess_bulk_min = float(mcmc_diag_df["ess_bulk"].min())
            ess_tail_min = float(mcmc_diag_df["ess_tail_5_95"].min())
            report_lines.append(
                f"  Max split R-hat: {rhat_max:.4f} (target <= 1.01)"
            )
            report_lines.append(
                f"  Min bulk ESS: {ess_bulk_min:.1f} (target > 400)"
            )
            report_lines.append(
                f"  Min tail ESS (5/95%): {ess_tail_min:.1f} (target > 400)"
            )
            report_lines.append(
                "  Saved diagnostics: mcmc_chain_diagnostics.csv, "
                "mcmc_fixed_effect_summary.csv, mcmc_ppc_summary.csv, "
                "plots/mcmc_diagnostic_dashboard.png, plots/ame_validators_composite.png"
            )
            if not mcmc_ppc_df.empty:
                ppc_row = mcmc_ppc_df.iloc[0]
                report_lines.append(
                    (
                        "  PPC go-count check: observed="
                        f"{int(ppc_row['observed_go_count'])}, "
                        f"pred mean={ppc_row['pp_go_count_mean']:.1f}, "
                        f"95% CrI [{ppc_row['pp_go_count_cri_2_5']:.1f}, {ppc_row['pp_go_count_cri_97_5']:.1f}], "
                        f"two-sided p={ppc_row['ppc_two_sided_p']:.3f}"
                    )
                )
    else:
        report_lines.append("  Skipped (use --run-mcmc-checklist to run full R-hat/ESS/PPC diagnostics).")

    report_lines.extend(["", "Factor-effect summary (AME-based):"])
    factor_source_df = mcmc_factor_effect_df if not mcmc_factor_effect_df.empty else factor_effect_df
    if factor_source_df.empty:
        report_lines.append("  Factor-effect summary unavailable.")
    else:
        source_label = "MCMC posterior" if not mcmc_factor_effect_df.empty else "Approximate posterior"
        report_lines.append(f"  Source: {source_label}")
        top_contrib = factor_source_df.sort_values("contribution_mean", ascending=False).head(8)
        for _, row in top_contrib.iterrows():
            report_lines.append(
                (
                    f"  {row['label']}: contribution={100.0 * row['contribution_mean']:.1f}% "
                    f"(95% CrI {100.0 * row['contribution_ci_2_5']:.1f}-{100.0 * row['contribution_ci_97_5']:.1f}%), "
                    f"AME={100.0 * row['ame_mean']:+.1f} pp "
                    f"(95% CrI {100.0 * row['ame_ci_2_5']:+.1f} to {100.0 * row['ame_ci_97_5']:+.1f}), "
                    f"pd={row.get('pd_ame', row['p_ame_gt_0']):.3f}"
                )
            )
    age_source_df = mcmc_age_group_effect_df if not mcmc_age_group_effect_df.empty else age_group_effect_df
    if not age_source_df.empty:
        report_lines.extend(
            [
                "",
                "Age-group model-implied P(go):",
            ]
        )
        for _, row in age_source_df.iterrows():
            report_lines.append(
                (
                    f"  {row['age_group']} (n={int(row['n_runs'])}, rep age={row['representative_age']:.1f}): "
                    f"P(go)={row['p_go_mean']:.3f} "
                    f"(95% CrI {row['p_go_ci_2_5']:.3f}-{row['p_go_ci_97_5']:.3f}), "
                    f"Delta vs 18-25={100.0 * row.get('delta_vs_18_25_mean', 0.0):+.1f} pp"
                )
            )

    report_lines.extend(
        [
            "",
            "Top fixed effects by absolute posterior mean:",
        ]
    )
    if not mcmc_param_df.empty:
        report_lines.append("  Source: MCMC posterior")
        top_effects = (
            mcmc_param_df.assign(abs_mean=lambda x: x["posterior_mean"].abs())
            .sort_values("abs_mean", ascending=False)
            .head(8)
        )
        for _, row in top_effects.iterrows():
            report_lines.append(
                (
                    f"  {row['term']}: mean={row['posterior_mean']:.4f}, "
                    f"95% CrI=({row['cri_2_5']:.4f}, {row['cri_97_5']:.4f}), "
                    f"pd={row['pd']:.3f}"
                )
            )
    else:
        report_lines.append("  Source: Approximate posterior")
        top_effects = (
            effect_df.assign(abs_mean=lambda x: x["posterior_mean"].abs())
            .sort_values("abs_mean", ascending=False)
            .head(8)
        )
        for _, row in top_effects.iterrows():
            report_lines.append(
                (
                    f"  {row['term']}: mean={row['posterior_mean']:.4f}, "
                    f"95% CrI=({row['ci_2_5']:.4f}, {row['ci_97_5']:.4f}), "
                    f"pd={row['pd']:.3f}"
                )
            )

    report_path = results_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n".join(report_lines))
    print(f"\nSaved cleaned data to: {clean_dir}")
    print(f"Saved model outputs to: {results_dir}")
    print(f"Saved plots to: {plots_dir}")
    print(f"Saved plotting data to: {plot_data_dir}")


if __name__ == "__main__":
    main()

