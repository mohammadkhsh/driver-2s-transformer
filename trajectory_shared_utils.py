from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
META_CSV = ROOT / "data" / "clean_data" / "all_runs_deduplicated.csv"
LOGS_DIR = ROOT / "data" / "logs"
OUT_DIR = ROOT / "results" / "trajectory_shared"
CACHE_DIR = OUT_DIR / "cache"
PUBLIC_CACHE_DIR = ROOT / "data" / "processed_trajectory_cache"
PLOTS_DIR = OUT_DIR / "plots"

STOP_SPEED_MAX_KMH = 0.2
STOP_DETECT_MIN_M = -2.0
STOP_DETECT_MAX_M = 6.5  # allow early valid stops around +6 m with sensor/sample noise
STOP_CONSTRAINT_MIN_M = -0.5
STOP_CONSTRAINT_MAX_M = 2.0
DEVICE = torch.device("cpu")
STOP_EVAL_SPEED_EPS_MPS = 0.3


def _first_stable_stop_time_s(a_row: np.ndarray, v_row: np.ndarray, T_s: float, v_eps: float = STOP_EVAL_SPEED_EPS_MPS) -> float | None:
    if not np.isfinite(T_s) or T_s <= 0 or len(v_row) == 0:
        return None
    t_src = np.linspace(0.0, float(T_s), len(v_row), dtype=np.float64)
    v = np.nan_to_num(np.asarray(v_row, dtype=np.float64), nan=np.inf, posinf=np.inf, neginf=0.0)
    cand = np.flatnonzero(v <= float(v_eps))
    for idx in cand:
        if np.nanmax(v[idx:]) <= float(v_eps) + 1e-6:
            return float(t_src[int(idx)])
    return None


def _interp_stop_padded_triplet(a_row: np.ndarray, d_row: np.ndarray, v_row: np.ndarray, T_s: float, t_eval: np.ndarray, stop_t_s: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_src = np.asarray(a_row, dtype=np.float64)
    d_src = np.asarray(d_row, dtype=np.float64)
    v_src = np.asarray(v_row, dtype=np.float64)
    T_s = float(max(T_s, 1e-6))
    t_src = np.linspace(0.0, T_s, a_src.size, dtype=np.float64)
    tq = np.asarray(t_eval, dtype=np.float64)
    t_clip = np.clip(tq, 0.0, T_s)
    a_out = np.interp(t_clip, t_src, a_src)
    d_out = np.interp(t_clip, t_src, d_src)
    v_out = np.interp(t_clip, t_src, v_src)
    after_src = tq > T_s
    if np.any(after_src):
        a_out[after_src] = float(a_src[-1]); d_out[after_src] = float(d_src[-1]); v_out[after_src] = float(v_src[-1])
    if stop_t_s is not None and np.isfinite(stop_t_s):
        ts = float(max(0.0, min(stop_t_s, T_s)))
        d_stop = float(np.interp(ts, t_src, d_src))
        mask = tq >= ts
        a_out[mask] = 0.0
        v_out[mask] = 0.0
        d_out[mask] = d_stop
    return a_out, d_out, v_out


def _compute_stop_extended_metrics(
    a_true: np.ndarray,
    d_true: np.ndarray,
    v_true: np.ndarray,
    T_true: np.ndarray,
    a_pred: np.ndarray,
    d_pred: np.ndarray,
    v_pred: np.ndarray,
    T_pred: np.ndarray,
    go: np.ndarray,
    d_end_obs: np.ndarray,
    v_end_obs: np.ndarray,
) -> dict[str, Any]:
    go = np.asarray(go).astype(int)
    n = len(go)
    a_abs_sum = a_sq_sum = 0.0
    n_a = 0
    d_term = np.zeros(n, dtype=np.float64)
    v_term = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if go[i] == 1:
            da = np.asarray(a_true[i], dtype=np.float64) - np.asarray(a_pred[i], dtype=np.float64)
            a_abs_sum += float(np.sum(np.abs(da))); a_sq_sum += float(np.sum(da * da)); n_a += da.size
            d_term[i] = float(np.asarray(d_pred[i], dtype=np.float64)[-1])
            v_term[i] = float(np.asarray(v_pred[i], dtype=np.float64)[-1])
            continue
        Tt = float(np.asarray(T_true, dtype=np.float64)[i]); Tp = float(np.asarray(T_pred, dtype=np.float64)[i])
        a_t = np.asarray(a_true[i], dtype=np.float64); d_t = np.asarray(d_true[i], dtype=np.float64); v_t = np.asarray(v_true[i], dtype=np.float64)
        a_p = np.asarray(a_pred[i], dtype=np.float64); d_p = np.asarray(d_pred[i], dtype=np.float64); v_p = np.asarray(v_pred[i], dtype=np.float64)
        t_stop_true = _first_stable_stop_time_s(a_t, v_t, Tt)
        t_stop_pred = _first_stable_stop_time_s(a_p, v_p, Tp)
        t_eval_end = max(Tt if t_stop_true is None else t_stop_true, Tp if t_stop_pred is None else t_stop_pred)
        dt_ref = min(max(Tt, 1e-3), max(Tp, 1e-3)) / max(len(a_t) - 1, 1)
        dt_ref = max(float(dt_ref), 1e-3)
        m = max(int(len(a_t)), int(math.ceil(t_eval_end / dt_ref)) + 1)
        t_eval = np.arange(m, dtype=np.float64) * dt_ref
        a_t_e, d_t_e, v_t_e = _interp_stop_padded_triplet(a_t, d_t, v_t, Tt, t_eval, t_stop_true)
        a_p_e, d_p_e, v_p_e = _interp_stop_padded_triplet(a_p, d_p, v_p, Tp, t_eval, t_stop_pred)
        da = a_t_e - a_p_e
        a_abs_sum += float(np.sum(np.abs(da))); a_sq_sum += float(np.sum(da * da)); n_a += da.size
        d_term[i] = float(d_p_e[-1]); v_term[i] = float(v_p_e[-1])
    return {
        "traj_mae_mps2": float(a_abs_sum / max(n_a, 1)),
        "traj_rmse_mps2": float(math.sqrt(a_sq_sum / max(n_a, 1))),
        "terminal_distance_mae_m": float(mean_absolute_error(np.asarray(d_end_obs, dtype=np.float64), d_term)),
        "terminal_speed_mae_mps": float(mean_absolute_error(np.asarray(v_end_obs, dtype=np.float64), v_term)),
        "d_term_eval": d_term,
        "v_term_eval": v_term,
    }


@dataclass
class Config:
    seed: int = 42
    traj_points: int = 64
    train_ratio: float = 0.60
    decision_epochs: int = 80
    decision_batch_size: int = 64
    decision_lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim_decision: int = 64
    a_clip_mps2: float = 12.0
    min_event_duration_s: float = 0.35
    max_event_duration_s: float = 20.0
    use_only_voluntary: bool = True
    extraction_logic_version: int = 3


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def choose_time_column(df_fix: pd.DataFrame, preferred: str | None = None) -> str:
    for c in ([preferred] if preferred else []) + ["t_wall", "ticks_utc"]:
        if c is None or c not in df_fix.columns:
            continue
        arr = pd.to_numeric(df_fix[c], errors="coerce")
        if np.isfinite(arr).sum() >= 2:
            return c
    raise ValueError("No valid time column.")


def read_fix_log(log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path, low_memory=False)
    if "type" not in df.columns:
        raise ValueError("Missing type column")
    fix = df[df["type"].astype(str).str.lower() == "fix"].copy()
    if fix.empty:
        raise ValueError("No fix rows")
    for c in ["t_wall", "ticks_utc", "speed_kmh", "light_distance_m", "ax_g", "hr_bpm"]:
        if c in fix.columns:
            fix[c] = pd.to_numeric(fix[c], errors="coerce")
    fix = fix.dropna(subset=["speed_kmh", "light_distance_m"])
    tcol = choose_time_column(fix)
    fix = fix.sort_values(tcol).reset_index(drop=True)
    return fix


def _interp_scalar(x0: float, x1: float, y0: float, y1: float, x: float) -> float:
    if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1)):
        return float("nan")
    if abs(x1 - x0) < 1e-12:
        return float(y0)
    a = (x - x0) / (x1 - x0)
    return float(y0 + a * (y1 - y0))


def interpolate_row_by_time(fix: pd.DataFrame, tcol: str, t_query: float) -> tuple[dict[str, float], int, int]:
    t = fix[tcol].to_numpy(dtype=float)
    if t_query < t[0] or t_query > t[-1]:
        raise ValueError("Query time out of range")
    idx = int(np.searchsorted(t, t_query, side="left"))
    if idx < len(t) and abs(t[idx] - t_query) <= 1e-9:
        row = {
            tcol: float(t_query),
            "speed_kmh": float(fix.loc[idx, "speed_kmh"]),
            "light_distance_m": float(fix.loc[idx, "light_distance_m"]),
            "ax_g": float(fix.loc[idx, "ax_g"]) if "ax_g" in fix.columns else float("nan"),
        }
        return row, idx, idx
    i1 = min(max(idx, 1), len(t) - 1)
    i0 = i1 - 1
    out = {tcol: float(t_query)}
    for c in ["speed_kmh", "light_distance_m", "ax_g"]:
        out[c] = (
            _interp_scalar(
                float(fix.loc[i0, tcol]),
                float(fix.loc[i1, tcol]),
                float(fix.loc[i0, c]),
                float(fix.loc[i1, c]),
                float(t_query),
            )
            if c in fix.columns
            else float("nan")
        )
    return out, i0, i1


def interpolate_crossing_at_distance(
    fix: pd.DataFrame, tcol: str, target_distance: float = 0.0, start_time: float | None = None
) -> dict[str, float]:
    work = fix.copy()
    if start_time is not None:
        work = work[work[tcol] >= float(start_time)].reset_index(drop=True)
    d = work["light_distance_m"].to_numpy(dtype=float)
    t = work[tcol].to_numpy(dtype=float)
    if len(work) < 2:
        raise ValueError("Insufficient rows for crossing interpolation")
    for i in range(1, len(work)):
        d0, d1 = d[i - 1], d[i]
        if not (np.isfinite(d0) and np.isfinite(d1)):
            continue
        exact0 = abs(d0 - target_distance) <= 1e-9
        exact1 = abs(d1 - target_distance) <= 1e-9
        crossed = (d0 - target_distance) * (d1 - target_distance) < 0
        if exact0 or exact1 or crossed:
            if exact0:
                tq = float(t[i - 1])
            elif exact1:
                tq = float(t[i])
            else:
                tq = _interp_scalar(d0, d1, t[i - 1], t[i], target_distance)
            return {
                tcol: float(tq),
                "light_distance_m": float(target_distance),
                "speed_kmh": _interp_scalar(t[i - 1], t[i], work.loc[i - 1, "speed_kmh"], work.loc[i, "speed_kmh"], tq),
                "ax_g": _interp_scalar(t[i - 1], t[i], work.loc[i - 1, "ax_g"], work.loc[i, "ax_g"], tq)
                if "ax_g" in work.columns
                else float("nan"),
            }
    raise ValueError("No light crossing found")


def find_stop_event_row(fix: pd.DataFrame, tcol: str, start_time: float) -> dict[str, float]:
    work = fix[fix[tcol] >= float(start_time)].copy().reset_index(drop=True)
    mask = (
        (work["speed_kmh"] <= STOP_SPEED_MAX_KMH)
        & (work["light_distance_m"] >= STOP_DETECT_MIN_M)
        & (work["light_distance_m"] <= STOP_DETECT_MAX_M)
    )
    if not mask.any():
        raise ValueError("No stop event in detection window")
    idx = int(np.flatnonzero(mask.to_numpy())[0])
    row = work.loc[idx]
    return {
        tcol: float(row[tcol]),
        "speed_kmh": float(row["speed_kmh"]),
        "light_distance_m": float(row["light_distance_m"]),
        "ax_g": float(row["ax_g"]) if "ax_g" in work.columns else float("nan"),
    }


def build_segment_with_endpoints(fix: pd.DataFrame, tcol: str, onset: dict[str, float], event: dict[str, float]) -> pd.DataFrame:
    t0 = float(onset[tcol])
    te = float(event[tcol])
    if te <= t0:
        raise ValueError("Invalid event time <= onset time")
    mid = fix[(fix[tcol] > t0) & (fix[tcol] < te)][[tcol, "speed_kmh", "light_distance_m", "ax_g"]].copy()
    endpoints = pd.DataFrame([onset, event])[[tcol, "speed_kmh", "light_distance_m", "ax_g"]]
    seg = pd.concat([endpoints.iloc[[0]], mid, endpoints.iloc[[1]]], ignore_index=True)
    seg = seg.drop_duplicates(subset=[tcol], keep="first").sort_values(tcol).reset_index(drop=True)
    return seg


def resample_segment(seg: pd.DataFrame, tcol: str, n_points: int, a_clip_mps2: float) -> dict[str, Any]:
    t = seg[tcol].to_numpy(dtype=float)
    v_kmh = seg["speed_kmh"].to_numpy(dtype=float)
    d_m = seg["light_distance_m"].to_numpy(dtype=float)
    ax = np.nan_to_num(seg["ax_g"].to_numpy(dtype=float), nan=0.0)
    duration = float(t[-1] - t[0])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid segment duration")
    u = (t - t[0]) / duration
    u_grid = np.linspace(0.0, 1.0, n_points, dtype=float)
    # Use signed longitudinal acceleration from ax_g (negative=braking, positive=acceleration).
    a = np.clip(ax, -a_clip_mps2, a_clip_mps2)
    return {
        "ax_target_mps2": np.interp(u_grid, u, a).astype(np.float32),
        "v_target_kmh": np.interp(u_grid, u, np.nan_to_num(v_kmh, nan=0.0)).astype(np.float32),
        "d_target_m": np.interp(u_grid, u, np.nan_to_num(d_m, nan=np.nanmedian(d_m))).astype(np.float32),
        "duration_s": duration,
        "v_end_obs_mps": float(v_kmh[-1] / 3.6),
        "d_end_obs_m": float(d_m[-1]),
    }


def extract_run_trajectory(row: pd.Series, n_points: int, a_clip_mps2: float) -> tuple[dict[str, Any] | None, str]:
    path = LOGS_DIR / str(row["source_file"])
    if not path.exists():
        return None, "missing_log_file"
    try:
        fix = read_fix_log(path)
        tcol = choose_time_column(fix, str(row.get("yellow_onset_time_source", "t_wall")))
        t_on = float(row["yellow_onset_time_value"])
        onset, _, _ = interpolate_row_by_time(fix, tcol, t_on)
        meta_go = int(row["go_decision"])
        meta_stop = int(row["stop_decision"])
        override_applied = 0
        override_reason = ""
        if int(row["stop_decision"]) == 1:
            event = find_stop_event_row(fix, tcol, t_on)
            event_type = "stop"
        elif int(row["go_decision"]) == 1:
            try:
                event = interpolate_crossing_at_distance(fix, tcol, 0.0, start_time=t_on)
                event_type = "go"
            except Exception as cross_err:  # noqa: BLE001
                # Robust fallback: some runs are mislabeled as go when they actually stop early (>+5 m).
                try:
                    event = find_stop_event_row(fix, tcol, t_on)
                    event_type = "stop"
                    override_applied = 1
                    override_reason = f"go_label_no_crossing_fallback_stop:{type(cross_err).__name__}"
                except Exception:
                    raise cross_err
        else:
            return None, "invalid_decision_label"
        seg = build_segment_with_endpoints(fix, tcol, onset, event)
        res = resample_segment(seg, tcol, n_points, a_clip_mps2)
        dur = float(res["duration_s"])
        if dur < CFG.min_event_duration_s or dur > CFG.max_event_duration_s:
            return None, "duration_out_of_range"
        out = {
            "source_file": str(row["source_file"]),
            "traj_event_type": event_type,
            "traj_time_col": tcol,
            "traj_t_onset": float(onset[tcol]),
            "traj_t_event": float(event[tcol]),
            "traj_duration_s": dur,
            "traj_onset_speed_kmh_raw_interp": float(onset["speed_kmh"]),
            "traj_onset_distance_m_raw_interp": float(onset["light_distance_m"]),
            "traj_event_speed_kmh_raw_interp": float(event["speed_kmh"]),
            "traj_event_distance_m_raw_interp": float(event["light_distance_m"]),
            "traj_v_end_obs_mps": float(res["v_end_obs_mps"]),
            "traj_d_end_obs_m": float(res["d_end_obs_m"]),
            "go_decision_meta_original": meta_go,
            "stop_decision_meta_original": meta_stop,
            "decision_label_override_applied": override_applied,
            "decision_label_override_reason": override_reason,
            "go_decision": int(1 if event_type == "go" else 0),
            "stop_decision": int(1 if event_type == "stop" else 0),
            "ax_target_mps2": res["ax_target_mps2"],
            "v_target_kmh": res["v_target_kmh"],
            "d_target_m": res["d_target_m"],
        }
        return out, "ok"
    except Exception as e:  # noqa: BLE001
        return None, f"error:{type(e).__name__}:{e}"


def build_or_load_trajectory_cache(meta_df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    cache_npz = CACHE_DIR / "trajectory_arrays.npz"
    cache_csv = CACHE_DIR / "trajectory_meta.csv"
    cache_info = CACHE_DIR / "trajectory_cache_info.json"
    info_key = {
        "meta_rows": int(len(meta_df)),
        "traj_points": int(cfg.traj_points),
        "a_clip_mps2": float(cfg.a_clip_mps2),
        "stop_detect_min_m": float(STOP_DETECT_MIN_M),
        "stop_detect_max_m": float(STOP_DETECT_MAX_M),
        "extraction_logic_version": int(cfg.extraction_logic_version),
    }
    if cache_npz.exists() and cache_csv.exists() and cache_info.exists():
        try:
            if json.loads(cache_info.read_text()) == info_key:
                m = pd.read_csv(cache_csv)
                npz = np.load(cache_npz)
                return m, {k: npz[k] for k in npz.files}, pd.read_csv(CACHE_DIR / "trajectory_extraction_summary.csv")
        except Exception:
            pass

    recs: list[dict[str, Any]] = []
    b_list: list[np.ndarray] = []
    v_list: list[np.ndarray] = []
    d_list: list[np.ndarray] = []
    statuses: list[dict[str, Any]] = []
    for _, row in meta_df.iterrows():
        traj, status = extract_run_trajectory(row, cfg.traj_points, cfg.a_clip_mps2)
        statuses.append({"source_file": row["source_file"], "status": status})
        if traj is None:
            continue
        arrs = {k: traj.pop(k) for k in ["ax_target_mps2", "v_target_kmh", "d_target_m"]}
        rec = dict(row.to_dict())
        rec.update(traj)
        recs.append(rec)
        b_list.append(np.asarray(arrs["ax_target_mps2"], dtype=np.float32))
        v_list.append(np.asarray(arrs["v_target_kmh"], dtype=np.float32))
        d_list.append(np.asarray(arrs["d_target_m"], dtype=np.float32))

    traj_meta = pd.DataFrame(recs).reset_index(drop=True)
    arrays = {
        "ax_target_mps2": np.stack(b_list, axis=0) if b_list else np.empty((0, cfg.traj_points), dtype=np.float32),
        "v_target_kmh": np.stack(v_list, axis=0) if v_list else np.empty((0, cfg.traj_points), dtype=np.float32),
        "d_target_m": np.stack(d_list, axis=0) if d_list else np.empty((0, cfg.traj_points), dtype=np.float32),
    }
    traj_meta.to_csv(cache_csv, index=False)
    np.savez_compressed(cache_npz, **arrays)
    cache_info.write_text(json.dumps(info_key, indent=2))
    status_df = pd.DataFrame(statuses)
    status_df.to_csv(CACHE_DIR / "trajectory_extraction_status.csv", index=False)
    summ = status_df.groupby("status", dropna=False)["source_file"].count().reset_index(name="count").sort_values("count", ascending=False)
    summ.to_csv(CACHE_DIR / "trajectory_extraction_summary.csv", index=False)
    return traj_meta, arrays, summ


def _load_public_trajectory_cache() -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame] | None:
    cache_npz = PUBLIC_CACHE_DIR / "trajectory_arrays.npz"
    cache_csv = PUBLIC_CACHE_DIR / "trajectory_meta.csv"
    summary_csv = PUBLIC_CACHE_DIR / "trajectory_extraction_summary.csv"
    if not (cache_npz.exists() and cache_csv.exists()):
        return None
    meta = pd.read_csv(cache_csv)
    npz = np.load(cache_npz)
    arrays = {k: npz[k] for k in npz.files}
    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
    else:
        summary = pd.DataFrame({"status": ["loaded_public_cache"], "count": [len(meta)]})
    return meta, arrays, summary


def prepare_modeling_dataframe(cfg: Config) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    if META_CSV.exists():
        meta = pd.read_csv(META_CSV)
        use_cached_directly = False
    else:
        cached = _load_public_trajectory_cache()
        if cached is None:
            raise FileNotFoundError(
                f"Missing {META_CSV}. Provide raw cleaned metadata or include "
                f"the processed public cache under {PUBLIC_CACHE_DIR}."
            )
        meta, arrays, extraction_summary = cached
        use_cached_directly = True
    if cfg.use_only_voluntary:
        meta = meta[meta["forced_stop_instruction"].fillna(0).astype(int) == 0].copy()
    req = [
        "source_file",
        "speed_at_yellow_kmh",
        "distance_threshold_m",
        "tti_s",
        "a_req_mps2",
        "driver_age",
        "go_decision",
        "stop_decision",
    ]
    meta = meta.dropna(subset=req).copy()
    meta["driver_age"] = pd.to_numeric(meta["driver_age"], errors="coerce")
    meta = meta.dropna(subset=["driver_age"]).copy()
    if "sex_female" not in meta.columns:
        meta["sex_female"] = 0.0
    meta["sex_female"] = pd.to_numeric(meta["sex_female"], errors="coerce").fillna(0.0)
    valid = meta["go_decision"].isin([0, 1]) & meta["stop_decision"].isin([0, 1]) & ((meta["go_decision"] + meta["stop_decision"]) == 1)
    meta = meta[valid].copy()
    meta["go_decision"] = meta["go_decision"].astype(int)
    meta["stop_decision"] = meta["stop_decision"].astype(int)

    if use_cached_directly:
        keep_idx = meta.index.to_numpy(dtype=int)
        traj_meta = meta.reset_index(drop=True)
        arrays = {k: v[keep_idx] for k, v in arrays.items()}
    else:
        traj_meta, arrays, extraction_summary = build_or_load_trajectory_cache(meta, cfg)
    ok = np.isfinite(traj_meta["traj_duration_s"].to_numpy(dtype=float))
    traj_meta = traj_meta[ok].reset_index(drop=True)
    arrays = {k: v[ok] for k, v in arrays.items()}
    extraction_summary = extraction_summary.copy()
    extraction_summary["meta_rows_after_filter"] = len(meta)
    extraction_summary.to_csv(OUT_DIR / "trajectory_extraction_summary.csv", index=False)
    return traj_meta, arrays, extraction_summary


def batch_iter(indices: np.ndarray, batch_size: int, shuffle: bool = True) -> list[np.ndarray]:
    idx = indices.copy()
    if shuffle:
        np.random.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]


class DecisionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_decision_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    feature_names: list[str] | None = None,
) -> tuple[DecisionMLP, dict[str, Any], dict[str, list[float]], StandardScaler]:
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_val_sc = scaler.transform(X_val).astype(np.float32)
    model = DecisionMLP(X_train_sc.shape[1], cfg.hidden_dim_decision).to(DEVICE)

    y_train_f = y_train.astype(np.float32)
    pos = float((y_train_f == 1).sum())
    neg = float((y_train_f == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.decision_lr, weight_decay=cfg.weight_decay)

    Xtr = torch.tensor(X_train_sc, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_train_f, dtype=torch.float32, device=DEVICE)
    Xva = torch.tensor(X_val_sc, dtype=torch.float32, device=DEVICE)
    yva = torch.tensor(y_val.astype(np.float32), dtype=torch.float32, device=DEVICE)

    hist = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -np.inf

    for _epoch in range(cfg.decision_epochs):
        model.train()
        loss_vals: list[float] = []
        for bidx in batch_iter(np.arange(len(X_train_sc)), cfg.decision_batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            logits = model(Xtr[bidx])
            loss = criterion(logits, ytr[bidx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            loss_vals.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(loss_vals)) if loss_vals else float("nan")

        model.eval()
        with torch.no_grad():
            logits_val = model(Xva)
            val_loss = float(criterion(logits_val, yva).detach().cpu())
            prob_val = torch.sigmoid(logits_val).cpu().numpy()
        pred_val = (prob_val >= 0.5).astype(int)
        f1 = float(f1_score(y_val, pred_val, zero_division=0))
        try:
            fpr, tpr, _ = roc_curve(y_val, prob_val)
            auc_val = float(auc(fpr, tpr))
        except Exception:
            auc_val = float("nan")
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_f1"].append(f1)
        hist["val_auc"].append(auc_val)
        score = f1 + 0.25 * (auc_val if np.isfinite(auc_val) else 0.0)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        prob_train = torch.sigmoid(model(Xtr)).cpu().numpy()
        prob_val = torch.sigmoid(model(Xva)).cpu().numpy()
    pred_train = (prob_train >= 0.5).astype(int)
    pred_val = (prob_val >= 0.5).astype(int)

    def _metrics(y_true: np.ndarray, p: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
        out = {
            "accuracy": float(accuracy_score(y_true, y_hat)),
            "precision": float(precision_score(y_true, y_hat, zero_division=0)),
            "recall": float(recall_score(y_true, y_hat, zero_division=0)),
            "f1": float(f1_score(y_true, y_hat, zero_division=0)),
            "brier": float(brier_score_loss(y_true, p)),
        }
        try:
            fpr, tpr, _ = roc_curve(y_true, p)
            out["auroc"] = float(auc(fpr, tpr))
        except Exception:
            out["auroc"] = float("nan")
        return out

    if feature_names is None:
        if X_train_sc.shape[1] == 5:
            feature_names_out = ["v0_kmh", "d_th_m", "tti_s", "a_req_mps2", "age_years"]
        else:
            feature_names_out = [f"x{i}" for i in range(X_train_sc.shape[1])]
    else:
        feature_names_out = [str(x) for x in feature_names]

    metrics = {
        "train": _metrics(y_train, prob_train, pred_train),
        "val": _metrics(y_val, prob_val, pred_val),
        "val_confusion_matrix": confusion_matrix(y_val, pred_val).tolist(),
        "feature_names": feature_names_out,
    }
    outputs = {
        "X_train_scaled": X_train_sc,
        "X_val_scaled": X_val_sc,
        "train_prob": prob_train,
        "val_prob": prob_val,
        "train_pred": pred_train,
        "val_pred": pred_val,
    }
    return model, {"metrics": metrics, "outputs": outputs}, hist, scaler



def save_json(obj: dict[str, Any], path: Path) -> None:
    def _conv(v: Any) -> Any:
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {str(k): _conv(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_conv(x) for x in v]
        return v

    path.write_text(json.dumps(_conv(obj), indent=2), encoding="utf-8")


def plot_decision_results(y_val: np.ndarray, prob_val: np.ndarray, pred_val: np.ndarray, hist: dict[str, list[float]], out_dir: Path) -> None:
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.2])
    ax_cm = fig.add_subplot(gs[0, 0])
    ax_roc = fig.add_subplot(gs[0, 1])
    ax_cur = fig.add_subplot(gs[0, 2])

    cm = confusion_matrix(y_val, pred_val)
    im = ax_cm.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontweight="bold")
    ax_cm.set_xticks([0, 1], labels=["Stop", "Go"])
    ax_cm.set_yticks([0, 1], labels=["Stop", "Go"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title("Decision MLP Confusion Matrix")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    try:
        fpr, tpr, _ = roc_curve(y_val, prob_val)
        auc_val = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2.2, color="#1d3557", label=f"AUC={auc_val:.3f}")
    except Exception:
        pass
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC (Validation)")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(alpha=0.25)

    ax_cur.plot(hist["train_loss"], lw=2.2, color="#2a9d8f", label="Train loss")
    ax_cur.plot(hist["val_loss"], lw=2.2, color="#e76f51", label="Val loss")
    ax_cur2 = ax_cur.twinx()
    ax_cur2.plot(hist["val_f1"], lw=2, ls="--", color="#264653", label="Val F1")
    ax_cur2.plot(hist["val_auc"], lw=2, ls=":", color="#6a4c93", label="Val AUROC")
    ax_cur.set_xlabel("Epoch")
    ax_cur.set_ylabel("BCE loss")
    ax_cur2.set_ylabel("F1 / AUROC")
    ax_cur.set_title("Decision Training Curves")
    ax_cur.grid(alpha=0.25)
    h1, l1 = ax_cur.get_legend_handles_labels()
    h2, l2 = ax_cur2.get_legend_handles_labels()
    ax_cur.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "decision_model_summary.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "decision_model_summary.pdf", bbox_inches="tight")
    plt.close(fig)


