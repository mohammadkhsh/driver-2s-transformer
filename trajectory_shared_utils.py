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
OUT_DIR = ROOT / "results" / "trajectory_mixture_primitives"
CACHE_DIR = OUT_DIR / "cache"
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
    trajectory_epochs: int = 180
    decision_batch_size: int = 64
    trajectory_batch_size: int = 64
    decision_lr: float = 1e-3
    trajectory_lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim_decision: int = 64
    hidden_dim_traj: int = 128
    primitive_count: int = 6
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


def prepare_modeling_dataframe(cfg: Config) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    meta = pd.read_csv(META_CSV)
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


class PrimitiveMixtureNet(nn.Module):
    """Conditional mixture-of-primitives trajectory model (6 primitives).

    Predicts signed longitudinal acceleration a_x(t) from yellow onset to event.
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_primitives: int, n_points: int) -> None:
        super().__init__()
        assert n_primitives == 6, "Expected 6 primitives."
        self.n_points = n_points
        self.n_primitives = n_primitives
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
        )
        self.gate_head = nn.Linear(hidden_dim, n_primitives)
        self.amp_head = nn.Linear(hidden_dim, n_primitives)
        # shape params: p2, p3, k4, k5, c5, mu6, sg6, baseline
        self.shape_head = nn.Linear(hidden_dim, 8)
        self.duration_head = nn.Linear(hidden_dim, 1)
        self.register_buffer("s_grid", torch.linspace(0.0, 1.0, n_points, dtype=torch.float32).view(1, -1))

    def _primitive_bank(self, amps: torch.Tensor, shp: torch.Tensor) -> torch.Tensor:
        B = amps.shape[0]
        s = self.s_grid.expand(B, -1)
        eps = 1e-6
        p2 = 1.0 + 3.0 * torch.sigmoid(shp[:, 0:1])
        p3 = 1.0 + 3.0 * torch.sigmoid(shp[:, 1:2])
        k4 = 0.2 + 5.8 * torch.sigmoid(shp[:, 2:3])
        k5 = 5.0 + 20.0 * torch.sigmoid(shp[:, 3:4])
        c5 = 0.05 + 0.90 * torch.sigmoid(shp[:, 4:5])
        mu6 = 0.10 + 0.80 * torch.sigmoid(shp[:, 5:6])
        sg6 = 0.05 + 0.30 * torch.sigmoid(shp[:, 6:7])
        base = 0.15 * torch.sigmoid(shp[:, 7:8])

        p1 = amps[:, 0:1] * (1.0 + base) * torch.ones_like(s)
        p2f = amps[:, 1:2] * torch.pow(torch.clamp(s, min=0.0), p2)
        p3f = amps[:, 2:3] * torch.pow(torch.clamp(1.0 - s, min=0.0), p3)
        exp_num = torch.exp(k4 * s) - 1.0
        exp_den = torch.exp(k4) - 1.0
        p4f = amps[:, 3:4] * (exp_num / torch.clamp(exp_den, min=eps))
        sig = torch.sigmoid(k5 * (s - c5))
        sig0 = torch.sigmoid(k5 * (0.0 - c5))
        sig1 = torch.sigmoid(k5 * (1.0 - c5))
        p5f = amps[:, 4:5] * ((sig - sig0) / torch.clamp(sig1 - sig0, min=eps))
        p6f = amps[:, 5:6] * torch.exp(-0.5 * torch.square((s - mu6) / torch.clamp(sg6, min=eps)))

        return torch.clamp(torch.stack([p1, p2f, p3f, p4f, p5f, p6f], dim=1), min=-12.0, max=12.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        gates = torch.softmax(self.gate_head(h), dim=-1)
        # Signed amplitudes let the model represent braking (negative) and acceleration (positive).
        amps = 12.0 * torch.tanh(self.amp_head(h))
        shp = self.shape_head(h)
        duration_s = 0.35 + 19.65 * torch.sigmoid(self.duration_head(h)).squeeze(-1)
        bank = self._primitive_bank(amps, shp)
        b_pred = torch.sum(gates.unsqueeze(-1) * bank, dim=1)
        b_pred = torch.clamp(b_pred, min=-12.0, max=12.0)
        return {"gates": gates, "amps": amps, "shape_raw": shp, "primitives": bank, "b_pred": b_pred, "duration_s": duration_s}


def integrate_kinematics_from_acceleration(
    a_mps2: torch.Tensor,
    duration_s: torch.Tensor,
    v0_mps: torch.Tensor,
    d0_m: torch.Tensor,
    go_label: torch.Tensor | None = None,
    stop_reaction_delay_steps: int = 4,
) -> dict[str, torch.Tensor]:
    """Integrate kinematics from signed longitudinal acceleration.

    Remaining distance to light:
      d_{k+1} = d_k - (v_k*dt + 0.5*a_k*dt^2)
    Speed:
      v_{k+1} = max(v_k + a_k*dt, 0)
    """
    B, N = a_mps2.shape
    dt = duration_s / max(N - 1, 1)
    v_list = [v0_mps]
    d_list = [d0_m]
    a_eff_list: list[torch.Tensor] = []
    if go_label is None:
        go_mask_all = torch.ones(B, dtype=torch.bool, device=a_mps2.device)
        stop_mask_all = torch.zeros(B, dtype=torch.bool, device=a_mps2.device)
    else:
        go_mask_all = go_label >= 0.5
        stop_mask_all = ~go_mask_all
    stopped = torch.zeros(B, dtype=torch.bool, device=a_mps2.device)
    eps_v = 1e-4

    for k in range(N - 1):
        ak = a_mps2[:, k]
        vk = v_list[-1]
        dk = d_list[-1]

        # After the stop event is reached, enforce exact zero acceleration and zero speed forever.
        ak = torch.where(stopped, torch.zeros_like(ak), ak)

        # Hard stop-conditioning after a short reaction-delay region:
        # once the stop decision is active, disallow positive acceleration and enforce enough decel
        # to remain stoppable before reaching the light.
        if k >= stop_reaction_delay_steps:
            active_stop = stop_mask_all & (~stopped)
            ak = torch.where(active_stop, torch.minimum(ak, torch.zeros_like(ak)), ak)
            valid_need = active_stop & (dk > 0.0) & (vk > eps_v)
            a_need = -torch.square(vk) / torch.clamp(2.0 * dk, min=1e-3)
            ak = torch.where(valid_need, torch.minimum(ak, a_need), ak)

        ds = vk * dt + 0.5 * ak * dt * dt
        d_next = dk - ds
        v_next = torch.clamp(vk + ak * dt, min=0.0)

        # Hard safety constraint for stop-conditioned samples:
        # if the line is reached, clamp to d=0 and force v=0, a=0 immediately.
        crossed_line_stop = stop_mask_all & (~stopped) & (d_next <= 0.0)
        d_next = torch.where(crossed_line_stop, torch.zeros_like(d_next), d_next)
        v_next = torch.where(crossed_line_stop, torch.zeros_like(v_next), v_next)
        ak = torch.where(crossed_line_stop, torch.zeros_like(ak), ak)

        # If speed reaches zero in stop-conditioned samples, clamp and freeze subsequent steps.
        reached_stop = stop_mask_all & (~stopped) & (v_next <= eps_v)
        v_next = torch.where(reached_stop, torch.zeros_like(v_next), v_next)
        ak = torch.where(reached_stop, torch.zeros_like(ak), ak)

        newly_stopped = crossed_line_stop | reached_stop
        stopped = stopped | newly_stopped

        # For samples already stopped before this step, keep distance constant exactly.
        d_next = torch.where(stopped & (~newly_stopped), dk, d_next)

        a_eff_list.append(ak)
        d_list.append(d_next)
        v_list.append(v_next)

    v = torch.stack(v_list, dim=1)
    d = torch.stack(d_list, dim=1)
    if a_eff_list:
        a_eff = torch.stack(a_eff_list + [torch.zeros_like(a_eff_list[-1])], dim=1)
    else:
        a_eff = torch.zeros_like(a_mps2)
    return {"v_mps": v, "d_m": d, "a_eff": a_eff, "dt_s": dt}


def make_stage2_features(base_X: np.ndarray, go_signal: np.ndarray, sex_female: np.ndarray) -> np.ndarray:
    return np.column_stack([base_X, go_signal.reshape(-1, 1), sex_female.reshape(-1, 1)]).astype(np.float32)


def train_trajectory_model(
    X2_train: np.ndarray,
    X2_val_oracle: np.ndarray,
    X2_val_cascade: np.ndarray,
    y_traj_train: np.ndarray,
    y_traj_val: np.ndarray,
    d_traj_train: np.ndarray,
    d_traj_val: np.ndarray,
    v_traj_train_mps: np.ndarray,
    v_traj_val_mps: np.ndarray,
    dur_train: np.ndarray,
    dur_val: np.ndarray,
    v0_train_mps: np.ndarray,
    v0_val_mps: np.ndarray,
    d0_train_m: np.ndarray,
    d0_val_m: np.ndarray,
    go_train: np.ndarray,
    go_val: np.ndarray,
    d_end_obs_train: np.ndarray,
    d_end_obs_val: np.ndarray,
    v_end_obs_train: np.ndarray,
    v_end_obs_val: np.ndarray,
    cfg: Config,
) -> tuple[PrimitiveMixtureNet, dict[str, Any], dict[str, list[float]], StandardScaler]:
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X2_train).astype(np.float32)
    Xva_or_sc = scaler.transform(X2_val_oracle).astype(np.float32)
    Xva_ca_sc = scaler.transform(X2_val_cascade).astype(np.float32)
    model = PrimitiveMixtureNet(Xtr_sc.shape[1], cfg.hidden_dim_traj, cfg.primitive_count, cfg.traj_points).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.trajectory_lr, weight_decay=cfg.weight_decay)

    Xtr = torch.tensor(Xtr_sc, dtype=torch.float32, device=DEVICE)
    Xva_or = torch.tensor(Xva_or_sc, dtype=torch.float32, device=DEVICE)
    Xva_ca = torch.tensor(Xva_ca_sc, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_traj_train, dtype=torch.float32, device=DEVICE)
    yva = torch.tensor(y_traj_val, dtype=torch.float32, device=DEVICE)
    Ttr = torch.tensor(dur_train, dtype=torch.float32, device=DEVICE)
    Tva = torch.tensor(dur_val, dtype=torch.float32, device=DEVICE)
    v0tr = torch.tensor(v0_train_mps, dtype=torch.float32, device=DEVICE)
    v0va = torch.tensor(v0_val_mps, dtype=torch.float32, device=DEVICE)
    d0tr = torch.tensor(d0_train_m, dtype=torch.float32, device=DEVICE)
    d0va = torch.tensor(d0_val_m, dtype=torch.float32, device=DEVICE)
    gotr = torch.tensor(go_train.astype(np.float32), dtype=torch.float32, device=DEVICE)
    gova = torch.tensor(go_val.astype(np.float32), dtype=torch.float32, device=DEVICE)
    dEndTr = torch.tensor(d_end_obs_train, dtype=torch.float32, device=DEVICE)
    dEndVa = torch.tensor(d_end_obs_val, dtype=torch.float32, device=DEVICE)
    vEndTr = torch.tensor(v_end_obs_train, dtype=torch.float32, device=DEVICE)
    vEndVa = torch.tensor(v_end_obs_val, dtype=torch.float32, device=DEVICE)

    hist = {
        "train_total": [],
        "val_total_oracle": [],
        "val_traj_mae_oracle": [],
        "val_d_end_mae_oracle": [],
        "val_stop_zone_rate_oracle": [],
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_score = np.inf

    def _loss(out: dict[str, torch.Tensor], y_true: torch.Tensor, T_true: torch.Tensor, v0: torch.Tensor, d0: torch.Tensor, go: torch.Tensor, d_end: torch.Tensor, v_end: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b_pred = out["b_pred"]
        T_pred = out["duration_s"]
        kin = integrate_kinematics_from_acceleration(b_pred, T_pred, v0, d0, go_label=go)
        a_phys = kin["a_eff"]
        d_term = kin["d_m"][:, -1]
        v_term = kin["v_mps"][:, -1]
        traj_loss = F.smooth_l1_loss(a_phys, y_true)
        edge_n = max(3, y_true.shape[1] // 12)
        edge_loss = F.smooth_l1_loss(a_phys[:, :edge_n], y_true[:, :edge_n]) + F.smooth_l1_loss(
            a_phys[:, -edge_n:], y_true[:, -edge_n:]
        )
        dur_loss = F.smooth_l1_loss(T_pred, T_true)
        term_d = F.smooth_l1_loss(d_term, d_end)
        term_v = F.smooth_l1_loss(v_term, v_end)
        stop_mask = (go < 0.5).float()
        go_mask = 1.0 - stop_mask
        stop_high = torch.relu(d_term - STOP_CONSTRAINT_MAX_M)
        stop_low = torch.relu(STOP_CONSTRAINT_MIN_M - d_term)
        stop_band = (((stop_high + stop_low) ** 2) * stop_mask).sum() / torch.clamp(stop_mask.sum(), min=1.0)
        stop_speed = (((v_term) ** 2) * stop_mask).sum() / torch.clamp(stop_mask.sum(), min=1.0)
        delay_idx = max(2, y_true.shape[1] // 6)
        stop_pos_after_delay = (
            (torch.relu(a_phys[:, delay_idx:]) ** 2).mean(dim=1) * stop_mask
        ).sum() / torch.clamp(stop_mask.sum(), min=1.0)
        go_cross = ((torch.relu(d_term) ** 2) * go_mask).sum() / torch.clamp(go_mask.sum(), min=1.0)
        go_speed = ((torch.relu(0.5 - v_term) ** 2) * go_mask).sum() / torch.clamp(go_mask.sum(), min=1.0)
        smooth = torch.mean((a_phys[:, 1:] - a_phys[:, :-1]) ** 2)
        entropy = -(out["gates"] * torch.log(torch.clamp(out["gates"], min=1e-8))).sum(dim=1).mean()
        # Encourage stop trajectories to settle near zero acceleration at the event time.
        stop_end_acc = (((a_phys[:, -3:]) ** 2).mean(dim=1) * stop_mask).sum() / torch.clamp(stop_mask.sum(), min=1.0)
        total = (
            1.00 * traj_loss
            + 0.35 * edge_loss
            + 2.00 * dur_loss
            + 0.40 * term_d
            + 3.00 * term_v
            + 1.80 * stop_band
            + 12.00 * stop_speed
            + 1.00 * stop_pos_after_delay
            + 0.70 * go_cross
            + 0.25 * go_speed
            + 0.04 * smooth
            + 0.20 * stop_end_acc
            + 0.02 * entropy
        )
        comps = {"d_term": d_term.detach(), "v_term": v_term.detach(), "a_phys": a_phys.detach()}
        return total, comps

    for _epoch in range(cfg.trajectory_epochs):
        model.train()
        tr_losses: list[float] = []
        for bidx in batch_iter(np.arange(len(Xtr_sc)), cfg.trajectory_batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            out = model(Xtr[bidx])
            loss, _ = _loss(out, ytr[bidx], Ttr[bidx], v0tr[bidx], d0tr[bidx], gotr[bidx], dEndTr[bidx], vEndTr[bidx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(float(loss.detach().cpu()))
        hist["train_total"].append(float(np.mean(tr_losses)) if tr_losses else float("nan"))

        model.eval()
        with torch.no_grad():
            out_or = model(Xva_or)
            val_loss, comps_or = _loss(out_or, yva, Tva, v0va, d0va, gova, dEndVa, vEndVa)
            b_or = comps_or["a_phys"].cpu().numpy()
            d_or = comps_or["d_term"].cpu().numpy()
            v_or = comps_or["v_term"].cpu().numpy()
        stop_mask = (go_val == 0)
        stop_zone_rate = float(
            np.mean(
                (d_or[stop_mask] >= STOP_CONSTRAINT_MIN_M)
                & (d_or[stop_mask] <= STOP_CONSTRAINT_MAX_M)
                & (v_or[stop_mask] <= 0.3)
            )
        ) if stop_mask.any() else float("nan")
        hist["val_total_oracle"].append(float(val_loss.detach().cpu()))
        hist["val_traj_mae_oracle"].append(float(np.mean(np.abs(b_or - y_traj_val))))
        hist["val_d_end_mae_oracle"].append(float(np.mean(np.abs(d_or - d_end_obs_val))))
        hist["val_stop_zone_rate_oracle"].append(stop_zone_rate)

        score = hist["val_traj_mae_oracle"][-1] + 0.2 * hist["val_d_end_mae_oracle"][-1] + (0.4 * (1 - stop_zone_rate) if np.isfinite(stop_zone_rate) else 0)
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    def _eval(
        Xsc: np.ndarray,
        y_true: np.ndarray,
        d_true: np.ndarray,
        v_true: np.ndarray,
        T_true: np.ndarray,
        v0: np.ndarray,
        d0: np.ndarray,
        go: np.ndarray,
        d_end: np.ndarray,
        v_end: np.ndarray,
    ) -> dict[str, Any]:
        Xt = torch.tensor(Xsc, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            out = model(Xt)
            go_t = torch.tensor(go, dtype=torch.float32, device=DEVICE)
            kin = integrate_kinematics_from_acceleration(
                out["b_pred"],
                out["duration_s"],
                torch.tensor(v0, dtype=torch.float32, device=DEVICE),
                torch.tensor(d0, dtype=torch.float32, device=DEVICE),
                go_label=go_t,
            )
        a_phys = kin["a_eff"].cpu().numpy()
        T_pred = out["duration_s"].cpu().numpy()
        d_traj_pred = kin["d_m"].cpu().numpy()
        v_traj_pred = kin["v_mps"].cpu().numpy()
        d_pred = d_traj_pred[:, -1]
        v_pred = v_traj_pred[:, -1]
        gates = out["gates"].cpu().numpy()
        stop_eval = _compute_stop_extended_metrics(
            a_true=y_true,
            d_true=d_true,
            v_true=v_true,
            T_true=T_true,
            a_pred=a_phys,
            d_pred=d_traj_pred,
            v_pred=v_traj_pred,
            T_pred=T_pred,
            go=go,
            d_end_obs=d_end,
            v_end_obs=v_end,
        )
        d_pred_eval = stop_eval["d_term_eval"]
        v_pred_eval = stop_eval["v_term_eval"]
        m = {
            "traj_mae_mps2": float(stop_eval["traj_mae_mps2"]),
            "traj_rmse_mps2": float(stop_eval["traj_rmse_mps2"]),
            "duration_mae_s": float(mean_absolute_error(T_true, T_pred)),
            "duration_rmse_s": float(math.sqrt(mean_squared_error(T_true, T_pred))),
            "terminal_distance_mae_m": float(mean_absolute_error(d_end, d_pred_eval)),
            "terminal_speed_mae_mps": float(mean_absolute_error(v_end, v_pred_eval)),
            "predicted_accel_peak_mae_mps2": float(
                mean_absolute_error(np.max(np.abs(y_true), axis=1), np.max(np.abs(a_phys), axis=1))
            ),
        }
        stop_mask_np = (go == 0)
        go_mask_np = ~stop_mask_np
        if stop_mask_np.any():
            m["stop_terminal_in_band_rate"] = float(
                np.mean((d_pred_eval[stop_mask_np] >= STOP_CONSTRAINT_MIN_M) & (d_pred_eval[stop_mask_np] <= STOP_CONSTRAINT_MAX_M) & (v_pred_eval[stop_mask_np] <= 0.3))
            )
            m["stop_terminal_distance_bias_m"] = float(np.mean(d_pred_eval[stop_mask_np] - d_end[stop_mask_np]))
        if go_mask_np.any():
            m["go_crossed_light_rate"] = float(np.mean(d_pred_eval[go_mask_np] <= 0.0))
            m["go_terminal_distance_bias_m"] = float(np.mean(d_pred_eval[go_mask_np] - d_end[go_mask_np]))
        return {
            "metrics": m,
            "b_pred": a_phys,
            "T_pred": T_pred,
            "d_pred": d_pred_eval,
            "v_pred": v_pred_eval,
            "gates": gates,
            "d_traj": d_traj_pred,
            "v_traj": v_traj_pred,
        }

    train_eval = _eval(Xtr_sc, y_traj_train, d_traj_train, v_traj_train_mps, dur_train, v0_train_mps, d0_train_m, go_train, d_end_obs_train, v_end_obs_train)
    val_eval_or = _eval(Xva_or_sc, y_traj_val, d_traj_val, v_traj_val_mps, dur_val, v0_val_mps, d0_val_m, go_val, d_end_obs_val, v_end_obs_val)
    val_eval_ca = _eval(Xva_ca_sc, y_traj_val, d_traj_val, v_traj_val_mps, dur_val, v0_val_mps, d0_val_m, go_val, d_end_obs_val, v_end_obs_val)
    metrics = {
        "train_oracle_decision_input": train_eval["metrics"],
        "val_oracle_decision_input": val_eval_or["metrics"],
        "val_cascaded_decision_input": val_eval_ca["metrics"],
        "primitive_names": ["const", "power_ramp_up", "power_ramp_down", "exp_rise", "logistic_step", "gaussian_bump"],
        "stage2_feature_names": ["v0_kmh", "d_th_m", "tti_s", "a_req_mps2", "age_years", "decision_go", "sex_female"],
    }
    outputs = {
        "X_train_scaled": Xtr_sc,
        "X_val_oracle_scaled": Xva_or_sc,
        "X_val_cascade_scaled": Xva_ca_sc,
        "train_eval": train_eval,
        "val_eval_oracle": val_eval_or,
        "val_eval_cascade": val_eval_ca,
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


def plot_trajectory_results(
    df_val: pd.DataFrame,
    y_traj_val: np.ndarray,
    d_traj_val: np.ndarray,
    traj_eval_oracle: dict[str, Any],
    traj_eval_cascade: dict[str, Any],
    hist: dict[str, list[float]],
    cfg: Config,
    out_dir: Path,
) -> None:
    # Curves
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(hist["train_total"], lw=2.2, color="#1d3557", label="Train total loss")
    ax.plot(hist["val_total_oracle"], lw=2.2, color="#e76f51", label="Val total loss (oracle)")
    ax2 = ax.twinx()
    ax2.plot(hist["val_traj_mae_oracle"], lw=2.0, ls="--", color="#2a9d8f", label="Val trajectory MAE")
    ax2.plot(hist["val_d_end_mae_oracle"], lw=2.0, ls=":", color="#6a4c93", label="Val terminal distance MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("MAE")
    ax.set_title("Mixture-of-Primitives (Physics-Informed) Training Curves")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_training_curves.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "trajectory_training_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    # Example trajectories: signed acceleration a_x(t) and remaining distance d(t)
    a_or = traj_eval_oracle["b_pred"]
    a_ca = traj_eval_cascade["b_pred"]
    d_or_traj = traj_eval_oracle["d_traj"]
    d_ca_traj = traj_eval_cascade["d_traj"]

    def _pick(mask: np.ndarray, k: int) -> list[int]:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return []
        score = (
            df_val.iloc[idx]["a_req_mps2"].to_numpy(dtype=float)
            + 0.01 * df_val.iloc[idx]["distance_threshold_m"].to_numpy(dtype=float)
        )
        idx_sorted = idx[np.argsort(score)]
        if len(idx_sorted) <= k:
            return idx_sorted.tolist()
        sel = np.linspace(0, len(idx_sorted) - 1, k, dtype=int)
        return idx_sorted[sel].tolist()

    go_mask_examples = df_val["go_decision"].to_numpy(dtype=int) == 1
    stop_mask_examples = ~go_mask_examples
    picks = _pick(stop_mask_examples, 6) + _pick(go_mask_examples, 6)
    if len(picks) > 0:
        cols = 4
        rows = int(math.ceil(len(picks) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows), squeeze=False)
        u = np.linspace(0.0, 1.0, cfg.traj_points)
        d_lo = float(np.nanmin(d_traj_val[picks]) - 2.0)
        d_hi = float(np.nanmax(d_traj_val[picks]) * 1.02)
        first_twin = None
        for ax, i in zip(axes.ravel(), picks):
            ax.plot(u, y_traj_val[i], color="#111827", lw=2.4, label="True a_x")
            ax.plot(u, a_or[i], color="#2a9d8f", lw=2.0, label="Pred a_x (oracle)")
            ax.plot(u, a_ca[i], color="#e76f51", lw=1.8, ls="--", label="Pred a_x (cascade)")
            ax.axhline(0.0, color="0.35", lw=1.0, ls=":")
            ax.set_ylim(-6.5, 3.5)
            ax_d = ax.twinx()
            if first_twin is None:
                first_twin = ax_d
            ax_d.plot(u, d_traj_val[i], color="#4c566a", lw=1.7, ls="-.", alpha=0.9, label="True d(t)")
            ax_d.plot(u, d_or_traj[i], color="#6a4c93", lw=1.5, ls=":", alpha=0.95, label="Pred d(t) oracle")
            ax_d.plot(u, d_ca_traj[i], color="#9d4edd", lw=1.1, ls=(0, (3, 2)), alpha=0.7, label="Pred d(t) cascade")
            ax_d.axhline(0.0, color="#6b7280", lw=0.9, ls="--")
            if int(df_val.iloc[i]["go_decision"]) == 0:
                ax_d.axhspan(STOP_CONSTRAINT_MIN_M, STOP_CONSTRAINT_MAX_M, color="#f4a261", alpha=0.08)
            ax_d.set_ylim(d_lo, d_hi)
            ax.set_title(
                f"{'GO' if int(df_val.iloc[i]['go_decision']) else 'STOP'} | "
                f"v0={df_val.iloc[i]['speed_at_yellow_kmh']:.0f} km/h, d0={df_val.iloc[i]['distance_threshold_m']:.0f} m\n"
                f"{df_val.iloc[i]['source_file']}",
                fontsize=8,
            )
            ax.set_xlabel("Normalized time")
            ax.set_ylabel("a_x (m/s^2)")
            ax_d.set_ylabel("d to light (m)")
            ax.grid(alpha=0.25)
        for ax in axes.ravel()[len(picks):]:
            ax.axis("off")
        h1, l1 = axes.ravel()[0].get_legend_handles_labels()
        h2, l2 = (first_twin.get_legend_handles_labels() if first_twin is not None else ([], []))
        fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=3, fontsize=9, frameon=True)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / "trajectory_examples.png", dpi=180, bbox_inches="tight")
        fig.savefig(out_dir / "trajectory_examples.pdf", bbox_inches="tight")
        plt.close(fig)

    # Terminal and primitives
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    d_true = df_val["traj_d_end_obs_m"].to_numpy(dtype=float)
    d_pred = traj_eval_oracle["d_pred"]
    go_mask = df_val["go_decision"].to_numpy(dtype=int) == 1
    stop_mask = ~go_mask
    if stop_mask.any():
        ax0.scatter(d_true[stop_mask], d_pred[stop_mask], color="#c1121f", s=35, alpha=0.75, label="Stop")
    if go_mask.any():
        ax0.scatter(d_true[go_mask], d_pred[go_mask], color="#2a9d8f", s=35, alpha=0.75, label="Go")
    lo = float(np.nanmin([d_true.min(), d_pred.min()]))
    hi = float(np.nanmax([d_true.max(), d_pred.max()]))
    ax0.plot([lo, hi], [lo, hi], "k--", lw=1.3, label="Ideal")
    ax0.axhspan(STOP_CONSTRAINT_MIN_M, STOP_CONSTRAINT_MAX_M, color="#f4a261", alpha=0.15, label="Stop target band")
    ax0.set_xlabel("Observed terminal distance to light (m)")
    ax0.set_ylabel("Predicted terminal distance (m)")
    ax0.set_title("Terminal Distance Prediction (Validation)")
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=9)

    gates = traj_eval_oracle["gates"]
    prim_names = ["Const", "Ramp↑", "Ramp↓", "Exp", "Logistic", "Gauss"]
    m_stop = gates[stop_mask].mean(axis=0) if stop_mask.any() else np.zeros(gates.shape[1])
    m_go = gates[go_mask].mean(axis=0) if go_mask.any() else np.zeros(gates.shape[1])
    x = np.arange(len(prim_names))
    w = 0.36
    ax1.bar(x - w / 2, m_stop, width=w, color="#c1121f", alpha=0.85, label="Stop")
    ax1.bar(x + w / 2, m_go, width=w, color="#2a9d8f", alpha=0.85, label="Go")
    ax1.set_xticks(x, prim_names, rotation=18)
    ax1.set_ylabel("Mean gate weight")
    ax1.set_title("Primitive Contribution (Validation)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_terminal_and_primitives.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "trajectory_terminal_and_primitives.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize_text(
    cfg: Config,
    extraction_summary: pd.DataFrame,
    decision_metrics: dict[str, Any],
    traj_metrics: dict[str, Any],
    decision_hist: dict[str, list[float]],
    traj_hist: dict[str, list[float]],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("Two-Stage Decision + Mixture-of-Primitives (Physics-Informed) Trajectory Modeling")
    lines.append("=" * 84)
    lines.append("")
    lines.append("Configuration")
    lines.append(json.dumps(asdict(cfg), indent=2))
    lines.append("")
    lines.append("Dataset and Extraction")
    lines.append(f"- Voluntary-only filter: {cfg.use_only_voluntary}")
    lines.append(f"- Trajectory runs extracted for modeling: {len(df_train) + len(df_val)}")
    lines.append(f"- Train/Val split: {len(df_train)}/{len(df_val)} ({cfg.train_ratio:.0%}/{1-cfg.train_ratio:.0%})")
    lines.append(f"- Train stop/go: {(df_train['go_decision']==0).sum()} / {(df_train['go_decision']==1).sum()}")
    lines.append(f"- Val stop/go: {(df_val['go_decision']==0).sum()} / {(df_val['go_decision']==1).sum()}")
    lines.append("- Extraction status counts (top):")
    for _, r in extraction_summary.head(10).iterrows():
        lines.append(f"  - {r['status']}: {int(r['count'])}")
    lines.append("")
    lines.append("Stage 1 (Decision MLP) metrics")
    for split in ("train", "val"):
        m = decision_metrics[split]
        lines.append(
            f"- {split}: acc={m['accuracy']:.3f}, precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}, auroc={m['auroc']:.3f}, brier={m['brier']:.3f}"
        )
    lines.append(f"- Val confusion matrix [[TN, FP],[FN, TP]]: {decision_metrics['val_confusion_matrix']}")
    lines.append("")
    lines.append("Stage 2 (Mixture-of-Primitives + PINN-style kinematic losses)")
    lines.append("- Target trajectory: signed longitudinal acceleration a_x(t)=ax_g(t), clipped to [-12,12] m/s^2")
    lines.append("- Inputs to stage 2: [v0, d_th, TTI, a_req, age, decision_go, sex_female]")
    lines.append("- Primitive set (6): constant, power-ramp-up, power-ramp-down, exponential rise, logistic step, gaussian bump")
    lines.append("")
    lines.append("Core equations")
    lines.append("  p(go|x)=sigma(f_theta(x))")
    lines.append("  a_hat(tau)=sum_{k=1}^6 w_k(x) phi_k(tau;psi_k(x)), with w=softmax(g_theta(x)), tau in [0,1]")
    lines.append("  v_{n+1}=max(v_n+a_n dt,0), d_{n+1}=d_n-(v_n dt+0.5 a_n dt^2)")
    lines.append(f"  Stop physics constraint: d_T in [{STOP_CONSTRAINT_MIN_M:.1f},{STOP_CONSTRAINT_MAX_M:.1f}] m and v_T~0")
    lines.append("")
    for key in ("train_oracle_decision_input", "val_oracle_decision_input", "val_cascaded_decision_input"):
        m = traj_metrics[key]
        lines.append(
            f"- {key}: traj_MAE={m['traj_mae_mps2']:.3f}, traj_RMSE={m['traj_rmse_mps2']:.3f}, duration_MAE={m['duration_mae_s']:.3f}s, d_end_MAE={m['terminal_distance_mae_m']:.3f}m, v_end_MAE={m['terminal_speed_mae_mps']:.3f}m/s"
        )
        if "stop_terminal_in_band_rate" in m:
            lines.append(f"  stop_terminal_in_band_rate={m['stop_terminal_in_band_rate']:.3f}")
        if "go_crossed_light_rate" in m:
            lines.append(f"  go_crossed_light_rate={m['go_crossed_light_rate']:.3f}")
    lines.append("")
    lines.append("Training summaries")
    lines.append(f"- Decision final val F1/AUROC in history: {decision_hist['val_f1'][-1]:.3f}/{decision_hist['val_auc'][-1]:.3f}")
    lines.append(
        f"- Trajectory final val (oracle) loss/trajMAE/dEndMAE/stopZoneRate: {traj_hist['val_total_oracle'][-1]:.3f}/{traj_hist['val_traj_mae_oracle'][-1]:.3f}/{traj_hist['val_d_end_mae_oracle'][-1]:.3f}/{traj_hist['val_stop_zone_rate_oracle'][-1]:.3f}"
    )
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    set_seed(CFG.seed)

    df, traj_arrays, extraction_summary = prepare_modeling_dataframe(CFG)
    if len(df) < 20:
        raise RuntimeError("Too few trajectory samples after extraction.")

    base_X_stage2 = np.column_stack(
        [
            df["speed_at_yellow_kmh"].to_numpy(dtype=float),
            df["distance_threshold_m"].to_numpy(dtype=float),
            df["tti_s"].to_numpy(dtype=float),
            df["a_req_mps2"].to_numpy(dtype=float),
            df["driver_age"].to_numpy(dtype=float),
        ]
    ).astype(np.float32)
    # Stage-1 decision model uses the best compact set from ablation: (v_onset, a_req).
    base_X_stage1 = np.column_stack(
        [
            df["speed_at_yellow_kmh"].to_numpy(dtype=float),
            df["a_req_mps2"].to_numpy(dtype=float),
        ]
    ).astype(np.float32)
    go = df["go_decision"].to_numpy(dtype=int)
    sex_female = df["sex_female"].to_numpy(dtype=float)

    all_idx = np.arange(len(df))
    idx_train, idx_val = train_test_split(
        all_idx,
        train_size=CFG.train_ratio,
        random_state=CFG.seed,
        stratify=go,
        shuffle=True,
    )
    idx_train = np.sort(idx_train)
    idx_val = np.sort(idx_val)
    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)

    X_train_stage1, X_val_stage1 = base_X_stage1[idx_train], base_X_stage1[idx_val]
    X_train_stage2, X_val_stage2 = base_X_stage2[idx_train], base_X_stage2[idx_val]
    y_train, y_val = go[idx_train], go[idx_val]

    decision_model, decision_bundle, decision_hist, decision_scaler = fit_decision_model(
        X_train_stage1,
        y_train,
        X_val_stage1,
        y_val,
        CFG,
        feature_names=["v0_kmh", "a_req_mps2"],
    )
    dec_metrics = decision_bundle["metrics"]
    dec_out = decision_bundle["outputs"]
    y_val_prob = dec_out["val_prob"]
    y_val_pred = dec_out["val_pred"]

    y_traj_train = traj_arrays["ax_target_mps2"][idx_train]
    y_traj_val = traj_arrays["ax_target_mps2"][idx_val]
    d_traj_train = traj_arrays["d_target_m"][idx_train].astype(np.float32)
    d_traj_val = traj_arrays["d_target_m"][idx_val].astype(np.float32)
    v_traj_train_mps = (traj_arrays["v_target_kmh"][idx_train] / 3.6).astype(np.float32)
    v_traj_val_mps = (traj_arrays["v_target_kmh"][idx_val] / 3.6).astype(np.float32)
    dur_train = df["traj_duration_s"].to_numpy(dtype=float)[idx_train].astype(np.float32)
    dur_val = df["traj_duration_s"].to_numpy(dtype=float)[idx_val].astype(np.float32)
    v0_train = (df["speed_at_yellow_kmh"].to_numpy(dtype=float)[idx_train] / 3.6).astype(np.float32)
    v0_val = (df["speed_at_yellow_kmh"].to_numpy(dtype=float)[idx_val] / 3.6).astype(np.float32)
    d0_train = df["distance_threshold_m"].to_numpy(dtype=float)[idx_train].astype(np.float32)
    d0_val = df["distance_threshold_m"].to_numpy(dtype=float)[idx_val].astype(np.float32)
    d_end_train = df["traj_d_end_obs_m"].to_numpy(dtype=float)[idx_train].astype(np.float32)
    d_end_val = df["traj_d_end_obs_m"].to_numpy(dtype=float)[idx_val].astype(np.float32)
    v_end_train = df["traj_v_end_obs_mps"].to_numpy(dtype=float)[idx_train].astype(np.float32)
    v_end_val = df["traj_v_end_obs_mps"].to_numpy(dtype=float)[idx_val].astype(np.float32)

    X2_train = make_stage2_features(X_train_stage2, y_train.astype(float), sex_female[idx_train])
    X2_val_oracle = make_stage2_features(X_val_stage2, y_val.astype(float), sex_female[idx_val])
    X2_val_cascade = make_stage2_features(X_val_stage2, y_val_pred.astype(float), sex_female[idx_val])

    traj_model, traj_bundle, traj_hist, traj_scaler = train_trajectory_model(
        X2_train, X2_val_oracle, X2_val_cascade,
        y_traj_train, y_traj_val,
        d_traj_train, d_traj_val,
        v_traj_train_mps, v_traj_val_mps,
        dur_train, dur_val,
        v0_train, v0_val,
        d0_train, d0_val,
        y_train, y_val,
        d_end_train, d_end_val,
        v_end_train, v_end_val,
        CFG,
    )
    traj_metrics = traj_bundle["metrics"]
    traj_out = traj_bundle["outputs"]

    save_json({"config": asdict(CFG), "decision_metrics": dec_metrics}, OUT_DIR / "decision_metrics.json")
    save_json({"config": asdict(CFG), "trajectory_metrics": traj_metrics}, OUT_DIR / "trajectory_metrics.json")
    pd.DataFrame({"source_file": df["source_file"], "split": np.where(np.isin(np.arange(len(df)), idx_train), "train", "val"), "go_decision": go}).to_csv(
        OUT_DIR / "split_manifest.csv", index=False
    )

    val_pred_df = df_val[
        [
            "source_file", "go_decision", "stop_decision", "speed_at_yellow_kmh", "distance_threshold_m",
            "tti_s", "a_req_mps2", "driver_age", "traj_duration_s", "traj_d_end_obs_m", "traj_v_end_obs_mps"
        ]
    ].copy()
    val_pred_df["decision_go_prob"] = y_val_prob
    val_pred_df["decision_go_pred"] = y_val_pred
    val_pred_df["traj_duration_pred_oracle_s"] = traj_out["val_eval_oracle"]["T_pred"]
    val_pred_df["traj_duration_pred_cascade_s"] = traj_out["val_eval_cascade"]["T_pred"]
    val_pred_df["traj_terminal_d_pred_oracle_m"] = traj_out["val_eval_oracle"]["d_pred"]
    val_pred_df["traj_terminal_d_pred_cascade_m"] = traj_out["val_eval_cascade"]["d_pred"]
    val_pred_df["traj_terminal_v_pred_oracle_mps"] = traj_out["val_eval_oracle"]["v_pred"]
    val_pred_df["traj_terminal_v_pred_cascade_mps"] = traj_out["val_eval_cascade"]["v_pred"]
    val_pred_df.to_csv(OUT_DIR / "validation_predictions.csv", index=False)

    np.savez_compressed(
        OUT_DIR / "validation_trajectory_arrays.npz",
        y_traj_val=y_traj_val,
        d_traj_val=traj_arrays["d_target_m"][idx_val],
        ax_pred_oracle=traj_out["val_eval_oracle"]["b_pred"],
        ax_pred_cascade=traj_out["val_eval_cascade"]["b_pred"],
        d_pred_traj_oracle=traj_out["val_eval_oracle"]["d_traj"],
        d_pred_traj_cascade=traj_out["val_eval_cascade"]["d_traj"],
        primitive_gates_oracle=traj_out["val_eval_oracle"]["gates"],
    )
    torch.save(decision_model.state_dict(), OUT_DIR / "decision_mlp_state_dict.pt")
    torch.save(traj_model.state_dict(), OUT_DIR / "trajectory_mop_pinn_state_dict.pt")
    np.savez_compressed(
        OUT_DIR / "feature_scalers.npz",
        decision_mean=decision_scaler.mean_, decision_scale=decision_scaler.scale_,
        trajectory_mean=traj_scaler.mean_, trajectory_scale=traj_scaler.scale_,
    )

    plot_decision_results(y_val, y_val_prob, y_val_pred, decision_hist, PLOTS_DIR)
    plot_trajectory_results(df_val, y_traj_val, traj_arrays["d_target_m"][idx_val], traj_out["val_eval_oracle"], traj_out["val_eval_cascade"], traj_hist, CFG, PLOTS_DIR)

    report = summarize_text(CFG, extraction_summary, dec_metrics, traj_metrics, decision_hist, traj_hist, df_train, df_val)
    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    (OUT_DIR / "README_run.txt").write_text(
        "Run with project venv:\n  .\\.venv\\Scripts\\python.exe trajectory_mixture_primitives.py\n", encoding="utf-8"
    )
    print(report)
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
