from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import trajectory_shared_utils as base


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "trajectory_transformer_ar"
CACHE_DIR = OUT_DIR / "cache"
PLOTS_DIR = OUT_DIR / "plots"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROGRESS_LOG = OUT_DIR / "run_progress.log"

if hasattr(torch, "set_float32_matmul_precision"):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

STOP_CONSTRAINT_MIN_M = base.STOP_CONSTRAINT_MIN_M
STOP_CONSTRAINT_MAX_M = base.STOP_CONSTRAINT_MAX_M
STOP_EVAL_SPEED_EPS_MPS = 0.3
STOP_EVAL_ACCEL_EPS_MPS2 = 0.5
STOP_EVAL_AR_MAX_HORIZON_FACTOR = 3.0


def _first_stable_stop_time_s(
    a_row: np.ndarray,
    v_row: np.ndarray,
    T_s: float,
    v_eps: float = STOP_EVAL_SPEED_EPS_MPS,
) -> float | None:
    if not np.isfinite(T_s) or T_s <= 0 or len(v_row) == 0:
        return None
    t_src = np.linspace(0.0, float(T_s), len(v_row), dtype=np.float64)
    v = np.nan_to_num(np.asarray(v_row, dtype=np.float64), nan=np.inf, posinf=np.inf, neginf=0.0)
    _ = np.asarray(a_row, dtype=np.float64)  # reserved for future tighter stop criteria
    cand = np.flatnonzero(v <= float(v_eps))
    for idx in cand:
        tail_v = v[idx:]
        if tail_v.size and np.nanmax(tail_v) <= float(v_eps) + 1e-6:
            return float(t_src[int(idx)])
    return None


def _interp_stop_padded_triplet(
    a_row: np.ndarray,
    d_row: np.ndarray,
    v_row: np.ndarray,
    T_s: float,
    t_eval: np.ndarray,
    stop_t_s: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_src = np.asarray(a_row, dtype=np.float64)
    d_src = np.asarray(d_row, dtype=np.float64)
    v_src = np.asarray(v_row, dtype=np.float64)
    if a_src.size == 0:
        z = np.zeros_like(t_eval, dtype=np.float64)
        return z.copy(), z.copy(), z.copy()
    T_s = float(max(T_s, 1e-6))
    t_src = np.linspace(0.0, T_s, a_src.size, dtype=np.float64)
    tq = np.asarray(t_eval, dtype=np.float64)
    t_clip = np.clip(tq, 0.0, T_s)
    a_out = np.interp(t_clip, t_src, a_src)
    d_out = np.interp(t_clip, t_src, d_src)
    v_out = np.interp(t_clip, t_src, v_src)
    after_src = tq > T_s
    if np.any(after_src):
        a_out[after_src] = float(a_src[-1])
        d_out[after_src] = float(d_src[-1])
        v_out[after_src] = float(v_src[-1])
    if stop_t_s is not None and np.isfinite(stop_t_s):
        ts = float(max(0.0, min(stop_t_s, T_s)))
        d_stop = float(np.interp(ts, t_src, d_src))
        after_stop = tq >= ts
        a_out[after_stop] = 0.0
        v_out[after_stop] = 0.0
        d_out[after_stop] = d_stop
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
    n = int(len(go))
    a_abs_sum = d_abs_sum = v_abs_sum = 0.0
    a_sq_sum = d_sq_sum = v_sq_sum = 0.0
    n_a = n_d = n_v = 0
    d_term = np.zeros(n, dtype=np.float64)
    v_term = np.zeros(n, dtype=np.float64)
    stop_completed_pred = np.zeros(n, dtype=bool)
    stop_completed_true = np.zeros(n, dtype=bool)

    for i in range(n):
        g = int(go[i])
        if g == 1:
            da = np.asarray(a_true[i], dtype=np.float64) - np.asarray(a_pred[i], dtype=np.float64)
            dd = np.asarray(d_true[i], dtype=np.float64) - np.asarray(d_pred[i], dtype=np.float64)
            dv = np.asarray(v_true[i], dtype=np.float64) - np.asarray(v_pred[i], dtype=np.float64)
            a_abs_sum += float(np.sum(np.abs(da))); a_sq_sum += float(np.sum(da * da)); n_a += da.size
            d_abs_sum += float(np.sum(np.abs(dd))); d_sq_sum += float(np.sum(dd * dd)); n_d += dd.size
            v_abs_sum += float(np.sum(np.abs(dv))); v_sq_sum += float(np.sum(dv * dv)); n_v += dv.size
            d_term[i] = float(np.asarray(d_pred[i], dtype=np.float64)[-1])
            v_term[i] = float(np.asarray(v_pred[i], dtype=np.float64)[-1])
            continue

        Tt = float(np.asarray(T_true, dtype=np.float64)[i])
        Tp = float(np.asarray(T_pred, dtype=np.float64)[i])
        a_t = np.asarray(a_true[i], dtype=np.float64)
        d_t = np.asarray(d_true[i], dtype=np.float64)
        v_t = np.asarray(v_true[i], dtype=np.float64)
        a_p = np.asarray(a_pred[i], dtype=np.float64)
        d_p = np.asarray(d_pred[i], dtype=np.float64)
        v_p = np.asarray(v_pred[i], dtype=np.float64)
        t_stop_true = _first_stable_stop_time_s(a_t, v_t, Tt)
        t_stop_pred = _first_stable_stop_time_s(a_p, v_p, Tp)
        stop_completed_true[i] = t_stop_true is not None
        stop_completed_pred[i] = t_stop_pred is not None
        t_done_true = Tt if t_stop_true is None else t_stop_true
        t_done_pred = Tp if t_stop_pred is None else t_stop_pred
        t_eval_end = max(float(t_done_true), float(t_done_pred))
        dt_ref = min(max(Tt, 1e-3), max(Tp, 1e-3)) / max(len(a_t) - 1, 1)
        dt_ref = max(float(dt_ref), 1e-3)
        m = max(int(len(a_t)), int(math.ceil(t_eval_end / dt_ref)) + 1)
        t_eval = np.arange(m, dtype=np.float64) * dt_ref

        a_t_e, d_t_e, v_t_e = _interp_stop_padded_triplet(a_t, d_t, v_t, Tt, t_eval, t_stop_true)
        a_p_e, d_p_e, v_p_e = _interp_stop_padded_triplet(a_p, d_p, v_p, Tp, t_eval, t_stop_pred)

        da = a_t_e - a_p_e
        dd = d_t_e - d_p_e
        dv = v_t_e - v_p_e
        a_abs_sum += float(np.sum(np.abs(da))); a_sq_sum += float(np.sum(da * da)); n_a += da.size
        d_abs_sum += float(np.sum(np.abs(dd))); d_sq_sum += float(np.sum(dd * dd)); n_d += dd.size
        v_abs_sum += float(np.sum(np.abs(dv))); v_sq_sum += float(np.sum(dv * dv)); n_v += dv.size
        d_term[i] = float(d_p_e[-1])
        v_term[i] = float(v_p_e[-1])

    d_end_obs_np = np.asarray(d_end_obs, dtype=np.float64)
    v_end_obs_np = np.asarray(v_end_obs, dtype=np.float64)
    out = {
        "traj_mae_mps2": float(a_abs_sum / max(n_a, 1)),
        "traj_rmse_mps2": float(math.sqrt(a_sq_sum / max(n_a, 1))),
        "distance_curve_mae_m": float(d_abs_sum / max(n_d, 1)),
        "distance_curve_rmse_m": float(math.sqrt(d_sq_sum / max(n_d, 1))),
        "speed_curve_mae_mps": float(v_abs_sum / max(n_v, 1)),
        "speed_curve_rmse_mps": float(math.sqrt(v_sq_sum / max(n_v, 1))),
        "terminal_distance_mae_m": float(mean_absolute_error(d_end_obs_np, d_term)),
        "terminal_speed_mae_mps": float(mean_absolute_error(v_end_obs_np, v_term)),
        "d_term_eval": d_term,
        "v_term_eval": v_term,
        "stop_completed_pred_mask": stop_completed_pred,
        "stop_completed_true_mask": stop_completed_true,
    }
    return out


@dataclass
class NodeConfig:
    seed: int = 42
    traj_points: int = 64
    train_ratio: float = 0.60
    use_only_voluntary: bool = True
    a_clip_mps2: float = 12.0
    min_event_duration_s: float = 0.35
    max_event_duration_s: float = 20.0
    extraction_logic_version: int = 3

    decision_epochs: int = 80
    decision_batch_size: int = 64
    decision_lr: float = 1e-3
    hidden_dim_decision: int = 64

    node_epochs: int = 150
    node_batch_size: int = 48
    node_lr: float = 8e-4
    weight_decay: float = 1e-5
    context_dim: int = 48
    hidden_state_dim: int = 12
    ode_hidden_dim: int = 96
    rk_substeps: int = 3
    max_abs_accel_mps2: float = 12.0
    max_abs_jerk_mps3: float = 15.0
    reaction_delay_max_frac: float = 0.25
    reaction_gate_sharpness: float = 18.0
    stop_reaction_delay_min_steps: int = 3
    # AR Transformer settings
    ar_epochs: int = 140
    ar_batch_size: int = 24
    ar_lr: float = 7e-4
    grad_clip: float = 4.0
    context_window: int = 24
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    ff_dim: int = 192
    tr_dropout: float = 0.08
    a_min_brake_mps2: float = -10.0
    a_max_accel_go_mps2: float = 3.0
    lambda_d_stop: float = 2.2
    lambda_v_stop: float = 1.2
    lambda_d_go: float = 0.8
    lambda_v_go: float = 1.5
    terminal_deadband_min_m: float = -0.5
    terminal_deadband_max_m: float = 1.5
    stop_eps_mps: float = 0.01
    w_accel: float = 1.0
    w_terminal_deadband: float = 2.2
    w_go_cross: float = 0.8
    w_d_curve: float = 0.20
    w_v_curve: float = 0.18
    w_duration: float = 0.7
    w_stop_terminal_speed: float = 0.8
    w_stop_terminal_accel: float = 0.5
    w_tail_smooth: float = 0.12
    w_curvature: float = 0.01
    w_jerk_reg: float = 0.004
    w_snap_reg: float = 0.0008
    curriculum_warmup_epochs: int = 45
    example_plot_total: int = 12
    example_plot_stop_ratio: float = 0.75


CFG = NodeConfig()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def log_progress(message: str) -> None:
    ensure_dirs()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    with PROGRESS_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def patch_base_for_extraction(cfg: NodeConfig) -> None:
    # Reuse the extraction/caching code from the mixture script but write into a new results folder.
    base.OUT_DIR = OUT_DIR
    base.CACHE_DIR = CACHE_DIR
    base.PLOTS_DIR = PLOTS_DIR
    base.DEVICE = DEVICE
    base.ensure_dirs()
    base.CFG.traj_points = cfg.traj_points
    base.CFG.a_clip_mps2 = cfg.a_clip_mps2
    base.CFG.min_event_duration_s = cfg.min_event_duration_s
    base.CFG.max_event_duration_s = cfg.max_event_duration_s
    base.CFG.use_only_voluntary = cfg.use_only_voluntary
    base.CFG.extraction_logic_version = cfg.extraction_logic_version
    # Keep the early-stop inclusion fix from the mixture pipeline.
    base.STOP_DETECT_MAX_M = 6.5


def batch_iter(indices: np.ndarray, batch_size: int, shuffle: bool = True) -> list[np.ndarray]:
    idx = indices.copy()
    if shuffle:
        np.random.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]


class ContextEncoder(nn.Module):
    def __init__(self, in_dim: int, context_dim: int, hidden_state_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.ctx_head = nn.Linear(64, context_dim)
        self.h0_head = nn.Linear(64, hidden_state_dim)
        self.a0_head = nn.Linear(64, 1)
        self.duration_head = nn.Linear(64, 1)
        self.delay_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        ctx = torch.tanh(self.ctx_head(h))
        h0 = torch.tanh(self.h0_head(h))
        # Start near zero acceleration at yellow onset (network can adjust slightly).
        a0 = 0.8 * torch.tanh(self.a0_head(h)).squeeze(-1)
        duration_s = 0.35 + 19.65 * torch.sigmoid(self.duration_head(h)).squeeze(-1)
        delay_frac = torch.sigmoid(self.delay_head(h)).squeeze(-1)
        ctx = torch.nan_to_num(ctx, nan=0.0, posinf=1.0, neginf=-1.0)
        h0 = torch.nan_to_num(h0, nan=0.0, posinf=1.0, neginf=-1.0)
        a0 = torch.nan_to_num(a0, nan=0.0, posinf=0.8, neginf=-0.8)
        duration_s = torch.nan_to_num(duration_s, nan=2.5, posinf=20.0, neginf=0.35)
        delay_frac = torch.nan_to_num(delay_frac, nan=0.08, posinf=1.0, neginf=0.0)
        return {"context": ctx, "h0": h0, "a0": a0, "duration_s": duration_s, "delay_frac": delay_frac}


class ODERHS(nn.Module):
    def __init__(self, context_dim: int, hidden_state_dim: int, ode_hidden_dim: int, max_abs_jerk: float) -> None:
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        self.max_abs_jerk = max_abs_jerk
        in_dim = 1 + 3 + hidden_state_dim + context_dim  # t + [d,v,a] + h + c
        self.net = nn.Sequential(
            nn.Linear(in_dim, ode_hidden_dim),
            nn.Tanh(),
            nn.Linear(ode_hidden_dim, ode_hidden_dim),
            nn.Tanh(),
            nn.Linear(ode_hidden_dim, 1 + hidden_state_dim),  # jerk + hdot
        )

    def forward(self, t_norm: torch.Tensor, y: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        d = y[:, 0:1]
        v = y[:, 1:2]
        a = y[:, 2:3]
        h = y[:, 3:]
        t_feat = t_norm.view(-1, 1)
        feat = torch.cat([t_feat, d, v, a, h, ctx], dim=1)
        out = self.net(feat)
        jerk = self.max_abs_jerk * torch.tanh(out[:, 0:1])
        hdot = torch.tanh(out[:, 1:])
        d_dot = -v
        v_dot = a
        a_dot = jerk
        return torch.cat([d_dot, v_dot, a_dot, hdot], dim=1)


class NeuralODETrajectoryModel(nn.Module):
    """Jerk-driven Neural ODE with hard event constraints for stop runs.

    State y = [d, v, a, h].
    Output trajectory uses constrained acceleration a_eff(t), not raw a(t), to ensure
    exact zero acceleration and zero speed after the stop event.
    """

    def __init__(self, in_dim: int, cfg: NodeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.context_encoder = ContextEncoder(in_dim, cfg.context_dim, cfg.hidden_state_dim)
        self.rhs = ODERHS(cfg.context_dim, cfg.hidden_state_dim, cfg.ode_hidden_dim, cfg.max_abs_jerk_mps3)

    def _rk4_step(self, y: torch.Tensor, t0: torch.Tensor, dt: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        f = self.rhs
        k1 = f(t0, y, ctx)
        k2 = f(t0 + 0.5 * dt, y + 0.5 * dt.view(-1, 1) * k1, ctx)
        k3 = f(t0 + 0.5 * dt, y + 0.5 * dt.view(-1, 1) * k2, ctx)
        k4 = f(t0 + dt, y + dt.view(-1, 1) * k3, ctx)
        y_next = y + (dt.view(-1, 1) / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Basic physical clipping (projection), done out-of-place to keep autograd stable.
        d_next = y_next[:, 0:1]
        v_next = torch.clamp(y_next[:, 1:2], min=0.0)  # speed
        a_next = torch.clamp(
            y_next[:, 2:3],
            min=-self.cfg.max_abs_accel_mps2,
            max=self.cfg.max_abs_accel_mps2,
        )  # accel
        h_next = y_next[:, 3:]
        y_proj = torch.cat([d_next, v_next, a_next, h_next], dim=1)
        return torch.nan_to_num(y_proj, nan=0.0, posinf=50.0, neginf=-50.0)

    def forward(
        self,
        x_scaled: torch.Tensor,
        v0_mps: torch.Tensor,
        d0_m: torch.Tensor,
        go_label: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B = x_scaled.shape[0]
        N = self.cfg.traj_points
        enc = self.context_encoder(x_scaled)
        ctx = enc["context"]
        h0 = enc["h0"]
        duration_s = enc["duration_s"]
        # Delay applies mainly to stop maneuvers; go trajectories can still have negligible delay.
        delay_frac = self.cfg.reaction_delay_max_frac * torch.sigmoid(enc["delay_frac"])
        a0 = enc["a0"]

        y = torch.cat([d0_m.view(-1, 1), v0_mps.view(-1, 1), a0.view(-1, 1), h0], dim=1)
        y_list = [y]
        a_eff_list: list[torch.Tensor] = [a0]
        stopped = torch.zeros(B, dtype=torch.bool, device=x_scaled.device)
        stop_mask_all = go_label < 0.5
        go_mask_all = ~stop_mask_all
        dt_macro = duration_s / max(N - 1, 1)

        for k in range(N - 1):
            yk = y_list[-1]
            # Integrate with substeps on normalized time for smoother NODE dynamics.
            y_next = yk
            t_norm0 = torch.full((B,), k / max(N - 1, 1), dtype=torch.float32, device=x_scaled.device)
            for sidx in range(self.cfg.rk_substeps):
                sub_dt = dt_macro / self.cfg.rk_substeps
                t_sub = t_norm0 + (sidx / self.cfg.rk_substeps) * (1.0 / max(N - 1, 1))
                y_next = self._rk4_step(y_next, t_sub, sub_dt, ctx)

            # Hard event-aware projection layer
            d_prev = yk[:, 0]
            v_prev = yk[:, 1]
            a_prev = yk[:, 2]
            d_new = y_next[:, 0]
            v_new = y_next[:, 1]
            a_new = y_next[:, 2]
            h_new = y_next[:, 3:]

            # Freeze after stop reached (exactly zero speed & accel, constant distance)
            a_new = torch.where(stopped, torch.zeros_like(a_new), a_new)
            v_new = torch.where(stopped, torch.zeros_like(v_new), v_new)
            d_new = torch.where(stopped, d_prev, d_new)

            # Reaction-delay gate for stop runs: use a smooth sigmoid gate to avoid a sharp release kink.
            tau_next = torch.full((B,), (k + 1) / max(N - 1, 1), dtype=torch.float32, device=x_scaled.device)
            active_stop_any = stop_mask_all & (~stopped)
            gate = torch.sigmoid(self.cfg.reaction_gate_sharpness * (tau_next - delay_frac))
            a_new = torch.where(active_stop_any, a_new * gate, a_new)

            # After delay, hard stop constraints:
            # 1) no positive acceleration for stop runs
            active_stop = stop_mask_all & (~stopped) & (tau_next >= delay_frac)
            a_new = torch.where(active_stop, torch.minimum(a_new, torch.zeros_like(a_new)), a_new)

            # 2) enforce stop-ability before line: a <= -v^2/(2d)
            valid_need = active_stop & (d_new > 0.0) & (v_new > 1e-4)
            a_need = -torch.square(v_new) / torch.clamp(2.0 * d_new, min=1e-3)
            a_new = torch.where(valid_need, torch.minimum(a_new, a_need), a_new)

            # NOTE: stop-end taper removed because it over-constrained trajectories when combined
            # with the hard stop freeze. We keep the smooth onset gate and hard event constraint.

            # If stop trajectory crosses line, clamp exact event and freeze
            crossed_line_stop = active_stop & (d_new <= 0.0)
            d_new = torch.where(crossed_line_stop, torch.zeros_like(d_new), d_new)
            v_new = torch.where(crossed_line_stop, torch.zeros_like(v_new), v_new)
            a_new = torch.where(crossed_line_stop, torch.zeros_like(a_new), a_new)

            # If stop trajectory reaches zero speed, freeze thereafter (stop before line allowed)
            reached_zero_v = active_stop & (v_new <= 1e-4)
            v_new = torch.where(reached_zero_v, torch.zeros_like(v_new), v_new)
            a_new = torch.where(reached_zero_v, torch.zeros_like(a_new), a_new)

            newly_stopped = crossed_line_stop | reached_zero_v
            # Hard jerk limit on the final constrained acceleration, using the actual macro dt (seconds).
            # Apply only while still moving; once a stop event is reached, exact zero is preserved.
            moving_mask = ~(stopped | newly_stopped)
            max_da = self.cfg.max_abs_jerk_mps3 * dt_macro
            a_lo = a_prev - max_da
            a_hi = a_prev + max_da
            a_limited = torch.maximum(torch.minimum(a_new, a_hi), a_lo)
            a_new = torch.where(moving_mask, a_limited, a_new)

            stopped = stopped | newly_stopped
            d_new = torch.where(stopped & (~newly_stopped), d_prev, d_new)

            # Reassemble state with constrained d,v,a and continuous latent h
            d_new = torch.nan_to_num(d_new, nan=0.0, posinf=200.0, neginf=-50.0)
            v_new = torch.nan_to_num(v_new, nan=0.0, posinf=50.0, neginf=0.0)
            a_new = torch.nan_to_num(a_new, nan=0.0, posinf=self.cfg.max_abs_accel_mps2, neginf=-self.cfg.max_abs_accel_mps2)
            h_new = torch.nan_to_num(h_new, nan=0.0, posinf=10.0, neginf=-10.0)
            y_next = torch.cat([d_new.view(-1, 1), v_new.view(-1, 1), a_new.view(-1, 1), h_new], dim=1)
            y_list.append(y_next)
            a_eff_list.append(a_new)

        y_traj = torch.stack(y_list, dim=1)              # [B, N, 3+H]
        a_eff = torch.stack(a_eff_list, dim=1)           # [B, N]
        return {
            "y_traj": y_traj,
            "a_eff": a_eff,
            "duration_s": duration_s,
            "delay_frac": delay_frac,
            "context": ctx,
        }


def make_stage2_features(
    base_X: np.ndarray,
    go_signal: np.ndarray,
    go_prob: np.ndarray,
    sex_female: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            base_X,
            go_signal.reshape(-1, 1),
            go_prob.reshape(-1, 1),
            sex_female.reshape(-1, 1),
        ]
    ).astype(np.float32)


def train_node_model(
    X2_train: np.ndarray,
    X2_val_oracle: np.ndarray,
    X2_val_cascade: np.ndarray,
    a_train: np.ndarray,
    a_val: np.ndarray,
    d_traj_train: np.ndarray,
    d_traj_val: np.ndarray,
    v_traj_train: np.ndarray,
    v_traj_val: np.ndarray,
    dur_train: np.ndarray,
    dur_val: np.ndarray,
    v0_train: np.ndarray,
    v0_val: np.ndarray,
    d0_train: np.ndarray,
    d0_val: np.ndarray,
    go_train: np.ndarray,
    go_val: np.ndarray,
    d_end_train: np.ndarray,
    d_end_val: np.ndarray,
    v_end_train: np.ndarray,
    v_end_val: np.ndarray,
    cfg: NodeConfig,
) -> tuple[NeuralODETrajectoryModel, dict[str, Any], dict[str, list[float]], StandardScaler]:
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X2_train).astype(np.float32)
    Xva_or_sc = scaler.transform(X2_val_oracle).astype(np.float32)
    Xva_ca_sc = scaler.transform(X2_val_cascade).astype(np.float32)

    model = NeuralODETrajectoryModel(in_dim=Xtr_sc.shape[1], cfg=cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.node_lr, weight_decay=cfg.weight_decay)

    # Tensors
    Xtr = torch.tensor(Xtr_sc, dtype=torch.float32, device=DEVICE)
    Xva_or = torch.tensor(Xva_or_sc, dtype=torch.float32, device=DEVICE)
    Xva_ca = torch.tensor(Xva_ca_sc, dtype=torch.float32, device=DEVICE)
    atr = torch.tensor(a_train, dtype=torch.float32, device=DEVICE)
    ava = torch.tensor(a_val, dtype=torch.float32, device=DEVICE)
    dtr = torch.tensor(d_traj_train, dtype=torch.float32, device=DEVICE)
    dva = torch.tensor(d_traj_val, dtype=torch.float32, device=DEVICE)
    vtr = torch.tensor(v_traj_train, dtype=torch.float32, device=DEVICE)
    vva = torch.tensor(v_traj_val, dtype=torch.float32, device=DEVICE)
    Ttr = torch.tensor(dur_train, dtype=torch.float32, device=DEVICE)
    Tva = torch.tensor(dur_val, dtype=torch.float32, device=DEVICE)
    v0tr = torch.tensor(v0_train, dtype=torch.float32, device=DEVICE)
    v0va = torch.tensor(v0_val, dtype=torch.float32, device=DEVICE)
    d0tr = torch.tensor(d0_train, dtype=torch.float32, device=DEVICE)
    d0va = torch.tensor(d0_val, dtype=torch.float32, device=DEVICE)
    gotr = torch.tensor(go_train.astype(np.float32), dtype=torch.float32, device=DEVICE)
    gova = torch.tensor(go_val.astype(np.float32), dtype=torch.float32, device=DEVICE)
    dEndTr = torch.tensor(d_end_train, dtype=torch.float32, device=DEVICE)
    dEndVa = torch.tensor(d_end_val, dtype=torch.float32, device=DEVICE)
    vEndTr = torch.tensor(v_end_train, dtype=torch.float32, device=DEVICE)
    vEndVa = torch.tensor(v_end_val, dtype=torch.float32, device=DEVICE)

    hist = {
        "train_total": [],
        "val_total_oracle": [],
        "val_accel_mae_oracle": [],
        "val_d_end_mae_oracle": [],
        "val_stop_zone_rate_oracle": [],
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_score = np.inf

    def _loss(
        out: dict[str, torch.Tensor],
        a_true: torch.Tensor,
        d_true: torch.Tensor,
        v_true: torch.Tensor,
        T_true: torch.Tensor,
        go: torch.Tensor,
        d_end: torch.Tensor,
        v_end: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a_hat = out["a_eff"]
        y_traj = out["y_traj"]
        d_hat = y_traj[:, :, 0]
        v_hat = y_traj[:, :, 1]
        a_state = y_traj[:, :, 2]
        T_hat = out["duration_s"]
        delay_frac = out["delay_frac"]

        d_term = d_hat[:, -1]
        v_term = v_hat[:, -1]
        a_term = a_hat[:, -1]

        stop_mask = (go < 0.5).float()
        go_mask = 1.0 - stop_mask
        stop_mask_bool = go < 0.5

        traj_acc_loss = F.smooth_l1_loss(a_hat, a_true)
        traj_dist_loss = 0.35 * F.smooth_l1_loss(d_hat, d_true)
        traj_speed_loss = 0.30 * F.smooth_l1_loss(v_hat, v_true)
        dur_loss = F.smooth_l1_loss(T_hat, T_true)
        term_d_loss = F.smooth_l1_loss(d_term, d_end)
        term_v_loss = F.smooth_l1_loss(v_term, v_end)

        # Boundary emphasis (near-zero onset response and settled ending behavior)
        edge_n = max(4, a_true.shape[1] // 10)
        edge_acc_loss = F.smooth_l1_loss(a_hat[:, :edge_n], a_true[:, :edge_n]) + F.smooth_l1_loss(
            a_hat[:, -edge_n:], a_true[:, -edge_n:]
        )

        # Hard constraints already applied in the decoder; losses keep the stop location in target band.
        stop_band_high = torch.relu(d_term - STOP_CONSTRAINT_MAX_M)
        stop_band_low = torch.relu(STOP_CONSTRAINT_MIN_M - d_term)
        stop_band_loss = (((stop_band_high + stop_band_low) ** 2) * stop_mask).sum() / torch.clamp(stop_mask.sum(), min=1.0)
        stop_v_zero_loss = ((torch.square(v_term) * stop_mask)).sum() / torch.clamp(stop_mask.sum(), min=1.0)
        stop_a_zero_loss = ((torch.square(a_term) * stop_mask)).sum() / torch.clamp(stop_mask.sum(), min=1.0)
        tail_k = max(4, a_hat.shape[1] // 12)
        stop_tail_acc_loss = (((a_hat[:, -tail_k:] ** 2).mean(dim=1)) * stop_mask).sum() / torch.clamp(stop_mask.sum(), min=1.0)

        # Go-event consistency (cross line with nonzero motion)
        go_cross_loss = ((torch.square(torch.relu(d_term)) * go_mask)).sum() / torch.clamp(go_mask.sum(), min=1.0)
        go_motion_loss = ((torch.square(torch.relu(0.3 - v_term)) * go_mask)).sum() / torch.clamp(go_mask.sum(), min=1.0)

        # Smoothness/regularity on acceleration and jerk
        da = a_state[:, 1:] - a_state[:, :-1]
        jerk_proxy = da * (a_true.shape[1] - 1) / torch.clamp(T_hat.view(-1, 1), min=0.2)
        smooth_acc = torch.mean(torch.square(a_hat[:, 1:] - a_hat[:, :-1]))
        jerk_reg = torch.mean(torch.square(torch.clamp(jerk_proxy, min=-25.0, max=25.0)))
        if a_hat.shape[1] >= 3:
            curv_reg = torch.mean(torch.square(a_hat[:, 2:] - 2.0 * a_hat[:, 1:-1] + a_hat[:, :-2]))
        else:
            curv_reg = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)

        # Delay prior only for stop runs: encourage short but nonzero delay region (regularization, not hard)
        if stop_mask_bool.any():
            delay_stop = delay_frac[stop_mask_bool]
            delay_reg = torch.mean(torch.square(torch.clamp(delay_stop - 0.08, min=-0.12, max=0.12)))
        else:
            delay_reg = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)

        total = (
            1.00 * traj_acc_loss
            + traj_dist_loss
            + traj_speed_loss
            + 0.80 * edge_acc_loss
            + 1.20 * dur_loss
            + 0.35 * term_d_loss
            + 0.90 * term_v_loss
            + 2.00 * stop_band_loss
            + 1.50 * stop_v_zero_loss
            + 1.50 * stop_a_zero_loss
            + 0.20 * stop_tail_acc_loss
            + 0.60 * go_cross_loss
            + 0.25 * go_motion_loss
            + 0.04 * smooth_acc
            + 0.004 * jerk_reg
            + 0.010 * curv_reg
            + 0.05 * delay_reg
        )
        comps = {
            "a_hat": a_hat.detach(),
            "d_hat": d_hat.detach(),
            "v_hat": v_hat.detach(),
            "d_term": d_term.detach(),
            "v_term": v_term.detach(),
        }
        return total, comps

    for _epoch in range(cfg.node_epochs):
        model.train()
        tr_losses: list[float] = []
        for bidx in batch_iter(np.arange(len(Xtr_sc)), cfg.node_batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            out = model(Xtr[bidx], v0tr[bidx], d0tr[bidx], gotr[bidx])
            loss, _ = _loss(out, atr[bidx], dtr[bidx], vtr[bidx], Ttr[bidx], gotr[bidx], dEndTr[bidx], vEndTr[bidx])
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(float(loss.detach().cpu()))
        hist["train_total"].append(float(np.mean(tr_losses)) if tr_losses else float("nan"))

        model.eval()
        with torch.no_grad():
            out_or = model(Xva_or, v0va, d0va, gova)
            val_loss, comps = _loss(out_or, ava, dva, vva, Tva, gova, dEndVa, vEndVa)
            a_pred = comps["a_hat"].cpu().numpy()
            d_term = comps["d_term"].cpu().numpy()
            v_term = comps["v_term"].cpu().numpy()
        stop_mask_np = (go_val == 0)
        stop_zone_rate = float(
            np.mean(
                (d_term[stop_mask_np] >= STOP_CONSTRAINT_MIN_M)
                & (d_term[stop_mask_np] <= STOP_CONSTRAINT_MAX_M)
                & (v_term[stop_mask_np] <= 0.3)
            )
        ) if stop_mask_np.any() else float("nan")
        hist["val_total_oracle"].append(float(val_loss.detach().cpu()))
        hist["val_accel_mae_oracle"].append(float(np.mean(np.abs(a_pred - a_val))))
        hist["val_d_end_mae_oracle"].append(float(np.mean(np.abs(d_term - d_end_val))))
        hist["val_stop_zone_rate_oracle"].append(stop_zone_rate)

        score = hist["val_accel_mae_oracle"][-1] + 0.25 * hist["val_d_end_mae_oracle"][-1] + (0.30 * (1 - stop_zone_rate) if np.isfinite(stop_zone_rate) else 0.0)
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    def _eval(
        X_sc: np.ndarray,
        a_true_np: np.ndarray,
        d_true_np: np.ndarray,
        v_true_np: np.ndarray,
        T_np: np.ndarray,
        v0_np: np.ndarray,
        d0_np: np.ndarray,
        go_np: np.ndarray,
        d_end_np: np.ndarray,
        v_end_np: np.ndarray,
    ) -> dict[str, Any]:
        Xt = torch.tensor(X_sc, dtype=torch.float32, device=DEVICE)
        v0t = torch.tensor(v0_np, dtype=torch.float32, device=DEVICE)
        d0t = torch.tensor(d0_np, dtype=torch.float32, device=DEVICE)
        got = torch.tensor(go_np.astype(np.float32), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            out = model(Xt, v0t, d0t, got)
        y_traj = out["y_traj"].cpu().numpy()
        a_hat = out["a_eff"].cpu().numpy()
        d_hat = y_traj[:, :, 0]
        v_hat = y_traj[:, :, 1]
        T_hat = out["duration_s"].cpu().numpy()
        delay_frac = out["delay_frac"].cpu().numpy()
        a_hat = np.nan_to_num(a_hat, nan=0.0, posinf=0.0, neginf=0.0)
        d_hat = np.nan_to_num(d_hat, nan=0.0, posinf=0.0, neginf=0.0)
        v_hat = np.nan_to_num(v_hat, nan=0.0, posinf=0.0, neginf=0.0)
        T_hat = np.nan_to_num(T_hat, nan=1.0, posinf=20.0, neginf=0.35)
        delay_frac = np.nan_to_num(delay_frac, nan=0.0, posinf=cfg.reaction_delay_max_frac, neginf=0.0)
        stop_eval = _compute_stop_extended_metrics(
            a_true=a_true_np,
            d_true=d_true_np,
            v_true=v_true_np,
            T_true=T_np,
            a_pred=a_hat,
            d_pred=d_hat,
            v_pred=v_hat,
            T_pred=T_hat,
            go=go_np,
            d_end_obs=d_end_np,
            v_end_obs=v_end_np,
        )
        d_term = stop_eval["d_term_eval"]
        v_term = stop_eval["v_term_eval"]
        metrics = {
            "traj_mae_mps2": float(stop_eval["traj_mae_mps2"]),
            "traj_rmse_mps2": float(stop_eval["traj_rmse_mps2"]),
            "distance_curve_mae_m": float(stop_eval["distance_curve_mae_m"]),
            "distance_curve_rmse_m": float(stop_eval["distance_curve_rmse_m"]),
            "speed_curve_mae_mps": float(stop_eval["speed_curve_mae_mps"]),
            "speed_curve_rmse_mps": float(stop_eval["speed_curve_rmse_mps"]),
            "duration_mae_s": float(mean_absolute_error(T_np, T_hat)),
            "duration_rmse_s": float(math.sqrt(mean_squared_error(T_np, T_hat))),
            "terminal_distance_mae_m": float(mean_absolute_error(d_end_np, d_term)),
            "terminal_speed_mae_mps": float(mean_absolute_error(v_end_np, v_term)),
            "mean_delay_frac": float(np.mean(delay_frac)),
            "median_delay_frac": float(np.median(delay_frac)),
        }
        stop_mask = (go_np == 0)
        go_mask = ~stop_mask
        if stop_mask.any():
            metrics["stop_terminal_in_band_rate"] = float(
                np.mean((d_term[stop_mask] >= STOP_CONSTRAINT_MIN_M) & (d_term[stop_mask] <= STOP_CONSTRAINT_MAX_M) & (v_term[stop_mask] <= 0.3))
            )
            metrics["stop_terminal_distance_bias_m"] = float(np.mean(d_term[stop_mask] - d_end_np[stop_mask]))
        if go_mask.any():
            metrics["go_crossed_light_rate"] = float(np.mean(d_term[go_mask] <= 0.0))
            metrics["go_terminal_distance_bias_m"] = float(np.mean(d_term[go_mask] - d_end_np[go_mask]))
        return {
            "metrics": metrics,
            "a_pred": a_hat,
            "d_traj_pred": d_hat,
            "v_traj_pred": v_hat,
            "T_pred": T_hat,
            "delay_frac": delay_frac,
            "d_pred": d_term,
            "v_pred": v_term,
        }

    train_eval = _eval(Xtr_sc, a_train, d_traj_train, v_traj_train, dur_train, v0_train, d0_train, go_train, d_end_train, v_end_train)
    val_eval_oracle = _eval(Xva_or_sc, a_val, d_traj_val, v_traj_val, dur_val, v0_val, d0_val, go_val, d_end_val, v_end_val)
    val_eval_cascade = _eval(Xva_ca_sc, a_val, d_traj_val, v_traj_val, dur_val, v0_val, d0_val, go_val, d_end_val, v_end_val)

    metrics = {
        "train_oracle_decision_input": train_eval["metrics"],
        "val_oracle_decision_input": val_eval_oracle["metrics"],
        "val_cascaded_decision_input": val_eval_cascade["metrics"],
        "stage2_feature_names": ["v0_kmh", "d_th_m", "tti_s", "a_req_mps2", "age_years", "decision_go_class", "decision_go_prob", "sex_female"],
        "node_config": asdict(cfg),
    }
    outputs = {
        "X_train_scaled": Xtr_sc,
        "X_val_oracle_scaled": Xva_or_sc,
        "X_val_cascade_scaled": Xva_ca_sc,
        "train_eval": train_eval,
        "val_eval_oracle": val_eval_oracle,
        "val_eval_cascade": val_eval_cascade,
    }
    return model, {"metrics": metrics, "outputs": outputs}, hist, scaler


def compute_tti_and_areq(v_mps: torch.Tensor, d_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v_pos = torch.clamp(v_mps, min=0.05)
    d_pos = torch.clamp(d_m, min=0.0)
    tti = d_pos / v_pos
    areq = torch.where(d_pos > 1e-3, torch.square(v_pos) / torch.clamp(2.0 * d_pos, min=0.5), torch.zeros_like(d_pos))
    areq = torch.clamp(areq, min=0.0, max=12.0)
    return tti, areq


def build_ar_token_scaler(
    a_train: np.ndarray,
    d_train: np.ndarray,
    v_train_mps: np.ndarray,
    age_train: np.ndarray,
    decision_class_train: np.ndarray,
    p_go_train: np.ndarray,
    sex_female_train: np.ndarray,
) -> StandardScaler:
    B, N = a_train.shape
    prev_a = np.concatenate([np.zeros((B, 1), dtype=np.float32), a_train[:, :-1].astype(np.float32)], axis=1)
    v_t = torch.tensor(v_train_mps.astype(np.float32), dtype=torch.float32)
    d_t = torch.tensor(d_train.astype(np.float32), dtype=torch.float32)
    tti_t, areq_t = compute_tti_and_areq(v_t, d_t)

    static_rep = np.zeros((B, N, 4), dtype=np.float32)
    static_rep[:, :, 0] = age_train[:, None]
    static_rep[:, :, 1] = decision_class_train[:, None]
    static_rep[:, :, 2] = p_go_train[:, None]
    static_rep[:, :, 3] = sex_female_train[:, None]
    goal_rep = np.zeros((B, N, 2), dtype=np.float32)
    goal_rep[:, :, 0] = 0.0
    goal_rep[:, :, 1] = 0.5

    toks = np.concatenate(
        [
            v_train_mps.astype(np.float32)[:, :, None],
            d_train.astype(np.float32)[:, :, None],
            areq_t.numpy()[:, :, None],
            tti_t.numpy()[:, :, None],
            prev_a[:, :, None],
            static_rep,
            goal_rep,
        ],
        axis=2,
    )  # [B, N, 11]
    scaler = StandardScaler()
    scaler.fit(toks.reshape(-1, toks.shape[-1]))
    return scaler


class ARTrajectoryTransformer(nn.Module):
    def __init__(self, feature_dim: int, cfg: NodeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_proj = nn.Linear(feature_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(cfg.context_window, cfg.d_model))
        self.in_ln = nn.LayerNorm(cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.tr_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.out_ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)
        self._mask_cache: dict[int, torch.Tensor] = {}
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        key = (L, device.type)
        if key in self._mask_cache:
            return self._mask_cache[key]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        self._mask_cache[key] = mask
        return mask

    def forward(self, x_seq_scaled: torch.Tensor) -> torch.Tensor:
        # [B, L, F] -> [B, L]
        B, L, _ = x_seq_scaled.shape
        h = self.token_proj(x_seq_scaled)
        h = h + self.pos_emb[:L].unsqueeze(0)
        h = self.in_ln(h)
        h = self.encoder(h, mask=self._causal_mask(L, x_seq_scaled.device))
        h = self.out_ln(h)
        return self.head(h).squeeze(-1)


def rollout_ar_transformer(
    model: ARTrajectoryTransformer,
    token_mean: torch.Tensor,
    token_scale: torch.Tensor,
    cfg: NodeConfig,
    v0_mps: torch.Tensor,
    d0_m: torch.Tensor,
    duration_s: torch.Tensor,
    age_years: torch.Tensor,
    decision_go_class: torch.Tensor,
    decision_go_prob: torch.Tensor,
    sex_female: torch.Tensor | None = None,
    token_feature_keep_mask: torch.Tensor | None = None,
    n_steps_override: int | None = None,
    dt_override_s: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    B = v0_mps.shape[0]
    N = int(n_steps_override) if n_steps_override is not None else int(cfg.traj_points)
    if dt_override_s is None:
        dt = torch.clamp(duration_s / max(N - 1, 1), min=0.01)
    else:
        dt = torch.clamp(dt_override_s, min=0.01)
    if sex_female is None:
        sex_female = torch.zeros_like(age_years)
    token_keep = None
    token_fill_raw = None
    if token_feature_keep_mask is not None:
        # Use the training-token mean as a neutral "unknown" value in raw token space.
        token_keep = (token_feature_keep_mask.view(1, -1) > 0.5)
        token_fill_raw = token_mean.view(1, -1)
    keep_decision_binary = True if token_keep is None else bool(token_keep[0, 6].item())
    keep_decision_prob = True if token_keep is None else bool(token_keep[0, 7].item())
    if keep_decision_binary:
        decision_go_class_ctrl = decision_go_class
    elif keep_decision_prob:
        # If only p(go) is visible, derive the control branch mode from visible information only.
        decision_go_class_ctrl = (decision_go_prob >= 0.5).float()
    else:
        # If both decision label and confidence are hidden, avoid leaking run type into the
        # closed-loop branch; use a neutral unknown state.
        decision_go_class_ctrl = torch.full_like(decision_go_class, 0.5)
    stop_mask = decision_go_class_ctrl < 0.5

    v_curr = v0_mps.clone()
    d_curr = d0_m.clone()
    a_prev = torch.zeros_like(v_curr)
    stopped = torch.zeros(B, dtype=torch.bool, device=v_curr.device)

    tokens_hist: list[torch.Tensor] = []
    a_raw_seq: list[torch.Tensor] = []
    a_final_seq: list[torch.Tensor] = []
    d_seq: list[torch.Tensor] = []
    v_seq: list[torch.Tensor] = []
    gate_seq: list[torch.Tensor] = []
    jerk_hit_seq: list[torch.Tensor] = []
    actuator_hit_seq: list[torch.Tensor] = []
    stop_freeze_seq: list[torch.Tensor] = []

    for t in range(N):
        d_seq.append(d_curr)
        v_seq.append(v_curr)
        tti_t, areq_t = compute_tti_and_areq(v_curr, d_curr)
        goal_v = torch.zeros_like(v_curr)
        goal_d = torch.full_like(v_curr, 0.5)

        tok_t = torch.stack(
            [
                v_curr,
                d_curr,
                areq_t,
                tti_t,
                a_prev,
                age_years,
                decision_go_class_ctrl,
                decision_go_prob,
                sex_female,
                goal_v,
                goal_d,
            ],
            dim=1,
        )
        if token_keep is not None and token_fill_raw is not None:
            tok_t = torch.where(token_keep, tok_t, token_fill_raw.expand_as(tok_t))
        tokens_hist.append(tok_t)
        L = min(len(tokens_hist), cfg.context_window)
        seq = torch.stack(tokens_hist[-L:], dim=1)
        seq_scaled = (seq - token_mean.view(1, 1, -1)) / token_scale.view(1, 1, -1)
        seq_scaled = torch.nan_to_num(seq_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        if token_feature_keep_mask is not None:
            seq_scaled = seq_scaled * token_feature_keep_mask.view(1, 1, -1)

        a_raw = model(seq_scaled)[:, -1]

        # Terminal gate (stop/go variants)
        # Use the same masked observation values for gate inputs so hidden d/v do not
        # leak back into the visible a_prev token through closed-loop gating.
        if token_fill_raw is not None:
            d_gate_src = torch.where(
                token_keep[:, 1].expand_as(d_curr),
                d_curr,
                token_fill_raw[:, 1].expand_as(d_curr),
            )
            v_gate_src = torch.where(
                token_keep[:, 0].expand_as(v_curr),
                v_curr,
                token_fill_raw[:, 0].expand_as(v_curr),
            )
        else:
            d_gate_src = d_curr
            v_gate_src = v_curr
        d_safe_stop = torch.clamp(d_gate_src + 0.5, min=0.0)
        d_safe_go = torch.clamp(d_gate_src + 0.25, min=0.0)
        v_safe = torch.clamp(v_gate_src, min=0.0)
        gate_stop = (1.0 - torch.exp(-d_safe_stop / cfg.lambda_d_stop)) * (1.0 - torch.exp(-v_safe / cfg.lambda_v_stop))
        gate_go = (1.0 - torch.exp(-d_safe_go / cfg.lambda_d_go)) * (1.0 - torch.exp(-v_safe / cfg.lambda_v_go))
        gate = torch.where(stop_mask, gate_stop, gate_go)
        gate = torch.clamp(torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)
        a_gated = a_raw * gate

        # Hard jerk limit in real time
        delta_a_max = cfg.max_abs_jerk_mps3 * dt
        a_jerk = torch.maximum(torch.minimum(a_gated, a_prev + delta_a_max), a_prev - delta_a_max)
        jerk_hit = (torch.abs(a_jerk - a_gated) > 1e-6).float()

        # Actuator limits (decision-aware)
        a_stop = torch.clamp(a_jerk, min=cfg.a_min_brake_mps2, max=0.0)
        a_go = torch.clamp(a_jerk, min=cfg.a_min_brake_mps2, max=cfg.a_max_accel_go_mps2)
        a_final = torch.where(stop_mask, a_stop, a_go)
        actuator_hit = (torch.abs(a_final - a_jerk) > 1e-6).float()

        a_final = torch.where(stopped, torch.zeros_like(a_final), a_final)

        a_raw_seq.append(a_raw)
        a_final_seq.append(a_final)
        gate_seq.append(gate)
        jerk_hit_seq.append(jerk_hit)
        actuator_hit_seq.append(actuator_hit)
        stop_freeze_seq.append(stopped.float())

        if t < N - 1:
            ds = v_curr * dt + 0.5 * a_final * dt * dt
            d_next = d_curr - ds
            v_next = torch.clamp(v_curr + a_final * dt, min=0.0)

            crossed_stop = stop_mask & (~stopped) & (d_next <= cfg.terminal_deadband_min_m)
            reached_zero = stop_mask & (~stopped) & (v_next <= cfg.stop_eps_mps)
            newly_stopped = crossed_stop | reached_zero

            d_next = torch.where(crossed_stop, torch.full_like(d_next, cfg.terminal_deadband_min_m), d_next)
            v_next = torch.where(newly_stopped, torch.zeros_like(v_next), v_next)
            stopped = stopped | newly_stopped
            d_next = torch.where(stopped & (~newly_stopped), d_curr, d_next)

            v_curr = torch.nan_to_num(v_next, nan=0.0, posinf=50.0, neginf=0.0)
            d_curr = torch.nan_to_num(d_next, nan=0.0, posinf=200.0, neginf=-50.0)
            a_prev = torch.nan_to_num(a_final, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "a_raw": torch.stack(a_raw_seq, dim=1),
        "a_final": torch.stack(a_final_seq, dim=1),
        "d_traj": torch.stack(d_seq, dim=1),
        "v_traj": torch.stack(v_seq, dim=1),
        "d_end": torch.stack(d_seq, dim=1)[:, -1],
        "v_end": torch.stack(v_seq, dim=1)[:, -1],
        "gate_seq": torch.stack(gate_seq, dim=1),
        "jerk_hit_seq": torch.stack(jerk_hit_seq, dim=1),
        "actuator_hit_seq": torch.stack(actuator_hit_seq, dim=1),
        "stop_freeze_seq": torch.stack(stop_freeze_seq, dim=1),
        "duration_s": duration_s,
    }


def train_transformer_model(
    X2_train: np.ndarray,
    X2_val_oracle: np.ndarray,
    X2_val_cascade: np.ndarray,
    a_train: np.ndarray,
    a_val: np.ndarray,
    d_traj_train: np.ndarray,
    d_traj_val: np.ndarray,
    v_traj_train: np.ndarray,
    v_traj_val: np.ndarray,
    dur_train: np.ndarray,
    dur_val: np.ndarray,
    v0_train: np.ndarray,
    v0_val: np.ndarray,
    d0_train: np.ndarray,
    d0_val: np.ndarray,
    go_train: np.ndarray,
    go_val: np.ndarray,
    d_end_train: np.ndarray,
    d_end_val: np.ndarray,
    v_end_train: np.ndarray,
    v_end_val: np.ndarray,
    cfg: NodeConfig,
    token_feature_keep_mask: np.ndarray | None = None,
) -> tuple[ARTrajectoryTransformer, dict[str, Any], dict[str, list[float]], StandardScaler]:
    age_train = X2_train[:, 4].astype(np.float32)
    decision_class_train = X2_train[:, 5].astype(np.float32)
    p_go_train = X2_train[:, 6].astype(np.float32)
    sex_female_train = X2_train[:, 7].astype(np.float32)
    token_scaler = build_ar_token_scaler(a_train, d_traj_train, v_traj_train, age_train, decision_class_train, p_go_train, sex_female_train)
    token_mean = torch.tensor(token_scaler.mean_.astype(np.float32), dtype=torch.float32, device=DEVICE)
    token_scale = torch.tensor(token_scaler.scale_.astype(np.float32), dtype=torch.float32, device=DEVICE)
    token_scale = torch.where(token_scale < 1e-6, torch.ones_like(token_scale), token_scale)
    if token_feature_keep_mask is None:
        token_feature_keep_mask_arr = np.ones_like(token_scaler.mean_, dtype=np.float32)
        token_feature_keep_mask_t = None
    else:
        token_feature_keep_mask_arr = np.asarray(token_feature_keep_mask, dtype=np.float32).reshape(-1)
        if token_feature_keep_mask_arr.shape[0] != token_scaler.mean_.shape[0]:
            raise ValueError(
                f"token_feature_keep_mask length {token_feature_keep_mask_arr.shape[0]} "
                f"!= token feature dim {token_scaler.mean_.shape[0]}"
            )
        token_feature_keep_mask_t = torch.tensor(token_feature_keep_mask_arr, dtype=torch.float32, device=DEVICE)
    keep_decision_binary = bool(token_feature_keep_mask_arr[6] > 0.5)
    keep_decision_prob = bool(token_feature_keep_mask_arr[7] > 0.5)

    model = ARTrajectoryTransformer(feature_dim=int(token_scaler.mean_.shape[0]), cfg=cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ar_lr, weight_decay=cfg.weight_decay)

    # Tensors
    Xtr = torch.tensor(X2_train, dtype=torch.float32, device=DEVICE)
    Xva_or = torch.tensor(X2_val_oracle, dtype=torch.float32, device=DEVICE)
    Xva_ca = torch.tensor(X2_val_cascade, dtype=torch.float32, device=DEVICE)
    atr = torch.tensor(a_train, dtype=torch.float32, device=DEVICE)
    ava = torch.tensor(a_val, dtype=torch.float32, device=DEVICE)
    dtr = torch.tensor(d_traj_train, dtype=torch.float32, device=DEVICE)
    dva = torch.tensor(d_traj_val, dtype=torch.float32, device=DEVICE)
    vtr = torch.tensor(v_traj_train, dtype=torch.float32, device=DEVICE)
    vva = torch.tensor(v_traj_val, dtype=torch.float32, device=DEVICE)
    Ttr = torch.tensor(dur_train, dtype=torch.float32, device=DEVICE)
    Tva = torch.tensor(dur_val, dtype=torch.float32, device=DEVICE)
    v0tr = torch.tensor(v0_train, dtype=torch.float32, device=DEVICE)
    v0va = torch.tensor(v0_val, dtype=torch.float32, device=DEVICE)
    d0tr = torch.tensor(d0_train, dtype=torch.float32, device=DEVICE)
    d0va = torch.tensor(d0_val, dtype=torch.float32, device=DEVICE)
    gotr = torch.tensor(go_train.astype(np.float32), dtype=torch.float32, device=DEVICE)
    gova = torch.tensor(go_val.astype(np.float32), dtype=torch.float32, device=DEVICE)
    dEndTr = torch.tensor(d_end_train, dtype=torch.float32, device=DEVICE)
    dEndVa = torch.tensor(d_end_val, dtype=torch.float32, device=DEVICE)
    vEndTr = torch.tensor(v_end_train, dtype=torch.float32, device=DEVICE)
    vEndVa = torch.tensor(v_end_val, dtype=torch.float32, device=DEVICE)

    hist = {
        "train_total": [],
        "val_total_oracle": [],
        "val_accel_mae_oracle": [],
        "val_d_end_mae_oracle": [],
        "val_stop_zone_rate_oracle": [],
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_score = np.inf

    def _roll_from_x2(
        Xb: torch.Tensor,
        v0b: torch.Tensor,
        d0b: torch.Tensor,
        Tb: torch.Tensor,
        *,
        n_steps_override: int | None = None,
        dt_override_s: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        age = Xb[:, 4]
        dclass = Xb[:, 5]
        pgo = Xb[:, 6]
        sex = Xb[:, 7]
        return rollout_ar_transformer(
            model=model,
            token_mean=token_mean,
            token_scale=token_scale,
            cfg=cfg,
            v0_mps=v0b,
            d0_m=d0b,
            duration_s=Tb,
            age_years=age,
            decision_go_class=dclass,
            decision_go_prob=pgo,
            sex_female=sex,
            token_feature_keep_mask=token_feature_keep_mask_t,
            n_steps_override=n_steps_override,
            dt_override_s=dt_override_s,
        )

    def _loss(
        roll: dict[str, torch.Tensor],
        a_true: torch.Tensor,
        d_true: torch.Tensor,
        v_true: torch.Tensor,
        T_true: torch.Tensor,
        go_true: torch.Tensor,
        d_end_obs: torch.Tensor,
        v_end_obs: torch.Tensor,
        epoch_idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a_hat = roll["a_final"]
        d_hat = roll["d_traj"]
        v_hat = roll["v_traj"]
        d_end = roll["d_end"]
        v_end = roll["v_end"]
        go_soft = torch.clamp(go_true, min=0.0, max=1.0)
        stop_m = 1.0 - go_soft
        go_m = go_soft
        stop_cnt = torch.clamp(stop_m.sum(), min=1.0)
        go_cnt = torch.clamp(go_m.sum(), min=1.0)

        acc_loss = F.mse_loss(a_hat, a_true)
        d_curve_loss = F.smooth_l1_loss(d_hat, d_true)
        v_curve_loss = F.smooth_l1_loss(v_hat, v_true)
        duration_loss = F.smooth_l1_loss(roll["duration_s"], T_true)

        deadband_low = torch.relu(cfg.terminal_deadband_min_m - d_end)
        deadband_high = torch.relu(d_end - cfg.terminal_deadband_max_m)
        deadband_loss = (((deadband_low + deadband_high) ** 2) * stop_m).sum() / stop_cnt
        go_cross_loss = ((torch.square(torch.relu(d_end)) * go_m)).sum() / go_cnt
        stop_v_term = ((torch.square(v_end) * stop_m)).sum() / stop_cnt
        stop_a_term = ((torch.square(a_hat[:, -1]) * stop_m)).sum() / stop_cnt

        tail_k = max(4, a_hat.shape[1] // 10)
        tail_smooth = torch.mean(torch.square(a_hat[:, -tail_k:]))
        if a_hat.shape[1] >= 3:
            curv_loss = torch.mean(torch.square(a_hat[:, 2:] - 2.0 * a_hat[:, 1:-1] + a_hat[:, :-2]))
        else:
            curv_loss = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
        # Real-time jerk/snap regularization to suppress autoregressive chatter while
        # avoiding penalizing the hard stop-freeze segment.
        if a_hat.shape[1] >= 2:
            dt_row = torch.clamp(roll["duration_s"] / max(cfg.traj_points - 1, 1), min=0.01).unsqueeze(1)
            jerk = (a_hat[:, 1:] - a_hat[:, :-1]) / dt_row
            free0 = (roll["stop_freeze_seq"][:, :-1] < 0.5)
            free1 = (roll["stop_freeze_seq"][:, 1:] < 0.5)
            jerk_mask = (free0 & free1).float()
            jerk_denom = torch.clamp(jerk_mask.sum(), min=1.0)
            jerk_reg = torch.sum(torch.square(jerk) * jerk_mask) / jerk_denom
        else:
            jerk_reg = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
        if a_hat.shape[1] >= 3:
            dt_row = torch.clamp(roll["duration_s"] / max(cfg.traj_points - 1, 1), min=0.01).unsqueeze(1)
            jerk_full = (a_hat[:, 1:] - a_hat[:, :-1]) / dt_row
            snap = (jerk_full[:, 1:] - jerk_full[:, :-1]) / dt_row
            freej0 = (roll["stop_freeze_seq"][:, :-2] < 0.5)
            freej1 = (roll["stop_freeze_seq"][:, 1:-1] < 0.5)
            freej2 = (roll["stop_freeze_seq"][:, 2:] < 0.5)
            snap_mask = (freej0 & freej1 & freej2).float()
            snap_denom = torch.clamp(snap_mask.sum(), min=1.0)
            snap_reg = torch.sum(torch.square(snap) * snap_mask) / snap_denom
        else:
            snap_reg = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)

        alpha = float(min(max(epoch_idx / max(cfg.curriculum_warmup_epochs, 1), 0.0), 1.0))
        total = (
            cfg.w_accel * acc_loss
            + cfg.w_terminal_deadband * deadband_loss
            + cfg.w_go_cross * go_cross_loss
            + (alpha * cfg.w_d_curve) * d_curve_loss
            + (alpha * cfg.w_v_curve) * v_curve_loss
            + 0.5 * cfg.w_duration * duration_loss
            + cfg.w_stop_terminal_speed * stop_v_term
            + cfg.w_stop_terminal_accel * stop_a_term
            + cfg.w_tail_smooth * tail_smooth
            + cfg.w_curvature * curv_loss
            + cfg.w_jerk_reg * jerk_reg
            + cfg.w_snap_reg * snap_reg
        )
        comps = {"a_hat": a_hat.detach(), "d_end": d_end.detach(), "v_end": v_end.detach()}
        return total, comps

    t_train0 = time.time()
    # Stage-2 training/control penalties are driven by stage-1 outputs (cascade), not ground-truth labels.
    if keep_decision_binary:
        go_train_stage2 = torch.clamp(Xtr[:, 5], 0.0, 1.0)
        go_val_stage2 = torch.clamp(Xva_ca[:, 5], 0.0, 1.0)
    elif keep_decision_prob:
        go_train_stage2 = torch.clamp(Xtr[:, 6], 0.0, 1.0)
        go_val_stage2 = torch.clamp(Xva_ca[:, 6], 0.0, 1.0)
    else:
        # Decision channels hidden: keep Stage-2 "pure" with no stop/go label input signal.
        go_train_stage2 = torch.full((Xtr.shape[0],), 0.5, dtype=torch.float32, device=DEVICE)
        go_val_stage2 = torch.full((Xva_ca.shape[0],), 0.5, dtype=torch.float32, device=DEVICE)
    for epoch in range(cfg.ar_epochs):
        t_ep0 = time.time()
        model.train()
        losses: list[float] = []
        for bidx in batch_iter(np.arange(len(X2_train)), cfg.ar_batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            roll = _roll_from_x2(Xtr[bidx], v0tr[bidx], d0tr[bidx], Ttr[bidx])
            loss, _ = _loss(roll, atr[bidx], dtr[bidx], vtr[bidx], Ttr[bidx], go_train_stage2[bidx], dEndTr[bidx], vEndTr[bidx], epoch)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        hist["train_total"].append(float(np.mean(losses)) if losses else float("nan"))

        model.eval()
        with torch.no_grad():
            # Cascade-first validation for ablations: select the checkpoint using stage-1 outputs,
            # not oracle decision labels on validation runs.
            roll_ca = _roll_from_x2(Xva_ca, v0va, d0va, Tva)
            val_loss, comps = _loss(roll_ca, ava, dva, vva, Tva, go_val_stage2, dEndVa, vEndVa, epoch)
            a_pred = comps["a_hat"].cpu().numpy()
            d_term = comps["d_end"].cpu().numpy()
            v_term = comps["v_end"].cpu().numpy()
        stop_mask_np = (go_val == 0)
        stop_zone_rate = float(
            np.mean(
                (d_term[stop_mask_np] >= STOP_CONSTRAINT_MIN_M)
                & (d_term[stop_mask_np] <= STOP_CONSTRAINT_MAX_M)
                & (v_term[stop_mask_np] <= 0.3)
            )
        ) if stop_mask_np.any() else float("nan")
        hist["val_total_oracle"].append(float(val_loss.detach().cpu()))
        hist["val_accel_mae_oracle"].append(float(np.mean(np.abs(a_pred - a_val))))
        hist["val_d_end_mae_oracle"].append(float(np.mean(np.abs(d_term - d_end_val))))
        hist["val_stop_zone_rate_oracle"].append(stop_zone_rate)
        score = hist["val_accel_mae_oracle"][-1] + 0.22 * hist["val_d_end_mae_oracle"][-1] + (
            0.30 * (1.0 - stop_zone_rate) if np.isfinite(stop_zone_rate) else 0.0
        )
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        ep_time = time.time() - t_ep0
        elapsed = time.time() - t_train0
        eta = (cfg.ar_epochs - (epoch + 1)) * (elapsed / max(epoch + 1, 1))
        log_progress(
            "AR epoch {}/{} | train_loss={:.4f} | val_loss={:.4f} | val_aMAE={:.4f} | val_dEndMAE={:.4f} | stopBandRate={} | {:.1f}s/ep | ETA {:.1f} min".format(
                epoch + 1,
                cfg.ar_epochs,
                hist["train_total"][-1],
                hist["val_total_oracle"][-1],
                hist["val_accel_mae_oracle"][-1],
                hist["val_d_end_mae_oracle"][-1],
                "nan" if not np.isfinite(hist["val_stop_zone_rate_oracle"][-1]) else f"{hist['val_stop_zone_rate_oracle'][-1]:.3f}",
                ep_time,
                eta / 60.0,
            )
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    def _eval(X2: np.ndarray, a_true: np.ndarray, d_true: np.ndarray, v_true: np.ndarray, T: np.ndarray, v0: np.ndarray, d0: np.ndarray, go: np.ndarray, d_end_obs: np.ndarray, v_end_obs: np.ndarray) -> dict[str, Any]:
        X2t = torch.tensor(X2, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            roll = _roll_from_x2(
                X2t,
                torch.tensor(v0, dtype=torch.float32, device=DEVICE),
                torch.tensor(d0, dtype=torch.float32, device=DEVICE),
                torch.tensor(T, dtype=torch.float32, device=DEVICE),
            )
        a_hat = torch.nan_to_num(roll["a_final"], nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
        d_hat = torch.nan_to_num(roll["d_traj"], nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
        v_hat = torch.nan_to_num(roll["v_traj"], nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
        d_end = torch.nan_to_num(roll["d_end"], nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
        v_end = torch.nan_to_num(roll["v_end"], nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
        gate_seq = roll["gate_seq"].cpu().numpy()
        jerk_hit_seq = roll["jerk_hit_seq"].cpu().numpy()
        actuator_hit_seq = roll["actuator_hit_seq"].cpu().numpy()
        stop_freeze_seq = roll["stop_freeze_seq"].cpu().numpy()
        T_pred_eval = np.asarray(T, dtype=np.float64).copy()
        a_for_metrics = a_hat
        d_for_metrics = d_hat
        v_for_metrics = v_hat
        stop_mask_np = (go == 0)
        if np.any(stop_mask_np):
            stop_idx = np.flatnonzero(stop_mask_np)
            dt_stop_np = np.clip(np.asarray(T, dtype=np.float64)[stop_idx] / max(cfg.traj_points - 1, 1), 0.01, None)
            n_steps_ext = max(cfg.traj_points, int(math.ceil(cfg.traj_points * STOP_EVAL_AR_MAX_HORIZON_FACTOR)))
            X2_stop_t = torch.tensor(X2[stop_idx], dtype=torch.float32, device=DEVICE)
            v0_stop_t = torch.tensor(v0[stop_idx], dtype=torch.float32, device=DEVICE)
            d0_stop_t = torch.tensor(d0[stop_idx], dtype=torch.float32, device=DEVICE)
            T_stop_t = torch.tensor(T[stop_idx], dtype=torch.float32, device=DEVICE)
            dt_stop_t = torch.tensor(dt_stop_np.astype(np.float32), dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                roll_ext = _roll_from_x2(
                    X2_stop_t,
                    v0_stop_t,
                    d0_stop_t,
                    T_stop_t,
                    n_steps_override=n_steps_ext,
                    dt_override_s=dt_stop_t,
                )
            a_ext = torch.nan_to_num(roll_ext["a_final"], nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
            d_ext = torch.nan_to_num(roll_ext["d_traj"], nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
            v_ext = torch.nan_to_num(roll_ext["v_traj"], nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
            a_for_metrics = a_hat.copy()
            d_for_metrics = d_hat.copy()
            v_for_metrics = v_hat.copy()
            # For stop rows, use extended rollout arrays in stop-aware metric computation.
            a_stop_rows = a_ext
            d_stop_rows = d_ext
            v_stop_rows = v_ext
            T_pred_eval_stop = dt_stop_np * max(n_steps_ext - 1, 0)
        else:
            stop_idx = np.empty((0,), dtype=int)
            a_stop_rows = d_stop_rows = v_stop_rows = np.empty((0, 0), dtype=np.float32)
            T_pred_eval_stop = np.empty((0,), dtype=np.float64)

        if np.any(stop_mask_np):
            a_metric_in: list[np.ndarray] = [np.asarray(a_for_metrics[i], dtype=np.float64) for i in range(len(go))]
            d_metric_in: list[np.ndarray] = [np.asarray(d_for_metrics[i], dtype=np.float64) for i in range(len(go))]
            v_metric_in: list[np.ndarray] = [np.asarray(v_for_metrics[i], dtype=np.float64) for i in range(len(go))]
            T_metric_in = T_pred_eval.astype(np.float64, copy=True)
            for j, ridx in enumerate(stop_idx.tolist()):
                a_metric_in[ridx] = np.asarray(a_stop_rows[j], dtype=np.float64)
                d_metric_in[ridx] = np.asarray(d_stop_rows[j], dtype=np.float64)
                v_metric_in[ridx] = np.asarray(v_stop_rows[j], dtype=np.float64)
                T_metric_in[ridx] = float(T_pred_eval_stop[j])
            stop_eval = _compute_stop_extended_metrics(
                a_true=a_true,
                d_true=d_true,
                v_true=v_true,
                T_true=T,
                a_pred=a_metric_in,
                d_pred=d_metric_in,
                v_pred=v_metric_in,
                T_pred=T_metric_in,
                go=go,
                d_end_obs=d_end_obs,
                v_end_obs=v_end_obs,
            )
        else:
            stop_eval = _compute_stop_extended_metrics(
                a_true=a_true,
                d_true=d_true,
                v_true=v_true,
                T_true=T,
                a_pred=a_hat,
                d_pred=d_hat,
                v_pred=v_hat,
                T_pred=T_pred_eval,
                go=go,
                d_end_obs=d_end_obs,
                v_end_obs=v_end_obs,
            )

        d_end_eval = stop_eval["d_term_eval"]
        v_end_eval = stop_eval["v_term_eval"]
        m = {
            "traj_mae_mps2": float(stop_eval["traj_mae_mps2"]),
            "traj_rmse_mps2": float(stop_eval["traj_rmse_mps2"]),
            "distance_curve_mae_m": float(stop_eval["distance_curve_mae_m"]),
            "distance_curve_rmse_m": float(stop_eval["distance_curve_rmse_m"]),
            "speed_curve_mae_mps": float(stop_eval["speed_curve_mae_mps"]),
            "speed_curve_rmse_mps": float(stop_eval["speed_curve_rmse_mps"]),
            "duration_mae_s": 0.0,
            "duration_rmse_s": 0.0,
            "terminal_distance_mae_m": float(mean_absolute_error(d_end_obs, d_end_eval)),
            "terminal_speed_mae_mps": float(mean_absolute_error(v_end_obs, v_end_eval)),
            "jerk_limit_hit_rate": float(np.mean(jerk_hit_seq)),
            "actuator_clamp_hit_rate": float(np.mean(actuator_hit_seq)),
            "mean_terminal_gate": float(np.mean(gate_seq)),
        }
        go_mask_np = ~stop_mask_np
        if stop_mask_np.any():
            m["stop_terminal_in_band_rate"] = float(np.mean((d_end_eval[stop_mask_np] >= STOP_CONSTRAINT_MIN_M) & (d_end_eval[stop_mask_np] <= STOP_CONSTRAINT_MAX_M) & (v_end_eval[stop_mask_np] <= 0.3)))
            m["stop_terminal_distance_bias_m"] = float(np.mean(d_end_eval[stop_mask_np] - d_end_obs[stop_mask_np]))
            m["stop_freeze_activation_rate"] = float(np.mean(np.max(stop_freeze_seq[stop_mask_np], axis=1) > 0.5))
        if go_mask_np.any():
            m["go_crossed_light_rate"] = float(np.mean(d_end_eval[go_mask_np] <= 0.0))
            m["go_terminal_distance_bias_m"] = float(np.mean(d_end_eval[go_mask_np] - d_end_obs[go_mask_np]))
        return {
            "metrics": m,
            "a_pred": a_hat,
            "d_traj_pred": d_hat,
            "v_traj_pred": v_hat,
            "d_pred": d_end_eval,
            "v_pred": v_end_eval,
            "gate_seq": gate_seq,
            "jerk_hit_seq": jerk_hit_seq,
            "actuator_hit_seq": actuator_hit_seq,
            "stop_freeze_seq": stop_freeze_seq,
        }

    train_eval = _eval(X2_train, a_train, d_traj_train, v_traj_train, dur_train, v0_train, d0_train, go_train, d_end_train, v_end_train)
    val_eval_or = _eval(X2_val_oracle, a_val, d_traj_val, v_traj_val, dur_val, v0_val, d0_val, go_val, d_end_val, v_end_val)
    val_eval_ca = _eval(X2_val_cascade, a_val, d_traj_val, v_traj_val, dur_val, v0_val, d0_val, go_val, d_end_val, v_end_val)

    metrics = {
        "train_oracle_decision_input": train_eval["metrics"],
        "val_oracle_decision_input": val_eval_or["metrics"],
        "val_cascaded_decision_input": val_eval_ca["metrics"],
        "stage2_feature_names": ["v0_kmh", "d_th_m", "tti_s", "a_req_mps2", "age_years", "decision_go_class", "decision_go_prob", "sex_female"],
        "token_feature_names": ["v_t_mps", "d_t_m", "a_req_t_mps2", "TTI_t_s", "a_prev_mps2", "age", "decision_go_class", "p_go", "sex_female", "v_target_mps", "d_center_m"],
        "token_feature_keep_mask": token_feature_keep_mask_arr.astype(float).tolist(),
        "transformer_config": asdict(cfg),
    }
    outputs = {
        "train_eval": train_eval,
        "val_eval_oracle": val_eval_or,
        "val_eval_cascade": val_eval_ca,
        "token_scaler_mean": token_scaler.mean_.astype(np.float32),
        "token_scaler_scale": token_scaler.scale_.astype(np.float32),
        "token_feature_keep_mask": token_feature_keep_mask_arr.astype(np.float32),
    }
    return model, {"metrics": metrics, "outputs": outputs}, hist, token_scaler


def plot_transformer_results(
    df_val: pd.DataFrame,
    a_true_val: np.ndarray,
    d_true_val: np.ndarray,
    traj_eval_oracle: dict[str, Any],
    traj_eval_cascade: dict[str, Any],
    hist: dict[str, list[float]],
    cfg: NodeConfig,
    out_dir: Path,
    val_go_pred: np.ndarray | None = None,
    val_go_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(hist["train_total"], lw=2.2, color="#1d3557", label="Train total loss")
    ax.plot(hist["val_total_oracle"], lw=2.2, color="#e76f51", label="Val total loss (oracle)")
    ax2 = ax.twinx()
    ax2.plot(hist["val_accel_mae_oracle"], lw=2.0, ls="--", color="#2a9d8f", label="Val a_x MAE")
    ax2.plot(hist["val_d_end_mae_oracle"], lw=2.0, ls=":", color="#6a4c93", label="Val terminal d MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("MAE")
    ax.set_title("AR Transformer (Kinematic-Constrained) Training Curves")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "transformer_training_curves.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "transformer_training_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    a_or = traj_eval_oracle["a_pred"]
    a_ca = traj_eval_cascade["a_pred"]
    d_or = traj_eval_oracle["d_traj_pred"]
    d_ca = traj_eval_cascade["d_traj_pred"]
    go_mask = df_val["go_decision"].to_numpy(dtype=int) == 1
    stop_mask = ~go_mask

    def _pick(mask: np.ndarray, k: int) -> list[int]:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return []
        score = df_val.iloc[idx]["a_req_mps2"].to_numpy(dtype=float) + 0.01 * df_val.iloc[idx]["distance_threshold_m"].to_numpy(dtype=float)
        idx = idx[np.argsort(score)]
        if len(idx) <= k:
            return idx.tolist()
        return idx[np.linspace(0, len(idx) - 1, k, dtype=int)].tolist()

    total_k = max(4, int(cfg.example_plot_total))
    stop_k = int(round(total_k * float(cfg.example_plot_stop_ratio)))
    stop_k = max(1, min(total_k - 1, stop_k))
    go_k = total_k - stop_k
    picks = _pick(stop_mask, stop_k) + _pick(go_mask, go_k)
    if len(picks) < total_k:
        used = set(int(i) for i in picks)
        remaining = [int(i) for i in np.argsort(df_val["a_req_mps2"].to_numpy(dtype=float)) if int(i) not in used]
        for i in remaining:
            picks.append(i)
            if len(picks) >= total_k:
                break
    plot_info: dict[str, Any] = {"sample_indices": [int(i) for i in picks], "sample_source_files": df_val.iloc[picks]["source_file"].tolist() if picks else []}
    if picks:
        cols = 4
        rows = int(math.ceil(len(picks) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4.3 * rows), squeeze=False)
        d_lo = float(np.nanmin(d_true_val[picks]) - 2.0)
        d_hi = float(np.nanmax(d_true_val[picks]) * 1.03)
        first_twin = None
        for ax, i in zip(axes.ravel(), picks):
            T_i = float(df_val.iloc[i]["traj_duration_s"])
            t_axis = np.linspace(0.0, max(T_i, 1e-3), cfg.traj_points)
            ax.plot(t_axis, a_true_val[i], color="#111827", lw=2.4, label="True a_x")
            ax.plot(t_axis, a_or[i], color="#2a9d8f", lw=2.0, label="AR a_x (oracle)")
            ax.plot(t_axis, a_ca[i], color="#e76f51", lw=1.8, ls="--", label="AR a_x (cascade)")
            ax.axhline(0.0, color="0.35", lw=1.0, ls=":")
            ax.set_ylim(-6.5, 3.5)
            axd = ax.twinx()
            if first_twin is None:
                first_twin = axd
            axd.plot(t_axis, d_true_val[i], color="#4c566a", lw=1.7, ls="-.", alpha=0.9, label="True d(t)")
            axd.plot(t_axis, d_or[i], color="#6a4c93", lw=1.4, ls=":", alpha=0.95, label="AR d(t) oracle")
            axd.plot(t_axis, d_ca[i], color="#9d4edd", lw=1.1, ls=(0, (3, 2)), alpha=0.7, label="AR d(t) cascade")
            axd.axhline(0.0, color="#6b7280", lw=0.9, ls="--")
            if int(df_val.iloc[i]["go_decision"]) == 0:
                axd.axhspan(STOP_CONSTRAINT_MIN_M, STOP_CONSTRAINT_MAX_M, color="#f4a261", alpha=0.08)
            axd.set_ylim(d_lo, d_hi)
            ax.set_xlim(0.0, 1.02 * T_i)
            title_parts = [
                f"true={'GO' if int(df_val.iloc[i]['go_decision']) else 'STOP'}",
            ]
            if val_go_pred is not None:
                title_parts.append(f"pred={'GO' if int(val_go_pred[i]) else 'STOP'}")
            if val_go_prob is not None:
                title_parts.append(f"p(go)={float(val_go_prob[i]):.2f}")
            title_parts.append(f"v0={df_val.iloc[i]['speed_at_yellow_kmh']:.0f} km/h")
            title_parts.append(f"d0={df_val.iloc[i]['distance_threshold_m']:.0f} m")
            title_parts.append(f"TTI={df_val.iloc[i]['tti_s']:.2f} s")
            ax.set_title(" | ".join(title_parts), fontsize=8)
            ax.set_xlabel("Time since yellow onset (s)")
            ax.set_ylabel("a_x (m/s^2)")
            axd.set_ylabel("d to light (m)")
            ax.grid(alpha=0.25)
        for ax in axes.ravel()[len(picks):]:
            ax.axis("off")
        h1, l1 = axes.ravel()[0].get_legend_handles_labels()
        h2, l2 = (first_twin.get_legend_handles_labels() if first_twin is not None else ([], []))
        fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=3, fontsize=9, frameon=True)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / "transformer_examples.png", dpi=180, bbox_inches="tight")
        fig.savefig(out_dir / "transformer_examples.pdf", bbox_inches="tight")
        plt.close(fig)

    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    d_true_end = df_val["traj_d_end_obs_m"].to_numpy(dtype=float)
    d_pred_end = traj_eval_oracle["d_pred"]
    if stop_mask.any():
        ax0.scatter(d_true_end[stop_mask], d_pred_end[stop_mask], color="#c1121f", s=35, alpha=0.75, label="Stop")
    if go_mask.any():
        ax0.scatter(d_true_end[go_mask], d_pred_end[go_mask], color="#2a9d8f", s=35, alpha=0.75, label="Go")
    lo = float(np.nanmin([d_true_end.min(), d_pred_end.min()]))
    hi = float(np.nanmax([d_true_end.max(), d_pred_end.max()]))
    ax0.plot([lo, hi], [lo, hi], "k--", lw=1.3, label="Ideal")
    ax0.axhspan(STOP_CONSTRAINT_MIN_M, STOP_CONSTRAINT_MAX_M, color="#f4a261", alpha=0.15, label="Stop target band")
    ax0.set_xlabel("Observed terminal distance (m)")
    ax0.set_ylabel("Predicted terminal distance (m)")
    ax0.set_title("AR Transformer Terminal Distance (Validation)")
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=9)

    gate = traj_eval_oracle["gate_seq"].reshape(-1)
    jerk_hit = traj_eval_oracle["jerk_hit_seq"].reshape(-1)
    act_hit = traj_eval_oracle["actuator_hit_seq"].reshape(-1)
    ax1.hist(gate, bins=np.linspace(0, 1, 16), density=True, color="#457b9d", alpha=0.75, label="Terminal gate")
    ax1.axvline(float(np.mean(gate)), color="#1d3557", lw=2.0, ls="--", label=f"mean={np.mean(gate):.2f}")
    txt = f"Jerk-limit hit rate = {np.mean(jerk_hit):.1%}\nActuator-clamp hit rate = {np.mean(act_hit):.1%}"
    ax1.text(0.98, 0.98, txt, ha="right", va="top", transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa", alpha=0.95))
    ax1.set_xlabel("Constraint gate value")
    ax1.set_ylabel("Density")
    ax1.set_title("Constraint Diagnostics (Validation)")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "transformer_terminal_and_constraints.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "transformer_terminal_and_constraints.pdf", bbox_inches="tight")
    plt.close(fig)
    return plot_info


def plot_node_results(
    df_val: pd.DataFrame,
    a_true_val: np.ndarray,
    d_true_val: np.ndarray,
    node_eval_oracle: dict[str, Any],
    node_eval_cascade: dict[str, Any],
    hist: dict[str, list[float]],
    cfg: NodeConfig,
    out_dir: Path,
) -> dict[str, Any]:
    # Training curves
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(hist["train_total"], lw=2.2, color="#1d3557", label="Train total loss")
    ax.plot(hist["val_total_oracle"], lw=2.2, color="#e76f51", label="Val total loss (oracle)")
    ax2 = ax.twinx()
    ax2.plot(hist["val_accel_mae_oracle"], lw=2.0, ls="--", color="#2a9d8f", label="Val a_x MAE")
    ax2.plot(hist["val_d_end_mae_oracle"], lw=2.0, ls=":", color="#6a4c93", label="Val terminal d MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("MAE")
    ax.set_title("Neural ODE Trajectory Training Curves")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "node_training_curves.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "node_training_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    # Example trajectories (more samples)
    a_or = node_eval_oracle["a_pred"]
    a_ca = node_eval_cascade["a_pred"]
    d_or = node_eval_oracle["d_traj_pred"]
    d_ca = node_eval_cascade["d_traj_pred"]
    delay_or = node_eval_oracle["delay_frac"]
    go_mask = df_val["go_decision"].to_numpy(dtype=int) == 1
    stop_mask = ~go_mask

    def _pick(mask: np.ndarray, k: int) -> list[int]:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return []
        score = df_val.iloc[idx]["a_req_mps2"].to_numpy(dtype=float) + 0.01 * df_val.iloc[idx]["distance_threshold_m"].to_numpy(dtype=float)
        idx = idx[np.argsort(score)]
        if len(idx) <= k:
            return idx.tolist()
        return idx[np.linspace(0, len(idx) - 1, k, dtype=int)].tolist()

    picks = _pick(stop_mask, 6) + _pick(go_mask, 6)
    plot_info: dict[str, Any] = {"sample_indices": [], "sample_source_files": []}
    if picks:
        plot_info["sample_indices"] = [int(i) for i in picks]
        plot_info["sample_source_files"] = df_val.iloc[picks]["source_file"].tolist()
        cols = 4
        rows = int(math.ceil(len(picks) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4.3 * rows), squeeze=False)
        d_lo = float(np.nanmin(d_true_val[picks]) - 2.0)
        d_hi = float(np.nanmax(d_true_val[picks]) * 1.03)
        first_twin = None
        for ax, i in zip(axes.ravel(), picks):
            T_true_i = float(df_val.iloc[i]["traj_duration_s"])
            T_or_i = float(node_eval_oracle["T_pred"][i])
            T_ca_i = float(node_eval_cascade["T_pred"][i])
            t_true = np.linspace(0.0, max(T_true_i, 1e-3), cfg.traj_points)
            t_or = np.linspace(0.0, max(T_or_i, 1e-3), cfg.traj_points)
            t_ca = np.linspace(0.0, max(T_ca_i, 1e-3), cfg.traj_points)

            ax.plot(t_true, a_true_val[i], color="#111827", lw=2.4, label="True a_x")
            ax.plot(t_or, a_or[i], color="#2a9d8f", lw=2.0, label="NODE a_x (oracle)")
            ax.plot(t_ca, a_ca[i], color="#e76f51", lw=1.8, ls="--", label="NODE a_x (cascade)")
            ax.axhline(0.0, color="0.35", lw=1.0, ls=":")
            ax.set_ylim(-6.5, 3.5)
            axd = ax.twinx()
            if first_twin is None:
                first_twin = axd
            axd.plot(t_true, d_true_val[i], color="#4c566a", lw=1.7, ls="-.", alpha=0.9, label="True d(t)")
            axd.plot(t_or, d_or[i], color="#6a4c93", lw=1.4, ls=":", alpha=0.95, label="NODE d(t) oracle")
            axd.plot(t_ca, d_ca[i], color="#9d4edd", lw=1.1, ls=(0, (3, 2)), alpha=0.7, label="NODE d(t) cascade")
            axd.axhline(0.0, color="#6b7280", lw=0.9, ls="--")
            if int(df_val.iloc[i]["go_decision"]) == 0:
                axd.axhspan(STOP_CONSTRAINT_MIN_M, STOP_CONSTRAINT_MAX_M, color="#f4a261", alpha=0.08)
            axd.set_ylim(d_lo, d_hi)
            ax.axvline(float(delay_or[i]) * T_or_i, color="#264653", lw=0.9, ls="--", alpha=0.7)
            ax.set_xlim(0.0, 1.02 * max(T_true_i, T_or_i, T_ca_i))
            ax.set_title(
                f"{'GO' if int(df_val.iloc[i]['go_decision']) else 'STOP'} | "
                f"v0={df_val.iloc[i]['speed_at_yellow_kmh']:.0f} km/h, d0={df_val.iloc[i]['distance_threshold_m']:.0f} m\n"
                f"{df_val.iloc[i]['source_file']}",
                fontsize=8,
            )
            ax.set_xlabel("Time since yellow onset (s)")
            ax.set_ylabel("a_x (m/s^2)")
            axd.set_ylabel("d to light (m)")
            ax.grid(alpha=0.25)
        for ax in axes.ravel()[len(picks):]:
            ax.axis("off")
        h1, l1 = axes.ravel()[0].get_legend_handles_labels()
        h2, l2 = (first_twin.get_legend_handles_labels() if first_twin is not None else ([], []))
        fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=3, fontsize=9, frameon=True)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / "node_examples.png", dpi=180, bbox_inches="tight")
        fig.savefig(out_dir / "node_examples.pdf", bbox_inches="tight")
        plt.close(fig)

    # Terminal + delay diagnostics
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    d_true_end = df_val["traj_d_end_obs_m"].to_numpy(dtype=float)
    d_pred_end = node_eval_oracle["d_pred"]
    if stop_mask.any():
        ax0.scatter(d_true_end[stop_mask], d_pred_end[stop_mask], color="#c1121f", s=35, alpha=0.75, label="Stop")
    if go_mask.any():
        ax0.scatter(d_true_end[go_mask], d_pred_end[go_mask], color="#2a9d8f", s=35, alpha=0.75, label="Go")
    lo = float(np.nanmin([d_true_end.min(), d_pred_end.min()]))
    hi = float(np.nanmax([d_true_end.max(), d_pred_end.max()]))
    ax0.plot([lo, hi], [lo, hi], "k--", lw=1.3, label="Ideal")
    ax0.axhspan(STOP_CONSTRAINT_MIN_M, STOP_CONSTRAINT_MAX_M, color="#f4a261", alpha=0.15, label="Stop target band")
    ax0.set_xlabel("Observed terminal distance (m)")
    ax0.set_ylabel("Predicted terminal distance (m)")
    ax0.set_title("Neural ODE Terminal Distance (Validation)")
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=9)

    delay = node_eval_oracle["delay_frac"]
    delay_stop = delay[stop_mask]
    delay_go = delay[go_mask]
    bins = np.linspace(0, cfg.reaction_delay_max_frac, 14)
    if len(delay_stop):
        ax1.hist(delay_stop, bins=bins, alpha=0.8, color="#c1121f", label="Stop", density=True)
    if len(delay_go):
        ax1.hist(delay_go, bins=bins, alpha=0.7, color="#2a9d8f", label="Go", density=True)
    ax1.set_xlabel("Predicted reaction-delay fraction")
    ax1.set_ylabel("Density")
    ax1.set_title("NODE Learned Delay Distribution (Validation)")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "node_terminal_and_delay.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "node_terminal_and_delay.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize_transformer_text(
    cfg: NodeConfig,
    extraction_summary: pd.DataFrame,
    decision_metrics: dict[str, Any],
    tfm_metrics: dict[str, Any],
    decision_hist: dict[str, list[float]],
    tfm_hist: dict[str, list[float]],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("Two-Stage Decision + Kinematic-Constrained Autoregressive Transformer")
    lines.append("=" * 84)
    lines.append("")
    lines.append("Configuration")
    lines.append(json.dumps(asdict(cfg), indent=2))
    lines.append("")
    lines.append("Dataset and Extraction")
    lines.append(f"- Voluntary-only filter: {cfg.use_only_voluntary}")
    lines.append(f"- Trajectory runs extracted: {len(df_train) + len(df_val)}")
    lines.append(f"- Train/Val split: {len(df_train)}/{len(df_val)} ({cfg.train_ratio:.0%}/{1-cfg.train_ratio:.0%})")
    lines.append(f"- Train stop/go: {(df_train['go_decision']==0).sum()} / {(df_train['go_decision']==1).sum()}")
    lines.append(f"- Val stop/go: {(df_val['go_decision']==0).sum()} / {(df_val['go_decision']==1).sum()}")
    lines.append("- Extraction status counts:")
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
    lines.append("Stage 2 (AR Transformer)")
    lines.append("- Causal decoder predicts acceleration autoregressively using rolling context tokens")
    lines.append("- Fixed onset conditioning for full rollout: decision class and confidence p(go)")
    lines.append("- Token standardization is fit on training tokens only and applied during rollout (no leakage)")
    lines.append("- Constraint pipeline at each step: terminal gate -> jerk limiter (15 m/s^3) -> actuator clamp -> kinematic update")
    lines.append("- Multi-task supervision uses acceleration, distance, and speed trajectories with a shared rollout state")
    lines.append("")
    lines.append("Core equations")
    lines.append("  a_raw,t = Transformer(x_{t-K+1:t})")
    lines.append("  a_gated,t = a_raw,t * (1-exp(-d_safe/lambda_d)) * (1-exp(-v_t/lambda_v))")
    lines.append("  a_t = clamp(clamp(a_gated,t, a_{t-1}-J*dt, a_{t-1}+J*dt), a_min, a_max(decision))")
    lines.append("  v_{t+1}=max(v_t+a_t dt,0), d_{t+1}=d_t-(v_t dt + 0.5 a_t dt^2)")
    lines.append("")
    for key in ("train_oracle_decision_input", "val_oracle_decision_input", "val_cascaded_decision_input"):
        m = tfm_metrics[key]
        lines.append(
            f"- {key}: a_MAE={m['traj_mae_mps2']:.3f}, a_RMSE={m['traj_rmse_mps2']:.3f}, dcurve_MAE={m['distance_curve_mae_m']:.3f}m, vcurve_MAE={m['speed_curve_mae_mps']:.3f}m/s, d_end_MAE={m['terminal_distance_mae_m']:.3f}m, v_end_MAE={m['terminal_speed_mae_mps']:.3f}m/s"
        )
        if "stop_terminal_in_band_rate" in m:
            lines.append(f"  stop_terminal_in_band_rate={m['stop_terminal_in_band_rate']:.3f}")
        if "go_crossed_light_rate" in m:
            lines.append(f"  go_crossed_light_rate={m['go_crossed_light_rate']:.3f}")
        lines.append(
            f"  jerk_hit_rate={m.get('jerk_limit_hit_rate', float('nan')):.3f}, actuator_hit_rate={m.get('actuator_clamp_hit_rate', float('nan')):.3f}, mean_gate={m.get('mean_terminal_gate', float('nan')):.3f}"
        )
    lines.append("")
    lines.append("Training summaries")
    lines.append(f"- Decision final val F1/AUROC in history: {decision_hist['val_f1'][-1]:.3f}/{decision_hist['val_auc'][-1]:.3f}")
    lines.append(
        f"- Transformer final val (oracle) loss/aMAE/dEndMAE/stopZoneRate: {tfm_hist['val_total_oracle'][-1]:.3f}/{tfm_hist['val_accel_mae_oracle'][-1]:.3f}/{tfm_hist['val_d_end_mae_oracle'][-1]:.3f}/{tfm_hist['val_stop_zone_rate_oracle'][-1]:.3f}"
    )
    return "\n".join(lines)


def archive_transformer_ablation_run(cfg: NodeConfig, extra: dict[str, Any] | None = None) -> Path:
    ab_root = OUT_DIR / "ablation_runs"
    ab_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ab_root / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    files_to_copy = [
        OUT_DIR / "report.txt",
        OUT_DIR / "decision_metrics.json",
        OUT_DIR / "transformer_metrics.json",
        OUT_DIR / "validation_predictions.csv",
        OUT_DIR / "validation_trajectory_arrays.npz",
        OUT_DIR / "split_manifest.csv",
        OUT_DIR / "feature_scalers.npz",
        OUT_DIR / "decision_mlp_state_dict.pt",
        OUT_DIR / "trajectory_transformer_state_dict.pt",
        OUT_DIR / "transformer_examples_sample_manifest.csv",
    ]
    for p in files_to_copy:
        if p.exists():
            shutil.copy2(p, run_dir / p.name)
    if PLOTS_DIR.exists():
        shutil.copytree(PLOTS_DIR, run_dir / "plots", dirs_exist_ok=True)
    cache_subset = run_dir / "cache_subset"
    cache_subset.mkdir(exist_ok=True)
    for name in ["trajectory_cache_info.json", "trajectory_meta.csv", "trajectory_extraction_summary.csv"]:
        p = CACHE_DIR / name
        if p.exists():
            shutil.copy2(p, cache_subset / name)
    shutil.copy2(Path(__file__), run_dir / "trajectory_transformer_ar.py")
    meta = {"timestamp": stamp, "config": asdict(cfg), "notes": "AR Transformer ablation snapshot"}
    if extra:
        meta.update(extra)
    (run_dir / "ablation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir
    return plot_info


def archive_ablation_run(cfg: NodeConfig, extra: dict[str, Any] | None = None) -> Path:
    ab_root = OUT_DIR / "ablation_runs"
    ab_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ab_root / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Core files for comparison across variants
    files_to_copy = [
        OUT_DIR / "report.txt",
        OUT_DIR / "decision_metrics.json",
        OUT_DIR / "node_metrics.json",
        OUT_DIR / "validation_predictions.csv",
        OUT_DIR / "validation_trajectory_arrays.npz",
        OUT_DIR / "split_manifest.csv",
        OUT_DIR / "feature_scalers.npz",
        OUT_DIR / "trajectory_extraction_summary.csv",
        OUT_DIR / "trajectory_node_state_dict.pt",
        OUT_DIR / "decision_mlp_state_dict.pt",
    ]
    for p in files_to_copy:
        if p.exists():
            shutil.copy2(p, run_dir / p.name)

    # Plots and selected cache diagnostics
    if PLOTS_DIR.exists():
        shutil.copytree(PLOTS_DIR, run_dir / "plots", dirs_exist_ok=True)
    cache_subset = run_dir / "cache_subset"
    cache_subset.mkdir(exist_ok=True)
    for name in [
        "trajectory_cache_info.json",
        "trajectory_extraction_summary.csv",
        "trajectory_meta.csv",
    ]:
        p = CACHE_DIR / name
        if p.exists():
            shutil.copy2(p, cache_subset / name)

    # Snapshot the script for reproducibility
    shutil.copy2(Path(__file__), run_dir / "trajectory_neural_ode.py")

    meta = {
        "timestamp": stamp,
        "config": asdict(cfg),
        "notes": "Neural ODE trajectory run archived for ablation comparison",
    }
    if extra:
        meta.update(extra)
    (run_dir / "ablation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir


def summarize_text(
    cfg: NodeConfig,
    extraction_summary: pd.DataFrame,
    decision_metrics: dict[str, Any],
    node_metrics: dict[str, Any],
    decision_hist: dict[str, list[float]],
    node_hist: dict[str, list[float]],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("Two-Stage Decision + Neural ODE Trajectory Model (Hard-Constrained Stop Event)")
    lines.append("=" * 84)
    lines.append("")
    lines.append("Configuration")
    lines.append(json.dumps(asdict(cfg), indent=2))
    lines.append("")
    lines.append("Dataset and Extraction")
    lines.append(f"- Voluntary-only filter: {cfg.use_only_voluntary}")
    lines.append(f"- Trajectory runs extracted: {len(df_train) + len(df_val)}")
    lines.append(f"- Train/Val split: {len(df_train)}/{len(df_val)} ({cfg.train_ratio:.0%}/{1-cfg.train_ratio:.0%})")
    lines.append(f"- Train stop/go: {(df_train['go_decision']==0).sum()} / {(df_train['go_decision']==1).sum()}")
    lines.append(f"- Val stop/go: {(df_val['go_decision']==0).sum()} / {(df_val['go_decision']==1).sum()}")
    lines.append("- Extraction status counts:")
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
    lines.append("Stage 2 (Neural ODE, jerk-driven, multi-task, hard event constraints)")
    lines.append("- State: [d(t), v(t), a(t), h(t)] with learned jerk j_theta = da/dt")
    lines.append("- Target trajectory: signed longitudinal acceleration a_x(t)=ax_g(t)")
    lines.append("- Multi-task supervision: a_x(t), d(t), and v(t) with shared latent state")
    lines.append("- Hard stop constraint in decoder: after stop event, enforce a=0, v=0, and d=constant exactly")
    lines.append("- Additional hard safety projection for stop runs (post-reaction): no positive acceleration and stop-ability bound a<=-v^2/(2d)")
    lines.append("- Stage-2 conditioning uses fixed onset decision class and decision confidence p(go) for the entire trajectory")
    lines.append("- Feature normalization (train-only): StandardScaler fit on training features only, applied to val; outputs remain in physical units")
    lines.append("")
    lines.append("Core equations")
    lines.append("  p(go|x)=sigma(f_theta(x))")
    lines.append("  d_dot=-v, v_dot=a, a_dot=j_theta(t, d, v, a, h, z), h_dot=f_theta(t, d, v, a, h, z)")
    lines.append("  z=[v0, d_th, TTI, a_req, age, decision_class, p_go, sex]")
    lines.append("  Stop freeze (hard): if stop event reached => (a,v,d)=(0,0,d_stop) for all future steps")
    lines.append("")
    for key in ("train_oracle_decision_input", "val_oracle_decision_input", "val_cascaded_decision_input"):
        m = node_metrics[key]
        lines.append(
            f"- {key}: a_MAE={m['traj_mae_mps2']:.3f}, a_RMSE={m['traj_rmse_mps2']:.3f}, dcurve_MAE={m['distance_curve_mae_m']:.3f}m, vcurve_MAE={m.get('speed_curve_mae_mps', float('nan')):.3f}m/s, duration_MAE={m['duration_mae_s']:.3f}s, d_end_MAE={m['terminal_distance_mae_m']:.3f}m, v_end_MAE={m['terminal_speed_mae_mps']:.3f}m/s"
        )
        if "stop_terminal_in_band_rate" in m:
            lines.append(f"  stop_terminal_in_band_rate={m['stop_terminal_in_band_rate']:.3f}")
        if "go_crossed_light_rate" in m:
            lines.append(f"  go_crossed_light_rate={m['go_crossed_light_rate']:.3f}")
        lines.append(f"  mean_delay_frac={m['mean_delay_frac']:.3f}, median_delay_frac={m['median_delay_frac']:.3f}")
    lines.append("")
    lines.append("Training summaries")
    lines.append(f"- Decision final val F1/AUROC in history: {decision_hist['val_f1'][-1]:.3f}/{decision_hist['val_auc'][-1]:.3f}")
    lines.append(
        f"- NODE final val (oracle) loss/aMAE/dEndMAE/stopZoneRate: {node_hist['val_total_oracle'][-1]:.3f}/{node_hist['val_accel_mae_oracle'][-1]:.3f}/{node_hist['val_d_end_mae_oracle'][-1]:.3f}/{node_hist['val_stop_zone_rate_oracle'][-1]:.3f}"
    )
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    set_seed(CFG.seed)
    patch_base_for_extraction(CFG)
    PROGRESS_LOG.write_text("", encoding="utf-8")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log_progress(f"Run started | device={DEVICE} | torch={torch.__version__} | accelerator={gpu_name}")

    # Reuse extraction + decision modeling infrastructure from the mixture script.
    log_progress("Stage 0: extracting/loading trajectory dataset cache...")
    df, traj_arrays, extraction_summary = base.prepare_modeling_dataframe(base.CFG)
    if len(df) < 20:
        raise RuntimeError("Too few trajectory samples after extraction.")
    log_progress(f"Stage 0 done | extracted_runs={len(df)}")

    base_X = np.column_stack(
        [
            df["speed_at_yellow_kmh"].to_numpy(dtype=float),
            df["distance_threshold_m"].to_numpy(dtype=float),
            df["tti_s"].to_numpy(dtype=float),
            df["a_req_mps2"].to_numpy(dtype=float),
            df["driver_age"].to_numpy(dtype=float),
        ]
    ).astype(np.float32)
    go = df["go_decision"].to_numpy(dtype=int)
    sex_female = df["sex_female"].to_numpy(dtype=float)

    idx_all = np.arange(len(df))
    idx_train, idx_val = train_test_split(
        idx_all,
        train_size=CFG.train_ratio,
        random_state=CFG.seed,
        stratify=go,
        shuffle=True,
    )
    idx_train = np.sort(idx_train)
    idx_val = np.sort(idx_val)

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    X_train, X_val = base_X[idx_train], base_X[idx_val]
    y_train, y_val = go[idx_train], go[idx_val]

    # Stage 1 decision MLP (reuse)
    log_progress("Stage 1: training decision MLP...")
    base.CFG.decision_epochs = CFG.decision_epochs
    base.CFG.decision_batch_size = CFG.decision_batch_size
    base.CFG.decision_lr = CFG.decision_lr
    base.CFG.hidden_dim_decision = CFG.hidden_dim_decision
    base.CFG.weight_decay = CFG.weight_decay
    decision_model, decision_bundle, decision_hist, decision_scaler = base.fit_decision_model(X_train, y_train, X_val, y_val, base.CFG)
    dec_metrics = decision_bundle["metrics"]
    dec_out = decision_bundle["outputs"]
    y_train_prob = dec_out["train_prob"]
    y_train_pred = dec_out["train_pred"]
    y_val_prob = dec_out["val_prob"]
    y_val_pred = dec_out["val_pred"]
    log_progress(
        "Stage 1 done | val_acc={:.3f} | val_f1={:.3f} | val_auroc={:.3f}".format(
            dec_metrics["val"]["accuracy"], dec_metrics["val"]["f1"], dec_metrics["val"]["auroc"]
        )
    )

    # Stage 2 trajectory dataset (same extracted trajectories)
    a_train = traj_arrays["ax_target_mps2"][idx_train]
    a_val = traj_arrays["ax_target_mps2"][idx_val]
    d_traj_train = traj_arrays["d_target_m"][idx_train]
    d_traj_val = traj_arrays["d_target_m"][idx_val]
    v_traj_train = (traj_arrays["v_target_kmh"][idx_train] / 3.6).astype(np.float32)
    v_traj_val = (traj_arrays["v_target_kmh"][idx_val] / 3.6).astype(np.float32)
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

    # Decision is made once at yellow onset and kept fixed over the whole trajectory.
    # We pass both decision class and decision confidence p(go) to stage 2.
    X2_train = make_stage2_features(X_train, y_train_pred.astype(float), y_train_prob.astype(float), sex_female[idx_train])
    X2_val_oracle = make_stage2_features(X_val, y_val.astype(float), y_val_prob.astype(float), sex_female[idx_val])
    X2_val_cascade = make_stage2_features(X_val, y_val_pred.astype(float), y_val_prob.astype(float), sex_female[idx_val])

    log_progress("Stage 2: training autoregressive transformer trajectory model...")
    tfm_model, tfm_bundle, tfm_hist, token_scaler = train_transformer_model(
        X2_train, X2_val_oracle, X2_val_cascade,
        a_train, a_val,
        d_traj_train, d_traj_val,
        v_traj_train, v_traj_val,
        dur_train, dur_val,
        v0_train, v0_val,
        d0_train, d0_val,
        y_train, y_val,
        d_end_train, d_end_val,
        v_end_train, v_end_val,
        CFG,
    )
    tfm_metrics = tfm_bundle["metrics"]
    tfm_out = tfm_bundle["outputs"]
    same_cond_mask = (y_val_pred.astype(int) == y_val.astype(int))
    if np.any(same_cond_mask):
        a_same_diff = float(np.mean(np.abs(
            tfm_out["val_eval_oracle"]["a_pred"][same_cond_mask] - tfm_out["val_eval_cascade"]["a_pred"][same_cond_mask]
        )))
        d_same_diff = float(np.mean(np.abs(
            tfm_out["val_eval_oracle"]["d_traj_pred"][same_cond_mask] - tfm_out["val_eval_cascade"]["d_traj_pred"][same_cond_mask]
        )))
        log_progress(
            "Oracle-vs-cascade consistency on same decision label subset (n={}) | mean |Delta a|={:.6f} | mean |Delta d|={:.6f}".format(
                int(np.sum(same_cond_mask)), a_same_diff, d_same_diff
            )
        )
    log_progress(
        "Stage 2 done | val_oracle aMAE={:.3f} | dEndMAE={:.3f} | stopBand={:.3f} | goCross={:.3f}".format(
            tfm_metrics["val_oracle_decision_input"]["traj_mae_mps2"],
            tfm_metrics["val_oracle_decision_input"]["terminal_distance_mae_m"],
            tfm_metrics["val_oracle_decision_input"].get("stop_terminal_in_band_rate", float("nan")),
            tfm_metrics["val_oracle_decision_input"].get("go_crossed_light_rate", float("nan")),
        )
    )

    base.save_json({"config": asdict(CFG), "decision_metrics": dec_metrics}, OUT_DIR / "decision_metrics.json")
    base.save_json({"config": asdict(CFG), "transformer_metrics": tfm_metrics}, OUT_DIR / "transformer_metrics.json")

    pd.DataFrame({"source_file": df["source_file"], "split": np.where(np.isin(np.arange(len(df)), idx_train), "train", "val"), "go_decision": go}).to_csv(
        OUT_DIR / "split_manifest.csv", index=False
    )
    val_pred = df_val[
        ["source_file", "go_decision", "stop_decision", "speed_at_yellow_kmh", "distance_threshold_m", "tti_s", "a_req_mps2", "driver_age", "traj_duration_s", "traj_d_end_obs_m", "traj_v_end_obs_mps"]
    ].copy()
    val_pred["decision_go_prob"] = y_val_prob
    val_pred["decision_go_pred"] = y_val_pred
    val_pred["ar_terminal_d_pred_oracle_m"] = tfm_out["val_eval_oracle"]["d_pred"]
    val_pred["ar_terminal_d_pred_cascade_m"] = tfm_out["val_eval_cascade"]["d_pred"]
    val_pred["ar_terminal_v_pred_oracle_mps"] = tfm_out["val_eval_oracle"]["v_pred"]
    val_pred["ar_terminal_v_pred_cascade_mps"] = tfm_out["val_eval_cascade"]["v_pred"]
    val_pred["ar_mean_gate_oracle"] = np.mean(tfm_out["val_eval_oracle"]["gate_seq"], axis=1)
    val_pred["ar_jerk_hit_rate_oracle"] = np.mean(tfm_out["val_eval_oracle"]["jerk_hit_seq"], axis=1)
    val_pred.to_csv(OUT_DIR / "validation_predictions.csv", index=False)

    np.savez_compressed(
        OUT_DIR / "validation_trajectory_arrays.npz",
        a_true_val=a_val,
        d_true_val=d_traj_val,
        v_true_val=v_traj_val,
        a_pred_oracle=tfm_out["val_eval_oracle"]["a_pred"],
        a_pred_cascade=tfm_out["val_eval_cascade"]["a_pred"],
        d_pred_traj_oracle=tfm_out["val_eval_oracle"]["d_traj_pred"],
        d_pred_traj_cascade=tfm_out["val_eval_cascade"]["d_traj_pred"],
        v_pred_traj_oracle=tfm_out["val_eval_oracle"]["v_traj_pred"],
        v_pred_traj_cascade=tfm_out["val_eval_cascade"]["v_traj_pred"],
        gate_seq_oracle=tfm_out["val_eval_oracle"]["gate_seq"],
        jerk_hit_seq_oracle=tfm_out["val_eval_oracle"]["jerk_hit_seq"],
        actuator_hit_seq_oracle=tfm_out["val_eval_oracle"]["actuator_hit_seq"],
    )
    torch.save(decision_model.state_dict(), OUT_DIR / "decision_mlp_state_dict.pt")
    torch.save(tfm_model.state_dict(), OUT_DIR / "trajectory_transformer_state_dict.pt")
    np.savez_compressed(
        OUT_DIR / "feature_scalers.npz",
        decision_mean=decision_scaler.mean_, decision_scale=decision_scaler.scale_,
        token_mean=token_scaler.mean_, token_scale=token_scaler.scale_,
    )

    base.plot_decision_results(y_val, y_val_prob, y_val_pred, decision_hist, PLOTS_DIR)
    log_progress("Generating plots...")
    plot_info = plot_transformer_results(
        df_val,
        a_val,
        d_traj_val,
        tfm_out["val_eval_oracle"],
        tfm_out["val_eval_cascade"],
        tfm_hist,
        CFG,
        PLOTS_DIR,
        val_go_pred=y_val_pred,
        val_go_prob=y_val_prob,
    )
    if plot_info.get("sample_indices"):
        sample_manifest = df_val.iloc[plot_info["sample_indices"]].copy()
        sample_manifest.insert(0, "val_row_index", plot_info["sample_indices"])
        sample_manifest.to_csv(OUT_DIR / "transformer_examples_sample_manifest.csv", index=False)

    report = summarize_transformer_text(CFG, extraction_summary, dec_metrics, tfm_metrics, decision_hist, tfm_hist, df_train, df_val)
    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    (OUT_DIR / "README_run.txt").write_text(
        "Run with project venv:\n  .\\.venv\\Scripts\\python.exe trajectory_transformer_ar.py\n", encoding="utf-8"
    )
    archive_dir = archive_transformer_ablation_run(
        CFG,
        extra={
            "stage1_val_metrics": dec_metrics.get("val", {}),
            "stage2_val_oracle_metrics": tfm_metrics.get("val_oracle_decision_input", {}),
            "stage2_val_cascade_metrics": tfm_metrics.get("val_cascaded_decision_input", {}),
            "plot_info": plot_info,
        },
    )
    log_progress(f"Ablation archive saved to: {archive_dir}")
    print(report)
    print(f"Ablation archive written to: {archive_dir}")
    print(f"\nOutputs written to: {OUT_DIR}")
    log_progress("Run finished successfully.")


if __name__ == "__main__":
    main()
