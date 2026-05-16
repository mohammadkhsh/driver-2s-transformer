from __future__ import annotations

import json
import argparse
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
class TransformerConfig:
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

    weight_decay: float = 1e-5
    max_abs_jerk_mps3: float = 15.0
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


CFG = TransformerConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate the two-stage stop/go + autoregressive Transformer trajectory model."
    )
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--train-ratio", type=float, default=CFG.train_ratio)
    parser.add_argument("--decision-epochs", type=int, default=CFG.decision_epochs)
    parser.add_argument("--epochs", type=int, default=CFG.ar_epochs, help="Stage 2 Transformer epochs.")
    parser.add_argument("--out-dir", type=str, default="results/trajectory_transformer_ar")
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


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


def patch_base_for_extraction(cfg: TransformerConfig) -> None:
    # Reuse the shared extraction and caching utilities, but write into the Transformer results folder.
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
    # Keep valid early stops around +6 m, which occurred in the recorded data.
    base.STOP_DETECT_MAX_M = 6.5


def batch_iter(indices: np.ndarray, batch_size: int, shuffle: bool = True) -> list[np.ndarray]:
    idx = indices.copy()
    if shuffle:
        np.random.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]


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
    def __init__(self, feature_dim: int, cfg: TransformerConfig) -> None:
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
    cfg: TransformerConfig,
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
    cfg: TransformerConfig,
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
    cfg: TransformerConfig,
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


def summarize_transformer_text(
    cfg: TransformerConfig,
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


def archive_transformer_ablation_run(cfg: TransformerConfig, extra: dict[str, Any] | None = None) -> Path:
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


def main() -> None:
    global OUT_DIR, CACHE_DIR, PLOTS_DIR, PROGRESS_LOG, DEVICE
    args = parse_args()
    CFG.seed = int(args.seed)
    CFG.train_ratio = float(args.train_ratio)
    CFG.decision_epochs = int(args.decision_epochs)
    CFG.ar_epochs = int(args.epochs)
    OUT_DIR = ROOT / args.out_dir
    CACHE_DIR = OUT_DIR / "cache"
    PLOTS_DIR = OUT_DIR / "plots"
    PROGRESS_LOG = OUT_DIR / "run_progress.log"
    if args.force_cpu:
        DEVICE = torch.device("cpu")
    ensure_dirs()
    set_seed(CFG.seed)
    patch_base_for_extraction(CFG)
    PROGRESS_LOG.write_text("", encoding="utf-8")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log_progress(f"Run started | device={DEVICE} | torch={torch.__version__} | accelerator={gpu_name}")

    # Reuse shared extraction and decision modeling utilities.
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
