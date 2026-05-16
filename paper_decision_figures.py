from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
from matplotlib.tri import LinearTriInterpolator, Triangulation
from sklearn.neighbors import KNeighborsRegressor

SPEED_MIN_KMH = 30.0
SPEED_MAX_KMH = 80.0
DIST_MIN_M = 0.0
DIST_MAX_M = 85.0

DECISION_CMAP = LinearSegmentedColormap.from_list(
    "decision_light_red_to_green",
    ["#e26d84", "#ffe8c2", "#63c783"],
)
A_REQ_CMAP = LinearSegmentedColormap.from_list(
    "a_req_purple_gradient",
    ["#ede9fe", "#c4b5fd", "#8b5cf6", "#5b21b6"],
)
GO_POINT_COLOR = "#0f8a3a"
STOP_POINT_COLOR = "#e11d48"
GO_EDGE_COLOR = "#f8fafc"
STOP_EDGE_COLOR = "#f8fafc"


def _configure_ieee_pdf_fonts() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.rm"] = "STIXGeneral"


def load_plot_data(plot_data_dir: Path) -> dict[str, np.ndarray]:
    npz_path = plot_data_dir / "surface_arrays.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing file: {npz_path}. Run analysis_decision.py first.")

    arrays = dict(np.load(npz_path))

    boundary_summary_path = plot_data_dir / "decision_boundary_summary.csv"
    surface_long_path = plot_data_dir / "surface_grid_long.csv"
    samples_path = plot_data_dir / "observed_decision_samples.csv"
    metadata_path = plot_data_dir / "plot_metadata.json"

    if not boundary_summary_path.exists():
        raise FileNotFoundError(f"Missing file: {boundary_summary_path}")
    if not surface_long_path.exists():
        raise FileNotFoundError(f"Missing file: {surface_long_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing file: {samples_path}")

    arrays["boundary_summary"] = pd.read_csv(boundary_summary_path)
    arrays["surface_grid_long"] = pd.read_csv(surface_long_path)
    arrays["samples"] = pd.read_csv(samples_path)
    arrays["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return arrays


def save_figure(fig: plt.Figure, out_base: Path, dpi: int = 320) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _valid_speed_mask(speed_grid: np.ndarray) -> np.ndarray:
    mask = (speed_grid >= SPEED_MIN_KMH) & (speed_grid <= SPEED_MAX_KMH)
    if not np.any(mask):
        return np.ones_like(speed_grid, dtype=bool)
    return mask


def plot_figure_3d_decision_surface(
    data: dict[str, np.ndarray],
    out_base: Path,
    sample_size: int,
    seed: int,
) -> None:
    speed_grid = data["speed_grid"]
    tti_grid = data["tti_grid"]
    speed_mask = _valid_speed_mask(speed_grid)
    speed_plot = speed_grid[speed_mask]

    p_mean = data["p_go_mean"][:, speed_mask]
    p_low = data["p_go_ci_low"][:, speed_mask]
    p_high = data["p_go_ci_high"][:, speed_mask]
    boundary = data["boundary_summary"].copy()
    boundary = boundary[(boundary["speed_kmh"] >= SPEED_MIN_KMH) & (boundary["speed_kmh"] <= SPEED_MAX_KMH)].copy()
    samples = data["samples"].copy()
    samples = samples[(samples["initial_speed_kmh"] >= SPEED_MIN_KMH) & (samples["initial_speed_kmh"] <= SPEED_MAX_KMH)].copy()

    sp_mesh, tti_mesh = np.meshgrid(speed_plot, tti_grid)

    rng = np.random.default_rng(seed)
    if len(samples) > sample_size:
        samples = samples.iloc[rng.choice(len(samples), size=sample_size, replace=False)].copy()

    fig = plt.figure(figsize=(12.5, 9.0))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        sp_mesh,
        tti_mesh,
        p_mean,
        cmap="viridis",
        alpha=0.95,
        linewidth=0.0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    ax.plot_surface(sp_mesh, tti_mesh, p_low, color="#3182bd", alpha=0.15, linewidth=0)
    ax.plot_surface(sp_mesh, tti_mesh, p_high, color="#de2d26", alpha=0.12, linewidth=0)

    valid = boundary["valid_draw_fraction"] >= 0.5
    b = boundary[valid]
    if not b.empty:
        ax.plot(
            b["speed_kmh"],
            b["tti_p50_median"],
            np.full(len(b), 0.5),
            color="black",
            linewidth=2.2,
        )
        ax.plot(
            b["speed_kmh"],
            b["tti_p50_ci_low"],
            np.full(len(b), 0.5),
            color="black",
            linewidth=1.2,
            linestyle="--",
        )
        ax.plot(
            b["speed_kmh"],
            b["tti_p50_ci_high"],
            np.full(len(b), 0.5),
            color="black",
            linewidth=1.2,
            linestyle="--",
        )

    ax.scatter(
        samples["initial_speed_kmh"],
        samples["tti_s"],
        samples["go_decision"],
        c=samples["go_decision"],
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        s=28,
        alpha=0.65,
        depthshade=False,
    )

    ax.contour(
        sp_mesh,
        tti_mesh,
        p_mean,
        levels=[0.5],
        zdir="z",
        offset=0.0,
        colors="black",
        linewidths=1.2,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.66, pad=0.08)
    cbar.set_label("Posterior mean P(go)")

    ax.set_xlabel("Speed at yellow onset (km/h)", labelpad=10)
    ax.set_ylabel("TTI at yellow onset (s)", labelpad=10)
    ax.set_zlabel("P(go)", labelpad=10)
    ax.set_xlim(SPEED_MIN_KMH, SPEED_MAX_KMH)
    ax.set_zlim(0.0, 1.0)
    ax.view_init(elev=29, azim=-129)
    ax.set_title("3D Bayesian Decision Manifold with 95% Uncertainty Envelopes")

    handles = [
        Line2D([0], [0], color="black", lw=2.2, label="Median P(go)=0.5 boundary"),
        Line2D([0], [0], color="black", lw=1.2, ls="--", label="95% boundary CI"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d53f8c", markersize=8, label="Observed runs"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True)

    save_figure(fig, out_base)


def plot_figure_2d_uncertainty_map(data: dict[str, np.ndarray], out_base: Path) -> None:
    speed_grid = data["speed_grid"]
    tti_grid = data["tti_grid"]
    speed_mask = _valid_speed_mask(speed_grid)
    speed_plot = speed_grid[speed_mask]

    p_mean = data["p_go_mean"][:, speed_mask]
    p_low = data["p_go_ci_low"][:, speed_mask]
    p_high = data["p_go_ci_high"][:, speed_mask]
    p_sd = data["p_go_post_sd"][:, speed_mask]
    p_dis = data["p_go_disagreement_var"][:, speed_mask]
    hesitation_mask = data["hesitation_mask"][:, speed_mask]
    boundary = data["boundary_summary"].copy()
    boundary = boundary[(boundary["speed_kmh"] >= SPEED_MIN_KMH) & (boundary["speed_kmh"] <= SPEED_MAX_KMH)].copy()
    samples = data["samples"].copy()
    samples = samples[(samples["initial_speed_kmh"] >= SPEED_MIN_KMH) & (samples["initial_speed_kmh"] <= SPEED_MAX_KMH)].copy()

    uncertainty_index, sd_excess, dis_excess = _orthogonal_uncertainty_index(
        p_mean=p_mean,
        post_sd=p_sd,
        disagreement=p_dis,
    )

    boundary_valid = boundary["valid_draw_fraction"] >= 0.5
    b = boundary[boundary_valid]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), constrained_layout=True)

    ax1 = axes[0]
    lv = np.linspace(0.0, 1.0, 21)
    c1 = ax1.contourf(speed_plot, tti_grid, p_mean, levels=lv, cmap="RdYlGn", alpha=0.97)
    fig.colorbar(c1, ax=ax1, label="Posterior mean P(go)")

    con = ax1.contour(speed_plot, tti_grid, p_mean, levels=[0.25, 0.5, 0.75], colors="black", linewidths=[1.0, 1.8, 1.0])
    ax1.clabel(con, inline=True, fontsize=8, fmt="%.2f")
    ax1.contourf(speed_plot, tti_grid, hesitation_mask, levels=[0.5, 1.5], colors="none", hatches=["////"])

    if not b.empty:
        ax1.fill_between(
            b["speed_kmh"],
            b["tti_p50_ci_low"],
            b["tti_p50_ci_high"],
            color="white",
            alpha=0.30,
            label="95% credible band of P(go)=0.5 boundary",
        )
        ax1.plot(b["speed_kmh"], b["tti_p50_median"], color="black", linewidth=2.2, label="Median P(go)=0.5 boundary")

    ax1.scatter(
        samples["initial_speed_kmh"],
        samples["tti_s"],
        c=samples["go_decision"],
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        s=16,
        alpha=0.35,
        edgecolors="none",
    )
    ax1.set_title("Decision Surface + Hesitation Region")
    ax1.set_xlabel("Speed at yellow onset (km/h)")
    ax1.set_ylabel("TTI at yellow onset (s)")
    ax1.set_xlim(SPEED_MIN_KMH, SPEED_MAX_KMH)
    ax1.legend(loc="lower right", frameon=True)

    ax2 = axes[1]
    scale = float(np.nanpercentile(np.abs(uncertainty_index), 95))
    if (not np.isfinite(scale)) or scale <= 0:
        scale = 1.0
    uncertainty_plot = uncertainty_index / scale
    levels_unc = np.linspace(-1.4, 1.4, 25)
    c2 = ax2.contourf(speed_plot, tti_grid, uncertainty_plot, levels=levels_unc, cmap="RdBu_r", extend="both")
    fig.colorbar(c2, ax=ax2, label="Excess uncertainty index (orthogonal to P(go))")

    dis_pos = dis_excess[dis_excess > 0]
    if dis_pos.size >= 10:
        dis_levels = np.quantile(dis_pos, [0.70, 0.85, 0.95])
        dcon = ax2.contour(speed_plot, tti_grid, dis_excess, levels=dis_levels, colors="cyan", linewidths=[1.0, 1.3, 1.8])
        ax2.clabel(dcon, inline=True, fontsize=8, fmt="ExD=%.2f")
    ax2.contour(speed_plot, tti_grid, p_mean, levels=[0.5], colors="white", linewidths=1.2)

    max_idx = int(np.nanargmax(uncertainty_plot))
    r, c = np.unravel_index(max_idx, uncertainty_index.shape)
    ax2.scatter([speed_plot[c]], [tti_grid[r]], color="white", edgecolor="black", s=90, marker="*", zorder=5)
    ax2.text(speed_plot[c], tti_grid[r], " max", color="white", fontsize=9, va="center")

    ax2.set_title("Orthogonal Uncertainty Atlas (Beyond Decision-Boundary Ambiguity)")
    ax2.set_xlabel("Speed at yellow onset (km/h)")
    ax2.set_ylabel("TTI at yellow onset (s)")
    ax2.set_xlim(SPEED_MIN_KMH, SPEED_MAX_KMH)

    save_figure(fig, out_base)


def _levels_in_range(candidates: list[float], lo: float, hi: float) -> list[float]:
    vals = [v for v in candidates if lo < v < hi]
    return vals


def _nice_step(span: float, target_ticks: int) -> float:
    span = float(span)
    if (not np.isfinite(span)) or span <= 0:
        return 1.0
    target_ticks = int(max(4, target_ticks))
    raw_step = span / float(target_ticks - 1)
    mag = 10.0 ** np.floor(np.log10(raw_step))
    candidates = np.array([1.0, 2.0, 2.5, 5.0, 10.0], dtype=float) * mag
    counts = span / candidates + 1.0
    idx = int(np.argmin(np.abs(counts - target_ticks)))
    return float(candidates[idx])


def _dense_ticks_with_endpoints(lo: float, hi: float, target_ticks: int) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)):
        return np.array([], dtype=float)
    if hi <= lo:
        return np.array([lo], dtype=float)

    step = _nice_step(hi - lo, target_ticks=target_ticks)
    start = np.ceil(lo / step) * step
    end = np.floor(hi / step) * step
    ticks = np.arange(start, end + 0.5 * step, step, dtype=float)
    ticks = ticks[(ticks > lo + 1e-9) & (ticks < hi - 1e-9)]
    ticks = np.concatenate(([lo], ticks, [hi]))
    ticks = np.unique(np.round(ticks, 6))
    return ticks


def _ticks_with_fixed_step(lo: float, hi: float, step: float) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)
    step = float(step)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (not np.isfinite(step)) or step <= 0:
        return np.array([], dtype=float)
    if hi <= lo:
        return np.array([lo], dtype=float)

    start = np.ceil(lo / step) * step
    ticks = np.arange(start, hi + 0.5 * step, step, dtype=float)
    ticks = ticks[(ticks > lo + 1e-9) & (ticks < hi - 1e-9)]
    ticks = np.concatenate(([lo], ticks, [hi]))
    ticks = np.unique(np.round(ticks, 6))
    return ticks


def _tick_decimals(ticks: np.ndarray) -> int:
    t = np.asarray(ticks, dtype=float)
    if t.size == 0:
        return 0
    if np.all(np.isclose(t, np.round(t), atol=1e-6)):
        return 0
    return 1


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    lo = np.nanmin(v)
    hi = np.nanmax(v)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo)


def _poly_residual(y: np.ndarray, x: np.ndarray, degree: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    out = np.zeros_like(y, dtype=float)
    if valid.sum() < max(6, degree + 2):
        center = float(np.nanmean(y[valid])) if valid.any() else 0.0
        out[:] = y - center
        return out

    xv = x[valid]
    yv = y[valid]
    deg = int(max(1, min(degree, len(np.unique(xv)) - 1)))
    try:
        coeff = np.polyfit(xv, yv, deg)
        pred = np.polyval(coeff, x)
        out = y - pred
    except Exception:
        center = float(np.nanmean(yv))
        out = y - center
    out[~np.isfinite(out)] = 0.0
    return out


def _orthogonal_uncertainty_index(
    p_mean: np.ndarray,
    post_sd: np.ndarray,
    disagreement: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(p_mean, dtype=float).reshape(-1)
    sd = np.asarray(post_sd, dtype=float).reshape(-1)
    dis = np.asarray(disagreement, dtype=float).reshape(-1)

    ambiguity = 1.0 - np.abs(2.0 * p - 1.0)
    sd_norm = _minmax_scale(sd)
    dis_norm = _minmax_scale(dis)

    sd_excess = _poly_residual(sd_norm, ambiguity, degree=3)
    dis_excess = _poly_residual(dis_norm, ambiguity, degree=3)

    eps = 1e-9
    sd_scale = float(np.nanpercentile(np.abs(sd_excess), 95))
    dis_scale = float(np.nanpercentile(np.abs(dis_excess), 95))
    if (not np.isfinite(sd_scale)) or sd_scale < eps:
        sd_scale = 1.0
    if (not np.isfinite(dis_scale)) or dis_scale < eps:
        dis_scale = 1.0

    sd_excess_scaled = sd_excess / sd_scale
    dis_excess_scaled = dis_excess / dis_scale
    orth = 0.5 * sd_excess_scaled + 0.5 * dis_excess_scaled

    # Remove any remaining smooth dependency on P(go) itself.
    orth = _poly_residual(orth, p, degree=3)

    shape = np.asarray(p_mean).shape
    return orth.reshape(shape), sd_excess_scaled.reshape(shape), dis_excess_scaled.reshape(shape)


def _triangular_field_with_slope_extrapolation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    df = pd.DataFrame({"x": x, "y": y, "z": z}).dropna().copy()
    if df.empty:
        xx, yy = np.meshgrid(x_grid, y_grid)
        return np.zeros_like(xx, dtype=float)

    # Aggregate duplicate coordinates to avoid invalid triangulations.
    df = df.groupby(["x", "y"], as_index=False)["z"].mean()
    x_u = df["x"].to_numpy(dtype=float)
    y_u = df["y"].to_numpy(dtype=float)
    z_u = df["z"].to_numpy(dtype=float)

    xx, yy = np.meshgrid(x_grid, y_grid)
    if len(z_u) < 3:
        z_grid = np.full_like(xx, float(np.nanmedian(z_u)), dtype=float)
        return z_grid

    try:
        tri = Triangulation(x_u, y_u)
        interp = LinearTriInterpolator(tri, z_u)
        z_raw = interp(xx, yy)
    except Exception:
        # Robust fallback if triangulation is degenerate.
        nn = int(max(3, min(20, len(z_u))))
        knn = KNeighborsRegressor(n_neighbors=nn, weights="distance")
        knn.fit(np.column_stack([x_u, y_u]), z_u)
        z_pred = knn.predict(np.column_stack([xx.ravel(), yy.ravel()]))
        return z_pred.reshape(len(y_grid), len(x_grid))

    if np.ma.isMaskedArray(z_raw):
        z_grid = z_raw.filled(np.nan).astype(float)
    else:
        z_grid = np.asarray(z_raw, dtype=float)

    fallback = float(np.nanmedian(z_u)) if np.isfinite(np.nanmedian(z_u)) else 0.0
    row_idx = np.arange(len(y_grid))
    col_idx = np.arange(len(x_grid))

    for j in range(z_grid.shape[1]):
        col = z_grid[:, j].copy()
        valid = np.isfinite(col)
        n_valid = int(valid.sum())

        if n_valid == 0:
            z_grid[:, j] = col
            continue

        if n_valid == 1:
            col[:] = col[valid][0]
            z_grid[:, j] = col
            continue

        idx = row_idx[valid]
        val = col[valid]

        col_full = np.interp(row_idx, idx, val)

        i0, i1 = int(idx[0]), int(idx[1])
        y0, y1 = float(y_grid[i0]), float(y_grid[i1])
        slope_low = (float(val[1]) - float(val[0])) / (y1 - y0) if y1 != y0 else 0.0
        if i0 > 0:
            col_full[:i0] = float(val[0]) + slope_low * (y_grid[:i0] - y0)

        iu0, iu1 = int(idx[-2]), int(idx[-1])
        yu0, yu1 = float(y_grid[iu0]), float(y_grid[iu1])
        slope_high = (float(val[-1]) - float(val[-2])) / (yu1 - yu0) if yu1 != yu0 else 0.0
        if iu1 < len(y_grid) - 1:
            col_full[iu1 + 1 :] = float(val[-1]) + slope_high * (y_grid[iu1 + 1 :] - yu1)

        z_grid[:, j] = col_full

    for i in range(z_grid.shape[0]):
        row = z_grid[i, :].copy()
        valid = np.isfinite(row)
        n_valid = int(valid.sum())

        if n_valid == 0:
            z_grid[i, :] = fallback
            continue

        if n_valid == 1:
            z_grid[i, :] = row[valid][0]
            continue

        idx = col_idx[valid]
        val = row[valid]
        row_full = np.interp(col_idx, idx, val)

        j0 = int(idx[0])
        if j0 > 0:
            row_full[:j0] = float(val[0])

        ju1 = int(idx[-1])
        if ju1 < len(x_grid) - 1:
            row_full[ju1 + 1 :] = float(val[-1])

        z_grid[i, :] = row_full

    z_grid[~np.isfinite(z_grid)] = fallback
    return z_grid


def plot_figure_distance_speed_advanced(data: dict[str, np.ndarray], out_base: Path) -> None:
    grid = data["surface_grid_long"].copy()
    samples = data["samples"].copy()

    grid = grid.dropna(subset=["distance_threshold_m", "speed_kmh"]).copy()
    samples = samples.dropna(subset=["distance_threshold_m", "initial_speed_kmh"]).copy()

    if not samples.empty:
        x_min = float(samples["distance_threshold_m"].min())
        x_max = float(samples["distance_threshold_m"].max())
        y_min = float(samples["initial_speed_kmh"].min())
        y_max = float(samples["initial_speed_kmh"].max())
    else:
        x_min = float(grid["distance_threshold_m"].min())
        x_max = float(grid["distance_threshold_m"].max())
        y_min = float(grid["speed_kmh"].min())
        y_max = float(grid["speed_kmh"].max())

    x_plot_min, x_plot_max = x_min, x_max
    y_plot_min, y_plot_max = y_min, y_max

    grid = grid[
        (grid["distance_threshold_m"] >= x_plot_min)
        & (grid["distance_threshold_m"] <= x_plot_max)
        & (grid["speed_kmh"] >= y_plot_min)
        & (grid["speed_kmh"] <= y_plot_max)
    ].copy()
    samples = samples[
        (samples["distance_threshold_m"] >= x_plot_min)
        & (samples["distance_threshold_m"] <= x_plot_max)
        & (samples["initial_speed_kmh"] >= y_plot_min)
        & (samples["initial_speed_kmh"] <= y_plot_max)
    ].copy()

    x = grid["distance_threshold_m"].to_numpy()
    y = grid["speed_kmh"].to_numpy()
    p = np.clip(grid["p_go_mean"].to_numpy(), 1e-9, 1 - 1e-9)

    xg = np.linspace(x_plot_min, x_plot_max, 240)
    yg = np.linspace(y_plot_min, y_plot_max, 220)
    xx, yy = np.meshgrid(xg, yg)

    p_mesh = _triangular_field_with_slope_extrapolation(x, y, p, xg, yg)
    p_mesh = np.clip(p_mesh, 0.0, 1.0)
    hesitation_mesh = ((p_mesh >= 0.4) & (p_mesh <= 0.6)).astype(float)

    fs_axis_tick = 18
    fs_axis_label = 23
    fs_colorbar_tick = 21
    fs_colorbar_label = 19
    fs_legend = 14

    fig, ax = plt.subplots(1, 1, figsize=(11.0, 7.6), constrained_layout=True)
    c = ax.contourf(xx, yy, p_mesh, levels=np.linspace(0, 1, 21), cmap=DECISION_CMAP, alpha=0.98)
    cbar = fig.colorbar(c, ax=ax, label="Posterior mean P(go)")
    cbar.ax.tick_params(labelsize=fs_colorbar_tick, length=9.0, width=1.7, direction="out")
    cbar.ax.yaxis.label.set_size(fs_colorbar_label)

    pcon = ax.contour(
        xx,
        yy,
        p_mesh,
        levels=[0.25, 0.5, 0.75],
        colors=["#fdba74", "#f97316", "#fdba74"],
        linewidths=[1.2, 3.0, 1.2],
        linestyles=["--", "-", "--"],
    )
    ax.clabel(pcon, fmt="%.2f", fontsize=8, inline=True)

    ax.contourf(xx, yy, hesitation_mesh, levels=[0.5, 1.5], colors="none", hatches=["////"])

    # Smooth and consistent analytical iso-a_req curves in distance-speed plane.
    a_req_axis = (yy / 3.6) ** 2 / np.clip(2.0 * xx, 1e-6, None)
    a_req_levels = _levels_in_range(
        [0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
        float(np.nanmin(a_req_axis)),
        float(np.nanmax(a_req_axis)),
    )
    areq_legend_label = r"Iso-$a_{\mathrm{req}}$ lines"
    if a_req_levels:
        acon = ax.contour(
            xx,
            yy,
            a_req_axis,
            levels=a_req_levels,
            cmap=A_REQ_CMAP,
            linewidths=1.5,
            linestyles="-",
        )
        ax.clabel(acon, fmt="a_req=%.2f", fontsize=8, inline=True)
        areq_legend_label = r"Iso-$a_{\mathrm{req}}$ lines"

    sample_age = pd.to_numeric(samples.get("driver_age"), errors="coerce")
    if sample_age.isna().all():
        sample_age = pd.Series(np.full(len(samples), 40.0), index=samples.index)
    else:
        sample_age = sample_age.fillna(sample_age.median())
    age_norm = np.clip((sample_age - 18.0) / (70.0 - 18.0), 0.0, 1.0)
    sample_sizes = 28.0 + 160.0 * age_norm

    go_mask = samples["go_decision"] == 1
    stop_mask = ~go_mask
    ax.scatter(
        samples.loc[go_mask, "distance_threshold_m"],
        samples.loc[go_mask, "initial_speed_kmh"],
        color=GO_POINT_COLOR,
        s=sample_sizes[go_mask],
        alpha=0.78,
        edgecolors=GO_EDGE_COLOR,
        linewidths=0.35,
        zorder=3,
    )
    ax.scatter(
        samples.loc[stop_mask, "distance_threshold_m"],
        samples.loc[stop_mask, "initial_speed_kmh"],
        color=STOP_POINT_COLOR,
        s=sample_sizes[stop_mask],
        alpha=0.78,
        edgecolors=STOP_EDGE_COLOR,
        linewidths=0.35,
        zorder=3,
    )
    ax.set_xlabel("Distance to traffic light at yellow onset (m)")
    ax.set_ylabel("Speed at yellow onset (km/h)")
    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(y_plot_min, y_plot_max)

    x_span = float(x_plot_max - x_plot_min)
    y_span = float(y_plot_max - y_plot_min)
    x_step = 5.0 if x_span <= 90.0 else 10.0
    y_step = 2.5 if y_span <= 50.0 else 5.0
    xticks = _ticks_with_fixed_step(x_plot_min, x_plot_max, step=x_step)
    yticks = _ticks_with_fixed_step(y_plot_min, y_plot_max, step=y_step)
    if xticks.size > 0:
        ax.set_xticks(xticks)
        x_decimals = _tick_decimals(xticks)
        ax.set_xticklabels([f"{v:.{x_decimals}f}" for v in xticks])
    if yticks.size > 0:
        ax.set_yticks(yticks)
        y_decimals = _tick_decimals(yticks)
        y_tick_labels = [f"{v:.{y_decimals}f}" for v in yticks]
        if len(y_tick_labels) >= 2:
            y_tick_labels[0] = ""
            y_tick_labels[-1] = ""
        ax.set_yticklabels(y_tick_labels)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # Draw clear external dash-like tick indicators on the main axes,
    # matching the visual convention used by the posterior-mean colorbar.
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fs_axis_tick,
        length=15.0,
        width=2.3,
        direction="out",
        bottom=True,
        left=True,
        top=False,
        right=False,
        pad=8,
        color="#111827",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=0.0,
        width=0.0,
        direction="out",
        bottom=True,
        left=True,
        top=False,
        right=False,
        color="#111827",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("#111827")
    ax.grid(which="major", color="#6b7280", alpha=0.34, linewidth=0.78)
    ax.grid(which="minor", color="#9ca3af", alpha=0.22, linewidth=0.5)

    ax.set_title("Decision Surface with Hesitation and Smooth Required-Deceleration Isolines", fontsize=18)
    ax.set_xlabel("Distance to traffic light at yellow onset (m)", fontsize=fs_axis_label)
    ax.set_ylabel("Speed at yellow onset (km/h)", fontsize=fs_axis_label)

    legend_handles = [
        Line2D([0], [0], color="#f97316", lw=3.0, label="P(go)=0.5 decision boundary"),
        Line2D([0], [0], color="#7c3aed", lw=1.7, label=areq_legend_label),
        Patch(facecolor="white", edgecolor="#374151", hatch="////", label="Hesitation region (0.4 <= P(go) <= 0.6)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GO_POINT_COLOR, markeredgecolor=GO_EDGE_COLOR, markersize=6, lw=0, label="Go sample"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STOP_POINT_COLOR, markeredgecolor=STOP_EDGE_COLOR, markersize=6, lw=0, label="Stop sample"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=fs_legend)

    save_figure(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create IEEE-style figures from decision-model plot data exported by analysis_decision.py"
    )
    parser.add_argument("--results-dir", type=str, default="results/decision_model")
    parser.add_argument("--out-dir", type=str, default="results/decision_model/paper_figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sns.set_theme(style="white")
    _configure_ieee_pdf_fonts()
    plt.rcParams["axes.titleweight"] = "bold"

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    plot_data_dir = results_dir / "plot_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_plot_data(plot_data_dir)

    plot_figure_distance_speed_advanced(
        data=data,
        out_base=out_dir / "figure_distance_speed_decision_enhanced",
    )

    notes = [
        "Generated figures:",
        "  figure_distance_speed_decision_enhanced.(png/pdf)",
        "Figure contents:",
        "  posterior decision surface P(go) + hesitation hatching + smooth iso-a_req lines",
        "  sample marker size normalized by driver age in [18, 70]",
        "",
        "Data source:",
        f"  {plot_data_dir}",
    ]
    (out_dir / "figure_manifest.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
