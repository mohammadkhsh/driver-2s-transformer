from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess


def _configure_ieee_pdf_fonts() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.rm"] = "STIXGeneral"


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _select_time_column(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "t_wall" in df.columns:
        t = _to_num(df["t_wall"]).to_numpy(dtype=float)
        if np.isfinite(t).sum() >= 3:
            return t, "t_wall"
    if "ticks_utc" in df.columns:
        t = _to_num(df["ticks_utc"]).to_numpy(dtype=float)
        if np.isfinite(t).sum() >= 3:
            return t, "ticks_utc"
    return np.arange(len(df), dtype=float), "row_index"


def compute_run_metrics(log_path: Path) -> dict[str, Any]:
    try:
        raw = pd.read_csv(log_path, dtype=str, keep_default_na=False)
    except Exception:
        return {
            "source_file": log_path.name,
            "hr_min_bpm": np.nan,
            "hr_max_bpm": np.nan,
            "hr_delta_bpm": np.nan,
            "hr_mean_bpm": np.nan,
            "hr_delta_pct": np.nan,
            "max_decel_abs_mps2": np.nan,
            "max_decel_signed_mps2": np.nan,
            "p95_decel_abs_mps2": np.nan,
            "decel_method": "read_error",
        }

    t_all, _ = _select_time_column(raw)
    hr = _to_num(raw.get("hr_bpm", pd.Series(index=raw.index, dtype=float))).to_numpy(dtype=float)
    valid_hr = np.isfinite(hr) & (hr >= 35.0) & (hr <= 220.0) & np.isfinite(t_all)
    if valid_hr.any():
        hr_vals = hr[valid_hr]
        hr_min = float(np.min(hr_vals))
        hr_max = float(np.max(hr_vals))
        hr_delta = float(hr_max - hr_min)
        hr_mean = float(np.mean(hr_vals))
        hr_delta_pct = float(100.0 * hr_delta / hr_min) if hr_min > 0 else np.nan
    else:
        hr_min = np.nan
        hr_max = np.nan
        hr_delta = np.nan
        hr_mean = np.nan
        hr_delta_pct = np.nan

    fix = raw[raw["type"].astype(str).str.lower() == "fix"].copy()
    if fix.empty:
        return {
            "source_file": log_path.name,
            "hr_min_bpm": hr_min,
            "hr_max_bpm": hr_max,
            "hr_delta_bpm": hr_delta,
            "hr_mean_bpm": hr_mean,
            "hr_delta_pct": hr_delta_pct,
            "max_decel_abs_mps2": np.nan,
            "max_decel_signed_mps2": np.nan,
            "p95_decel_abs_mps2": np.nan,
            "decel_method": "missing_fix",
        }
    if "ax_g" not in fix.columns:
        return {
            "source_file": log_path.name,
            "hr_min_bpm": hr_min,
            "hr_max_bpm": hr_max,
            "hr_delta_bpm": hr_delta,
            "hr_mean_bpm": hr_mean,
            "hr_delta_pct": hr_delta_pct,
            "max_decel_abs_mps2": np.nan,
            "max_decel_signed_mps2": np.nan,
            "p95_decel_abs_mps2": np.nan,
            "decel_method": "missing_ax_g",
        }

    # User-requested deceleration source: direct acceleration channel in logs.
    # Values are treated as m/s^2 from ax_g.
    ax = _to_num(fix["ax_g"]).to_numpy(dtype=float)
    valid_ax = np.isfinite(ax) & (np.abs(ax) <= 12.0)
    ax = ax[valid_ax]
    if ax.size == 0:
        return {
            "source_file": log_path.name,
            "hr_min_bpm": hr_min,
            "hr_max_bpm": hr_max,
            "hr_delta_bpm": hr_delta,
            "hr_mean_bpm": hr_mean,
            "max_decel_abs_mps2": np.nan,
            "max_decel_signed_mps2": np.nan,
            "p95_decel_abs_mps2": np.nan,
            "decel_method": "no_valid_ax_g",
        }

    decel_abs = np.maximum(-ax, 0.0)
    max_decel_abs = float(np.max(decel_abs))
    p95_decel_abs = float(np.quantile(decel_abs, 0.95))

    return {
        "source_file": log_path.name,
        "hr_min_bpm": hr_min,
        "hr_max_bpm": hr_max,
        "hr_delta_bpm": hr_delta,
        "hr_mean_bpm": hr_mean,
        "hr_delta_pct": hr_delta_pct,
        "max_decel_abs_mps2": max_decel_abs,
        "max_decel_signed_mps2": -max_decel_abs,
        "p95_decel_abs_mps2": p95_decel_abs,
        "decel_method": "ax_g_direct",
    }


def _bootstrap_slope(x: np.ndarray, y: np.ndarray, n_boot: int = 4000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan
    slopes = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        if np.allclose(np.std(xb), 0):
            slopes[i] = np.nan
        else:
            slopes[i] = np.polyfit(xb, yb, deg=1)[0]
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(slopes)), float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))


def _bayes_adjacent_boundaries_from_df(
    df: pd.DataFrame,
    x_col: str = "a_abs",
    rating_col: str = "comfort_rating",
    rating_order: list[int] | None = None,
) -> list[float]:
    if rating_order is None:
        rating_order = [5, 4, 3, 2, 1]

    tmp = df[[x_col, rating_col]].copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[rating_col] = pd.to_numeric(tmp[rating_col], errors="coerce")
    tmp = tmp[tmp[x_col].notna() & tmp[rating_col].notna()].copy()
    tmp["rating_int"] = np.round(tmp[rating_col]).astype(int)
    tmp = tmp[tmp["rating_int"].isin(rating_order)].copy()
    if tmp.empty:
        return []

    stats = tmp.groupby("rating_int", observed=True)[x_col].agg(["count", "mean", "std"]).reindex(rating_order)
    stats = stats.dropna(subset=["count", "mean", "std"])
    if len(stats) < 2:
        return []
    n_tot = float(stats["count"].sum())
    stats["prior"] = stats["count"] / max(n_tot, 1.0)

    bounds: list[float] = []
    for i, j in zip(rating_order[:-1], rating_order[1:]):
        if i not in stats.index or j not in stats.index:
            continue
        mu_i = float(stats.loc[i, "mean"])
        mu_j = float(stats.loc[j, "mean"])
        sd_i = float(stats.loc[i, "std"])
        sd_j = float(stats.loc[j, "std"])
        pi_i = float(stats.loc[i, "prior"])
        pi_j = float(stats.loc[j, "prior"])
        if not (np.isfinite(sd_i) and np.isfinite(sd_j) and sd_i > 0 and sd_j > 0 and pi_i > 0 and pi_j > 0):
            continue

        vi = sd_i * sd_i
        vj = sd_j * sd_j
        A = (1.0 / (2.0 * vi)) - (1.0 / (2.0 * vj))
        B = (-mu_i / vi) + (mu_j / vj)
        C = (mu_i * mu_i / (2.0 * vi)) - (mu_j * mu_j / (2.0 * vj)) - np.log((pi_i * sd_j) / (pi_j * sd_i))

        roots: list[float] = []
        if abs(A) < 1e-12:
            if abs(B) > 1e-12:
                roots = [float(-C / B)]
        else:
            disc = B * B - 4.0 * A * C
            if disc >= 0.0 and np.isfinite(disc):
                sdisc = float(np.sqrt(disc))
                roots = [float((-B + sdisc) / (2.0 * A)), float((-B - sdisc) / (2.0 * A))]

        lo, hi = sorted([mu_i, mu_j])
        in_between = [r for r in roots if np.isfinite(r) and lo <= r <= hi]
        if len(in_between) > 0:
            mid = 0.5 * (mu_i + mu_j)
            pick = sorted(in_between, key=lambda r: abs(r - mid))[0]
        elif len(roots) > 0:
            mid = 0.5 * (mu_i + mu_j)
            pick = sorted(roots, key=lambda r: abs(r - mid))[0]
        else:
            pick = 0.5 * (mu_i + mu_j)
        bounds.append(float(pick))

    return sorted(bounds)


def build_figure(df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    sns.set_theme(style="whitegrid")
    _configure_ieee_pdf_fonts()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

    fs_axis = 21.0
    fs_tick = 16.6
    fs_title = 18.0
    fs_legend = 11.8

    fig = plt.figure(figsize=(14.8, 11.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])

    d = df.copy()
    d["a_abs"] = _to_num(d["max_decel_abs_mps2"])
    d["hr_delta_pct"] = _to_num(d["hr_delta_pct"])
    d["driver_age"] = _to_num(d["driver_age"])
    d["group"] = np.where(d["forced_stop_instruction"] == 1, "anyway_stop", "voluntary_stop")
    d = d[d["a_abs"].notna() & d["comfort_rating"].notna() & d["hr_delta_pct"].notna()].copy()

    # User-requested absolute-deceleration axis, low->high from left to right.
    a_lo = 0.0
    a_hi = 11.0
    d = d[(d["a_abs"] >= a_lo) & (d["a_abs"] <= a_hi)].copy()
    age_order = ["18-25", "25-32", "32+"]
    age_palette = {"18-25": "#2a9d8f", "25-32": "#e9c46a", "32+": "#7b2cbf"}
    d["age_group"] = pd.cut(
        d["driver_age"],
        bins=[18.0, 25.0, 32.0, np.inf],
        right=False,
        labels=age_order,
    ).astype(str)
    d.loc[~d["age_group"].isin(age_order), "age_group"] = np.nan
    xticks = np.arange(a_lo, a_hi + 0.001, 0.5)
    palette = {"anyway_stop": "#c1121f", "voluntary_stop": "#1d3557"}
    # Shared rating color convention across subplots (light R5 -> dark R1).
    rating_line_colors = {
        5: "#fde0dd",  # light pink
        4: "#fcbba1",
        3: "#fc9272",
        2: "#ef3b2c",
        1: "#cb181d",  # sharp red
    }

    # Bayes boundaries between adjacent comfort classes on |a_min| axis.
    comfort_bounds = _bayes_adjacent_boundaries_from_df(d, x_col="a_abs", rating_col="comfort_rating", rating_order=[5, 4, 3, 2, 1])
    x_edges = [a_lo] + [float(b) for b in comfort_bounds if np.isfinite(b)] + [a_hi]
    x_edges = sorted([float(v) for v in x_edges])
    # enforce strictly increasing edges
    uniq_edges = [x_edges[0]]
    for v in x_edges[1:]:
        if v > uniq_edges[-1] + 1e-6:
            uniq_edges.append(v)
    x_edges = uniq_edges
    region_rating = [5, 4, 3, 2, 1]
    if len(x_edges) >= 2:
        n_seg = min(len(region_rating), len(x_edges) - 1)
        for k in range(n_seg):
            xl, xr = x_edges[k], x_edges[k + 1]
            rr = region_rating[k]
            ax_a.axvspan(
                xl,
                xr,
                facecolor=rating_line_colors[rr],
                alpha=0.18,
                edgecolor="none",
                zorder=0.15,
            )
        # draw boundary markers
        for b in x_edges[1:-1]:
            ax_a.axvline(b, color="#7f1d1d", linestyle="--", linewidth=1.45, alpha=0.75, zorder=0.2)

    # (a) Advanced bin-wise effect summary using 0.5 m/s^2 bins.
    edges_a = np.arange(a_lo, a_hi + 0.5, 0.5)
    d["a_bin_05"] = pd.cut(d["a_abs"], bins=edges_a, include_lowest=True, right=False)
    bin_stats = (
        d.groupby(["group", "a_bin_05"], observed=True)
        .agg(
            n=("comfort_rating", "size"),
            comfort_mean=("comfort_rating", "mean"),
            comfort_sd=("comfort_rating", "std"),
        )
        .reset_index()
    )
    if not bin_stats.empty:
        bin_stats["comfort_sd"] = bin_stats["comfort_sd"].fillna(0.0)
        bin_stats["x"] = bin_stats["a_bin_05"].apply(lambda iv: 0.5 * (iv.left + iv.right)).astype(float)
        bin_stats["comfort_se"] = bin_stats["comfort_sd"] / np.sqrt(bin_stats["n"].clip(lower=1))
        bin_stats["comfort_ci95"] = 1.96 * bin_stats["comfort_se"]
        err_palette = {"anyway_stop": "#7f1d1d", "voluntary_stop": "#1e3a8a"}
        for grp in ["anyway_stop", "voluntary_stop"]:
            s = bin_stats[bin_stats["group"] == grp].sort_values("x")
            if s.empty:
                continue
            x = s["x"].to_numpy()
            y = s["comfort_mean"].to_numpy()
            yerr = s["comfort_ci95"].to_numpy()
            nvals = s["n"].to_numpy(dtype=float)
            size = 52.0 + 18.0 * np.sqrt(nvals)
            ax_a.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=err_palette[grp],
                elinewidth=1.35,
                capsize=3.6,
                alpha=0.32,
            )
            ax_a.scatter(x, y, s=size, c=palette[grp], alpha=0.94, edgecolors="white", linewidths=0.9, label=grp)
            ax_a.plot(x, y, color=palette[grp], linewidth=3.2, alpha=0.98)

    rng = np.random.default_rng(42)
    for grp in ["voluntary_stop", "anyway_stop"]:
        s = d[d["group"] == grp]
        if s.empty:
            continue
        yj = s["comfort_rating"].to_numpy(dtype=float) + rng.uniform(-0.035, 0.035, size=len(s))
        ax_a.scatter(
            s["a_abs"].to_numpy(dtype=float),
            yj,
            s=10,
            c=palette[grp],
            alpha=0.12,
            edgecolors="none",
        )

    ax_a.set_title("(a) Comfort vs Peak Braking Magnitude (0.5 m/s$^2$ bins)", fontsize=fs_title, fontweight="bold")
    ax_a.set_xlabel(r"Peak braking magnitude $|a_{\min}|$ (m/s$^2$)", fontsize=fs_axis)
    ax_a.set_ylabel("Comfort rating (bin mean ±95% CI)", fontsize=fs_axis)
    ax_a.set_xlim(a_lo, a_hi)
    ax_a.set_xticks(xticks)
    ax_a.tick_params(axis="x", rotation=55, labelsize=fs_tick)
    ax_a.tick_params(axis="y", labelsize=fs_tick)
    ax_a.set_ylim(0.8, 5.2)
    ax_a.legend(loc="lower left", frameon=True, fontsize=fs_legend)
    ax_a.text(
        0.02,
        0.18,
        "Markers: bin mean (size~n), bars: 95% CI, faint dots: raw runs\nShaded regions: Bayes boundaries (R5->R1)",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.0,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#9aa0a6"},
    )
    rating_stats = (
        d.assign(comfort_rating_int=np.round(d["comfort_rating"]).astype(int))
        .groupby("comfort_rating_int", observed=True)
        .agg(mean_abs=("a_abs", "mean"), sd_abs=("a_abs", "std"))
        .reindex([1, 2, 3, 4, 5])
    )
    stats_lines = []
    for r in [1, 2, 3, 4, 5]:
        m = rating_stats.loc[r, "mean_abs"]
        s = rating_stats.loc[r, "sd_abs"]
        if np.isfinite(m):
            stats_lines.append(f"R{r}: {m:.2f} ± {s:.2f}")
    ax_a.text(
        0.985,
        0.98,
        "Mean±SD of |a_min| by rating\n" + "\n".join(stats_lines),
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=10.9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#9aa0a6"},
    )
    ax_a.grid(alpha=0.25)

    # (b) Rating matrix for all stop-valid runs + side summary of |a_min| by rating.
    bm = d.copy()
    edges_b = np.arange(a_lo, a_hi + 1.5, 1.5)
    bm["a_bin_15"] = pd.cut(bm["a_abs"], bins=edges_b, include_lowest=True, right=False)
    bm["comfort_rating_int"] = np.round(bm["comfort_rating"]).astype(int)
    rating_order = [5, 4, 3, 2, 1]  # requested: R5 on left, R1 on right
    heat = pd.crosstab(bm["a_bin_15"], bm["comfort_rating_int"])
    heat = heat.reindex(columns=rating_order, fill_value=0)
    heat = heat[heat.sum(axis=1) > 0].copy()
    # requested: acceleration increases toward top
    heat = heat.iloc[::-1]
    heat_pct = 100.0 * heat.div(heat.sum(axis=1).replace(0, np.nan), axis=0)
    heat_main = heat_pct.rename(columns={5: "R5", 4: "R4", 3: "R3", 2: "R2", 1: "R1"})
    annot_pct = heat_main.apply(lambda col: col.map(lambda v: f"{v:.0f}%" if np.isfinite(v) else ""))
    heat_blank = np.zeros_like(heat_main.to_numpy(dtype=float))
    sns.heatmap(
        heat_blank,
        cmap=sns.color_palette(["#ffffff"], as_cmap=True),
        vmin=0.0,
        vmax=1.0,
        annot=False,
        fmt="",
        cbar=False,
        xticklabels=heat_main.columns,
        yticklabels=heat_main.index,
        linewidths=0.35,
        linecolor="white",
        ax=ax_b,
    )
    # Very light column-wise background by rating (same hue as panel-c rating lines).
    age_bar_palette = {"18-25": "#8fd7ce", "25-32": "#f3e6ad", "32+": "#c8b6e8"}
    rating_to_col = {5: 0, 4: 1, 3: 2, 2: 3, 1: 4}
    n_rows_b = len(heat_main.index)
    for rv in rating_order:
        j = rating_to_col[rv]
        ax_b.add_patch(
            Rectangle(
                (float(j), 0.0),
                1.0,
                float(n_rows_b),
                facecolor=rating_line_colors[rv],
                edgecolor="none",
                alpha=0.24,
                zorder=1.6,
            )
        )

    # One horizontal bar per age-group per decel row: center=mean rating, width=±1SD.
    # Keep clear margins from row separators for readability.
    y_offsets = [0.50, 0.66, 0.82]
    for i, iv in enumerate(heat_main.index):
        row_sub = bm[(bm["a_bin_15"] == iv) & bm["age_group"].isin(age_order)].copy()
        for ag, y_off in zip(age_order, y_offsets):
            sub = row_sub[row_sub["age_group"] == ag]
            if len(sub) < 1:
                continue
            rvals = pd.to_numeric(sub["comfort_rating"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(rvals) < 1:
                continue
            mu = float(np.mean(rvals))
            sd = float(np.std(rvals, ddof=1)) if len(rvals) > 1 else 0.0
            # Keep bars visible even when SD=0 (e.g., repeated identical ratings).
            half_width = max(1.0 * sd, 0.10)
            lo = max(1.0, mu - half_width)
            hi = min(5.0, mu + half_width)
            # map rating scale (R5 left ... R1 right): x = 5.5 - rating
            x_left = 5.5 - hi
            x_right = 5.5 - lo
            x_center = 5.5 - mu
            y = float(i) + y_off
            bar_h = 0.09
            ax_b.add_patch(
                Rectangle(
                    (x_left, y - 0.5 * bar_h),
                    max(0.0, x_right - x_left),
                    bar_h,
                    facecolor=age_bar_palette[ag],
                    edgecolor=age_palette[ag],
                    linewidth=0.85,
                    alpha=0.98,
                    zorder=3.9,
                )
            )
            ax_b.add_patch(
                Rectangle(
                    (x_center - 0.013, y - 0.082),
                    0.026,
                    0.164,
                    facecolor=age_palette[ag],
                    edgecolor=age_palette[ag],
                    linewidth=0.0,
                    alpha=1.0,
                    zorder=4.1,
                )
            )
    # Main percentage text in upper 40% of each cell.
    for i, iv in enumerate(heat_main.index):
        for j, col in enumerate(heat_main.columns):
            v = float(heat_main.loc[iv, col])
            if not np.isfinite(v):
                continue
            ax_b.text(
                float(j) + 0.5,
                float(i) + 0.30,
                f"{v:.0f}%",
                ha="center",
                va="center",
                fontsize=16.2,
                fontweight="bold",
                color="#111827",
                zorder=5.0,
            )
    ylabels = []
    for iv in heat_main.index:
        p = heat_main.loc[iv].to_numpy(dtype=float) / 100.0
        p = p[np.isfinite(p)]
        ai = float(np.sum(p**2)) if len(p) else np.nan
        ylabels.append(f"[{iv.left:.1f},{iv.right:.1f})\nA={ai:.2f}")
    # Row separators for readability.
    for y_sep in np.arange(1, n_rows_b, 1):
        ax_b.hlines(y_sep, xmin=0.0, xmax=float(len(rating_order)), colors="black", linewidth=0.85, alpha=0.72, zorder=4.6)
    ax_b.set_yticklabels(ylabels, rotation=0, fontsize=14.0)
    ax_b.set_title("(b) Stop-Valid Rating Matrix by |a_min| Bin (1.5 m/s$^2$)", fontsize=fs_title, fontweight="bold")
    ax_b.set_xlabel("Comfort rating class", fontsize=fs_axis)
    ax_b.set_ylabel(r"Peak braking magnitude $|a_{\min}|$ bin (m/s$^2$)", fontsize=fs_axis)
    ax_b.tick_params(axis="x", labelsize=fs_tick)
    age_handles_b = [
        Patch(facecolor=age_bar_palette[a], edgecolor=age_palette[a], linewidth=0.85, label=f"{a}: mean±1SD")
        for a in age_order
    ]
    ax_b.legend(
        handles=age_handles_b,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.995),
        frameon=True,
        fontsize=8.9,
        title="Per-row age bars (comfort mean±1SD)",
        title_fontsize=9.0,
    )
    # (c) Participant-wise robust lines + ensemble beam from line-parameter distribution.
    cdat = d.copy()
    cdat["hr_delta_bpm"] = _to_num(cdat["hr_delta_bpm"])
    cdat["participant_group"] = _to_num(cdat["participant_group"])
    cdat = cdat[
        cdat["hr_delta_bpm"].notna() & cdat["participant_group"].notna() & cdat["age_group"].isin(age_order)
    ].copy()
    x_grid = np.linspace(a_lo, a_hi, 240)
    fit_rows: list[dict[str, float]] = []
    for pid, gp in cdat.groupby("participant_group", observed=True):
        gp = gp.dropna(subset=["a_abs", "hr_delta_bpm"])
        if len(gp) < 3:
            continue
        xv = gp["a_abs"].to_numpy(dtype=float)
        yv = gp["hr_delta_bpm"].to_numpy(dtype=float)
        if (np.nanmax(xv) - np.nanmin(xv)) < 1.0:
            continue
        slope, intercept, _, _ = stats.theilslopes(yv, xv)
        age_mode = gp["age_group"].dropna().mode()
        if age_mode.empty:
            continue
        age_grp = str(age_mode.iloc[0])
        if np.isfinite(slope) and np.isfinite(intercept):
            fit_rows.append(
                {
                    "participant_group": float(pid),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "n": float(len(gp)),
                    "age_group": age_grp,
                }
            )

    fit_df = pd.DataFrame(fit_rows)
    n_lines = int(len(fit_df))
    if n_lines > 0:
        for ag in age_order:
            sub = fit_df[fit_df["age_group"] == ag]
            for r in sub.itertuples(index=False):
                y_pred = r.intercept + r.slope * x_grid
                ax_c.plot(x_grid, y_pred, color=age_palette[ag], linewidth=1.25, alpha=0.42)

        b = fit_df[["intercept", "slope"]].to_numpy(dtype=float)
        b_mean = b.mean(axis=0)
        if n_lines >= 2:
            b_cov = np.cov(b, rowvar=False, ddof=1)
        else:
            b_cov = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)

        design = np.column_stack([np.ones_like(x_grid), x_grid])
        mean_line = design @ b_mean
        var_line = np.einsum("ij,jk,ik->i", design, b_cov, design)
        sd_line = np.sqrt(np.clip(var_line, 0.0, None))

        ax_c.fill_between(
            x_grid,
            mean_line - 1.0 * sd_line,
            mean_line + 1.0 * sd_line,
            color="#1d3557",
            alpha=0.18,
            label="Ensemble beam (mean +/- 1 SD)",
        )
        ax_c.fill_between(
            x_grid,
            mean_line - 0.5 * sd_line,
            mean_line + 0.5 * sd_line,
            color="#1d3557",
            alpha=0.30,
            label="Ensemble beam (mean +/- 0.5 SD)",
        )
        ax_c.plot(
            x_grid,
            mean_line,
            color="#0b132b",
            linewidth=4.6,
            alpha=0.98,
            label="Average participant line",
        )
    # Dashed reference lines: mean |a_min| for ratings R1..R5 from panel (a) stats.
    rating_line_vals: list[tuple[int, float]] = []
    for r in [1, 2, 3, 4, 5]:
        if r in rating_stats.index:
            x0 = float(rating_stats.loc[r, "mean_abs"])
            if np.isfinite(x0) and a_lo <= x0 <= a_hi:
                rating_line_vals.append((r, x0))
                ax_c.axvline(
                    x0,
                    color=rating_line_colors[r],
                    linestyle="--",
                    linewidth=2.15,
                    alpha=0.92,
                )

    ax_c.set_title("(c) Within-Participant $\\Delta HR$ vs Peak Braking Magnitude", fontsize=fs_title, fontweight="bold")
    ax_c.set_xlabel(r"Peak braking magnitude $|a_{\min}|$ (m/s$^2$)", fontsize=fs_axis)
    ax_c.set_ylabel(r"Run-level absolute HR increase $\Delta HR = HR_{\max}-HR_{\min}$ (bpm)", fontsize=fs_axis)
    ax_c.set_xlim(a_lo, a_hi)
    ax_c.set_xticks(xticks)
    ax_c.tick_params(axis="x", rotation=55, labelsize=fs_tick)
    ax_c.tick_params(axis="y", labelsize=fs_tick)
    y_cap = float(np.nanquantile(cdat["hr_delta_bpm"].to_numpy(dtype=float), 0.995)) if len(cdat) else 30.0
    y_top = max(20.0, np.ceil((y_cap + 3.0) / 5.0) * 5.0)
    ax_c.set_ylim(0.0, y_top)
    y_label_base = max(0.65, 0.065 * y_top)
    for k, (r, x0) in enumerate(sorted(rating_line_vals, key=lambda t: t[1])):
        ax_c.text(
            x0 + 0.06,
            y_label_base + 0.10 * (k % 2),
            f"R{r}= {x0:.2f} m/s^2",
            rotation=90,
            ha="left",
            va="bottom",
            fontsize=9.1,
            color=rating_line_colors[r],
            alpha=0.98,
            clip_on=True,
        )
    p10 = float((d["hr_delta_pct"] >= 10.0).mean() * 100.0) if len(d) else np.nan
    p20 = float((d["hr_delta_pct"] >= 20.0).mean() * 100.0) if len(d) else np.nan
    p30 = float((d["hr_delta_pct"] >= 30.0).mean() * 100.0) if len(d) else np.nan
    slope_med = float(fit_df["slope"].median()) if n_lines > 0 else np.nan
    slope_q1 = float(fit_df["slope"].quantile(0.25)) if n_lines > 0 else np.nan
    slope_q3 = float(fit_df["slope"].quantile(0.75)) if n_lines > 0 else np.nan
    slope_mean = float(fit_df["slope"].mean()) if n_lines > 0 else np.nan
    slope_sd = float(fit_df["slope"].std(ddof=1)) if n_lines > 1 else np.nan
    mean_hr = float(cdat["hr_delta_bpm"].mean()) if len(cdat) else np.nan
    sd_hr = float(cdat["hr_delta_bpm"].std(ddof=1)) if len(cdat) > 1 else np.nan
    if n_lines > 0:
        ax_c.text(
            0.98,
            0.02,
            f"Average slope = {slope_mean:.2f} bpm/(m/s$^2$)",
            transform=ax_c.transAxes,
            fontsize=10.9,
            color="#0b132b",
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#9aa0a6"},
        )
    ax_c.text(
        0.02,
        0.98,
        (
            f"Slope (bpm per m/s^2): mean={slope_mean:.2f}±{slope_sd:.2f}, median={slope_med:.2f}, IQR=[{slope_q1:.2f},{slope_q3:.2f}]\n"
            f"HR delta mean={mean_hr:.2f}±{sd_hr:.2f} bpm\n"
            f"HR elevation share: >=10% {p10:.1f}%, >=20% {p20:.1f}%, >=30% {p30:.1f}%"
        ),
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#9aa0a6"},
    )
    age_slope_labels = {}
    for ag in age_order:
        sub = fit_df.loc[fit_df["age_group"] == ag, "slope"] if not fit_df.empty else pd.Series(dtype=float)
        if len(sub) > 0:
            s_mu = float(sub.mean())
            s_sd = float(sub.std(ddof=1)) if len(sub) > 1 else 0.0
            age_slope_labels[ag] = f"{ag} slope={s_mu:.2f}±{s_sd:.2f}"
        else:
            age_slope_labels[ag] = f"{ag} slope=NA"

    guide_handles = [
        Patch(facecolor="#1d3557", alpha=0.30, edgecolor="none", label="Ensemble beam (mean +/- 0.5 SD)"),
        Patch(facecolor="#1d3557", alpha=0.18, edgecolor="none", label="Ensemble beam (mean +/- 1 SD)"),
        Line2D([0], [0], color="#0b132b", linewidth=4.6, label="Average participant line"),
        Line2D([0], [0], color=age_palette["18-25"], linewidth=1.8, label=age_slope_labels["18-25"]),
        Line2D([0], [0], color=age_palette["25-32"], linewidth=1.8, label=age_slope_labels["25-32"]),
        Line2D([0], [0], color=age_palette["32+"], linewidth=1.8, label=age_slope_labels["32+"]),
    ]
    ax_c.legend(handles=guide_handles, loc="upper left", bbox_to_anchor=(0.01, 0.90), frameon=True, fontsize=8.9, ncol=2)
    ax_c.grid(alpha=0.25)

    fig.savefig(out_png, dpi=320)
    fig.savefig(out_pdf, dpi=320)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Heart-rate and comfort analysis for stop-valid runs.")
    parser.add_argument("--log-dir", type=str, default="data/logs")
    parser.add_argument("--runs-csv", type=str, default="data/clean_data/all_runs_deduplicated.csv")
    parser.add_argument("--out-dir", type=str, default="results/hr_comfort")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(args.runs_csv)
    runs["comfort_rating"] = _to_num(runs["comfort_rating"])
    runs["stop_decision"] = _to_num(runs["stop_decision"]).fillna(0).astype(int)
    runs["forced_stop_instruction"] = _to_num(runs["forced_stop_instruction"]).fillna(0).astype(int)
    runs["experiment_kind"] = runs["experiment_kind"].astype(str).str.lower()

    # Comfort is valid only for stop cases.
    valid = runs[(runs["stop_decision"] == 1) & runs["comfort_rating"].notna()].copy()

    metric_rows = []
    for sf in valid["source_file"].astype(str).unique():
        metric_rows.append(compute_run_metrics(log_dir / sf))
    metrics = pd.DataFrame(metric_rows)

    merged = valid.merge(metrics, on="source_file", how="left")
    merged = merged[
        merged["max_decel_abs_mps2"].notna()
        & merged["comfort_rating"].notna()
        & merged["hr_delta_bpm"].notna()
        & merged["hr_delta_pct"].notna()
        & np.isfinite(merged["hr_delta_pct"])
    ].copy()
    pid = pd.to_numeric(merged["participant_group"], errors="coerce")
    merged = merged[pid.notna()].copy()
    merged["participant_group"] = pid[pid.notna()].astype(int).values
    merged["group"] = np.where(merged["forced_stop_instruction"] == 1, "anyway_stop", "voluntary_stop")
    merged["hr_delta_pct_within_pp"] = (
        merged["hr_delta_pct"] - merged.groupby("participant_group")["hr_delta_pct"].transform("mean")
    )
    merged["decel_within_pp"] = (
        merged["max_decel_abs_mps2"] - merged.groupby("participant_group")["max_decel_abs_mps2"].transform("mean")
    )
    merged.to_csv(out_dir / "run_level_hr_comfort_metrics.csv", index=False)

    aw = merged[merged["group"] == "anyway_stop"].copy()
    vw = merged[merged["group"] == "voluntary_stop"].copy()

    def _spearman(df: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
        if len(df) < 5:
            return np.nan, np.nan
        rho, p = stats.spearmanr(df[x].to_numpy(dtype=float), df[y].to_numpy(dtype=float))
        return float(rho), float(p)

    rho_c_all, p_c_all = _spearman(merged, "max_decel_abs_mps2", "comfort_rating")
    rho_c_aw, p_c_aw = _spearman(aw, "max_decel_abs_mps2", "comfort_rating")
    rho_hr_aw, p_hr_aw = _spearman(aw, "max_decel_abs_mps2", "hr_delta_pct")
    rho_hr_within, p_hr_within = _spearman(merged, "decel_within_pp", "hr_delta_pct_within_pp")

    slope_c_aw, slope_c_aw_l, slope_c_aw_h = _bootstrap_slope(
        aw["max_decel_abs_mps2"].to_numpy(dtype=float),
        aw["comfort_rating"].to_numpy(dtype=float),
        n_boot=4000,
        seed=77,
    )
    slope_hr_aw, slope_hr_aw_l, slope_hr_aw_h = _bootstrap_slope(
        aw["max_decel_abs_mps2"].to_numpy(dtype=float),
        aw["hr_delta_pct"].to_numpy(dtype=float),
        n_boot=4000,
        seed=91,
    )
    fe_model = smf.ols("hr_delta_pct ~ max_decel_abs_mps2 + C(participant_group)", data=merged).fit(cov_type="HC3")
    fe_slope = float(fe_model.params.get("max_decel_abs_mps2", np.nan))
    fe_ci = fe_model.conf_int().loc["max_decel_abs_mps2"]
    fe_slope_lo = float(fe_ci.iloc[0])
    fe_slope_hi = float(fe_ci.iloc[1])
    fe_p = float(fe_model.pvalues.get("max_decel_abs_mps2", np.nan))

    max_decel_vals = aw["max_decel_abs_mps2"].to_numpy(dtype=float)
    q80 = float(np.quantile(max_decel_vals, 0.8)) if len(max_decel_vals) > 0 else np.nan
    top20 = aw[aw["max_decel_abs_mps2"] >= q80].copy() if np.isfinite(q80) else aw.iloc[0:0].copy()

    summary_lines = [
        "Heart-Rate and Comfort Analysis (Stop-Valid Runs)",
        "===============================================",
        f"Total stop-valid runs with HR + decel + rating: {len(merged)}",
        f"  anyway_stop runs: {len(aw)}",
        f"  voluntary_stop runs: {len(vw)}",
        "",
        "Run-level max braking deceleration |a_min| (m/s^2):",
        f"  anyway_stop mean = {aw['max_decel_abs_mps2'].mean():.3f}, SD = {aw['max_decel_abs_mps2'].std(ddof=1):.3f}",
        f"  voluntary_stop mean = {vw['max_decel_abs_mps2'].mean():.3f}, SD = {vw['max_decel_abs_mps2'].std(ddof=1):.3f}",
        f"  anyway_stop top-20% threshold (Q80) = {q80:.3f}",
    ]
    if len(top20) > 0:
        summary_lines.extend(
            [
                f"  top-20% anyway_stop mean |a_min| = {top20['max_decel_abs_mps2'].mean():.3f}",
                f"  top-20% anyway_stop range |a_min| = [{top20['max_decel_abs_mps2'].min():.3f}, {top20['max_decel_abs_mps2'].max():.3f}]",
            ]
        )

    summary_lines.extend(
        [
            "",
            "Heart-rate range DeltaHR = HR_max - HR_min (bpm):",
            f"  anyway_stop mean = {aw['hr_delta_bpm'].mean():.3f}, SD = {aw['hr_delta_bpm'].std(ddof=1):.3f}",
            f"  voluntary_stop mean = {vw['hr_delta_bpm'].mean():.3f}, SD = {vw['hr_delta_bpm'].std(ddof=1):.3f}",
            "",
            "Relative heart-rate elevation DeltaHR% = 100*(HR_max-HR_min)/HR_min:",
            f"  anyway_stop mean = {aw['hr_delta_pct'].mean():.2f}%, SD = {aw['hr_delta_pct'].std(ddof=1):.2f}%",
            f"  voluntary_stop mean = {vw['hr_delta_pct'].mean():.2f}%, SD = {vw['hr_delta_pct'].std(ddof=1):.2f}%",
            f"  share with DeltaHR% >= 10%: {(merged['hr_delta_pct'] >= 10.0).mean()*100.0:.1f}%",
            f"  share with DeltaHR% >= 20%: {(merged['hr_delta_pct'] >= 20.0).mean()*100.0:.1f}%",
            f"  share with DeltaHR% >= 30%: {(merged['hr_delta_pct'] >= 30.0).mean()*100.0:.1f}%",
            "",
            "Monotonic association (Spearman rho, p-value):",
            f"  all stop-valid: comfort vs |a_min| -> rho = {rho_c_all:.3f}, p = {p_c_all:.4f}",
            f"  anyway_stop: comfort vs |a_min| -> rho = {rho_c_aw:.3f}, p = {p_c_aw:.4f}",
            f"  anyway_stop: DeltaHR% vs |a_min| -> rho = {rho_hr_aw:.3f}, p = {p_hr_aw:.4f}",
            f"  all stop-valid (within-participant centered): DeltaHR% vs |a_min| -> rho = {rho_hr_within:.3f}, p = {p_hr_within:.4f}",
            "",
            "Participant-fixed-effect validation (controls individual baseline physiology):",
            f"  DeltaHR% ~ |a_min| + C(participant): slope = {fe_slope:.3f} %-points/(m/s^2), 95% CI [{fe_slope_lo:.3f}, {fe_slope_hi:.3f}], p = {fe_p:.4f}",
            "",
            "Bootstrap slope estimates in anyway_stop:",
            f"  comfort ~ |a_min| slope = {slope_c_aw:.3f} rating/(m/s^2), 95% CI [{slope_c_aw_l:.3f}, {slope_c_aw_h:.3f}]",
            f"  DeltaHR% ~ |a_min| slope = {slope_hr_aw:.3f} %-points/(m/s^2), 95% CI [{slope_hr_aw_l:.3f}, {slope_hr_aw_h:.3f}]",
        ]
    )

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    build_figure(
        df=merged,
        out_png=plot_dir / "hr_comfort_advanced.png",
        out_pdf=plot_dir / "hr_comfort_advanced.pdf",
    )

    print("\n".join(summary_lines))
    print(f"\nSaved run-level metrics to: {out_dir / 'run_level_hr_comfort_metrics.csv'}")
    print(f"Saved summary to: {out_dir / 'summary.txt'}")
    print(f"Saved figure to: {plot_dir / 'hr_comfort_advanced.pdf'}")


if __name__ == "__main__":
    main()

