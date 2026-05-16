from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


def _configure_ieee_pdf_fonts() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.rm"] = "STIXGeneral"


@dataclass
class Stats:
    n_participants_all: int
    n_participants_complete: int
    n_participants_partial: int
    n_male: int
    n_female: int
    male_pct: float
    female_pct: float
    age_mean: float
    age_std: float
    age_median: float
    age_min: float
    age_max: float
    age_q25: float
    age_q75: float
    age_cluster_18_24_n: int
    age_cluster_24_32_n: int
    age_cluster_32_plus_n: int
    age_cluster_18_24_pct: float
    age_cluster_24_32_pct: float
    age_cluster_32_plus_pct: float
    n_days_active: int
    day_min: str
    day_max: str
    n_runs_total: int
    n_runs_main: int
    n_runs_forced: int
    n_runs_highspeed: int
    n_runs_voluntary: int
    n_fix_total: int
    n_fix_voluntary: int
    n_go: int
    n_stop: int
    go_rate: float
    stop_rate: float
    speed_inst_min: float
    speed_inst_median: float
    speed_inst_max: float
    speed_inst_mean: float
    speed_inst_std: float
    dist_thr_min: float
    dist_thr_q25: float
    dist_thr_median: float
    dist_thr_q75: float
    dist_thr_max: float
    dist_thr_mean: float
    dist_thr_std: float
    tti_min: float
    tti_q25: float
    tti_median: float
    tti_mean: float
    tti_std: float
    tti_q75: float
    tti_max: float
    tti_skew: float
    tti_kurt: float
    areq_min: float
    areq_q25: float
    areq_median: float
    areq_mean: float
    areq_std: float
    areq_q75: float
    areq_max: float
    areq_skew: float
    areq_kurt: float
    onset_interp_n: int
    onset_interp_rate: float
    speed_delta_mean: float
    speed_delta_std: float
    male_decision_n: int
    female_decision_n: int


def _quantile(series: pd.Series, q: float) -> float:
    return float(pd.to_numeric(series, errors="coerce").quantile(q))


def compute_stats() -> tuple[Stats, dict[str, object], pd.DataFrame]:
    all_runs = pd.read_csv("data/clean_data/all_runs_deduplicated.csv")
    decision = pd.read_csv("data/clean_data/decision_runs.csv")
    fix_rows = pd.read_csv("data/clean_data/fix_rows_labeled.csv")
    duplicates = pd.read_csv("data/clean_data/duplicate_runs_removed.csv")

    remove_files = set(duplicates["source_file"]) if not duplicates.empty else set()
    fix_rows = fix_rows[~fix_rows["source_file"].isin(remove_files)].copy()

    all_runs["driver_age"] = pd.to_numeric(all_runs["driver_age"], errors="coerce")
    all_runs["comfort_rating"] = pd.to_numeric(all_runs["comfort_rating"], errors="coerce")
    all_runs["instructed_speed_kmh"] = pd.to_numeric(all_runs["instructed_speed_kmh"], errors="coerce")
    all_runs["distance_threshold_m"] = pd.to_numeric(all_runs["distance_threshold_m"], errors="coerce")
    all_runs["speed_delta_from_instruction_kmh"] = pd.to_numeric(
        all_runs["speed_delta_from_instruction_kmh"], errors="coerce"
    )
    all_runs["forced_stop_instruction"] = (
        pd.to_numeric(all_runs["forced_stop_instruction"], errors="coerce").fillna(0).astype(int)
    )
    all_runs["file_timestamp"] = pd.to_datetime(all_runs["file_timestamp"], errors="coerce")
    all_runs["driver_sex"] = all_runs["driver_sex"].fillna("unknown").astype(str).str.lower().str.strip()

    decision["tti_s"] = pd.to_numeric(decision["tti_s"], errors="coerce")
    decision["a_req_mps2"] = pd.to_numeric(decision["a_req_mps2"], errors="coerce")
    decision["driver_sex"] = decision["driver_sex"].fillna("unknown").astype(str).str.lower().str.strip()
    decision["speed_delta_from_instruction_kmh"] = pd.to_numeric(
        decision["speed_delta_from_instruction_kmh"], errors="coerce"
    )

    participant_df = (
        all_runs.sort_values("file_timestamp")
        .groupby("participant_group", as_index=False)
        .first()
    )
    participant_df["participant_group"] = participant_df["participant_group"].astype(str)

    participant_counts_main = (
        all_runs[(all_runs["experiment_kind"] == "main") & (all_runs["forced_stop_instruction"] == 0)]
        .groupby("participant_group")
        .size()
    )
    complete_ids = set(participant_counts_main[participant_counts_main == 12].index.astype(str))

    n_participants_all = int(participant_df["participant_group"].nunique())
    n_male = int((participant_df["driver_sex"] == "male").sum())
    n_female = int((participant_df["driver_sex"] == "female").sum())
    male_pct = 100.0 * n_male / n_participants_all
    female_pct = 100.0 * n_female / n_participants_all

    age_clusters = pd.cut(
        participant_df["driver_age"],
        bins=[18, 24, 32, np.inf],
        right=False,
        include_lowest=True,
        labels=["18-24", "24-32", "32+"],
    )
    age_cluster_counts = age_clusters.value_counts().reindex(["18-24", "24-32", "32+"], fill_value=0).astype(int)
    age_cluster_pcts = 100.0 * age_cluster_counts / n_participants_all

    run_dates = all_runs["file_timestamp"].dt.date.dropna()
    main_runs = all_runs[(all_runs["experiment_kind"] == "main") & (all_runs["forced_stop_instruction"] == 0)]
    forced_runs = all_runs[all_runs["forced_stop_instruction"] == 1]
    highspeed_runs = all_runs[
        (all_runs["forced_stop_instruction"] == 0)
        & (all_runs["experiment_kind"].fillna("") == "")
        & (all_runs["instructed_speed_kmh"] >= 69.0)
    ]
    voluntary_runs = all_runs[all_runs["forced_stop_instruction"] == 0].copy()

    speed_counts_vol = (
        voluntary_runs["instructed_speed_kmh"]
        .dropna()
        .round(1)
        .value_counts()
        .sort_index()
    )
    distance_values_vol = voluntary_runs["distance_threshold_m"].dropna().astype(float)

    design_df = voluntary_runs[["distance_threshold_m", "instructed_speed_kmh"]].dropna().copy()
    design_df["distance_bin_left"] = (
        np.floor(design_df["distance_threshold_m"] / 5.0) * 5.0
    )
    design_df["distance_bin_label"] = (
        design_df["distance_bin_left"].astype(int).astype(str)
        + "-"
        + (design_df["distance_bin_left"] + 5.0).astype(int).astype(str)
    )
    design_matrix_binned = pd.pivot_table(
        design_df,
        index="distance_bin_label",
        columns="instructed_speed_kmh",
        values="distance_threshold_m",
        aggfunc="count",
        fill_value=0,
    )
    design_matrix_binned = design_matrix_binned.reindex(
        sorted(design_matrix_binned.index, key=lambda s: int(str(s).split("-")[0]))
    )
    design_matrix_binned = design_matrix_binned.reindex(
        sorted(design_matrix_binned.columns.tolist()), axis=1
    )

    onset_interp_n = int((decision["yellow_onset_method"] == "linear_interpolation").sum())
    onset_interp_rate = onset_interp_n / len(decision)

    tti_vals = pd.to_numeric(decision["tti_s"], errors="coerce").dropna()
    areq_vals = pd.to_numeric(decision["a_req_mps2"], errors="coerce").dropna()
    age_vals = pd.to_numeric(participant_df["driver_age"], errors="coerce").dropna()
    speed_inst_vals = voluntary_runs["instructed_speed_kmh"].dropna()
    dist_vals = distance_values_vol
    dec_sex = decision["driver_sex"]

    s = Stats(
        n_participants_all=n_participants_all,
        n_participants_complete=len(complete_ids),
        n_participants_partial=int(n_participants_all - len(complete_ids)),
        n_male=n_male,
        n_female=n_female,
        male_pct=male_pct,
        female_pct=female_pct,
        age_mean=float(age_vals.mean()),
        age_std=float(age_vals.std()),
        age_median=float(age_vals.median()),
        age_min=float(age_vals.min()),
        age_max=float(age_vals.max()),
        age_q25=float(age_vals.quantile(0.25)),
        age_q75=float(age_vals.quantile(0.75)),
        age_cluster_18_24_n=int(age_cluster_counts["18-24"]),
        age_cluster_24_32_n=int(age_cluster_counts["24-32"]),
        age_cluster_32_plus_n=int(age_cluster_counts["32+"]),
        age_cluster_18_24_pct=float(age_cluster_pcts["18-24"]),
        age_cluster_24_32_pct=float(age_cluster_pcts["24-32"]),
        age_cluster_32_plus_pct=float(age_cluster_pcts["32+"]),
        n_days_active=int(run_dates.nunique()),
        day_min=str(run_dates.min()),
        day_max=str(run_dates.max()),
        n_runs_total=len(all_runs),
        n_runs_main=len(main_runs),
        n_runs_forced=len(forced_runs),
        n_runs_highspeed=len(highspeed_runs),
        n_runs_voluntary=int(len(voluntary_runs)),
        n_fix_total=len(fix_rows),
        n_fix_voluntary=int((fix_rows["forced_stop_instruction"] == 0).sum()),
        n_go=int(decision["go_decision"].sum()),
        n_stop=int(decision["stop_decision"].sum()),
        go_rate=float(decision["go_decision"].mean()),
        stop_rate=float(decision["stop_decision"].mean()),
        speed_inst_min=float(speed_inst_vals.min()),
        speed_inst_median=float(speed_inst_vals.median()),
        speed_inst_max=float(speed_inst_vals.max()),
        speed_inst_mean=float(speed_inst_vals.mean()),
        speed_inst_std=float(speed_inst_vals.std()),
        dist_thr_min=float(dist_vals.min()),
        dist_thr_q25=float(dist_vals.quantile(0.25)),
        dist_thr_median=float(dist_vals.median()),
        dist_thr_q75=float(dist_vals.quantile(0.75)),
        dist_thr_max=float(dist_vals.max()),
        dist_thr_mean=float(dist_vals.mean()),
        dist_thr_std=float(dist_vals.std()),
        tti_min=float(tti_vals.min()),
        tti_q25=float(tti_vals.quantile(0.25)),
        tti_median=float(tti_vals.median()),
        tti_mean=float(tti_vals.mean()),
        tti_std=float(tti_vals.std()),
        tti_q75=float(tti_vals.quantile(0.75)),
        tti_max=float(tti_vals.max()),
        tti_skew=float(tti_vals.skew()),
        tti_kurt=float(tti_vals.kurt()),
        areq_min=float(areq_vals.min()),
        areq_q25=float(areq_vals.quantile(0.25)),
        areq_median=float(areq_vals.median()),
        areq_mean=float(areq_vals.mean()),
        areq_std=float(areq_vals.std()),
        areq_q75=float(areq_vals.quantile(0.75)),
        areq_max=float(areq_vals.max()),
        areq_skew=float(areq_vals.skew()),
        areq_kurt=float(areq_vals.kurt()),
        onset_interp_n=onset_interp_n,
        onset_interp_rate=float(onset_interp_rate),
        speed_delta_mean=float(decision["speed_delta_from_instruction_kmh"].mean()),
        speed_delta_std=float(decision["speed_delta_from_instruction_kmh"].std()),
        male_decision_n=int((dec_sex == "male").sum()),
        female_decision_n=int((dec_sex == "female").sum()),
    )

    aux: dict[str, object] = {
        "age_cluster_counts": age_cluster_counts,
        "sex_counts": pd.Series({"Male": n_male, "Female": n_female}),
        "speed_counts_vol": speed_counts_vol,
        "distance_values_vol": distance_values_vol,
        "design_matrix_binned": design_matrix_binned,
        "voluntary_design_pairs": voluntary_runs[["instructed_speed_kmh", "distance_threshold_m"]].dropna().copy(),
        "voluntary_tracking_df": voluntary_runs[
            ["instructed_speed_kmh", "driver_age", "initial_speed_kmh", "speed_delta_from_instruction_kmh"]
        ].dropna().copy(),
    }
    return s, aux, decision


def generate_instruction_design_figure(aux: dict[str, object], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    _configure_ieee_pdf_fonts()
    plt.rcParams["axes.titleweight"] = "bold"

    speed_counts = pd.Series(aux["speed_counts_vol"]).copy()
    distance_values = pd.Series(aux["distance_values_vol"]).copy()
    design_matrix = pd.DataFrame(aux["design_matrix_binned"]).copy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 3.8),
        gridspec_kw={"width_ratios": [1.0, 1.1, 1.45]},
        constrained_layout=True,
    )

    ax = axes[0]
    speed_counts = speed_counts.sort_index()
    bars = ax.bar(
        [f"{int(v)}" for v in speed_counts.index.to_numpy(dtype=float)],
        speed_counts.to_numpy(dtype=int),
        color=["#1d3557", "#457b9d", "#2a9d8f", "#264653"][: len(speed_counts)],
        alpha=0.92,
    )
    for b, val in zip(bars, speed_counts.to_numpy(dtype=int)):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 1.0, f"{int(val)}", ha="center", va="bottom", fontsize=8)
    ax.set_title("(a) Instructed Speed Setpoints")
    ax.set_xlabel(r"$v_{inst}$ (km/h)")
    ax.set_ylabel("Run count")

    ax = axes[1]
    sns.histplot(
        distance_values,
        bins=np.arange(10, 85, 5),
        color="#6a4c93",
        edgecolor="white",
        alpha=0.85,
        ax=ax,
    )
    ax.axvline(float(distance_values.median()), color="#c1121f", linewidth=2.0, linestyle="--")
    ax.set_title("(b) Threshold-Distance Allocation")
    ax.set_xlabel(r"$d_{th}$ (m)")
    ax.set_ylabel("Run count")
    ax.text(
        0.98,
        0.95,
        f"median={float(distance_values.median()):.1f} m\nrange=[{float(distance_values.min()):.1f}, {float(distance_values.max()):.1f}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#9e9e9e", "boxstyle": "round,pad=0.22"},
    )

    ax = axes[2]
    mat = design_matrix.to_numpy(dtype=float)
    hm = sns.heatmap(
        mat,
        cmap="YlGnBu",
        cbar=True,
        ax=ax,
        linewidths=0.2,
        linecolor="white",
        xticklabels=[f"{int(v)}" for v in design_matrix.columns.to_numpy(dtype=float)],
        yticklabels=design_matrix.index.tolist(),
        cbar_kws={"shrink": 0.9, "pad": 0.02, "label": "Run count"},
    )
    hm.collections[0].colorbar.ax.tick_params(labelsize=8)
    ax.set_title("(c) Parameter Occupancy Map")
    ax.set_xlabel(r"$v_{inst}$ (km/h)")
    ax.set_ylabel(r"$d_{th}$ bin (m)")
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelsize=7)

    out_png = out_dir / "fig_instruction_design_tiny.png"
    out_pdf = out_dir / "fig_instruction_design_tiny.pdf"
    fig.savefig(out_png, dpi=360, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def generate_distance_threshold_by_speed_tiny_figure(aux: dict[str, object], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    _configure_ieee_pdf_fonts()
    plt.rcParams["axes.titleweight"] = "bold"
    # Enlarged typography for IEEE column rendering. The figure is usually
    # inserted at \columnwidth, so small annotation fonts become unreadable.
    fs_title = 18.0
    fs_title_c = 15.6
    fs_axis = 20.0
    fs_tick = 15.0
    fs_annot = 13.0
    fs_legend = 11.6
    fs_heat_annot = 13.0
    fs_foot = 10.0

    df = pd.DataFrame(aux["voluntary_design_pairs"]).copy()
    df_track = pd.DataFrame(aux["voluntary_tracking_df"]).copy()
    df["instructed_speed_kmh"] = pd.to_numeric(df["instructed_speed_kmh"], errors="coerce")
    df["distance_threshold_m"] = pd.to_numeric(df["distance_threshold_m"], errors="coerce")
    df = df.dropna(subset=["instructed_speed_kmh", "distance_threshold_m"]).copy()
    df_track["instructed_speed_kmh"] = pd.to_numeric(df_track["instructed_speed_kmh"], errors="coerce")
    df_track["driver_age"] = pd.to_numeric(df_track["driver_age"], errors="coerce")
    df_track["initial_speed_kmh"] = pd.to_numeric(df_track["initial_speed_kmh"], errors="coerce")
    df_track["speed_delta_from_instruction_kmh"] = pd.to_numeric(
        df_track["speed_delta_from_instruction_kmh"], errors="coerce"
    )
    df_track = df_track.dropna(
        subset=["instructed_speed_kmh", "driver_age", "initial_speed_kmh", "speed_delta_from_instruction_kmh"]
    ).copy()

    speeds = sorted(df["instructed_speed_kmh"].unique().tolist())
    speed_labels = [f"{int(s)} km/h" for s in speeds]
    df["speed_label"] = df["instructed_speed_kmh"].map({s: f"{int(s)} km/h" for s in speeds})
    df_track["speed_label"] = df_track["instructed_speed_kmh"].map({s: f"{int(s)} km/h" for s in speeds})

    # Use a clearer speed palette; 40 and 50 are intentionally far apart in hue.
    palette_vals = ["#1f3b73", "#7b2cbf", "#2a9d8f", "#e76f51"]
    palette = {lbl: palette_vals[i % len(palette_vals)] for i, lbl in enumerate(speed_labels)}

    def _lighten_hex(color: str, frac_to_white: float = 0.45) -> str:
        rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
        mixed = rgb * (1.0 - frac_to_white) + frac_to_white * np.ones(3, dtype=float)
        return mcolors.to_hex(np.clip(mixed, 0.0, 1.0))

    # Panel (b) uses the same speed mapping but in a lighter tone.
    palette_b = {k: _lighten_hex(v, frac_to_white=0.45) for k, v in palette.items()}

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.6, 5.35),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.24, 1.24, 0.70]},
    )

    # (a) D_thr summary by speed with full tested range (min-max) plus mean and SD bars.
    ax = axes[0]
    xpos = np.arange(len(speed_labels), dtype=float)
    means: list[float] = []
    sds: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    for lbl in speed_labels:
        vals = df.loc[df["speed_label"] == lbl, "distance_threshold_m"].to_numpy(dtype=float)
        if len(vals) == 0:
            means.append(np.nan)
            sds.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
            continue
        means.append(float(np.mean(vals)))
        sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        mins.append(float(np.min(vals)))
        maxs.append(float(np.max(vals)))

    means_arr = np.asarray(means, dtype=float)
    sds_arr = np.asarray(sds, dtype=float)
    mins_arr = np.asarray(mins, dtype=float)
    maxs_arr = np.asarray(maxs, dtype=float)
    valid = np.isfinite(means_arr)
    valid_range = np.isfinite(mins_arr) & np.isfinite(maxs_arr)

    sd_band_width = 0.30
    mean_line_half = 0.18
    for i in range(len(xpos)):
        if not valid[i]:
            continue
        # SD shown as a vertical band centered at the mean (mu +/- SD).
        ax.add_patch(
            plt.Rectangle(
                (xpos[i] - sd_band_width / 2.0, means_arr[i] - sds_arr[i]),
                sd_band_width,
                2.0 * sds_arr[i],
                facecolor=palette[speed_labels[i]],
                edgecolor=palette[speed_labels[i]],
                linewidth=0.8,
                alpha=0.40,
                zorder=2,
            )
        )
        # Mean shown as a thick horizontal line.
        ax.hlines(
            means_arr[i],
            xpos[i] - mean_line_half,
            xpos[i] + mean_line_half,
            color="#111827",
            linewidth=3.2,
            zorder=5,
        )
    # Thin full-range indicator shows tested extremes for each speed.
    ax.vlines(xpos[valid_range], mins_arr[valid_range], maxs_arr[valid_range], color="#111827", linewidth=1.5, zorder=4)
    cap_w = 0.08
    for i in range(len(xpos)):
        if not valid_range[i]:
            continue
        ax.hlines([mins_arr[i], maxs_arr[i]], xpos[i] - cap_w, xpos[i] + cap_w, color="#111827", linewidth=1.5, zorder=4)
        ax.text(
            xpos[i],
            maxs_arr[i] + 1.4,
            f"mu={means_arr[i]:.1f}\nSD={sds_arr[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=fs_annot,
            color="#111827",
        )

    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{int(s)}" for s in speeds])
    y_min = float(np.nanmin(mins_arr))
    y_max = float(np.nanmax(maxs_arr))
    ax.set_ylim(y_min - 2.0, y_max + 6.4)
    ax.set_title("(a) $D_{thr}$ Full Range and Mean/SD by $v_{inst}$", fontsize=fs_title)
    ax.set_xlabel(r"$V_{\mathrm{inst}}$ (km/h)", fontsize=fs_axis)
    ax.set_ylabel(r"$D_{thr}$ (m)", fontsize=fs_axis)
    summary_handles = [
        Line2D([0], [0], color="#111827", lw=3.2, label="Mean $D_{thr}$"),
        Patch(facecolor="#6b7280", alpha=0.40, edgecolor="#6b7280", label="$\\mu \\pm 1\\sigma$ band"),
        Line2D([0], [0], color="#111827", lw=1.5, label="Full tested range (min-max)"),
    ]
    ax.legend(handles=summary_handles, fontsize=fs_legend, frameon=True, loc="lower right")
    ax.tick_params(axis="both", labelsize=fs_tick)

    # (b) Actual speed distributions vs instructed-speed reference (dot-free; summary-focused).
    ax = axes[1]
    sns.violinplot(
        data=df_track,
        x="speed_label",
        y="initial_speed_kmh",
        hue="speed_label",
        order=speed_labels,
        palette=palette_b,
        legend=False,
        dodge=False,
        inner="quartile",
        cut=0,
        linewidth=1.0,
        ax=ax,
    )
    above_color = "#111827"
    below_color = "#d7263d"
    y_global_min = float(df_track["initial_speed_kmh"].min())
    y_global_max = float(df_track["initial_speed_kmh"].max())
    ax.set_ylim(y_global_min - 1.8, y_global_max + 3.6)
    for i, (sp, lbl) in enumerate(zip(speeds, speed_labels)):
        sub = df_track.loc[df_track["speed_label"] == lbl, :].copy()
        if sub.empty:
            continue
        ax.hlines(float(sp), i - 0.34, i + 0.34, color="#111827", linewidth=1.3, linestyle="--", zorder=4)
        vy = sub["initial_speed_kmh"].to_numpy(dtype=float)
        vy_mean = float(np.mean(vy))
        dv_sd = float(np.std(sub["speed_delta_from_instruction_kmh"].to_numpy(dtype=float), ddof=1)) if len(sub) > 1 else 0.0
        # Mean marker without x-jitter points.
        ax.hlines(vy_mean, i - 0.16, i + 0.16, color="#0f172a", linewidth=1.9, zorder=5)
        y_top = min(float(np.nanmax(sub["initial_speed_kmh"])) + 0.25, y_global_max + 2.05)
        if i == len(speed_labels) - 1:
            y_top -= 0.95
        ax.text(
            i,
            y_top,
            f"mean speed={vy_mean:.1f}\nSD error={dv_sd:.1f}",
            ha="center",
            va="bottom",
            fontsize=fs_annot,
            color="#111827",
            clip_on=True,
            rotation=0,
        )
    above_all = df_track.loc[df_track["speed_delta_from_instruction_kmh"] > 0, "speed_delta_from_instruction_kmh"]
    below_all = df_track.loc[df_track["speed_delta_from_instruction_kmh"] < 0, "speed_delta_from_instruction_kmh"]
    n_all = max(float(len(df_track)), 1.0)
    p_above_all = 100.0 * len(above_all) / n_all
    p_below_all = 100.0 * len(below_all) / n_all
    dv_sd_all = float(df_track["speed_delta_from_instruction_kmh"].std(ddof=1)) if len(df_track) > 1 else 0.0
    ax.text(
        0.02,
        0.97,
        (
            f"Overall speed-tracking SD = {dv_sd_all:.2f} km/h\n"
            f"Above target: {p_above_all:.1f}%  |  Below target: {p_below_all:.1f}%"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fs_legend,
        color="#111827",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#9ca3af", "boxstyle": "round,pad=0.25"},
    )
    ax.set_title("(b) Actual $V_{onset}$ Distribution vs $v_{inst}$", fontsize=fs_title)
    ax.set_xlabel(r"$V_{\mathrm{inst}}$ (km/h)", fontsize=fs_axis)
    ax.set_ylabel(r"Actual speed at yellow onset $V_{onset}$ (km/h)", fontsize=fs_axis)
    ax.set_xticklabels([f"{int(s)}" for s in speeds])
    speed_handles = [
        Line2D([0], [0], color="#2b2b2b", lw=6.0, alpha=0.35, label="Violin width = density of actual $V_{onset}$"),
        Line2D([0], [0], color="#111827", lw=1.3, ls="--", label="Instructed speed line"),
        Line2D([0], [0], color="#0f172a", lw=1.9, label="Mean actual $V_{onset}$"),
    ]
    ax.legend(handles=speed_handles, fontsize=fs_legend, frameon=True, loc="lower right")
    ax.tick_params(axis="both", labelsize=fs_tick)

    # (c) Age effect by speed: raw tracking-error map (speed columns interpreted independently).
    ax = axes[2]
    age_edges = [18, 25, 32, 120]
    age_labels = ["18-25", "25-32", "32+"]
    age_labels_plot = ["32+", "25-32", "18-25"]
    df_track["age_bin"] = pd.cut(
        df_track["driver_age"],
        bins=age_edges,
        right=False,
        include_lowest=True,
        labels=age_labels,
    )
    hm_df = (
        df_track.groupby(["age_bin", "speed_label"], observed=True)["speed_delta_from_instruction_kmh"]
        .mean()
        .unstack()
        .reindex(index=age_labels_plot, columns=speed_labels)
    )
    cnt_df = (
        df_track.groupby(["age_bin", "speed_label"], observed=True)["speed_delta_from_instruction_kmh"]
        .size()
        .unstack()
        .reindex(index=age_labels_plot, columns=speed_labels)
    )
    # Show raw Delta-v values directly (negative = underspeed).
    hm_plot = hm_df.copy()
    hm_vals = hm_plot.to_numpy(dtype=float)
    finite = np.isfinite(hm_vals)
    if np.any(finite):
        vmin = float(np.nanmin(hm_vals))
        vmax = float(np.nanmax(hm_vals))
    else:
        vmin, vmax = -1.0, 1.0
    # Keep 0 visible as the neutral line and emphasize underspeed (more negative) with darker blue.
    if vmax > 0.0:
        vmax = 0.0
    if not np.isfinite(vmin) or vmin >= vmax - 1e-9:
        vmin = min(vmax - 1.0, -1.0)

    sns.heatmap(
        hm_plot,
        cmap="Blues_r",
        vmin=vmin,
        vmax=vmax,
        linewidths=1.0,
        linecolor="white",
        cbar=True,
        cbar_kws={
            "shrink": 0.9,
            "pad": 0.02,
            "label": "Mean tracking error ($\\Delta v$) [km/h]",
        },
        annot=False,
        ax=ax,
    )
    # Visual separators between speed columns to emphasize non-comparability across speeds.
    y0, y1 = ax.get_ylim()
    for x in range(1, len(speed_labels)):
        ax.vlines(float(x), y0, y1, colors="#111827", linewidth=1.2, alpha=0.8)

    for r in range(hm_df.shape[0]):
        for c in range(hm_df.shape[1]):
            val = hm_df.iloc[r, c]
            n = cnt_df.iloc[r, c]
            if pd.notna(val) and pd.notna(n):
                intensity = hm_plot.iloc[r, c]
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    f"{val:+.1f}\n(n={int(n)})",
                    ha="center",
                    va="center",
                    fontsize=fs_heat_annot,
                    fontweight="bold",
                    color=("white" if abs(float(intensity)) >= 0.45 * abs(vmin) else "#111827"),
                )
    ax.set_xticklabels([f"{int(s)}" for s in speeds], rotation=0)
    ax.set_title("(c) Age Effect on Tracking Error ($\\Delta v$)", fontsize=fs_title_c)
    ax.set_xlabel(r"$V_{\mathrm{inst}}$ (km/h)", fontsize=fs_axis)
    ax.set_ylabel("Age bin (years)", fontsize=fs_axis)
    ax.tick_params(axis="both", labelsize=fs_tick)
    if ax.collections and ax.collections[0].colorbar is not None:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=fs_tick)
        cbar.ax.yaxis.label.set_size(fs_axis)
    out_png = out_dir / "fig_distance_threshold_by_speed_tiny.png"
    out_pdf = out_dir / "fig_distance_threshold_by_speed_tiny.pdf"
    fig.savefig(out_png, dpi=360, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return out_pdf

def _plot_distribution_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_col: str,
    xlabel: str,
    panel_title: str,
    value_fmt: str,
) -> None:
    fs_panel_title = 18.5
    fs_axis = 20.2
    fs_tick = 20.0
    fs_stats = 13.0

    palette = {"male": "#355070", "female": "#b56576"}
    age_styles = {
        "18-25": {"color": "#2a9d8f", "linestyle": ":"},
        "25-32": {"color": "#e9c46a", "linestyle": "-."},
        "32+": {"color": "#7b2cbf", "linestyle": "--"},
    }

    panel_df = data[[value_col, "driver_sex", "driver_age"]].copy()
    panel_df[value_col] = pd.to_numeric(panel_df[value_col], errors="coerce")
    panel_df["driver_age"] = pd.to_numeric(panel_df["driver_age"], errors="coerce")
    panel_df = panel_df.dropna()
    panel_df["driver_sex"] = panel_df["driver_sex"].where(panel_df["driver_sex"].isin(["male", "female"]), "other")
    panel_df["age_group"] = pd.cut(
        panel_df["driver_age"],
        bins=[18.0, 25.0, 32.0, np.inf],
        right=False,
        labels=["18-25", "25-32", "32+"],
    ).astype(str)

    values = panel_df[value_col].to_numpy()
    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    mean_v = float(np.mean(values))
    sd_v = float(np.std(values, ddof=1))
    skew_v = float(pd.Series(values).skew())
    kurt_v = float(pd.Series(values).kurt())

    sns.histplot(values, bins=28, stat="density", color="#4f6078", alpha=0.42, edgecolor=None, ax=ax)
    for sex in ["male", "female"]:
        subset = panel_df.loc[panel_df["driver_sex"] == sex, value_col].to_numpy()
        if len(subset) >= 2:
            sns.kdeplot(subset, ax=ax, color=palette[sex], fill=False, linewidth=3.1)

    for grp, style in age_styles.items():
        subset = panel_df.loc[panel_df["age_group"] == grp, value_col].to_numpy()
        if len(subset) >= 2:
            sns.kdeplot(
                subset,
                ax=ax,
                color=style["color"],
                fill=False,
                linewidth=2.6,
                linestyle=style["linestyle"],
                alpha=0.95,
            )

    ax.axvline(q25, color="#6a4c93", linestyle="--", linewidth=2.3)
    ax.axvline(q50, color="#c1121f", linestyle="-", linewidth=3.2)
    ax.axvline(q75, color="#6a4c93", linestyle="--", linewidth=2.3)

    male_vals = panel_df.loc[panel_df["driver_sex"] == "male", value_col].to_numpy()
    female_vals = panel_df.loc[panel_df["driver_sex"] == "female", value_col].to_numpy()
    male_n = int(len(male_vals))
    female_n = int(len(female_vals))
    male_mean = float(np.mean(male_vals)) if male_n > 0 else np.nan
    female_mean = float(np.mean(female_vals)) if female_n > 0 else np.nan
    age_text_parts = []
    for grp in ["18-25", "25-32", "32+"]:
        vals_grp = panel_df.loc[panel_df["age_group"] == grp, value_col].to_numpy()
        if len(vals_grp) > 0:
            grp_mean = float(np.mean(vals_grp))
            grp_sd = float(np.std(vals_grp, ddof=1)) if len(vals_grp) > 1 else 0.0
            age_text_parts.append(
                f"{grp}: mean={value_fmt.format(grp_mean)}, SD={value_fmt.format(grp_sd)}"
            )
    age_text = "\n".join(age_text_parts)

    stat_text = (
        f"male mean = {value_fmt.format(male_mean)}, female mean = {value_fmt.format(female_mean)}\n"
        f"age-bin mean/SD:\n{age_text}\n"
        f"mean = {value_fmt.format(mean_v)}, SD = {value_fmt.format(sd_v)}\n"
        f"median = {value_fmt.format(q50)}\n"
        f"IQR = [{value_fmt.format(q25)}, {value_fmt.format(q75)}]\n"
        f"skew = {skew_v:.2f}, excess kurtosis = {kurt_v:.2f}"
    )
    ax.text(
        0.985,
        0.975,
        stat_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fs_stats,
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "#9e9e9e", "boxstyle": "round,pad=0.34"},
    )

    ax.set_title(panel_title, fontweight="bold", fontsize=fs_panel_title)
    ax.set_xlabel(xlabel, fontsize=fs_axis)
    ax.set_ylabel("Probability Density", fontsize=fs_axis)
    ax.tick_params(axis="both", labelsize=fs_tick)


def generate_tti_areq_figure(decision: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    _configure_ieee_pdf_fonts()
    plt.rcParams["axes.titleweight"] = "bold"

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.15))

    _plot_distribution_panel(
        ax=axes[0],
        data=decision,
        value_col="tti_s",
        xlabel="TTI at Yellow Onset (s)",
        panel_title="(a) TTI Distribution With Gender + Age Composition",
        value_fmt="{:.2f}",
    )
    _plot_distribution_panel(
        ax=axes[1],
        data=decision,
        value_col="a_req_mps2",
        xlabel=r"Required Deceleration $a_{\mathrm{req}}$ (m/s$^2$)",
        panel_title=r"(b) $a_{\mathrm{req}}$ Distribution With Gender + Age Composition",
        value_fmt="{:.2f}",
    )

    legend_handles = [
        Patch(facecolor="#4f6078", edgecolor="none", alpha=0.42, label="Overall histogram"),
        Line2D([0], [0], color="#355070", lw=2.4, label="Male KDE"),
        Line2D([0], [0], color="#b56576", lw=2.4, label="Female KDE"),
        Line2D([0], [0], color="#2a9d8f", lw=2.0, ls=":", label="Age 18-25 KDE"),
        Line2D([0], [0], color="#e9c46a", lw=2.0, ls="-.", label="Age 25-32 KDE"),
        Line2D([0], [0], color="#7b2cbf", lw=2.0, ls="--", label="Age 32+ KDE"),
        Line2D([0], [0], color="#6a4c93", lw=1.8, ls="--", label="Q1 and Q3"),
        Line2D([0], [0], color="#c1121f", lw=2.6, ls="-", label="Median"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        framealpha=0.92,
        bbox_to_anchor=(0.5, 0.03),
        borderaxespad=0.2,
        columnspacing=1.4,
        handlelength=2.2,
        fontsize=14.0,
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))

    out_png = out_dir / "fig_tti_areq_rich_stats.png"
    out_pdf = out_dir / "fig_tti_areq_rich_stats.pdf"
    fig.savefig(out_png, dpi=360, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def generate_latex(
    stats: Stats,
    aux: dict[str, object],
    out_tex: Path,
    instruction_fig_rel_path: str,
    tti_areq_fig_rel_path: str,
    protocol_text: str,
) -> None:
    speed_counts = pd.Series(aux["speed_counts_vol"]).sort_index()
    speed_count_text = ", ".join([f"{int(s)} km/h: {int(c)}" for s, c in speed_counts.items()])

    latex = f"""% Auto-generated by generate_data_collection_section.py
\\section{{Data Collection and Protocol}}
\\label{{sec:data_collection}}

The data campaign lasted eight calendar days ({stats.day_min} to {stats.day_max}), with {stats.n_days_active} active recording days. After deduplication, the dataset contains {stats.n_runs_total} runs and {stats.n_fix_total:,} timestamped vehicle-state samples. The run composition is {stats.n_runs_main} main decision runs, {stats.n_runs_forced} forced-stop comfort runs (``anyway\\_stop''), and {stats.n_runs_highspeed} high-speed add-on runs (70~km/h condition).

Each run was parameterized by two control variables: instructed speed $v_{{inst}}$ and yellow-onset threshold distance $D_{{thr}}$. For voluntary runs ($n={stats.n_runs_voluntary}$), the instructed-speed allocation was: {speed_count_text}. Threshold distances were distributed over [{stats.dist_thr_min:.1f}, {stats.dist_thr_max:.1f}]~m (median {stats.dist_thr_median:.1f}~m; IQR [{stats.dist_thr_q25:.1f}, {stats.dist_thr_q75:.1f}]~m). One randomly placed always-green control case was included in each participant sequence to reduce expectancy bias.

\\begin{{figure*}}[t]
\\centering
\\includegraphics[width=0.98\\textwidth]{{{instruction_fig_rel_path}}}
\\caption{{Instruction-space characterization. (a) instructed-speed setpoint counts; (b) threshold-distance allocation; (c) occupancy of the instructed parameter space in binned distance-speed coordinates.}}
\\label{{fig:instruction_design_tiny}}
\\end{{figure*}}

The participant cohort included {stats.n_participants_all} anonymized drivers ({stats.n_participants_complete} complete 12-run protocols, {stats.n_participants_partial} partial/add-on only). Age ranged from {stats.age_min:.0f} to {stats.age_max:.0f} years (mean {stats.age_mean:.2f}, SD {stats.age_std:.2f}, median {stats.age_median:.2f}, IQR [{stats.age_q25:.1f}, {stats.age_q75:.1f}]). Sex distribution was male={stats.n_male} ({stats.male_pct:.1f}\\%) and female={stats.n_female} ({stats.female_pct:.1f}\\%).

\\begin{{table}}[t]
\\centering
\\caption{{Participant demographics (age bins: [18,24), [24,32), [32,$\\infty$)).}}
\\label{{tab:participant_demographics}}
\\begin{{tabular}}{{lcc}}
\\hline
Group & Count & Share (\\%) \\\\
\\hline
Age 18--24 & {stats.age_cluster_18_24_n} & {stats.age_cluster_18_24_pct:.1f} \\\\
Age 24--32 & {stats.age_cluster_24_32_n} & {stats.age_cluster_24_32_pct:.1f} \\\\
Age 32+ & {stats.age_cluster_32_plus_n} & {stats.age_cluster_32_plus_pct:.1f} \\\\
\\hline
Male & {stats.n_male} & {stats.male_pct:.1f} \\\\
Female & {stats.n_female} & {stats.female_pct:.1f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}

Yellow-onset states were reconstructed by linear interpolation at exact threshold crossing:
\\begin{{align}}
\\alpha &= \\frac{{D_{{thr}}-d_0}}{{d_1-d_0}}, &
t_y &= t_0 + \\alpha (t_1-t_0), &
V_{{onset}} &= v_0 + \\alpha (v_1-v_0).
\\end{{align}}
Interpolation succeeded in {stats.onset_interp_n}/{stats.n_runs_voluntary} runs ({100.0*stats.onset_interp_rate:.2f}\\%). Derived variables were
\\begin{{align}}
\\mathrm{{TTI}} = \\frac{{D_{{thr}}}}{{V_{{onset}}}}, \\qquad
a_{{req}} = \\frac{{V_{{onset}}^2}}{{2D_{{thr}}}}.
\\end{{align}}
The voluntary decision subset contains go={stats.n_go} ({100.0*stats.go_rate:.1f}\\%) and stop={stats.n_stop} ({100.0*stats.stop_rate:.1f}\\%), with sex composition male={stats.male_decision_n}, female={stats.female_decision_n}. TTI statistics: mean {stats.tti_mean:.2f}s, SD {stats.tti_std:.2f}s, median {stats.tti_median:.2f}s, IQR [{stats.tti_q25:.2f}, {stats.tti_q75:.2f}]s, range [{stats.tti_min:.2f}, {stats.tti_max:.2f}]s, skewness {stats.tti_skew:.2f}, excess kurtosis {stats.tti_kurt:.2f}. Required deceleration statistics: mean {stats.areq_mean:.2f}~m/s$^2$, SD {stats.areq_std:.2f}, median {stats.areq_median:.2f}, IQR [{stats.areq_q25:.2f}, {stats.areq_q75:.2f}], range [{stats.areq_min:.2f}, {stats.areq_max:.2f}]~m/s$^2$, skewness {stats.areq_skew:.2f}, excess kurtosis {stats.areq_kurt:.2f}.

\\begin{{figure*}}[t]
\\centering
\\includegraphics[width=0.98\\textwidth]{{{tti_areq_fig_rel_path}}}
\\caption{{TTI and required-deceleration distributions for voluntary decision runs, with overall histograms, sex-conditioned and age-bin-conditioned KDE overlays, quartile/median markers, and higher-order distribution descriptors.}}
\\label{{fig:tti_areq_rich_stats}}
\\end{{figure*}}
"""

    latex += "\n% Source protocol notes extracted from Driver Instructions.pdf:\n"
    for line in protocol_text.splitlines():
        line = line.rstrip()
        if line:
            latex += f"% {line}\n"
        else:
            latex += "%\n"

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(latex, encoding="utf-8")


def main() -> None:
    out_root = Path("results/paper")
    fig_dir = out_root / "figures"
    tex_path = out_root / "section_data_collection.tex"
    protocol_path = Path("results/driver_instructions.txt")
    if not protocol_path.exists():
        raise FileNotFoundError("Missing results/driver_instructions.txt. Extract Driver Instructions.pdf first.")

    stats, aux, decision = compute_stats()
    instruction_fig_pdf = generate_instruction_design_figure(aux, fig_dir)
    dth_speed_tiny_fig_pdf = generate_distance_threshold_by_speed_tiny_figure(aux, fig_dir)
    tti_areq_fig_pdf = generate_tti_areq_figure(decision, fig_dir)

    instruction_fig_rel = "figures/" + instruction_fig_pdf.name
    tti_areq_fig_rel = "figures/" + tti_areq_fig_pdf.name
    protocol_text = protocol_path.read_text(encoding="utf-8")
    generate_latex(
        stats=stats,
        aux=aux,
        out_tex=tex_path,
        instruction_fig_rel_path=instruction_fig_rel,
        tti_areq_fig_rel_path=tti_areq_fig_rel,
        protocol_text=protocol_text,
    )

    print(f"Generated figure: {instruction_fig_pdf}")
    print(f"Generated figure: {dth_speed_tiny_fig_pdf}")
    print(f"Generated figure: {tti_areq_fig_pdf}")
    print(f"Generated LaTeX section: {tex_path}")


if __name__ == "__main__":
    main()


