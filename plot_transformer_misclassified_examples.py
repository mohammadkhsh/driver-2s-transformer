from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("results/trajectory_transformer_ar")
PRED_CSV = BASE_DIR / "validation_predictions.csv"
ARR_NPZ = BASE_DIR / "validation_trajectory_arrays.npz"
OUT_DIR = BASE_DIR / "plots"


def _pick_low_high(df: pd.DataFrame, n: int = 2) -> list[int]:
    if df.empty:
        return []
    if len(df) <= n:
        return df.index.astype(int).tolist()
    order = df["a_req_mps2"].to_numpy(dtype=float).argsort()
    idx = df.index.to_numpy(dtype=int)[order]
    picks = [int(idx[0]), int(idx[-1])]
    # if duplicates for len==1 or same index (defensive)
    picks = list(dict.fromkeys(picks))
    if len(picks) < n:
        for i in idx:
            if int(i) not in picks:
                picks.append(int(i))
            if len(picks) >= n:
                break
    return picks[:n]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED_CSV)
    arr = np.load(ARR_NPZ)

    a_true = arr["a_true_val"]
    d_true = arr["d_true_val"]
    a_pred = arr["a_pred_cascade"]
    d_pred = arr["d_pred_traj_cascade"]

    gt_go = df["go_decision"].to_numpy(dtype=int)
    gt_stop = df["stop_decision"].to_numpy(dtype=int)
    pred_go = df["decision_go_pred"].to_numpy(dtype=int)
    pred_stop = 1 - pred_go

    mask_gt_stop_pred_go = (gt_stop == 1) & (pred_go == 1)
    mask_gt_go_pred_stop = (gt_go == 1) & (pred_stop == 1)

    df_m1 = df.loc[mask_gt_stop_pred_go].copy()
    df_m2 = df.loc[mask_gt_go_pred_stop].copy()

    picks = _pick_low_high(df_m1, 2) + _pick_low_high(df_m2, 2)
    if len(picks) < 4:
        raise RuntimeError(
            f"Need 4 misclassified samples, found {len(picks)} (GTstop->Predgo={mask_gt_stop_pred_go.sum()}, GTgo->Predstop={mask_gt_go_pred_stop.sum()})"
        )

    # Plot settings
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.6), squeeze=False)
    axes = axes.ravel()

    d_min = float(min(np.nanmin(d_true[picks]), np.nanmin(d_pred[picks])) - 2.0)
    d_max = float(max(np.nanmax(d_true[picks]), np.nanmax(d_pred[picks])) + 2.0)
    a_min = float(min(np.nanmin(a_true[picks]), np.nanmin(a_pred[picks])) - 0.4)
    a_max = float(max(np.nanmax(a_true[picks]), np.nanmax(a_pred[picks])) + 0.4)
    # clamp to a readable range without hiding extremes
    a_min = min(a_min, -7.0)
    a_max = max(a_max, 2.5)

    first_twin = None
    for j, (ax, i) in enumerate(zip(axes, picks)):
        row = df.iloc[i]
        T = float(row["traj_duration_s"])
        t = np.linspace(0.0, max(T, 1e-3), a_true.shape[1])

        # Left y-axis: acceleration
        ax.plot(t, a_true[i], color="#111827", lw=2.2, label="GT $a_x$")
        ax.plot(t, a_pred[i], color="#e76f51", lw=2.0, ls="--", label="Pred $a_x$")
        ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
        ax.set_ylim(a_min, a_max)
        ax.grid(alpha=0.23)
        ax.set_xlabel("Time (s)")
        if j == 0:
            ax.set_ylabel("$a_x$ (m/s$^2$)")

        # Right y-axis: distance-to-light
        axd = ax.twinx()
        if first_twin is None:
            first_twin = axd
        axd.plot(t, d_true[i], color="#1d3557", lw=1.7, alpha=0.95, label="GT $d(t)$")
        axd.plot(t, d_pred[i], color="#2a9d8f", lw=1.7, ls="-.", alpha=0.95, label="Pred $d(t)$")
        axd.axhline(0.0, color="#6b7280", lw=0.8, ls="--")
        axd.set_ylim(d_min, d_max)
        if j == len(axes) - 1:
            axd.set_ylabel("$d$ to light (m)")

        gt_lbl = "STOP" if int(row["stop_decision"]) == 1 else "GO"
        pr_lbl = "STOP" if int(row["decision_go_pred"]) == 0 else "GO"
        mismatch_type = "GT=STOP, Pred=GO" if (int(row["stop_decision"]) == 1 and int(row["decision_go_pred"]) == 1) else "GT=GO, Pred=STOP"

        title = (
            f"{mismatch_type}\n"
            f"TTI={float(row['tti_s']):.2f} s, "
            f"$a_{{req}}$={float(row['a_req_mps2']):.2f} m/s$^2$"
        )
        ax.set_title(title)

        # small corner chip with p(go), v0, d0
        chip = (
            f"$p_{{go}}$={float(row['decision_go_prob']):.2f}\n"
            f"$v_0$={float(row['speed_at_yellow_kmh']):.1f} km/h, "
            f"$d_0$={float(row['distance_threshold_m']):.1f} m"
        )
        ax.text(
            0.98,
            0.02,
            chip,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bbbbbb", alpha=0.9),
        )

    # Shared legend
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = (first_twin.get_legend_handles_labels() if first_twin is not None else ([], []))
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=4, frameon=True)

    fig.suptitle(
        "Misclassified Full-Pipeline Rollouts (2 x GT=STOP→Pred=GO, 2 x GT=GO→Pred=STOP)",
        y=0.995,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    out_png = OUT_DIR / "transformer_misclassified_4panel.png"
    out_pdf = OUT_DIR / "transformer_misclassified_4panel.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # save manifest for reproducibility
    manifest = df.loc[picks, [
        "source_file",
        "go_decision",
        "stop_decision",
        "decision_go_pred",
        "decision_go_prob",
        "speed_at_yellow_kmh",
        "distance_threshold_m",
        "tti_s",
        "a_req_mps2",
        "traj_duration_s",
    ]].copy()
    manifest.insert(0, "panel_order", [1, 2, 3, 4])
    manifest.to_csv(OUT_DIR / "transformer_misclassified_4panel_manifest.csv", index=False)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {OUT_DIR / 'transformer_misclassified_4panel_manifest.csv'}")


if __name__ == "__main__":
    main()

