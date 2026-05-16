from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("results/trajectory_transformer_ar")


STOP_EPS_MPS = 0.05


def _configure_ieee_pdf_fonts() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["mathtext.rm"] = "STIXGeneral"


def _pick_spread(df: pd.DataFrame, n: int, prefer_mask: np.ndarray | None = None) -> list[int]:
    if df.empty:
        return []
    work = df.copy()
    if prefer_mask is not None and len(prefer_mask) == len(df):
        # Keep preferred rows if enough; otherwise fallback to all
        if int(np.sum(prefer_mask)) >= n:
            work = work.loc[prefer_mask].copy()
    if len(work) <= n:
        return work.index.astype(int).tolist()
    # Sort by a_req then sample evenly
    order = np.argsort(work["a_req_mps2"].to_numpy(dtype=float))
    idx = work.index.to_numpy(dtype=int)[order]
    take = np.linspace(0, len(idx) - 1, n, dtype=int)
    return [int(idx[k]) for k in take]


def _first_stop_distance(d: np.ndarray, v: np.ndarray, eps: float = STOP_EPS_MPS) -> tuple[float | None, bool]:
    # returns (distance, exact_flag) with exact_flag=False if fallback used
    hit = np.where(v <= eps)[0]
    if hit.size > 0:
        return float(d[int(hit[0])]), True
    # fallback to minimum-speed point
    j = int(np.argmin(v))
    return float(d[j]), False


def _speed_at_distance_zero(d: np.ndarray, v: np.ndarray) -> tuple[float | None, bool]:
    # interpolate first crossing from d>0 to d<=0
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    for k in range(1, len(d)):
        d0, d1 = d[k - 1], d[k]
        if d0 == 0.0:
            return float(v[k - 1]), True
        if d1 == 0.0:
            return float(v[k]), True
        if (d0 > 0.0 and d1 < 0.0) or (d0 < 0.0 and d1 > 0.0):
            # linear interpolation in distance
            w = (0.0 - d0) / (d1 - d0)
            vv = v[k - 1] + w * (v[k] - v[k - 1])
            return float(vv), True
    return None, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 12 correctly classified Transformer examples.")
    parser.add_argument("--base-dir", type=str, default=str(BASE_DIR))
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    pred_csv = base_dir / "validation_predictions.csv"
    arr_npz = base_dir / "validation_trajectory_arrays.npz"
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (base_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)
    arr = np.load(arr_npz)
    a_true = arr["a_true_val"]
    d_true = arr["d_true_val"]
    v_true = arr["v_true_val"]
    a_pred = arr["a_pred_cascade"]
    d_pred = arr["d_pred_traj_cascade"]
    v_pred = arr["v_pred_traj_cascade"]

    gt_go = df["go_decision"].to_numpy(dtype=int)
    gt_stop = df["stop_decision"].to_numpy(dtype=int)
    pred_go = df["decision_go_pred"].to_numpy(dtype=int)
    pred_stop = 1 - pred_go

    stop_ok = (gt_stop == 1) & (pred_stop == 1)
    go_ok = (gt_go == 1) & (pred_go == 1)

    # Prefer stop cases where predicted trajectory actually reaches near-zero speed
    stop_reaches_zero = np.array([np.min(v_pred[i]) <= STOP_EPS_MPS for i in range(len(df))], dtype=bool)
    # Prefer go cases where predicted trajectory crosses the line
    go_crosses = np.array([np.min(d_pred[i]) <= 0.0 for i in range(len(df))], dtype=bool)

    stop_df = df.loc[stop_ok].copy()
    go_df = df.loc[go_ok].copy()
    stop_pref = stop_reaches_zero[stop_ok]
    go_pref = go_crosses[go_ok]

    stop_picks = _pick_spread(stop_df, 8, prefer_mask=stop_pref)
    go_picks = _pick_spread(go_df, 4, prefer_mask=go_pref)
    picks = stop_picks + go_picks
    if len(stop_picks) < 8 or len(go_picks) < 4:
        raise RuntimeError(f"Not enough correct samples for layout (stop={len(stop_picks)}, go={len(go_picks)})")

    # Plot style
    _configure_ieee_pdf_fonts()
    plt.rcParams.update(
        {
            "font.size": 13.0,
            "axes.titlesize": 13.4,
            "axes.labelsize": 17.4,
            "xtick.labelsize": 14.6,
            "ytick.labelsize": 14.6,
            "legend.fontsize": 12.4,
        }
    )

    fs_panel_note = 11.6
    fs_panel_title = 11.6
    fs_global_title = 20.0
    fs_global_legend = 16.0

    fig, axes = plt.subplots(3, 4, figsize=(17.6, 10.9), squeeze=False)
    axes = axes.ravel()

    a_min = float(min(np.nanmin(a_true[picks]), np.nanmin(a_pred[picks])) - 0.3)
    a_max = float(max(np.nanmax(a_true[picks]), np.nanmax(a_pred[picks])) + 0.3)
    a_min = min(a_min, -7.0)
    a_max = max(a_max, 2.0)
    d_min = float(min(np.nanmin(d_true[picks]), np.nanmin(d_pred[picks])) - 1.5)
    d_max = float(max(np.nanmax(d_true[picks]), np.nanmax(d_pred[picks])) + 1.5)

    first_twin = None
    for j, (ax, i) in enumerate(zip(axes, picks)):
        row = df.iloc[i]
        T = float(row["traj_duration_s"])
        t = np.linspace(0.0, max(T, 1e-3), a_true.shape[1])

        # acceleration curves
        ax.plot(t, a_true[i], color="#111827", lw=3.05, label="GT $a_t$")
        ax.plot(t, a_pred[i], color="#e76f51", lw=2.85, ls="--", label="Pred $a_t$")
        ax.axhline(0.0, color="0.45", lw=1.1, ls=":")
        ax.set_ylim(a_min, a_max)
        ax.set_xlim(0.0, 1.02 * T)
        ax.grid(alpha=0.22)

        # distance curves on twin axis
        axd = ax.twinx()
        if first_twin is None:
            first_twin = axd
        axd.plot(t, d_true[i], color="#1d3557", lw=2.45, alpha=0.95, label="GT $d(t)$")
        axd.plot(t, d_pred[i], color="#2a9d8f", lw=2.35, ls="-.", alpha=0.95, label="Pred $d(t)$")
        axd.axhline(0.0, color="#6b7280", lw=1.0, ls="--")
        if int(row["stop_decision"]) == 1:
            axd.axhspan(-0.5, 1.5, color="#f4a261", alpha=0.07)
        axd.set_ylim(d_min, d_max)

        # labels only on outer plots
        r, c = divmod(j, 4)
        if c == 0:
            ax.set_ylabel("$a_t$ (m/s$^2$)")
        else:
            ax.set_ylabel("")
        if c == 3:
            axd.set_ylabel("$d$ to light (m)")
        else:
            axd.set_yticklabels([])
        if r == 2:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xlabel("")

        pred_label = "GO" if int(row["decision_go_pred"]) == 1 else "STOP"
        ax.set_title(
            f"Pred={pred_label} | TTI={float(row['tti_s']):.2f}s | "
            f"$V_{{onset}}$={float(row['speed_at_yellow_kmh']):.1f} km/h | "
            f"$a_{{req}}$={float(row['a_req_mps2']):.2f}",
            pad=4.5,
            fontsize=fs_panel_title,
            fontweight="bold",
        )

        mae_a = float(np.mean(np.abs(a_pred[i] - a_true[i])))
        if int(row["stop_decision"]) == 1:
            d_gt, d_gt_exact = _first_stop_distance(d_true[i], v_true[i], STOP_EPS_MPS)
            d_pr, d_pr_exact = _first_stop_distance(d_pred[i], v_pred[i], STOP_EPS_MPS)
            txt = (
                f"MAE$_a$={mae_a:.2f} | "
                f"GT $d_f$={d_gt:.2f} m{'*' if not d_gt_exact else ''} | "
                f"Pred $d_f$={d_pr:.2f} m{'*' if not d_pr_exact else ''}"
            )
        else:
            v_gt0, ok_gt = _speed_at_distance_zero(d_true[i], v_true[i])
            v_pr0, ok_pr = _speed_at_distance_zero(d_pred[i], v_pred[i])
            gt_str = f"{v_gt0:.2f} m/s" if ok_gt and v_gt0 is not None else "NC"
            pr_str = f"{v_pr0:.2f} m/s" if ok_pr and v_pr0 is not None else "NC"
            txt = f"MAE$_a$={mae_a:.2f} | GT $v_f$={gt_str} | Pred $v_f$={pr_str}"

        ax.text(
            0.50,
            0.97,
            txt,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=fs_panel_note,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#c7c7c7", alpha=0.92),
        )

    # global legend
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = (first_twin.get_legend_handles_labels() if first_twin is not None else ([], []))
    uniq = {}
    for h, l in list(zip(h1, l1)) + list(zip(h2, l2)):
        if l not in uniq:
            uniq[l] = h
    fig.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc="upper center",
        ncol=4,
        frameon=True,
        fontsize=fs_global_legend,
        handlelength=2.9,
        handletextpad=0.85,
        columnspacing=1.85,
        borderpad=0.55,
        bbox_to_anchor=(0.5, 0.966),
    )
    fig.suptitle(
        "Representative Trajectory Examples (Rows 1-2: STOP, Row 3: GO)",
        y=0.984,
        fontsize=fs_global_title,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.957), w_pad=0.55, h_pad=0.85)

    out_png = out_dir / "transformer_correct12_3x4.png"
    out_pdf = out_dir / "transformer_correct12_3x4.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

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
    manifest.insert(0, "panel_order", np.arange(1, len(picks) + 1))
    manifest.to_csv(out_dir / "transformer_correct12_3x4_manifest.csv", index=False)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_dir / 'transformer_correct12_3x4_manifest.csv'}")


if __name__ == "__main__":
    main()

