from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("results/trajectory_transformer_ar")
PRED_CSV = BASE_DIR / "validation_predictions.csv"
ARR_NPZ = BASE_DIR / "validation_trajectory_arrays.npz"
OUT_DIR = BASE_DIR / "plots"


def _select_failed_go_cases(df: pd.DataFrame, n_total: int = 6) -> list[int]:
    """Select representative GT=GO cases with failed line crossing.

    We include both decision-error failures (Pred=STOP) and rollout failures (Pred=GO but d_f>0),
    then spread by final-position residual.
    """
    is_gt_go = df["go_decision"].astype(int) == 1
    cross_fail = df["ar_terminal_d_pred_cascade_m"].astype(float) > 0.0
    sub = df.loc[is_gt_go & cross_fail].copy()
    if sub.empty:
        return []

    # Split by predicted label to show both failure modes if present
    pred_go = sub["decision_go_pred"].astype(int) == 1
    sub_go = sub.loc[pred_go].copy()
    sub_stop = sub.loc[~pred_go].copy()

    # Target allocation (prefer 4 rollout-fail Pred=GO and 2 decision-fail Pred=STOP)
    n_stop = min(2, len(sub_stop))
    n_go = min(max(n_total - n_stop, 0), len(sub_go))
    # Fill remainder from whichever bucket has more
    remaining = n_total - (n_go + n_stop)
    if remaining > 0:
        extra_go = min(remaining, len(sub_go) - n_go)
        n_go += max(extra_go, 0)
        remaining = n_total - (n_go + n_stop)
    if remaining > 0:
        extra_stop = min(remaining, len(sub_stop) - n_stop)
        n_stop += max(extra_stop, 0)

    def _pick_spread(work: pd.DataFrame, n: int) -> list[int]:
        if n <= 0 or work.empty:
            return []
        if len(work) <= n:
            return work.index.astype(int).tolist()
        order = np.argsort(work["ar_terminal_d_pred_cascade_m"].to_numpy(dtype=float))
        idx = work.index.to_numpy(dtype=int)[order]
        take = np.linspace(0, len(idx) - 1, n, dtype=int)
        return [int(idx[k]) for k in take]

    picks = _pick_spread(sub_go, n_go) + _pick_spread(sub_stop, n_stop)
    # Final ordering by failure severity (small to large d_f)
    picks = sorted(set(picks), key=lambda i: float(df.loc[i, "ar_terminal_d_pred_cascade_m"]))
    return picks[:n_total]


def _speed_at_distance_zero(d: np.ndarray, v: np.ndarray) -> tuple[float | None, bool]:
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    for k in range(1, len(d)):
        d0, d1 = d[k - 1], d[k]
        if d0 == 0.0:
            return float(v[k - 1]), True
        if d1 == 0.0:
            return float(v[k]), True
        if (d0 > 0.0 and d1 < 0.0) or (d0 < 0.0 and d1 > 0.0):
            w = (0.0 - d0) / (d1 - d0)
            return float(v[k - 1] + w * (v[k] - v[k - 1])), True
    return None, False


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED_CSV)
    arr = np.load(ARR_NPZ)

    a_true = arr["a_true_val"]
    d_true = arr["d_true_val"]
    v_true = arr["v_true_val"]
    a_pred = arr["a_pred_cascade"]
    d_pred = arr["d_pred_traj_cascade"]
    v_pred = arr["v_pred_traj_cascade"]

    picks = _select_failed_go_cases(df, n_total=6)
    if not picks:
        raise RuntimeError("No GT=GO / cross-failure cases found.")

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.4,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 9.0,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.0), squeeze=False)
    axes = axes.ravel()

    a_min = float(min(np.nanmin(a_true[picks]), np.nanmin(a_pred[picks])) - 0.25)
    a_max = float(max(np.nanmax(a_true[picks]), np.nanmax(a_pred[picks])) + 0.25)
    d_min = float(min(np.nanmin(d_true[picks]), np.nanmin(d_pred[picks])) - 1.5)
    d_max = float(max(np.nanmax(d_true[picks]), np.nanmax(d_pred[picks])) + 1.5)
    a_min = min(a_min, -8.0)
    a_max = max(a_max, 2.5)

    first_twin = None
    for j, (ax, i) in enumerate(zip(axes, picks)):
        row = df.iloc[i]
        T = float(row["traj_duration_s"])
        t = np.linspace(0.0, max(T, 1e-3), a_true.shape[1])

        ax.plot(t, a_true[i], color="#111827", lw=2.35, label="GT $a_x$")
        ax.plot(t, a_pred[i], color="#e76f51", lw=2.15, ls="--", label="Pred $a_x$")
        ax.axhline(0.0, color="0.45", lw=0.9, ls=":")
        ax.set_ylim(a_min, a_max)
        ax.set_xlim(0.0, 1.02 * T)
        ax.grid(alpha=0.22)

        axd = ax.twinx()
        if first_twin is None:
            first_twin = axd
        axd.plot(t, d_true[i], color="#1d3557", lw=1.95, alpha=0.95, label="GT $d(t)$")
        axd.plot(t, d_pred[i], color="#2a9d8f", lw=1.90, ls="-.", alpha=0.95, label="Pred $d(t)$")
        axd.axhline(0.0, color="#6b7280", lw=0.85, ls="--")
        axd.set_ylim(d_min, d_max)

        r, c = divmod(j, 3)
        if c == 0:
            ax.set_ylabel("$a_x$ (m/s$^2$)")
        else:
            ax.set_ylabel("")
        if c == 2:
            axd.set_ylabel("$d$ to light (m)")
        else:
            axd.set_yticklabels([])
        if r == 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xlabel("")

        pred_lbl = "GO" if int(row["decision_go_pred"]) == 1 else "STOP"
        title = (
            f"GT=GO | Pred={pred_lbl} | TTI={float(row['tti_s']):.2f}s\n"
            f"$v_0$={float(row['speed_at_yellow_kmh']):.1f} km/h | "
            f"$a_{{req}}$={float(row['a_req_mps2']):.2f}"
        )
        ax.set_title(title, pad=4.5)

        mae_a = float(np.mean(np.abs(a_pred[i] - a_true[i])))
        d_f = float(row["ar_terminal_d_pred_cascade_m"])
        v_f = float(row["ar_terminal_v_pred_cascade_mps"])
        v_gt0, ok_gt = _speed_at_distance_zero(d_true[i], v_true[i])
        gt_cross = f"{v_gt0:.2f}" if (ok_gt and v_gt0 is not None) else "NC"
        txt = (
            f"MAE$_a$={mae_a:.2f} | $d_f$={d_f:.2f} m | $v_f$={v_f:.2f} m/s | "
            f"$p_{{go}}$={float(row['decision_go_prob']):.2f} | GT $v_{{d=0}}$={gt_cross} m/s"
        )
        ax.text(
            0.5,
            0.97,
            txt,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.4,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#c7c7c7", alpha=0.93),
        )

    # Shared legend
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
        bbox_to_anchor=(0.5, 0.975),
    )
    fig.suptitle(
        "GT=GO Cases with Failed Line-Cross Success (Cascaded Full Pipeline)",
        y=0.988,
        fontsize=12.4,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.935), w_pad=0.65, h_pad=0.9)

    out_png = OUT_DIR / "transformer_gocross_fail_2x3.png"
    out_pdf = OUT_DIR / "transformer_gocross_fail_2x3.pdf"
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    manifest_cols = [
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
        "ar_terminal_d_pred_cascade_m",
        "ar_terminal_v_pred_cascade_mps",
    ]
    manifest = df.loc[picks, manifest_cols].copy()
    manifest.insert(0, "panel_order", np.arange(1, len(manifest) + 1))
    manifest.to_csv(OUT_DIR / "transformer_gocross_fail_2x3_manifest.csv", index=False)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {OUT_DIR / 'transformer_gocross_fail_2x3_manifest.csv'}")


if __name__ == "__main__":
    main()

