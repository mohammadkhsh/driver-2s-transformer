from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import trajectory_shared_utils as base


ROOT = Path(__file__).resolve().parent

FEATURES = {
    "speed": ("speed_at_yellow_kmh", "speed_at_yellow_kmh"),
    "dth": ("distance_threshold_m", "distance_threshold_m"),
    "tti": ("tti_s", "tti_s"),
    "a_req": ("a_req_mps2", "a_req_mps2"),
    "age": ("driver_age", "driver_age"),
    "sex": ("sex_female", "sex_female"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate the Stage 1 stop/go decision MLP.")
    parser.add_argument("--features", default="speed,a_req", help="Comma-separated feature keys.")
    parser.add_argument("--split-mode", choices=["run_stratified", "participant_disjoint"], default="run_stratified")
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/stage1_decision")
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def participant_series(df: pd.DataFrame) -> pd.Series:
    if "participant_group" in df.columns:
        return df["participant_group"].astype(str)
    if "participant_id_raw" in df.columns:
        return df["participant_id_raw"].astype(str)
    if "participant_id" in df.columns:
        return df["participant_id"].astype(str)
    return df["source_file"].astype(str).str.extract(r"^([^_]+)", expand=False).fillna("unknown")


def make_split(df: pd.DataFrame, y: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    idx_all = np.arange(len(df))
    if args.split_mode == "run_stratified":
        tr, va = train_test_split(
            idx_all,
            train_size=args.train_ratio,
            random_state=args.seed,
            stratify=y,
            shuffle=True,
        )
        return np.sort(tr), np.sort(va)

    rng = np.random.default_rng(args.seed)
    p = participant_series(df)
    unique_p = np.array(sorted(p.unique()))
    for _ in range(500):
        rng.shuffle(unique_p)
        n_train = max(1, int(round(len(unique_p) * args.train_ratio)))
        train_p = set(unique_p[:n_train])
        train_mask = p.isin(train_p).to_numpy()
        tr = idx_all[train_mask]
        va = idx_all[~train_mask]
        if len(tr) and len(va) and len(np.unique(y[tr])) == 2 and len(np.unique(y[va])) == 2:
            return np.sort(tr), np.sort(va)
    raise RuntimeError("Could not create a valid participant-disjoint split with both classes in each subset.")


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if args.force_cpu:
        base.DEVICE = torch.device("cpu")

    base.CFG.seed = args.seed
    base.CFG.train_ratio = args.train_ratio
    base.CFG.decision_epochs = args.epochs
    base.set_seed(args.seed)

    df, _, _ = base.prepare_modeling_dataframe(base.CFG)
    keys = [k.strip() for k in args.features.split(",") if k.strip()]
    unknown = [k for k in keys if k not in FEATURES]
    if unknown:
        raise ValueError(f"Unknown feature keys {unknown}. Valid keys are {sorted(FEATURES)}")
    cols = [FEATURES[k][0] for k in keys]
    names = [FEATURES[k][1] for k in keys]
    X = np.column_stack([df[c].to_numpy(dtype=float) for c in cols]).astype(np.float32)
    y = df["go_decision"].to_numpy(dtype=int)
    idx_train, idx_val = make_split(df, y, args)

    model, bundle, hist, scaler = base.fit_decision_model(
        X[idx_train], y[idx_train], X[idx_val], y[idx_val], base.CFG
    )
    metrics = bundle["metrics"]
    outputs = bundle["outputs"]

    base.save_json(
        {
            "features": names,
            "split_mode": args.split_mode,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "epochs": args.epochs,
            "metrics": metrics,
        },
        out_dir / "decision_metrics.json",
    )
    pred = df.iloc[idx_val][["source_file", "go_decision", "stop_decision", "speed_at_yellow_kmh", "distance_threshold_m", "tti_s", "a_req_mps2", "driver_age", "sex_female"]].copy()
    pred["p_go"] = outputs["val_prob"]
    pred["pred_go"] = outputs["val_pred"]
    pred.to_csv(out_dir / "validation_predictions.csv", index=False)
    torch.save(model.state_dict(), out_dir / "decision_mlp_state_dict.pt")
    np.savez_compressed(out_dir / "decision_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    base.plot_decision_results(y[idx_val], outputs["val_prob"], outputs["val_pred"], hist, plots_dir)

    lines = [
        "Stage 1 stop/go decision MLP",
        f"features={names}",
        f"split={args.split_mode}, train/val={len(idx_train)}/{len(idx_val)}",
        f"val accuracy={metrics['val']['accuracy']:.3f}",
        f"val precision={metrics['val']['precision']:.3f}",
        f"val recall={metrics['val']['recall']:.3f}",
        f"val F1={metrics['val']['f1']:.3f}",
        f"val AUROC={metrics['val']['auroc']:.3f}",
    ]
    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
