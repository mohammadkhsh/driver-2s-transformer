from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import trajectory_shared_utils as base


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "results" / "decision_mlp_stage1_ablation_64combo_stratified"
RUNS_ROOT = OUT_ROOT / "runs"
SPLITS_ROOT = OUT_ROOT / "splits"
PROGRESS_LOG = OUT_ROOT / "ablation_progress.log"
SUMMARY_CSV = OUT_ROOT / "ablation_summary.csv"

STAGE1_MODALITIES = ["speed", "dth", "tti", "a_req", "age", "gender"]
STAGE1_FEATURE_COLUMNS = {
    "speed": ("speed_at_yellow_kmh", "v0_kmh"),
    "dth": ("distance_threshold_m", "d_th_m"),
    "tti": ("tti_s", "tti_s"),
    "a_req": ("a_req_mps2", "a_req_mps2"),
    "age": ("driver_age", "age_years"),
    "gender": ("sex_female", "sex_female"),
}


def set_out_root(root: Path) -> None:
    global OUT_ROOT, RUNS_ROOT, SPLITS_ROOT, PROGRESS_LOG, SUMMARY_CSV
    OUT_ROOT = root
    RUNS_ROOT = OUT_ROOT / "runs"
    SPLITS_ROOT = OUT_ROOT / "splits"
    PROGRESS_LOG = OUT_ROOT / "ablation_progress.log"
    SUMMARY_CSV = OUT_ROOT / "ablation_summary.csv"


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)


def log_progress(msg: str) -> None:
    ensure_dirs()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with PROGRESS_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def append_summary(row: dict[str, Any]) -> None:
    ensure_dirs()
    df_row = pd.DataFrame([row])
    if SUMMARY_CSV.exists():
        df_row.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(SUMMARY_CSV, index=False)


def make_combo_list() -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for bits in itertools.product([0, 1], repeat=len(STAGE1_MODALITIES)):
        include = {m: bool(b) for m, b in zip(STAGE1_MODALITIES, bits)}
        key = "".join("1" if include[m] else "0" for m in STAGE1_MODALITIES)
        included = [m for m in STAGE1_MODALITIES if include[m]]
        excluded = [m for m in STAGE1_MODALITIES if not include[m]]
        combos.append(
            {
                "combo_key": key,
                "include": include,
                "included_modalities": included,
                "excluded_modalities": excluded,
            }
        )
    return combos


def get_participant_series(df: pd.DataFrame) -> pd.Series:
    if "participant_group" in df.columns:
        s = df["participant_group"].astype(str)
        s = s.replace({"nan": "", "None": ""})
        bad = s.str.strip() == ""
        if bad.any():
            s.loc[bad] = df.loc[bad, "participant_id_raw"].astype(str)
        return s
    if "participant_id_raw" in df.columns:
        return df["participant_id_raw"].astype(str)
    return df["source_file"].astype(str).str.extract(r"^([^_]+)", expand=False).fillna("unknown")


def make_split_indices(
    df: pd.DataFrame,
    y: np.ndarray,
    mode: str,
    train_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = len(df)
    idx_all = np.arange(n)
    meta: dict[str, Any] = {"split_mode": mode, "train_ratio": train_ratio, "seed": seed}
    if mode == "run_stratified":
        idx_train, idx_val = train_test_split(
            idx_all,
            train_size=train_ratio,
            random_state=seed,
            stratify=y,
            shuffle=True,
        )
        idx_train = np.sort(idx_train)
        idx_val = np.sort(idx_val)
        meta["participant_overlap"] = "allowed"
        return idx_train, idx_val, meta

    if mode != "participant_disjoint":
        raise ValueError(f"Unsupported split mode: {mode}")

    pser = get_participant_series(df)
    uniq = pd.Index(sorted(pser.unique().tolist()))
    rng = np.random.default_rng(seed)
    uniq_arr = uniq.to_numpy(dtype=object)
    rng.shuffle(uniq_arr)
    n_train_p = max(1, min(len(uniq_arr) - 1, int(round(train_ratio * len(uniq_arr)))))
    p_train = set(uniq_arr[:n_train_p].tolist())
    train_mask = pser.isin(p_train).to_numpy()
    idx_train = np.flatnonzero(train_mask)
    idx_val = np.flatnonzero(~train_mask)

    if len(np.unique(y[idx_train])) < 2 or len(np.unique(y[idx_val])) < 2:
        for k in range(1, 200):
            rng = np.random.default_rng(seed + k)
            uniq_arr = uniq.to_numpy(dtype=object)
            rng.shuffle(uniq_arr)
            p_train = set(uniq_arr[:n_train_p].tolist())
            train_mask = pser.isin(p_train).to_numpy()
            idx_train = np.flatnonzero(train_mask)
            idx_val = np.flatnonzero(~train_mask)
            if len(idx_train) > 0 and len(idx_val) > 0 and len(np.unique(y[idx_train])) >= 2 and len(np.unique(y[idx_val])) >= 2:
                meta["resample_attempt"] = k
                break

    meta["participant_overlap"] = "none"
    meta["n_participants_total"] = int(len(uniq))
    meta["n_participants_train"] = int(len(set(pser.iloc[idx_train].tolist())))
    meta["n_participants_val"] = int(len(set(pser.iloc[idx_val].tolist())))
    meta["train_participants"] = sorted(set(pser.iloc[idx_train].tolist()))
    meta["val_participants"] = sorted(set(pser.iloc[idx_val].tolist()))
    return np.sort(idx_train), np.sort(idx_val), meta


def build_stage1_matrix(df: pd.DataFrame, combo: dict[str, Any]) -> tuple[np.ndarray, list[str], bool]:
    cols = [STAGE1_FEATURE_COLUMNS[m][0] for m in STAGE1_MODALITIES if combo["include"][m]]
    names = [STAGE1_FEATURE_COLUMNS[m][1] for m in STAGE1_MODALITIES if combo["include"][m]]
    if not cols:
        # Bias-only baseline: no modalities visible. We feed a constant dummy feature so the MLP
        # can still train a bias term and produce a valid probability baseline.
        X = np.zeros((len(df), 1), dtype=np.float32)
        return X, [], True
    X = np.column_stack([df[c].to_numpy(dtype=float) for c in cols]).astype(np.float32)
    return X, names, False


def snapshot_source_code() -> None:
    shutil.copy2(ROOT / "decision_mlp_stage1_ablation.py", OUT_ROOT / "decision_mlp_stage1_ablation.py")
    shutil.copy2(ROOT / "trajectory_shared_utils.py", OUT_ROOT / "trajectory_shared_utils.py")


def run_ablation(seed: int, epochs: int, split_mode: str, train_ratio: float, force_cpu: bool) -> None:
    ensure_dirs()
    PROGRESS_LOG.write_text("", encoding="utf-8")
    if force_cpu:
        base.DEVICE = torch.device("cpu")
    log_progress(
        f"Stage-1 ablation started | device={base.DEVICE} | torch={torch.__version__} | "
        f"epochs={epochs} | split={split_mode} | train_ratio={train_ratio} | combos=64"
    )

    # Reuse the same extraction settings as the transformer pipeline.
    base.CFG.seed = seed
    base.CFG.decision_epochs = int(epochs)
    base.CFG.use_only_voluntary = True
    base.CFG.traj_points = 64
    base.CFG.a_clip_mps2 = 12.0
    base.CFG.extraction_logic_version = 3

    df, _traj_arrays, extraction_summary = base.prepare_modeling_dataframe(base.CFG)
    log_progress(f"Dataset ready | runs={len(df)} | extraction={extraction_summary.to_dict(orient='records')}")

    y = df["go_decision"].to_numpy(dtype=int)
    idx_train, idx_val, split_meta = make_split_indices(df, y, split_mode, train_ratio, seed)

    split_tag = f"{split_mode}_seed{seed}"
    split_dir = SPLITS_ROOT / split_tag
    split_dir.mkdir(parents=True, exist_ok=True)
    split_meta.update(
        {
            "n_total": int(len(df)),
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
            "stage1_modalities_order": STAGE1_MODALITIES,
            "stage1_feature_columns": {k: v[0] for k, v in STAGE1_FEATURE_COLUMNS.items()},
            "stage1_feature_names": {k: v[1] for k, v in STAGE1_FEATURE_COLUMNS.items()},
            "note": "Combo 000000 is evaluated as a bias-only baseline via a constant dummy feature (no observed modalities).",
        }
    )
    save_json(split_meta, split_dir / "split_meta.json")
    pd.DataFrame(
        {
            "row_index": np.arange(len(df)),
            "split": np.where(np.isin(np.arange(len(df)), idx_train), "train", "val"),
            "source_file": df["source_file"].to_numpy(),
            "go_decision": y,
            "stop_decision": df["stop_decision"].to_numpy(dtype=int),
        }
    ).to_csv(split_dir / "split_manifest.csv", index=False)

    protocol = {
        "stage1_modalities_order": STAGE1_MODALITIES,
        "stage1_feature_columns": {k: v[0] for k, v in STAGE1_FEATURE_COLUMNS.items()},
        "stage1_feature_names": {k: v[1] for k, v in STAGE1_FEATURE_COLUMNS.items()},
        "n_combinations": 64,
        "epochs": int(epochs),
        "split_mode": split_mode,
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "device": str(base.DEVICE),
        "note": "All 64 subsets over [speed,dth,tti,a_req,age,gender]. 000000 is a bias-only baseline.",
    }
    save_json(protocol, OUT_ROOT / "ablation_protocol.json")
    snapshot_source_code()

    combos = make_combo_list()
    runs_dir = RUNS_ROOT / split_tag
    runs_dir.mkdir(parents=True, exist_ok=True)
    t_split0 = time.time()
    n_done = 0
    for i, combo in enumerate(combos, start=1):
        combo_key = combo["combo_key"]
        run_id = f"{split_tag}__combo_{combo_key}"
        run_dir = runs_dir / run_id
        done_flag = run_dir / "DONE.ok"
        if done_flag.exists() and (run_dir / "decision_metrics.json").exists():
            n_done += 1
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "RUNNING.flag").write_text(datetime.now().isoformat(), encoding="utf-8")

        X_all, feature_names, is_bias_only = build_stage1_matrix(df, combo)
        X_train = X_all[idx_train]
        X_val = X_all[idx_val]
        y_train = y[idx_train]
        y_val = y[idx_val]

        run_meta = {
            "run_id": run_id,
            "combo_index_1based": i,
            "combo_total": len(combos),
            "combo_key": combo_key,
            "included_modalities": combo["included_modalities"],
            "excluded_modalities": combo["excluded_modalities"],
            "feature_names": feature_names,
            "n_features_visible": int(len(feature_names)),
            "is_bias_only_baseline": bool(is_bias_only),
            "split_meta": split_meta,
            "decision_config": {
                "epochs": int(base.CFG.decision_epochs),
                "batch_size": int(base.CFG.decision_batch_size),
                "lr": float(base.CFG.decision_lr),
                "hidden_dim": int(base.CFG.hidden_dim_decision),
                "weight_decay": float(base.CFG.weight_decay),
            },
        }
        save_json(run_meta, run_dir / "ablation_spec.json")

        t0 = time.time()
        log_progress(f"{run_id}: start ({i}/{len(combos)}) | include={combo['included_modalities']}")
        try:
            model, bundle, hist, scaler = base.fit_decision_model(X_train, y_train, X_val, y_val, base.CFG)
            metrics = bundle["metrics"]
            outputs = bundle["outputs"]
            # Override the hardcoded feature_names from base with the actual visible feature names.
            metrics["feature_names"] = feature_names
            metrics["is_bias_only_baseline"] = bool(is_bias_only)

            save_json({"decision_metrics": metrics}, run_dir / "decision_metrics.json")
            save_json({"decision_history": {k: [float(x) for x in v] for k, v in hist.items()}}, run_dir / "decision_history.json")
            torch.save(model.state_dict(), run_dir / "decision_mlp_state_dict.pt")
            np.savez_compressed(run_dir / "decision_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

            val_df = df.iloc[idx_val][
                ["source_file", "go_decision", "stop_decision", "speed_at_yellow_kmh", "distance_threshold_m", "tti_s", "a_req_mps2", "driver_age", "sex_female"]
            ].reset_index(drop=True).copy()
            val_df["decision_go_prob"] = outputs["val_prob"]
            val_df["decision_go_pred"] = outputs["val_pred"]
            val_df.to_csv(run_dir / "validation_predictions.csv", index=False)

            summary_row = {
                "run_id": run_id,
                "split_mode": split_mode,
                "combo_key": combo_key,
                "included_modalities": ",".join(combo["included_modalities"]),
                "excluded_modalities": ",".join(combo["excluded_modalities"]),
                "n_features_visible": int(len(feature_names)),
                "is_bias_only_baseline": bool(is_bias_only),
                "n_train": int(len(idx_train)),
                "n_val": int(len(idx_val)),
                "decision_val_acc": float(metrics["val"]["accuracy"]),
                "decision_val_precision": float(metrics["val"]["precision"]),
                "decision_val_recall": float(metrics["val"]["recall"]),
                "decision_val_f1": float(metrics["val"]["f1"]),
                "decision_val_brier": float(metrics["val"]["brier"]),
                "decision_val_auroc": float(metrics["val"]["auroc"]),
                "decision_train_acc": float(metrics["train"]["accuracy"]),
                "decision_train_f1": float(metrics["train"]["f1"]),
                "decision_train_auroc": float(metrics["train"]["auroc"]),
                "runtime_sec": float(time.time() - t0),
            }
            append_summary(summary_row)

            done_flag.write_text(datetime.now().isoformat(), encoding="utf-8")
            (run_dir / "RUNNING.flag").unlink(missing_ok=True)
            n_done += 1
            elapsed = time.time() - t_split0
            avg_sec = elapsed / max(n_done, 1)
            rem = (len(combos) - n_done) * avg_sec
            log_progress(
                f"{run_id}: done | val_acc={summary_row['decision_val_acc']:.3f} | "
                f"val_f1={summary_row['decision_val_f1']:.3f} | auroc={summary_row['decision_val_auroc']:.3f} | "
                f"runtime={summary_row['runtime_sec']:.1f}s | ETA={rem/60.0:.1f} min"
            )
        except Exception as e:
            (run_dir / "ERROR.txt").write_text(repr(e), encoding="utf-8")
            (run_dir / "RUNNING.flag").unlink(missing_ok=True)
            log_progress(f"{run_id}: FAILED | {type(e).__name__}: {e}")

    log_progress(f"Stage-1 ablation finished | completed={n_done}/{len(combos)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 decision MLP ablation over all 64 feature subsets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--split-mode", type=str, default="run_stratified", choices=["run_stratified", "participant_disjoint"])
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--out-dir", type=str, default="results/decision_mlp_stage1_ablation_64combo_stratified")
    parser.add_argument("--force-cpu", action="store_true", help="Run decision MLP ablations on CPU to avoid GPU contention")
    args = parser.parse_args()

    set_out_root(ROOT / Path(args.out_dir))
    run_ablation(
        seed=int(args.seed),
        epochs=int(args.epochs),
        split_mode=str(args.split_mode),
        train_ratio=float(args.train_ratio),
        force_cpu=bool(args.force_cpu),
    )


if __name__ == "__main__":
    main()
