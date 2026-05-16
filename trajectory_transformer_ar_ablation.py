from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import trajectory_shared_utils as base
import trajectory_transformer_ar as ar


ROOT = Path(__file__).resolve().parent
ABL_ROOT = ROOT / "results" / "trajectory_transformer_ar_ablation"
RUNS_ROOT = ABL_ROOT / "runs"
PLOTS_ROOT = ABL_ROOT / "plots"
PROGRESS_LOG = ABL_ROOT / "ablation_progress.log"
SUMMARY_CSV = ABL_ROOT / "ablation_summary.csv"
COMBO_PRESET = "curated_user"
SELECTED_SPLIT_MODES = ["run_stratified", "participant_disjoint"]
STAGE1_FEATURE_NAMES = ["speed_at_yellow_kmh", "a_req_mps2"]
FORCE_HIDDEN_MODALITIES: set[str] = set()


ABLATABLE_MODALITIES = [
    "speed",
    "dth",
    "tti",
    "a_req",
    "age",
    "gender",
    "decision_binary",
    "decision_confidence",
]

# Token feature order in trajectory_transformer_ar.train_transformer_model outputs
TOKEN_FEATURE_ORDER = [
    "v_t_mps",          # 0
    "d_t_m",            # 1
    "a_req_t_mps2",     # 2
    "TTI_t_s",          # 3
    "a_prev_mps2",      # 4 (always kept)
    "age",              # 5
    "decision_go_class",# 6
    "p_go",             # 7
    "sex_female",       # 8
    "v_target_mps",     # 9 (always kept)
    "d_center_m",       # 10 (always kept)
]

MODALITY_TO_TOKEN_IDX = {
    "speed": 0,
    "dth": 1,
    "a_req": 2,
    "tti": 3,
    "age": 5,
    "decision_binary": 6,
    "decision_confidence": 7,
    "gender": 8,
}

CURATED17_MODALITY_SETS: list[tuple[str, ...]] = [
    # User-curated compact set for paper reporting (11 combinations).
    ("dth", "speed"),
    ("dth", "speed", "a_req"),
    ("dth", "speed", "a_req", "tti"),
    ("dth", "speed", "a_req", "tti", "age"),
    ("dth", "speed", "a_req", "tti", "age", "decision_binary"),
    ("dth", "speed", "a_req", "tti", "age", "decision_binary", "decision_confidence"),
    ("dth", "speed", "a_req", "tti", "decision_binary", "decision_confidence"),
    ("dth", "speed", "tti", "decision_binary", "age"),
    ("dth", "speed", "age", "decision_binary", "decision_confidence"),
    ("a_req", "tti", "age"),
    ("a_req", "tti", "decision_binary"),
]


def set_ablation_root(root: Path) -> None:
    global ABL_ROOT, RUNS_ROOT, PLOTS_ROOT, PROGRESS_LOG, SUMMARY_CSV
    ABL_ROOT = root
    RUNS_ROOT = ABL_ROOT / "runs"
    PLOTS_ROOT = ABL_ROOT / "plots"
    PROGRESS_LOG = ABL_ROOT / "ablation_progress.log"
    SUMMARY_CSV = ABL_ROOT / "ablation_summary.csv"


def ensure_dirs() -> None:
    ABL_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)


def log_progress(msg: str) -> None:
    ensure_dirs()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with PROGRESS_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def snapshot_current_transformer_run() -> Path | None:
    src = ROOT / "results" / "trajectory_transformer_ar"
    if not src.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = ABL_ROOT / "baseline_snapshots" / f"baseline_noise_resolved_{stamp}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Copy a compact but complete snapshot (no cache duplication beyond metadata)
    keep = [
        "report.txt",
        "decision_metrics.json",
        "transformer_metrics.json",
        "validation_predictions.csv",
        "validation_trajectory_arrays.npz",
        "feature_scalers.npz",
        "decision_mlp_state_dict.pt",
        "trajectory_transformer_state_dict.pt",
        "split_manifest.csv",
        "transformer_examples_sample_manifest.csv",
        "run_progress.log",
    ]
    dst.mkdir(parents=True, exist_ok=True)
    for name in keep:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
    if (src / "plots").exists():
        shutil.copytree(src / "plots", dst / "plots", dirs_exist_ok=True)
    if (src / "cache").exists():
        cache_dst = dst / "cache_subset"
        cache_dst.mkdir(exist_ok=True)
        for name in ["trajectory_cache_info.json", "trajectory_meta.csv"]:
            p = src / "cache" / name
            if p.exists():
                shutil.copy2(p, cache_dst / name)
    shutil.copy2(ROOT / "trajectory_transformer_ar.py", dst / "trajectory_transformer_ar.py")
    return dst


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
    # Fallback to filename prefix
    return df["source_file"].astype(str).str.extract(r"^([^_]+)", expand=False).fillna("unknown")


def make_split_indices(df: pd.DataFrame, go: np.ndarray, mode: str, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = len(df)
    idx_all = np.arange(n)
    meta: dict[str, Any] = {"split_mode": mode, "train_ratio": train_ratio, "seed": seed}
    if mode == "run_stratified":
        idx_train, idx_val = train_test_split(
            idx_all,
            train_size=train_ratio,
            random_state=seed,
            stratify=go,
            shuffle=True,
        )
        idx_train = np.sort(idx_train)
        idx_val = np.sort(idx_val)
        meta["participant_overlap"] = "allowed"
        return idx_train, idx_val, meta

    if mode != "participant_disjoint":
        raise ValueError(f"Unknown split mode: {mode}")

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

    # Ensure both sets contain both classes if possible; try a few reseeds.
    if len(np.unique(go[idx_train])) < 2 or len(np.unique(go[idx_val])) < 2:
        for k in range(1, 200):
            rng = np.random.default_rng(seed + k)
            uniq_arr = uniq.to_numpy(dtype=object)
            rng.shuffle(uniq_arr)
            p_train = set(uniq_arr[:n_train_p].tolist())
            train_mask = pser.isin(p_train).to_numpy()
            idx_train = np.flatnonzero(train_mask)
            idx_val = np.flatnonzero(~train_mask)
            if len(idx_train) > 0 and len(idx_val) > 0 and len(np.unique(go[idx_train])) >= 2 and len(np.unique(go[idx_val])) >= 2:
                meta["resample_attempt"] = k
                break

    meta["participant_overlap"] = "none"
    meta["n_participants_total"] = int(len(uniq))
    meta["n_participants_train"] = int(len(set(pser.iloc[idx_train].tolist())))
    meta["n_participants_val"] = int(len(set(pser.iloc[idx_val].tolist())))
    meta["train_participants"] = sorted(set(pser.iloc[idx_train].tolist()))
    meta["val_participants"] = sorted(set(pser.iloc[idx_val].tolist()))
    return np.sort(idx_train), np.sort(idx_val), meta


def combo_iter() -> list[dict[str, Any]]:
    def _mk_combo(modalities: tuple[str, ...]) -> dict[str, Any]:
        include = {
            name: ((name in modalities) and (name not in FORCE_HIDDEN_MODALITIES))
            for name in ABLATABLE_MODALITIES
        }
        mask = np.ones(len(TOKEN_FEATURE_ORDER), dtype=np.float32)
        for m in ABLATABLE_MODALITIES:
            if not include[m]:
                mask[MODALITY_TO_TOKEN_IDX[m]] = 0.0
        key = "".join("1" if include[m] else "0" for m in ABLATABLE_MODALITIES)
        included = [m for m in ABLATABLE_MODALITIES if include[m]]
        excluded = [m for m in ABLATABLE_MODALITIES if not include[m]]
        return {
            "combo_key": key,
            "include": include,
            "included_modalities": included,
            "excluded_modalities": excluded,
            "token_feature_keep_mask": mask,
        }

    if COMBO_PRESET == "curated_user":
        combos = [_mk_combo(tuple(mods)) for mods in CURATED17_MODALITY_SETS]
        return combos

    combos: list[dict[str, Any]] = []
    for bits in itertools.product([0, 1], repeat=len(ABLATABLE_MODALITIES)):
        include_mods = tuple(name for name, bit in zip(ABLATABLE_MODALITIES, bits) if bit)
        combos.append(_mk_combo(include_mods))
    return combos


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


def run_ablation() -> None:
    ensure_dirs()
    PROGRESS_LOG.write_text("", encoding="utf-8")
    log_progress(
        f"Ablation run started | device={ar.DEVICE} | torch={torch.__version__} | "
        f"preset={COMBO_PRESET} | splits={SELECTED_SPLIT_MODES}"
    )

    snap = snapshot_current_transformer_run()
    if snap is not None:
        log_progress(f"Saved baseline snapshot: {snap}")

    # Reuse extraction/cache from AR transformer pipeline.
    cfg = ar.CFG
    ar.patch_base_for_extraction(cfg)
    base.CFG.use_only_voluntary = cfg.use_only_voluntary
    base.CFG.traj_points = cfg.traj_points
    base.CFG.a_clip_mps2 = cfg.a_clip_mps2
    base.CFG.extraction_logic_version = cfg.extraction_logic_version
    df, traj_arrays, extraction_summary = base.prepare_modeling_dataframe(base.CFG)
    log_progress(f"Dataset ready | runs={len(df)} | extraction={extraction_summary.to_dict(orient='records')}")

    # Stage-2 base features keep the original layout expected by trajectory_transformer_ar.
    base_X_stage2 = np.column_stack(
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
    # Stage-1 decision model uses only (v_onset, a_req).
    base_X_stage1 = np.column_stack(
        [
            df["speed_at_yellow_kmh"].to_numpy(dtype=float),
            df["a_req_mps2"].to_numpy(dtype=float),
        ]
    ).astype(np.float32)

    all_combos = combo_iter()
    log_progress(f"Prepared {len(all_combos)} modality combinations (preset={COMBO_PRESET}) over {ABLATABLE_MODALITIES}")
    if FORCE_HIDDEN_MODALITIES:
        log_progress(f"Forced hidden modalities across all combos: {sorted(FORCE_HIDDEN_MODALITIES)}")
    split_modes = list(SELECTED_SPLIT_MODES)
    est_runs = len(all_combos) * len(split_modes)
    log_progress(f"Planned runs: {est_runs} ({len(split_modes)} split modes x {len(all_combos)} combos).")

    split_root = ABL_ROOT / "splits"
    split_root.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "trajectory_transformer_ar.py", ABL_ROOT / "trajectory_transformer_ar.py")
    save_json(
        {
            "ablatable_modalities": ABLATABLE_MODALITIES,
            "token_feature_order": TOKEN_FEATURE_ORDER,
            "modality_to_token_idx": MODALITY_TO_TOKEN_IDX,
            "stage1_feature_names": STAGE1_FEATURE_NAMES,
            "forced_hidden_modalities": sorted(FORCE_HIDDEN_MODALITIES),
            "transformer_config": asdict(cfg),
            "note": "Ablations hide transformer token channels at all rollout stages. Closed-loop gate inputs use the same masked observations for speed/distance. If decision label/confidence are ablated, the closed-loop stop/go control branch is restricted to visible decision information only (or neutral unknown if both are hidden).",
        },
        ABL_ROOT / "ablation_protocol.json",
    )

    for split_mode in split_modes:
        log_progress(f"=== Split mode: {split_mode} ===")
        idx_train, idx_val, split_meta = make_split_indices(df, go, split_mode, cfg.train_ratio, cfg.seed)
        split_tag = f"{split_mode}_seed{cfg.seed}"
        split_dir = split_root / split_tag
        split_dir.mkdir(parents=True, exist_ok=True)
        save_json(split_meta, split_dir / "split_meta.json")
        pd.DataFrame(
            {
                "row_index": np.arange(len(df)),
                "split": np.where(np.isin(np.arange(len(df)), idx_train), "train", "val"),
                "source_file": df["source_file"].to_numpy(),
                "participant_group": get_participant_series(df).to_numpy(),
                "go_decision": go,
            }
        ).to_csv(split_dir / "split_manifest.csv", index=False)

        # Stage-1 decision model once per split (shared across all combos in that split).
        ar.log_progress = log_progress  # route epoch logs if any inside ar; base.fit_decision_model does not log much
        base.CFG.decision_epochs = cfg.decision_epochs
        base.CFG.decision_batch_size = cfg.decision_batch_size
        base.CFG.decision_lr = cfg.decision_lr
        base.CFG.hidden_dim_decision = cfg.hidden_dim_decision
        base.CFG.weight_decay = cfg.weight_decay

        X1_train, X1_val = base_X_stage1[idx_train], base_X_stage1[idx_val]
        y_train, y_val = go[idx_train], go[idx_val]
        log_progress(f"{split_tag}: training decision MLP once for all combos (stage1 inputs={STAGE1_FEATURE_NAMES})...")
        decision_model, decision_bundle, decision_hist, decision_scaler = base.fit_decision_model(X1_train, y_train, X1_val, y_val, base.CFG)
        dec_metrics = decision_bundle["metrics"]
        dec_out = decision_bundle["outputs"]
        y_train_prob = dec_out["train_prob"]
        y_train_pred = dec_out["train_pred"]
        y_val_prob = dec_out["val_prob"]
        y_val_pred = dec_out["val_pred"]

        torch.save(decision_model.state_dict(), split_dir / "decision_mlp_state_dict.pt")
        np.savez_compressed(split_dir / "decision_scaler.npz", mean=decision_scaler.mean_, scale=decision_scaler.scale_)
        save_json({"decision_metrics": dec_metrics}, split_dir / "decision_metrics.json")
        save_json({"decision_history": {k: [float(x) for x in v] for k, v in decision_hist.items()}}, split_dir / "decision_history.json")

        # Shared stage-2 arrays for this split
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

        X2_base_train = base_X_stage2[idx_train]
        X2_base_val = base_X_stage2[idx_val]
        X2_train = ar.make_stage2_features(X2_base_train, y_train_pred.astype(float), y_train_prob.astype(float), sex_female[idx_train])
        # No oracle decision inputs to Stage-2: use cascade inputs for both paths.
        X2_val_cascade = ar.make_stage2_features(X2_base_val, y_val_pred.astype(float), y_val_prob.astype(float), sex_female[idx_val])
        X2_val_oracle = X2_val_cascade.copy()

        split_runs_dir = RUNS_ROOT / split_tag
        split_runs_dir.mkdir(parents=True, exist_ok=True)
        n_done = 0
        n_total = len(all_combos)
        t_split0 = time.time()

        for i, combo in enumerate(all_combos, start=1):
            combo_key = combo["combo_key"]
            run_id = f"{split_tag}__combo_{combo_key}"
            run_dir = split_runs_dir / run_id
            done_flag = run_dir / "DONE.ok"
            if done_flag.exists() and (run_dir / "transformer_metrics.json").exists():
                n_done += 1
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "RUNNING.flag").write_text(datetime.now().isoformat(), encoding="utf-8")

            token_mask = combo["token_feature_keep_mask"].astype(np.float32)
            run_meta = {
                "run_id": run_id,
                "split_mode": split_mode,
                "split_tag": split_tag,
                "combo_index_1based": i,
                "combo_total": n_total,
                "combo_key": combo_key,
                "included_modalities": combo["included_modalities"],
                "excluded_modalities": combo["excluded_modalities"],
                "token_feature_order": TOKEN_FEATURE_ORDER,
                "token_feature_keep_mask": token_mask.tolist(),
                "note": "Transformer-visible channels are masked at every rollout step (including closed-loop feedback). Decision-conditioned control uses only visible decision information for this ablation; if decision channels are hidden, run type is not exposed to the transformer through the control branch.",
                "config": asdict(cfg),
            }
            save_json(run_meta, run_dir / "ablation_spec.json")

            log_progress(f"{run_id}: start ({i}/{n_total} in {split_mode}) | include={combo['included_modalities']}")
            t0 = time.time()
            try:
                # Route any internal training logs to the ablation progress file.
                ar.log_progress = log_progress
                tfm_model, tfm_bundle, tfm_hist, token_scaler = ar.train_transformer_model(
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
                    cfg,
                    token_feature_keep_mask=token_mask,
                )

                tfm_metrics = tfm_bundle["metrics"]
                tfm_out = tfm_bundle["outputs"]
                # Save detailed outputs for later plotting/comparison.
                save_json({"transformer_metrics": tfm_metrics}, run_dir / "transformer_metrics.json")
                save_json({"decision_metrics": dec_metrics}, run_dir / "decision_metrics.json")
                save_json({"split_meta": split_meta}, run_dir / "split_meta.json")
                save_json({"history": {k: [float(x) for x in v] for k, v in tfm_hist.items()}}, run_dir / "transformer_history.json")

                val_pred = df.iloc[idx_val][
                    ["source_file", "go_decision", "stop_decision", "speed_at_yellow_kmh", "distance_threshold_m", "tti_s", "a_req_mps2", "driver_age", "sex_female", "traj_duration_s", "traj_d_end_obs_m", "traj_v_end_obs_mps"]
                ].reset_index(drop=True).copy()
                val_pred["decision_go_prob"] = y_val_prob
                val_pred["decision_go_pred"] = y_val_pred
                val_pred["ar_terminal_d_pred_oracle_m"] = tfm_out["val_eval_oracle"]["d_pred"]
                val_pred["ar_terminal_d_pred_cascade_m"] = tfm_out["val_eval_cascade"]["d_pred"]
                val_pred["ar_terminal_v_pred_oracle_mps"] = tfm_out["val_eval_oracle"]["v_pred"]
                val_pred["ar_terminal_v_pred_cascade_mps"] = tfm_out["val_eval_cascade"]["v_pred"]
                val_pred["ar_mean_gate_oracle"] = np.mean(tfm_out["val_eval_oracle"]["gate_seq"], axis=1)
                val_pred["ar_jerk_hit_rate_oracle"] = np.mean(tfm_out["val_eval_oracle"]["jerk_hit_seq"], axis=1)
                val_pred.to_csv(run_dir / "validation_predictions.csv", index=False)

                np.savez_compressed(
                    run_dir / "validation_trajectory_arrays.npz",
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
                    token_feature_keep_mask=token_mask,
                )

                np.savez_compressed(
                    run_dir / "feature_scalers.npz",
                    decision_mean=decision_scaler.mean_,
                    decision_scale=decision_scaler.scale_,
                    token_mean=token_scaler.mean_,
                    token_scale=token_scaler.scale_,
                    token_feature_keep_mask=token_mask,
                )
                torch.save(decision_model.state_dict(), run_dir / "decision_mlp_state_dict.pt")
                torch.save(tfm_model.state_dict(), run_dir / "trajectory_transformer_state_dict.pt")

                # Compact report text for quick browsing.
                report_lines = [
                    f"run_id: {run_id}",
                    f"split_mode: {split_mode}",
                    f"combo_key: {combo_key}",
                    f"included_modalities: {combo['included_modalities']}",
                    f"excluded_modalities: {combo['excluded_modalities']}",
                    "primary_eval: cascade",
                    "",
                    "Decision val metrics:",
                    json.dumps(dec_metrics.get("val", {}), indent=2),
                    "",
                    "Transformer val (cascade, PRIMARY):",
                    json.dumps(tfm_metrics.get("val_cascaded_decision_input", {}), indent=2),
                    "",
                    "Transformer val (oracle, diagnostic only):",
                    json.dumps(tfm_metrics.get("val_oracle_decision_input", {}), indent=2),
                ]
                (run_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")

                summary_row = {
                    "run_id": run_id,
                    "split_mode": split_mode,
                    "combo_key": combo_key,
                    "included_modalities": ",".join(combo["included_modalities"]),
                    "excluded_modalities": ",".join(combo["excluded_modalities"]),
                    "n_train": int(len(idx_train)),
                    "n_val": int(len(idx_val)),
                    "decision_val_acc": float(dec_metrics["val"]["accuracy"]),
                    "decision_val_f1": float(dec_metrics["val"]["f1"]),
                    "decision_val_auroc": float(dec_metrics["val"]["auroc"]),
                    "primary_eval": "cascade",
                    "ar_val_primary_a_mae": float(tfm_metrics["val_cascaded_decision_input"]["traj_mae_mps2"]),
                    "ar_val_primary_d_end_mae": float(tfm_metrics["val_cascaded_decision_input"]["terminal_distance_mae_m"]),
                    "ar_val_primary_stop_band": float(tfm_metrics["val_cascaded_decision_input"].get("stop_terminal_in_band_rate", np.nan)),
                    "ar_val_primary_go_cross": float(tfm_metrics["val_cascaded_decision_input"].get("go_crossed_light_rate", np.nan)),
                    "ar_val_oracle_a_mae": float(tfm_metrics["val_oracle_decision_input"]["traj_mae_mps2"]),
                    "ar_val_oracle_a_rmse": float(tfm_metrics["val_oracle_decision_input"]["traj_rmse_mps2"]),
                    "ar_val_oracle_dcurve_mae": float(tfm_metrics["val_oracle_decision_input"]["distance_curve_mae_m"]),
                    "ar_val_oracle_dcurve_rmse": float(tfm_metrics["val_oracle_decision_input"]["distance_curve_rmse_m"]),
                    "ar_val_oracle_vcurve_mae": float(tfm_metrics["val_oracle_decision_input"]["speed_curve_mae_mps"]),
                    "ar_val_oracle_vcurve_rmse": float(tfm_metrics["val_oracle_decision_input"]["speed_curve_rmse_mps"]),
                    "ar_val_oracle_d_end_mae": float(tfm_metrics["val_oracle_decision_input"]["terminal_distance_mae_m"]),
                    "ar_val_oracle_v_end_mae": float(tfm_metrics["val_oracle_decision_input"]["terminal_speed_mae_mps"]),
                    "ar_val_oracle_stop_band": float(tfm_metrics["val_oracle_decision_input"].get("stop_terminal_in_band_rate", np.nan)),
                    "ar_val_oracle_go_cross": float(tfm_metrics["val_oracle_decision_input"].get("go_crossed_light_rate", np.nan)),
                    "ar_val_oracle_jerk_hit": float(tfm_metrics["val_oracle_decision_input"].get("jerk_limit_hit_rate", np.nan)),
                    "ar_val_oracle_actuator_hit": float(tfm_metrics["val_oracle_decision_input"].get("actuator_clamp_hit_rate", np.nan)),
                    "ar_val_cascade_a_mae": float(tfm_metrics["val_cascaded_decision_input"]["traj_mae_mps2"]),
                    "ar_val_cascade_a_rmse": float(tfm_metrics["val_cascaded_decision_input"]["traj_rmse_mps2"]),
                    "ar_val_cascade_dcurve_mae": float(tfm_metrics["val_cascaded_decision_input"]["distance_curve_mae_m"]),
                    "ar_val_cascade_dcurve_rmse": float(tfm_metrics["val_cascaded_decision_input"]["distance_curve_rmse_m"]),
                    "ar_val_cascade_vcurve_mae": float(tfm_metrics["val_cascaded_decision_input"]["speed_curve_mae_mps"]),
                    "ar_val_cascade_vcurve_rmse": float(tfm_metrics["val_cascaded_decision_input"]["speed_curve_rmse_mps"]),
                    "ar_val_cascade_d_end_mae": float(tfm_metrics["val_cascaded_decision_input"]["terminal_distance_mae_m"]),
                    "ar_val_cascade_v_end_mae": float(tfm_metrics["val_cascaded_decision_input"]["terminal_speed_mae_mps"]),
                    "ar_val_cascade_stop_band": float(tfm_metrics["val_cascaded_decision_input"].get("stop_terminal_in_band_rate", np.nan)),
                    "ar_val_cascade_go_cross": float(tfm_metrics["val_cascaded_decision_input"].get("go_crossed_light_rate", np.nan)),
                    "runtime_sec": float(time.time() - t0),
                }
                append_summary(summary_row)

                done_flag.write_text(datetime.now().isoformat(), encoding="utf-8")
                (run_dir / "RUNNING.flag").unlink(missing_ok=True)
                n_done += 1
                elapsed_split = time.time() - t_split0
                avg_sec = elapsed_split / max(n_done, 1)
                remaining = (n_total - n_done) * avg_sec
                log_progress(
                    f"{run_id}: done | cascade aMAE={summary_row['ar_val_cascade_a_mae']:.3f} | dEndMAE={summary_row['ar_val_cascade_d_end_mae']:.3f} | "
                    f"runtime={summary_row['runtime_sec'] / 60.0:.1f} min | split ETA={remaining / 3600.0:.1f} h"
                )
            except Exception as e:
                (run_dir / "ERROR.txt").write_text(repr(e), encoding="utf-8")
                (run_dir / "RUNNING.flag").unlink(missing_ok=True)
                log_progress(f"{run_id}: FAILED | {type(e).__name__}: {e}")
            finally:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        log_progress(f"=== Split mode completed: {split_mode} | completed {n_done}/{n_total} combos ===")

    log_progress("Ablation run finished.")


def main() -> None:
    global COMBO_PRESET, SELECTED_SPLIT_MODES, FORCE_HIDDEN_MODALITIES
    parser = argparse.ArgumentParser(description="Full AR Transformer modality ablation runner (resumable)")
    parser.add_argument("--seed", type=int, default=ar.CFG.seed)
    parser.add_argument("--epochs", type=int, default=ar.CFG.ar_epochs, help="AR transformer epochs per ablation run")
    parser.add_argument("--decision-epochs", type=int, default=ar.CFG.decision_epochs)
    parser.add_argument("--max-combos", type=int, default=0, help="0 means all combinations; otherwise run only the first N combinations for smoke test")
    parser.add_argument("--combo-preset", type=str, default=COMBO_PRESET, choices=["curated_user", "all"])
    parser.add_argument(
        "--split-modes",
        type=str,
        default="run_stratified,participant_disjoint",
        help="Comma-separated subset of: run_stratified, participant_disjoint",
    )
    parser.add_argument(
        "--force-hide-modalities",
        type=str,
        default="",
        help="Comma-separated modalities to hide in every ablation combo (e.g., gender,age)",
    )
    parser.add_argument("--out-dir", type=str, default="results/trajectory_transformer_ar_ablation_curated")
    args = parser.parse_args()

    COMBO_PRESET = str(args.combo_preset)
    split_modes = [s.strip() for s in str(args.split_modes).split(",") if s.strip()]
    valid_split_modes = {"run_stratified", "participant_disjoint"}
    bad_split_modes = [s for s in split_modes if s not in valid_split_modes]
    if not split_modes or bad_split_modes:
        raise ValueError(
            f"Invalid --split-modes={args.split_modes!r}. "
            f"Allowed values: {sorted(valid_split_modes)}"
        )
    SELECTED_SPLIT_MODES = split_modes
    forced = {s.strip() for s in str(args.force_hide_modalities).split(",") if s.strip()}
    bad_forced = sorted(forced - set(ABLATABLE_MODALITIES))
    if bad_forced:
        raise ValueError(
            f"Invalid --force-hide-modalities entries: {bad_forced}. "
            f"Allowed values: {sorted(ABLATABLE_MODALITIES)}"
        )
    FORCE_HIDDEN_MODALITIES = forced
    set_ablation_root(ROOT / Path(args.out_dir))
    ar.CFG.seed = int(args.seed)
    ar.CFG.ar_epochs = int(args.epochs)
    ar.CFG.decision_epochs = int(args.decision_epochs)

    if args.max_combos and args.max_combos > 0:
        # Monkeypatch combo_iter output limit for smoke tests
        original_combo_iter = combo_iter

        def _limited() -> list[dict[str, Any]]:
            return original_combo_iter()[: int(args.max_combos)]

        globals()["combo_iter"] = _limited

    run_ablation()


if __name__ == "__main__":
    main()
