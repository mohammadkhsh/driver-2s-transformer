from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


STAMP = "20260226-200444"
ROOT = Path(__file__).resolve().parent
RERUN_ROOT = ROOT / "results" / f"stop_aligned_eval_reruns_{STAMP}"
CONDA_ENV = "riccardo"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _backup_baselines() -> None:
    for name in ["trajectory_mixture_primitives", "trajectory_neural_ode", "trajectory_transformer_ar"]:
        src = ROOT / "results" / name
        dst = ROOT / "results" / f"{name}_preStopAlignedEval_{STAMP}"
        if src.exists() and not dst.exists():
            try:
                src.rename(dst)
            except PermissionError:
                shutil.copytree(src, dst)


def _write_plus11_runner() -> Path:
    p = RERUN_ROOT / "run_plus_gender11_stopaligned.py"
    code = f"""from pathlib import Path
import trajectory_transformer_ar_ablation as abl

WANTED_KEYS = {{
    '11000100','11010100','11110100','11111100','11111110','11111111',
    '11110111','11101110','11001111','00111100','00110110'
}}
ORDER = ['11000100','11010100','11110100','11111100','11111110','11111111','11110111','11101110','11001111','00111100','00110110']
ROOT = Path(__file__).resolve().parents[2]
abl.COMBO_PRESET = 'all'
abl.SELECTED_SPLIT_MODES = ['run_stratified']
abl.set_ablation_root(ROOT / 'results' / 'trajectory_transformer_ar_ablation_curated_true_mask_stratified_e100_genderS1_cascadeOnly_plusGender11_stopAlignedEval')
abl.ar.CFG.seed = 42
abl.ar.CFG.ar_epochs = 100
abl.ar.CFG.decision_epochs = 80
_orig = abl.combo_iter

def _filtered_combo_iter():
    combos = [c for c in _orig() if str(c.get('combo_key','')).zfill(8) in WANTED_KEYS]
    by_key = {{str(c['combo_key']).zfill(8): c for c in combos}}
    return [by_key[k] for k in ORDER if k in by_key]

abl.combo_iter = _filtered_combo_iter
abl.run_ablation()
"""
    _write_text(p, code)
    return p


def _write_gpu_sequence_cmd(plus_runner: Path) -> Path:
    gpu_cmd = RERUN_ROOT / "run_gpu_sequence.cmd"
    gpu_master = RERUN_ROOT / "gpu_master.log"
    lines = [
        "@echo off",
        f'cd /d "{ROOT}"',
        f'echo %date% %time% START trajectory_transformer_ar>>"{gpu_master}"',
        f'conda run -n {CONDA_ENV} python trajectory_transformer_ar.py 1>>"{RERUN_ROOT / "transformer_ar_stdout.log"}" 2>>"{RERUN_ROOT / "transformer_ar_stderr.log"}"',
        "if errorlevel 1 goto :fail",
        f'echo %date% %time% DONE trajectory_transformer_ar>>"{gpu_master}"',
        f'echo %date% %time% START ablation_curated_base>>"{gpu_master}"',
        f'conda run -n {CONDA_ENV} python trajectory_transformer_ar_ablation.py --epochs 100 --decision-epochs 80 --combo-preset curated_user --split-modes run_stratified --out-dir results/trajectory_transformer_ar_ablation_curated_true_mask_stratified_e100_genderS1_cascadeOnly_stopAlignedEval 1>>"{RERUN_ROOT / "ablation_base_stdout.log"}" 2>>"{RERUN_ROOT / "ablation_base_stderr.log"}"',
        "if errorlevel 1 goto :fail",
        f'echo %date% %time% DONE ablation_curated_base>>"{gpu_master}"',
        f'echo %date% %time% START ablation_plus_gender11>>"{gpu_master}"',
        f'conda run -n {CONDA_ENV} python "{plus_runner}" 1>>"{RERUN_ROOT / "ablation_plus11_stdout.log"}" 2>>"{RERUN_ROOT / "ablation_plus11_stderr.log"}"',
        "if errorlevel 1 goto :fail",
        f'echo %date% %time% DONE ablation_plus_gender11>>"{gpu_master}"',
        f'echo %date% %time% ALL_DONE>>"{gpu_master}"',
        "goto :eof",
        ":fail",
        f'echo %date% %time% FAIL errorlevel=%errorlevel%>>"{gpu_master}"',
        "exit /b %errorlevel%",
    ]
    _write_text(gpu_cmd, "\n".join(lines))
    return gpu_cmd


def _start_background_processes(gpu_cmd: Path) -> dict[str, int]:
    RERUN_ROOT.mkdir(parents=True, exist_ok=True)
    mix = subprocess.Popen(
        f'conda run -n {CONDA_ENV} python trajectory_mixture_primitives.py',
        cwd=str(ROOT),
        stdout=open(RERUN_ROOT / "mixture_stdout.log", "ab"),
        stderr=open(RERUN_ROOT / "mixture_stderr.log", "ab"),
        shell=True,
    )
    node = subprocess.Popen(
        f'conda run -n {CONDA_ENV} python trajectory_neural_ode.py',
        cwd=str(ROOT),
        stdout=open(RERUN_ROOT / "node_stdout.log", "ab"),
        stderr=open(RERUN_ROOT / "node_stderr.log", "ab"),
        shell=True,
    )
    gpu = subprocess.Popen(
        [str(gpu_cmd)],
        cwd=str(ROOT),
        stdout=open(RERUN_ROOT / "gpu_pipeline_stdout.log", "ab"),
        stderr=open(RERUN_ROOT / "gpu_pipeline_stderr.log", "ab"),
        shell=True,
    )
    return {"mixture": mix.pid, "node": node.pid, "gpu_pipeline": gpu.pid}


def main() -> int:
    RERUN_ROOT.mkdir(parents=True, exist_ok=True)
    _backup_baselines()
    plus_runner = _write_plus11_runner()
    gpu_cmd = _write_gpu_sequence_cmd(plus_runner)
    pids = _start_background_processes(gpu_cmd)
    _write_text(RERUN_ROOT / "pids.json", json.dumps(pids, indent=2))
    print(f"Rerun root: {RERUN_ROOT}")
    print(json.dumps(pids, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
