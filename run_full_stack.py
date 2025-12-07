import argparse
import os
import subprocess
from pathlib import Path

from src.utils import ensure_dirs

ROOT = Path(__file__).resolve().parent


def run_script(script_name: str, extra_args=None):
    script_path = ROOT / script_name
    cmd = ["python", str(script_path)]
    if extra_args:
        cmd += extra_args
    print(f"Running {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main(models, force=False, no_ensemble=False, smoke=False, gpu=False):
    ensure_dirs()
    if gpu:
        os.environ["FORCE_GPU"] = "1"
    model_map = {
        "lgbm": "baseline_lgbm.py",
        "xgb": "baseline_xgboost.py",
        "cat": "baseline_catboost.py",
    }

    for m in models:
        if m not in model_map:
            print(f"Unknown model {m}, skipping")
            continue
        args = []
        if force:
            args.append("--force")
        if smoke:
            args.append("--smoke")
        run_script(model_map[m], args)

    if not no_ensemble:
        run_script("ensemble.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="lgbm,xgb,cat", help="comma-separated list of models to run")
    parser.add_argument("--force", action="store_true", help="retrain even if outputs exist")
    parser.add_argument("--no-ensemble", action="store_true", help="skip ensemble step")
    parser.add_argument("--smoke", action="store_true", help="fast 2-fold tiny runs")
    parser.add_argument("--gpu", action="store_true", help="force GPU mode and fail if unavailable")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    main(models=models, force=args.force, no_ensemble=args.no_ensemble, smoke=args.smoke, gpu=args.gpu)
