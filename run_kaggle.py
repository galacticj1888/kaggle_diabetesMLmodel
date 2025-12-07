import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = {"naive", "shift", "full", "stack"}


def run_cmd(cmd):
    print(f"[RUN] Executing: {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    if ret != 0:
        raise SystemExit(ret)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Kaggle entrypoint for baselines")
    parser.add_argument("--mode", type=str, choices=sorted(MODES), required=True, help="Which pipeline to run")
    parser.add_argument("--use_weights", action="store_true", help="Apply domain weights in shift baseline")
    parser.add_argument("--use_p_test_feature", action="store_true", help="Append p_test as a feature in shift baseline")
    parser.add_argument("--stratify_domain", action="store_true", help="Stratify CV by label x domain_bin")
    parser.add_argument("--force_adv", action="store_true", help="Force recomputing adversarial artifacts in full mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Fast tiny runs for validation")
    parser.add_argument("--gpu", action="store_true", help="Force GPU mode for stacking run")
    args = parser.parse_args(argv)

    mode = args.mode.lower()
    print(f"[RUN] Mode: {mode}")

    if mode == "naive":
        cmd = [sys.executable, str(ROOT / "baseline_lgbm.py")]
        if args.smoke:
            cmd.append("--smoke")
        run_cmd(cmd)
    elif mode == "shift":
        cmd = [
            sys.executable,
            str(ROOT / "baseline_shift_aware.py"),
            "--seed",
            str(args.seed),
        ]
        if args.use_weights:
            cmd.append("--use_weights")
        if args.use_p_test_feature:
            cmd.append("--use_p_test_feature")
        if args.stratify_domain:
            cmd.append("--stratify_domain")
        if args.smoke:
            cmd.append("--smoke")
        run_cmd(cmd)
    elif mode == "full":
        cmd_adv = [
            sys.executable,
            str(ROOT / "adversarial_mapping.py"),
            "--seed",
            str(args.seed),
        ]
        if args.force_adv:
            cmd_adv.append("--force")
        run_cmd(cmd_adv)

        cmd_shift = [
            sys.executable,
            str(ROOT / "baseline_shift_aware.py"),
            "--seed",
            str(args.seed),
            "--use_weights",
            "--use_p_test_feature",
            "--stratify_domain",
        ]
        if args.smoke:
            cmd_shift.append("--smoke")
        run_cmd(cmd_shift)
    elif mode == "stack":
        cmd_stack = [sys.executable, str(ROOT / "run_full_stack.py")]
        if args.smoke:
            cmd_stack.append("--smoke")
        if args.gpu:
            cmd_stack.append("--gpu")
        run_cmd(cmd_stack)

    print("[RUN] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
