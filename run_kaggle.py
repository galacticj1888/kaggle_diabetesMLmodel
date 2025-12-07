"""Orchestrator for Kaggle notebooks: naive vs shift-aware baselines."""
from __future__ import annotations

import argparse
import subprocess
import sys


MODES = {"naive", "shift", "full"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Kaggle entrypoint for baselines")
    parser.add_argument("--mode", type=str, choices=sorted(MODES), required=True, help="Which pipeline to run")
    parser.add_argument("--use_weights", action="store_true", help="Apply domain weights in shift baseline")
    parser.add_argument("--use_p_test_feature", action="store_true", help="Append p_test as a feature in shift baseline")
    parser.add_argument("--stratify_domain", action="store_true", help="Stratify CV by label x domain_bin")
    parser.add_argument("--force_adv", action="store_true", help="Force recomputing adversarial artifacts in full mode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    mode = args.mode.lower()
    print(f"[RUN] Mode: {mode}")

    commands = []
    if mode == "naive":
        commands.append([sys.executable, "baseline_naive.py", "--seed", str(args.seed)])
    elif mode == "shift":
        commands.append(
            [
                sys.executable,
                "baseline_shift_aware.py",
                "--seed",
                str(args.seed),
            ]
            + (["--use_weights"] if args.use_weights else [])
            + (["--use_p_test_feature"] if args.use_p_test_feature else [])
            + (["--stratify_domain"] if args.stratify_domain else [])
        )
    elif mode == "full":
        commands.append(
            [
                sys.executable,
                "adversarial_mapping.py",
                "--seed",
                str(args.seed),
            ]
            + (["--force"] if args.force_adv else [])
        )
        commands.append(
            [
                sys.executable,
                "baseline_shift_aware.py",
                "--seed",
                str(args.seed),
                "--use_weights",
                "--use_p_test_feature",
                "--stratify_domain",
            ]
        )

    for cmd in commands:
        print(f"[RUN] Executing: {' '.join(cmd)}")
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[RUN] Command failed with exit code {ret}: {' '.join(cmd)}")
            return ret
    print("[RUN] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
