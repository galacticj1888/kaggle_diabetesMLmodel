"""Project initializer for Kaggle Diabetes S5E12."""
import os
import random
import sys
import warnings
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import scipy
import sklearn
import tensorflow as tf
from sklearn import metrics, model_selection, preprocessing

# optional imports
try:
    import tensorflow_addons as tfa  # noqa: F401
except Exception:
    tfa = None
try:
    import torch
except Exception:
    torch = None
try:
    import lightgbm  # noqa: F401
except Exception:
    lightgbm = None
try:
    import xgboost  # noqa: F401
except Exception:
    xgboost = None
try:
    import catboost  # noqa: F401
except Exception:
    catboost = None

warnings.filterwarnings("ignore")

IN_KAGGLE = "KAGGLE_URL_BASE" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
DATA_DIR = Path("/kaggle/input/playground-series-s5e12") if IN_KAGGLE else Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUB_PATH = DATA_DIR / "sample_submission.csv"


def _detect_tpu():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        device = "tpu"
    except Exception:
        strategy = tf.distribute.get_strategy()
        device = "cpu"
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        elif torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    return device, strategy


DEVICE, STRATEGY = _detect_tpu()


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def print_env_report():
    def safe_version(pkg, name):
        if pkg is None:
            return f"{name}: MISSING"
        return f"{name}: {getattr(pkg, '__version__', 'unknown')}"

    print("[ENV] Kaggle environment:", IN_KAGGLE)
    print("[ENV] Data directory:", DATA_DIR)
    print("[ENV] Device:", DEVICE)

    # GPU visibility
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"[ENV] GPU: {result.stdout.strip()}")
        else:
            print("[ENV] GPU: nvidia-smi unavailable")
    except FileNotFoundError:
        print("[ENV] GPU: nvidia-smi not found")

    versions = [
        safe_version(np, "numpy"),
        safe_version(pd, "pandas"),
        safe_version(scipy, "scipy"),
        safe_version(sklearn, "sklearn"),
        safe_version(lightgbm, "lightgbm"),
        safe_version(xgboost, "xgboost"),
        safe_version(catboost, "catboost"),
        safe_version(tf, "tensorflow"),
    ]
    for v in versions:
        print("[ENV]", v)
    missing = [name for name, pkg in [("lightgbm", lightgbm), ("xgboost", xgboost), ("catboost", catboost)] if pkg is None]
    if missing:
        print("[ENV] Missing optional packages:", ", ".join(missing))


Path("models").mkdir(exist_ok=True)
Path("experiments").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)
Path("artifacts").mkdir(exist_ok=True)

print(f"[INIT] Kaggle environment: {IN_KAGGLE}")
print(f"[INIT] Device: {DEVICE}")
print(f"[INIT] Data directory: {DATA_DIR}")

__all__ = [
    "IN_KAGGLE",
    "DATA_DIR",
    "TRAIN_PATH",
    "TEST_PATH",
    "SUB_PATH",
    "DEVICE",
    "STRATEGY",
    "set_seed",
    "print_env_report",
    "np",
    "pd",
    "tf",
    "torch",
    "scipy",
    "sklearn",
]
