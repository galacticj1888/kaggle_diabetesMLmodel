"""
Project initializer for Kaggle diabetes competition.
Provides unified imports, environment detection, device setup, seeding,
and path configuration for both Kaggle and local execution.
"""

import os
import sys
import random
import json
import gc
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy import sparse

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import compose
from sklearn import impute
from sklearn import feature_extraction

import lightgbm as lgb
import xgboost as xgb
import catboost as cat

import torch

import tensorflow as tf
try:  # optional dependency
    import tensorflow_addons as tfa
except Exception:  # pragma: no cover - tfa may be absent in some envs
    tfa = None

# ---------------------------------------------------------------------------
# Kaggle environment detection
# ---------------------------------------------------------------------------
IN_KAGGLE = (
    "KAGGLE_URL_BASE" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
)

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
if IN_KAGGLE:
    DATA_DIR = "/kaggle/input/playground-series-s5e12"
else:
    DATA_DIR = "./data"

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

# ---------------------------------------------------------------------------
# Device detection (PyTorch + TensorFlow)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# TensorFlow strategy detection (TPU > GPU/CPU)
try:
    _tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(_tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(_tpu_resolver)
    STRATEGY = tf.distribute.TPUStrategy(_tpu_resolver)
    _tf_device = "tpu"
except Exception:
    STRATEGY = tf.distribute.get_strategy()
    # detect GPU visibility inside TF if available
    _tf_device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"

# ---------------------------------------------------------------------------
# Global seeding helper
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across common libraries."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

# ---------------------------------------------------------------------------
# Suppress warnings for cleaner notebooks/logs
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure project directories exist
# ---------------------------------------------------------------------------
for _dir in ["models", "experiments", "outputs"]:
    Path(_dir).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------
print(f"[INIT] Kaggle environment: {IN_KAGGLE}")
print(f"[INIT] Device: {DEVICE.type}")
print(f"[INIT] TensorFlow device: {_tf_device}")
print(f"[INIT] Data directory: {DATA_DIR}")

__all__ = [
    # stdlib
    "os", "sys", "random", "json", "gc", "time", "Path", "warnings",
    # core libs
    "np", "pd", "scipy", "stats", "sparse",
    # plotting
    "matplotlib", "plt", "sns",
    # sklearn modules
    "metrics", "model_selection", "preprocessing", "pipeline", "compose",
    "impute", "feature_extraction",
    # ml libs
    "lgb", "xgb", "cat",
    # torch / tf
    "torch", "tf", "tfa", "DEVICE", "STRATEGY",
    # env flags
    "IN_KAGGLE", "DATA_DIR", "TRAIN_PATH", "TEST_PATH", "SUB_PATH",
    # helpers
    "set_seed",
]
