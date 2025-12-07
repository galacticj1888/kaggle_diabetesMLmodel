import os
import json
import random
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 42


def in_kaggle() -> bool:
    return "KAGGLE_URL_BASE" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def get_paths():
    if in_kaggle():
        data_dir = Path("/kaggle/input/playground-series-s5e12")
    else:
        data_dir = Path("./data")
    return {
        "data_dir": data_dir,
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }


def load_data(paths=None):
    paths = paths or get_paths()
    train_df = pd.read_csv(paths["train"])
    test_df = pd.read_csv(paths["test"])
    sample_df = pd.read_csv(paths["sample"])
    return train_df, test_df, sample_df


def split_cols(train_df: pd.DataFrame, target: str, id_col: str) -> Tuple[List[str], List[str]]:
    num_cols, cat_cols = [], []
    for col in train_df.columns:
        if col in (target, id_col):
            continue
        if pd.api.types.is_object_dtype(train_df[col]) or isinstance(train_df[col].dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(train_df[col]):
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return num_cols, cat_cols


def prepare_common(train_df: pd.DataFrame, test_df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
    train_proc = train_df.copy()
    test_proc = test_df.copy()

    # numeric fill with train medians
    for col in num_cols:
        median = train_proc[col].median()
        train_proc[col] = train_proc[col].fillna(median)
        test_proc[col] = test_proc[col].fillna(median)

    # categorical fill and align categories
    for col in cat_cols:
        train_proc[col] = train_proc[col].astype("object").fillna("UNKNOWN")
        test_proc[col] = test_proc[col].astype("object").fillna("UNKNOWN")
        cats = pd.Index(pd.concat([train_proc[col], test_proc[col]], axis=0).unique())
        train_proc[col] = pd.Categorical(train_proc[col], categories=cats)
        test_proc[col] = pd.Categorical(test_proc[col], categories=cats)

    return train_proc, test_proc, num_cols, cat_cols


def ensure_dirs():
    for d in [Path("artifacts"), Path("outputs"), Path("submissions")]:
        d.mkdir(parents=True, exist_ok=True)


def make_folds(y: np.ndarray, n_splits: int = 5, seed: int = SEED, save_path: str = "artifacts/folds.npy", force: bool = False) -> np.ndarray:
    ensure_dirs()
    fold_path = Path(save_path)
    if fold_path.exists() and not force:
        folds = np.load(fold_path)
        return folds

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.zeros_like(y, dtype=np.int8)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros_like(y), y)):
        folds[val_idx] = fold
    np.save(fold_path, folds)
    return folds


def save_preds(oof: np.ndarray, test_pred: np.ndarray, prefix: str, out_dir: str = "outputs/", submission_dir: str = "submissions/"):
    ensure_dirs()
    out_dir_path = Path(out_dir)
    sub_dir_path = Path(submission_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    sub_dir_path.mkdir(parents=True, exist_ok=True)

    oof = oof.astype(np.float32)
    test_pred = test_pred.astype(np.float32)

    np.save(out_dir_path / f"oof_{prefix}.npy", oof)
    np.save(out_dir_path / f"test_{prefix}.npy", test_pred)

    paths = get_paths()
    _, _, sample = load_data(paths)
    submission = sample.copy()
    submission["diagnosed_diabetes"] = test_pred
    submission.to_csv(sub_dir_path / f"submission_{prefix}.csv", index=False)
    print(f"Saved predictions to {out_dir_path} and submissions to {sub_dir_path}")
    return submission


def save_metadata(meta: dict, path: str):
    ensure_dirs()
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

__all__ = [
    "SEED",
    "in_kaggle",
    "get_paths",
    "load_data",
    "split_cols",
    "prepare_common",
    "make_folds",
    "save_preds",
    "save_metadata",
    "ensure_dirs",
]
