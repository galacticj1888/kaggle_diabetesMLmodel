"""LightGBM naive baseline (no shift-awareness).

Runs 5-fold StratifiedKFold on the target only, logs per-fold AUC,
saves OOF/test predictions and a submission CSV.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"
DATA_DIR_KAGGLE = "/kaggle/input/playground-series-s5e12"


def in_kaggle() -> bool:
    return "KAGGLE_URL_BASE" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def detect_paths(train_path: str | None, test_path: str | None, sub_path: str | None) -> Tuple[str, str, str]:
    if train_path and test_path and sub_path:
        return train_path, test_path, sub_path
    if in_kaggle():
        base = DATA_DIR_KAGGLE
    else:
        base = "./data"
    return (
        train_path or os.path.join(base, "train.csv"),
        test_path or os.path.join(base, "test.csv"),
        sub_path or os.path.join(base, "sample_submission.csv"),
    )


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def identify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols: List[str] = []
    num_cols: List[str] = []
    for col in df.columns:
        if col in {TARGET_COL, ID_COL}:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_bool_dtype(
            df[col]
        ):
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return cat_cols, num_cols


def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    cat_cols, num_cols = identify_columns(train)
    # Align categorical categories across train+test
    for col in cat_cols:
        combined = pd.concat([train[col], test[col]], axis=0)
        combined = combined.astype("object").fillna("UNKNOWN")
        categories = pd.Categorical(combined).categories
        train[col] = pd.Categorical(train[col].astype("object").fillna("UNKNOWN"), categories=categories)
        test[col] = pd.Categorical(test[col].astype("object").fillna("UNKNOWN"), categories=categories)

    # Simple numeric imputation
    for col in num_cols:
        median = pd.concat([train[col], test[col]], axis=0).median()
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)

    feature_cols = [c for c in train.columns if c not in {TARGET_COL, ID_COL}]
    return train, test, feature_cols


def lightgbm_params(seed: int) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "verbose": -1,
        "seed": seed,
    }


def run_cv(train: pd.DataFrame, test: pd.DataFrame, feature_cols: List[str], cat_cols: List[str], seed: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    y = train[TARGET_COL].values
    X = train[feature_cols]
    X_test = test[feature_cols]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(train), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)
    best_iters: List[int] = []

    print(f"[CV] Starting 5-fold CV with {len(feature_cols)} features; categorical: {len(cat_cols)}")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        print(f"[Fold {fold}/5] train={len(tr_idx):,}, val={len(va_idx):,}")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = lgb.LGBMClassifier(**lightgbm_params(seed), n_estimators=2000)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=True)],
            categorical_feature=cat_cols,
        )

        va_pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
        oof[va_idx] = va_pred
        test_pred += model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1] / 5.0

        fold_auc = roc_auc_score(y_va, va_pred)
        best_iters.append(model.best_iteration_ or model.n_estimators_)
        print(f"[Fold {fold}/5] best_iter={model.best_iteration_} val_auc={fold_auc:.6f}")

    overall_auc = roc_auc_score(y, oof)
    print(f"[OOF] AUC={overall_auc:.6f}")
    print(f"[TEST] mean_pred={test_pred.mean():.4f} std={test_pred.std():.4f}")
    return oof, test_pred, best_iters


def save_outputs(oof: np.ndarray, test_pred: np.ndarray, submission_path: str, test: pd.DataFrame, sub_template_path: str) -> None:
    np.save("oof_naive.npy", oof.astype(np.float32))
    np.save("test_naive.npy", test_pred.astype(np.float32))

    submission = pd.read_csv(sub_template_path)
    submission[TARGET_COL] = test_pred
    submission.to_csv(submission_path, index=False)
    print(f"Saved: oof_naive.npy, test_naive.npy, {submission_path}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Naive LightGBM baseline (no shift awareness)")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--sample_submission_csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_submission", type=str, default="submission_naive.csv")
    args = parser.parse_args(argv)

    train_path, test_path, sub_path = detect_paths(args.train_csv, args.test_csv, args.sample_submission_csv)
    print(f"[Paths] train={train_path} test={test_path} submission_template={sub_path}")

    train_df, test_df = load_data(train_path, test_path)
    train_df, test_df, feature_cols = prepare_features(train_df, test_df)
    cat_cols, _ = identify_columns(train_df)

    start = time.time()
    oof, test_pred, _ = run_cv(train_df, test_df, feature_cols, cat_cols, args.seed)
    elapsed = time.time() - start
    print(f"[Runtime] {elapsed/60:.2f} minutes")

    save_outputs(oof, test_pred, args.output_submission, test_df, sub_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
