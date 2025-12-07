"""Compute adversarial domain mapping artifacts for shift-aware training."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"
DATA_DIR_KAGGLE = "/kaggle/input/playground-series-s5e12"
ARTIFACT_PATH = Path("artifacts/adversarial_mapping.parquet")
META_PATH = Path("artifacts/adversarial_metadata.json")


def in_kaggle() -> bool:
    return "KAGGLE_URL_BASE" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def detect_paths(train_path: str | None, test_path: str | None) -> Tuple[str, str]:
    if train_path and test_path:
        return train_path, test_path
    base = DATA_DIR_KAGGLE if in_kaggle() else "./data"
    return (
        train_path or os.path.join(base, "train.csv"),
        test_path or os.path.join(base, "test.csv"),
    )


def identify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols: List[str] = []
    num_cols: List[str] = []
    for col in df.columns:
        if col in {TARGET_COL, ID_COL}:
            continue
        if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(
            df[col]
        ):
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return cat_cols, num_cols


def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    cat_cols, num_cols = identify_columns(train)
    for col in cat_cols:
        combined = pd.concat([train[col], test[col]], axis=0)
        combined = combined.astype("object").fillna("UNKNOWN")
        categories = pd.Categorical(combined).categories
        train[col] = pd.Categorical(train[col].astype("object").fillna("UNKNOWN"), categories=categories)
        test[col] = pd.Categorical(test[col].astype("object").fillna("UNKNOWN"), categories=categories)

    for col in num_cols:
        median = pd.concat([train[col], test[col]], axis=0).median()
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)

    feature_cols = [c for c in train.columns if c not in {TARGET_COL, ID_COL}]
    return train, test, feature_cols, cat_cols


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


def compute_adversarial_mapping(train: pd.DataFrame, test: pd.DataFrame, feature_cols: List[str], cat_cols: List[str], seed: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    X_train = train[feature_cols]
    X_test = test[feature_cols]

    # Domain labels: train=0, test=1
    X_dom = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_dom = np.concatenate([
        np.zeros(len(train), dtype=int),
        np.ones(len(test), dtype=int),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_dom), dtype=np.float32)
    best_iters: List[int] = []

    print(f"[ADV] Training adversarial classifier with {len(feature_cols)} features; categorical={len(cat_cols)}")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_dom, y_dom), 1):
        print(f"[ADV Fold {fold}/5] train={len(tr_idx):,} val={len(va_idx):,}")
        X_tr, X_va = X_dom.iloc[tr_idx], X_dom.iloc[va_idx]
        y_tr, y_va = y_dom[tr_idx], y_dom[va_idx]

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
        fold_auc = roc_auc_score(y_va, va_pred)
        best_iters.append(model.best_iteration_ or model.n_estimators_)
        print(f"[ADV Fold {fold}/5] best_iter={model.best_iteration_} val_auc={fold_auc:.6f}")

    overall_auc = roc_auc_score(y_dom, oof)
    print(f"[ADV] OOF AUC={overall_auc:.6f}")

    avg_iter = int(np.mean(best_iters)) if best_iters else 2000
    print(f"[ADV] Training final model on all data with n_estimators={avg_iter}")
    final_model = lgb.LGBMClassifier(**lightgbm_params(seed), n_estimators=avg_iter)
    final_model.fit(X_dom, y_dom, categorical_feature=cat_cols)
    test_pred_full = final_model.predict_proba(X_test, num_iteration=final_model.best_iteration_)[:, 1]

    train_oof = oof[: len(train)]
    test_oof = oof[len(train) :]

    # Prefer cross-validated predictions for stability; fallback to full model where needed
    if np.isnan(test_oof).any():
        test_oof = test_pred_full
    else:
        # blend CV and full-model predictions slightly to reduce variance
        test_oof = 0.7 * test_oof + 0.3 * test_pred_full

    return train_oof, test_oof, best_iters


def save_artifacts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_p: np.ndarray,
    test_p: np.ndarray,
    weight_clip: Tuple[float, float],
    n_bins: int,
    best_iters: List[int],
    adversarial_auc: float,
) -> dict:
    eps = 1e-6
    weights = (train_p + eps) / (1.0 - train_p + eps)
    lo, hi = weight_clip
    weights = np.clip(weights, lo, hi)
    weights = weights / weights.mean()

    bins, bin_edges = pd.qcut(train_p, q=n_bins, labels=False, retbins=True, duplicates="drop")
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    test_bins = pd.cut(test_p, bins=bin_edges, labels=False, include_lowest=True)

    # Build combined mapping
    mapping_rows = []
    for idx, (row_id, p) in enumerate(zip(train[ID_COL].values, train_p)):
        mapping_rows.append(
            {
                ID_COL: int(row_id),
                "p_test": float(p),
                "domain_weight": float(weights[idx]),
                "domain_bin": int(bins[idx]) if not pd.isna(bins[idx]) else -1,
                "is_train": True,
            }
        )
    for idx, (row_id, p) in enumerate(zip(test[ID_COL].values, test_p)):
        bin_val = test_bins[idx]
        mapping_rows.append(
            {
                ID_COL: int(row_id),
                "p_test": float(p),
                "domain_weight": 1.0,
                "domain_bin": int(bin_val) if not pd.isna(bin_val) else -1,
                "is_train": False,
            }
        )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_map = pd.DataFrame(mapping_rows)
    df_map.to_parquet(ARTIFACT_PATH, index=False)
    print(f"[ADV] Saved mapping -> {ARTIFACT_PATH}")

    # Stratification sanity: target x domain_bin counts
    strat_labels = train[TARGET_COL].astype(str) + "_" + bins.astype(str)
    min_stratum = strat_labels.value_counts().min()

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adversarial_oof_auc": adversarial_auc,
        "weight_clip_low": float(lo),
        "weight_clip_high": float(hi),
        "n_bins": n_bins,
        "min_stratum_count": int(min_stratum),
        "avg_best_iteration": float(np.mean(best_iters)) if best_iters else None,
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[ADV] Saved metadata -> {META_PATH}")
    print("[ADV] Weight summary:", pd.Series(weights).describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
    print(f"[ADV] min_stratum_count={min_stratum}")
    return metadata


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute adversarial mapping artifacts")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Force recomputation even if artifacts exist")
    parser.add_argument("--weight_clip_low", type=float, default=0.01, help="Lower quantile for weight clipping")
    parser.add_argument("--weight_clip_high", type=float, default=0.99, help="Upper quantile for weight clipping")
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args(argv)

    if ARTIFACT_PATH.exists() and META_PATH.exists() and not args.force:
        print(f"[ADV] Artifacts already exist at {ARTIFACT_PATH}; use --force to recompute.")
        return 0

    train_path, test_path = detect_paths(args.train_csv, args.test_csv)
    print(f"[Paths] train={train_path} test={test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df, test_df, feature_cols, cat_cols = prepare_features(train_df, test_df)
    print(f"[Features] {len(feature_cols)} total; categorical={len(cat_cols)}")

    train_p, test_p, best_iters = compute_adversarial_mapping(train_df, test_df, feature_cols, cat_cols, args.seed)
    adversarial_auc = roc_auc_score(
        np.concatenate([np.zeros(len(train_df)), np.ones(len(test_df))]),
        np.concatenate([train_p, test_p]),
    )
    print(f"[ADV] Combined AUC (train+test preds)={adversarial_auc:.6f}")

    lo = float(np.quantile((train_p + 1e-6) / (1 - train_p + 1e-6), args.weight_clip_low))
    hi = float(np.quantile((train_p + 1e-6) / (1 - train_p + 1e-6), args.weight_clip_high))
    metadata = save_artifacts(train_df, test_df, train_p, test_p, (lo, hi), args.n_bins, best_iters, adversarial_auc)

    print("[ADV] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
