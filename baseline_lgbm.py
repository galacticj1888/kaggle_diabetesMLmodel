import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils import (
    SEED,
    ensure_dirs,
    get_paths,
    load_data,
    split_cols,
    prepare_common,
    make_folds,
    save_preds,
)


def detect_device_params():
    return {}


def run(force: bool = False, smoke: bool = False):
    ensure_dirs()
    paths = get_paths()
    train_df, test_df, _ = load_data(paths)

    target = "diagnosed_diabetes"
    id_col = "id"

    num_cols, cat_cols = split_cols(train_df, target, id_col)
    train_proc, test_proc, num_cols, cat_cols = prepare_common(train_df, test_df, num_cols, cat_cols)

    n_splits = 2 if smoke else 5
    folds_path = "artifacts/folds_smoke.npy" if smoke else "artifacts/folds.npy"
    folds = make_folds(train_proc[target].values, n_splits=n_splits, save_path=folds_path, force=force)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 100,
        "n_estimators": 5000 if not smoke else 10,
        "n_jobs": -1,
        "verbose": -1,
        "seed": SEED,
    }
    params.update(detect_device_params())

    feature_cols = [c for c in train_proc.columns if c not in (target, id_col)]
    cat_features = [c for c in feature_cols if c in cat_cols]

    oof = np.zeros(len(train_proc), dtype=np.float32)
    test_pred = np.zeros(len(test_proc), dtype=np.float32)

    for fold in range(n_splits):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        x_train = train_proc.iloc[train_idx][feature_cols]
        y_train = train_proc.iloc[train_idx][target]
        x_val = train_proc.iloc[val_idx][feature_cols]
        y_val = train_proc.iloc[val_idx][target]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
        )

        val_pred = model.predict_proba(x_val, num_iteration=model.best_iteration_)[:, 1].astype(np.float32)
        test_fold_pred = model.predict_proba(test_proc[feature_cols], num_iteration=model.best_iteration_)[:, 1].astype(np.float32)

        oof[val_idx] = val_pred
        test_pred += test_fold_pred / n_splits

        val_auc = roc_auc_score(y_val, val_pred)
        print(f"[Fold {fold+1}/{n_splits}] size train={len(train_idx)}, val={len(val_idx)}, best_iter={model.best_iteration_}, val_auc={val_auc:.6f}")

    oof_auc = roc_auc_score(train_proc[target], oof)
    print(f"[OOF] AUC={oof_auc:.6f}")

    save_preds(oof, test_pred, prefix="lgbm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="retrain even if outputs exist")
    parser.add_argument("--smoke", action="store_true", help="quick 2-fold, 10-iter run for validation")
    args = parser.parse_args()

    out_files = [Path("outputs/oof_lgbm.npy"), Path("outputs/test_lgbm.npy"), Path("submissions/submission_lgbm.csv")]
    if all(p.exists() for p in out_files) and not args.force:
        print("Outputs already exist for lgbm. Use --force to retrain.")
    else:
        run(force=args.force, smoke=args.smoke)
