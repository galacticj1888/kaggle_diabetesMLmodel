import argparse
import os
from pathlib import Path

import numpy as np
import xgboost as xgb
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


def detect_gpu(force_gpu: bool = False) -> bool:
    gpu_available = False
    exc = None
    try:
        tmp = xgb.DMatrix(np.random.randn(32, 4), label=np.random.randint(0, 2, 32))
        xgb.train({"tree_method": "gpu_hist", "device": "cuda", "verbosity": 0}, tmp, num_boost_round=1)
        gpu_available = True
        print("[XGB] GPU test passed - using gpu_hist")
    except Exception as err:  # pragma: no cover - diagnostic
        exc = err
        print(f"[XGB] GPU test failed ({err}); defaulting to CPU hist")
    if force_gpu and not gpu_available:
        raise RuntimeError(f"--gpu/ FORCE_GPU set but GPU unavailable or failed init: {exc}")
    return gpu_available


def run(force: bool = False, smoke: bool = False):
    ensure_dirs()
    paths = get_paths()
    train_df, test_df, _ = load_data(paths)

    target = "diagnosed_diabetes"
    id_col = "id"

    force_gpu = os.environ.get("FORCE_GPU", "0") == "1"
    gpu_available = detect_gpu(force_gpu=force_gpu)

    num_cols, cat_cols = split_cols(train_df, target, id_col)
    train_proc, test_proc, num_cols, cat_cols = prepare_common(
        train_df, test_df, num_cols, cat_cols, encode_cats_for_gpu=gpu_available
    )

    n_splits = 2 if smoke else 5
    folds_path = "artifacts/folds_smoke.npy" if smoke else "artifacts/folds.npy"
    folds = make_folds(train_proc[target].values, n_splits=n_splits, save_path=folds_path, force=force)

    feature_cols = [c for c in train_proc.columns if c not in (target, id_col)]

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "gpu_hist" if gpu_available else "hist",
        "device": "cuda" if gpu_available else None,
        "n_estimators": 2000 if not smoke else 10,
        "random_state": SEED,
    }

    oof = np.zeros(len(train_proc), dtype=np.float32)
    test_pred = np.zeros(len(test_proc), dtype=np.float32)

    for fold in range(n_splits):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        x_train = train_proc.iloc[train_idx][feature_cols]
        y_train = train_proc.iloc[train_idx][target]
        x_val = train_proc.iloc[val_idx][feature_cols]
        y_val = train_proc.iloc[val_idx][target]

        test_frame = test_proc[feature_cols]
        params = {k: v for k, v in base_params.items() if v is not None}
        enable_cats = not gpu_available

        model = xgb.XGBClassifier(**params, enable_categorical=enable_cats)
        try:
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                verbose=100,
                early_stopping_rounds=100,
            )
        except Exception as exc:
            if gpu_available:
                raise RuntimeError(f"[Fold {fold+1}] GPU training failed with encoded categories: {exc}")
            print(f"[Fold {fold+1}] categorical training failed on CPU ({exc}); switching to ordinal codes")
            x_train = x_train.copy()
            x_val = x_val.copy()
            test_frame = test_frame.copy()
            for c in cat_cols:
                x_train[c] = x_train[c].cat.codes.astype(np.int32)
                x_val[c] = x_val[c].cat.codes.astype(np.int32)
                test_frame[c] = test_frame[c].cat.codes.astype(np.int32)
            model = xgb.XGBClassifier(**params, enable_categorical=False)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                verbose=100,
                early_stopping_rounds=100,
            )

        val_pred = model.predict_proba(x_val, iteration_range=(0, model.best_iteration + 1))[:, 1].astype(np.float32)
        test_fold_pred = model.predict_proba(test_frame, iteration_range=(0, model.best_iteration + 1))[:, 1].astype(np.float32)

        oof[val_idx] = val_pred
        test_pred += test_fold_pred / n_splits

        val_auc = roc_auc_score(y_val, val_pred)
        print(
            f"[Fold {fold+1}/{n_splits}] tree_method={params['tree_method']}, "
            f"enable_cats={enable_cats}, best_iter={model.best_iteration}, val_auc={val_auc:.6f}"
        )

    oof_auc = roc_auc_score(train_proc[target], oof)
    print(f"[OOF] AUC={oof_auc:.6f}")

    save_preds(oof, test_pred, prefix="xgb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="retrain even if outputs exist")
    parser.add_argument("--smoke", action="store_true", help="quick 2-fold run with 10 trees")
    args = parser.parse_args()

    out_files = [Path("outputs/oof_xgb.npy"), Path("outputs/test_xgb.npy"), Path("submissions/submission_xgb.csv")]
    if all(p.exists() for p in out_files) and not args.force:
        print("Outputs already exist for xgboost. Use --force to retrain.")
    else:
        run(force=args.force, smoke=args.smoke)
