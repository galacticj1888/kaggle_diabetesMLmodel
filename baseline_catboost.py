import argparse
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool
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


def detect_task_type():
    try:
        import catboost

        devices = catboost.config.get_devices()
        if devices and any("GPU" in d for d in devices):
            return "GPU"
    except Exception:
        pass
    return "CPU"


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

    feature_cols = [c for c in train_proc.columns if c not in (target, id_col)]
    cat_indices = [feature_cols.index(c) for c in feature_cols if c in cat_cols]

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "iterations": 4000 if not smoke else 20,
        "random_seed": SEED,
        "task_type": detect_task_type(),
        "verbose": 200,
        "od_type": "Iter",
        "od_wait": 100,
    }

    oof = np.zeros(len(train_proc), dtype=np.float32)
    test_pred = np.zeros(len(test_proc), dtype=np.float32)

    # CatBoost needs raw strings for cats
    train_frame = train_proc.copy()
    test_frame = test_proc.copy()
    for c in cat_cols:
        train_frame[c] = train_frame[c].astype(str)
        test_frame[c] = test_frame[c].astype(str)

    for fold in range(n_splits):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        x_train = train_frame.iloc[train_idx][feature_cols]
        y_train = train_frame.iloc[train_idx][target]
        x_val = train_frame.iloc[val_idx][feature_cols]
        y_val = train_frame.iloc[val_idx][target]

        train_pool = Pool(x_train, label=y_train, cat_features=cat_indices)
        val_pool = Pool(x_val, label=y_val, cat_features=cat_indices)
        test_pool = Pool(test_frame[feature_cols], cat_features=cat_indices)

        model = CatBoostClassifier(**params)
        try:
            model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=200)
        except Exception as exc:
            if params["task_type"] == "GPU":
                print(f"[Fold {fold+1}] GPU failed ({exc}); retrying on CPU")
                params["task_type"] = "CPU"
                model = CatBoostClassifier(**params)
                model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=200)
            else:
                raise

        val_pred = model.predict_proba(val_pool)[:, 1].astype(np.float32)
        test_fold_pred = model.predict_proba(test_pool)[:, 1].astype(np.float32)

        oof[val_idx] = val_pred
        test_pred += test_fold_pred / n_splits

        val_auc = roc_auc_score(y_val, val_pred)
        print(f"[Fold {fold+1}/{n_splits}] task_type={params['task_type']} val_auc={val_auc:.6f}")

    oof_auc = roc_auc_score(train_proc[target], oof)
    print(f"[OOF] AUC={oof_auc:.6f}")

    save_preds(oof, test_pred, prefix="cat")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="retrain even if outputs exist")
    parser.add_argument("--smoke", action="store_true", help="quick 2-fold short-iteration run")
    args = parser.parse_args()

    out_files = [Path("outputs/oof_cat.npy"), Path("outputs/test_cat.npy"), Path("submissions/submission_cat.csv")]
    if all(p.exists() for p in out_files) and not args.force:
        print("Outputs already exist for catboost. Use --force to retrain.")
    else:
        run(force=args.force, smoke=args.smoke)
