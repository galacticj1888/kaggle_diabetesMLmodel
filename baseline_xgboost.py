import argparse
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


def pick_tree_method():
    preferred = "gpu_hist"
    try:
        dev = xgb.core._get_cuda_runtime_version()
        if dev is not None:
            return preferred
    except Exception:
        pass
    return "hist"


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
        "tree_method": pick_tree_method(),
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

        # track whether categorical enabled
        use_cats = True
        test_frame = test_proc[feature_cols]
        params = base_params.copy()

        model = xgb.XGBClassifier(**params, enable_categorical=True)
        try:
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                verbose=100,
                early_stopping_rounds=100,
            )
        except Exception as exc:
            if params["tree_method"] == "gpu_hist":
                print(f"[Fold {fold+1}] GPU training failed ({exc}); falling back to CPU hist")
                params["tree_method"] = "hist"
                model = xgb.XGBClassifier(**params, enable_categorical=True)
                try:
                    model.fit(
                        x_train,
                        y_train,
                        eval_set=[(x_val, y_val)],
                        verbose=100,
                        early_stopping_rounds=100,
                    )
                except Exception as exc2:
                    print(f"[Fold {fold+1}] categorical training failed on CPU ({exc2}); using ordinal codes")
                    use_cats = False
            else:
                print(f"[Fold {fold+1}] categorical training failed ({exc}); using ordinal codes")
                use_cats = False

        if not use_cats:
            # fallback to integer codes
            x_train = x_train.copy()
            x_val = x_val.copy()
            test_frame = test_frame.copy()
            for c in cat_cols:
                x_train[c] = x_train[c].cat.codes.astype(np.int32)
                x_val[c] = x_val[c].cat.codes.astype(np.int32)
                test_frame[c] = test_frame[c].cat.codes.astype(np.int32)
            params_fallback = params.copy()
            model = xgb.XGBClassifier(**params_fallback, enable_categorical=False)
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
            f"[Fold {fold+1}/{n_splits}] tree_method={params['tree_method']}, use_cats={use_cats}, "
            f"best_iter={model.best_iteration}, val_auc={val_auc:.6f}"
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
