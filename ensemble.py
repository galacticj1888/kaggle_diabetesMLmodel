import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.utils import get_paths, load_data, make_folds, ensure_dirs


BASE_MODELS = {
    "lgbm": "outputs/oof_lgbm.npy",
    "xgb": "outputs/oof_xgb.npy",
    "cat": "outputs/oof_cat.npy",
}


def load_predictions(prefix: str):
    oof_path = Path(f"outputs/oof_{prefix}.npy")
    test_path = Path(f"outputs/test_{prefix}.npy")
    if not (oof_path.exists() and test_path.exists()):
        return None, None
    return np.load(oof_path), np.load(test_path)


def mean_blend(arrs: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(arrs), axis=0)


def rank_mean_blend(arrs: List[np.ndarray]) -> np.ndarray:
    ranked = [rankdata(a) / len(a) for a in arrs]
    return np.mean(np.vstack(ranked), axis=0)


def logistic_stack(y: np.ndarray, folds: np.ndarray, features: Dict[str, np.ndarray], test_features: Dict[str, np.ndarray]):
    model_names = list(features.keys())
    X = np.vstack([features[k] for k in model_names]).T
    X_test = np.vstack([test_features[k] for k in model_names]).T
    oof_meta = np.zeros_like(y, dtype=np.float32)
    n_splits = folds.max() + 1

    for fold in range(n_splits):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        oof_meta[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    final_model = LogisticRegression(max_iter=1000)
    final_model.fit(X, y)
    coeff_report = dict(zip(model_names, final_model.coef_[0].tolist()))
    test_pred = final_model.predict_proba(X_test)[:, 1]
    return oof_meta, test_pred, coeff_report


def run(force: bool = False):
    ensure_dirs()
    paths = get_paths()
    train_df, test_df, _ = load_data(paths)
    y = train_df["diagnosed_diabetes"].values

    folds = make_folds(y, n_splits=5, save_path="artifacts/folds.npy")

    loaded = {}
    aucs = []
    for name in BASE_MODELS:
        oof, test_pred = load_predictions(name)
        if oof is None:
            print(f"Skipping {name} (predictions not found)")
            continue
        loaded[name] = {"oof": oof.astype(np.float32), "test": test_pred.astype(np.float32)}
        auc = roc_auc_score(y, oof)
        aucs.append(auc)
        print(f"[{name}] OOF AUC={auc:.6f}")

    if len(loaded) < 2:
        print("Need at least two models for blending/stacking. Aborting ensemble.")
        return

    if len(aucs) >= 2:
        spread = max(aucs) - min(aucs)
        if spread > 0.01:
            print(f"[WARN] Base model AUC spread {spread:.4f} exceeds 0.01; check preprocessing consistency")

    # mean blend
    mean_oof = mean_blend([v["oof"] for v in loaded.values()])
    mean_test = mean_blend([v["test"] for v in loaded.values()])
    mean_auc = roc_auc_score(y, mean_oof)
    print(f"[Blend-Mean] OOF AUC={mean_auc:.6f}")

    rank_oof = rank_mean_blend([v["oof"] for v in loaded.values()])
    rank_test = rank_mean_blend([v["test"] for v in loaded.values()])
    rank_auc = roc_auc_score(y, rank_oof)
    print(f"[Blend-Rank] OOF AUC={rank_auc:.6f}")

    # logistic stack
    feature_map = {k: v["oof"] for k, v in loaded.items()}
    feature_map_test = {k: v["test"] for k, v in loaded.items()}
    stack_oof, stack_test, coeffs = logistic_stack(y, folds, feature_map, feature_map_test)
    stack_auc = roc_auc_score(y, stack_oof)
    print(f"[Stack-LogReg] OOF AUC={stack_auc:.6f}")
    print("Meta coefficients:", coeffs)

    # save submissions
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(parents=True, exist_ok=True)

    def save_submission(arr: np.ndarray, name: str):
        sub = test_df[["id"]].copy()
        sub["diagnosed_diabetes"] = arr.astype(np.float32)
        out_path = submissions_dir / f"submission_{name}.csv"
        sub.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

    save_submission(mean_test, "blend_mean")
    save_submission(rank_test, "blend_rank")
    save_submission(stack_test, "stack_logreg")

    np.save("outputs/oof_stack_logreg.npy", stack_oof.astype(np.float32))
    np.save("outputs/test_stack_logreg.npy", stack_test.astype(np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="currently unused placeholder for symmetry")
    args = parser.parse_args()
    run(force=args.force)
