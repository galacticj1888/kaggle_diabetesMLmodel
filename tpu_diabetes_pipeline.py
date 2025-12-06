"""
TPU-ready training script for the Playground Series Season 5 Episode 12 diabetes competition.

Key design goals
-----------------
* Replatform the original GPU-first XGBoost/LightGBM pipeline onto TensorFlow so it can run on
  a Cloud TPU v5e-8 (or CPU/GPU when TPU is unavailable).
* Add domain-adaptation steps (adversarial validation, importance weighting, and domain bins)
  so the model trains on a distribution closer to the public test set.
* Keep feature engineering lightweight and deterministic to make CV and inference identical on
  Kaggle, Vertex AI, or Colab TPUs.

Usage (Kaggle notebook/CLI)
---------------------------
1. Point the CSV paths in `CFG` to the Kaggle input datasets.
2. Run `python tpu_diabetes_pipeline.py train` to produce:
   * `submission_tpu.csv` – predictions for the competition test set.
   * `oof_tpu.npy` – OOF predictions for local analysis.
3. The script auto-detects TPU; if none is found it falls back to CPU/GPU.

The code is intentionally self-contained (only depends on pandas/numpy/sklearn/tensorflow) to
avoid GPU-specific libraries that are unsupported on TPU (e.g., XGBoost CUDA, cuML).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


SEED = 2025
np.random.seed(SEED)
tf.random.set_seed(SEED)


@dataclass
class CFG:
    train_csv: str = "/kaggle/input/playground-series-s5e12/train.csv"
    test_csv: str = "/kaggle/input/playground-series-s5e12/test.csv"
    sample_submission_csv: str = "/kaggle/input/playground-series-s5e12/sample_submission.csv"
    target: str = "diagnosed_diabetes"
    id_col: str = "id"
    n_splits: int = 5
    batch_per_replica: int = 512
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-6
    hidden_per_group: int = 48
    signal_per_group: int = 24
    dropout_group: float = 0.25
    dropout_head: float = 0.3
    domain_bins: int = 10
    adv_estimators: int = 600
    adv_lr: float = 0.05
    adv_max_depth: int = 6
    adv_min_samples: int = 200
    weight_clip_low: float = 0.01
    weight_clip_high: float = 0.99
    hgb_estimators: int = 1200
    hgb_lr: float = 0.05
    hgb_max_depth: int = 6
    hgb_min_leaf: int = 20
    stack_C: float = 2.0
    output_submission: str = "submission_tpu.csv"
    output_oof: str = "oof_tpu.npy"
    metadata_json: str = "training_metadata.json"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def detect_tpu_strategy(batch_per_replica: int) -> Tuple[tf.distribute.Strategy, int]:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print(f"Running on TPU: {tpu.master()}")
    except Exception:
        strategy = tf.distribute.get_strategy()
        print("TPU not found; falling back to CPU/GPU")
    global_batch = batch_per_replica * strategy.num_replicas_in_sync
    print(f"Replicas: {strategy.num_replicas_in_sync} | Global batch size: {global_batch}")
    # Prefer bfloat16 on TPU when available
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("Mixed precision: bfloat16")
    except Exception as exc:  # pragma: no cover - safe fallback
        print(f"Mixed precision not set: {exc}")
    return strategy, global_batch


def load_data(cfg: CFG) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(cfg.train_csv)
    test = pd.read_csv(cfg.test_csv)
    sample = pd.read_csv(cfg.sample_submission_csv)
    return train, test, sample


def split_feature_types(df: pd.DataFrame, target: str, id_col: str) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c not in {target, id_col}]
    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return num_cols, cat_cols


def align_categories(train: pd.DataFrame, test: pd.DataFrame, cat_cols: Iterable[str]) -> Dict[str, List[str]]:
    cat_maps: Dict[str, List[str]] = {}
    for col in cat_cols:
        combined = pd.concat([train[col], test[col]], axis=0, ignore_index=True).astype("category")
        cats = combined.cat.categories
        cat_maps[col] = list(cats)
        train[col] = pd.Categorical(train[col], categories=cats)
        test[col] = pd.Categorical(test[col], categories=cats)
    return cat_maps


def add_percentile_features(train: pd.DataFrame, test: pd.DataFrame, num_cols: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_pct = train.copy()
    test_pct = test.copy()
    for col in num_cols:
        merged = pd.concat([train[col], test[col]], axis=0, ignore_index=True)
        pct_rank = merged.rank(pct=True, method="average")
        train_pct[f"{col}__pct"] = pct_rank.iloc[: len(train)].astype("float32")
        test_pct[f"{col}__pct"] = pct_rank.iloc[len(train) :].astype("float32")
    return train_pct, test_pct


def encode_categories(train: pd.DataFrame, test: pd.DataFrame, cat_cols: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    vocab_sizes: Dict[str, int] = {}
    for col in cat_cols:
        cats = train[col].cat.categories if hasattr(train[col], "cat") else pd.Categorical(train[col]).categories
        vocab_sizes[col] = len(cats)
        train[col] = pd.Categorical(train[col], categories=cats).codes.astype("int32")
        test[col] = pd.Categorical(test[col], categories=cats).codes.astype("int32")
    return train, test, vocab_sizes


# ---------------------------------------------------------------------------
# Domain adaptation
# ---------------------------------------------------------------------------

def adversarial_weights(
    train: pd.DataFrame,
    test: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    cfg: CFG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a simple adversarial classifier to approximate p(test|x).

    We keep this lightweight (HistGradientBoosting) so it runs on CPU/TPU hosts
    without GPU dependencies. Categorical columns are treated as integer codes.
    """

    df_train = train.copy()
    df_test = test.copy()

    # Ensure categorical codes
    df_train, df_test, _ = encode_categories(df_train, df_test, cat_cols)

    feature_cols = [c for c in df_train.columns if c not in {cfg.target, cfg.id_col}]
    X_combined = pd.concat([df_train[feature_cols], df_test[feature_cols]], axis=0, ignore_index=True)
    y_combined = np.concatenate([
        np.zeros(len(df_train), dtype=np.int8),
        np.ones(len(df_test), dtype=np.int8),
    ])

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_combined), dtype=np.float32)

    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=cfg.adv_lr,
        max_depth=cfg.adv_max_depth,
        max_bins=255,
        min_samples_leaf=cfg.adv_min_samples,
        max_iter=cfg.adv_estimators,
        random_state=SEED,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_combined, y_combined), 1):
        clf.fit(X_combined.iloc[tr_idx], y_combined[tr_idx])
        prob = clf.predict_proba(X_combined.iloc[va_idx])[:, 1]
        oof[va_idx] = prob.astype("float32")
        score = roc_auc_score(y_combined[va_idx], prob)
        print(f"Adversarial fold {fold}: AUC={score:.4f}")

    p_test = oof[: len(train)]
    eps = 1e-6
    weights = (p_test + eps) / (1.0 - p_test + eps)
    lo = np.quantile(weights, cfg.weight_clip_low)
    hi = np.quantile(weights, cfg.weight_clip_high)
    weights = np.clip(weights, lo, hi)
    weights = weights / weights.mean()
    print(f"Adversarial OOF AUC: {roc_auc_score(y_combined, oof):.4f}")
    return p_test, weights.astype("float32")


# ---------------------------------------------------------------------------
# TensorFlow input builders and model
# ---------------------------------------------------------------------------

def emb_dim(vocab: int) -> int:
    return int(min(32, max(4, np.ceil(np.log2(vocab + 1)) + 1)))


def build_model(
    num_cols: List[str],
    cat_cols: List[str],
    vocab_sizes: Dict[str, int],
    cfg: CFG,
) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(cfg.weight_decay)

    inputs = {}
    num_inputs = []
    for col in num_cols:
        inp = tf.keras.layers.Input(shape=(1,), name=col, dtype=tf.float32)
        num_inputs.append(inp)
        inputs[col] = inp
    for col in cat_cols:
        inp = tf.keras.layers.Input(shape=(1,), name=col, dtype=tf.int32)
        inputs[col] = inp

    if num_inputs:
        num_stack = tf.keras.layers.Concatenate()(num_inputs)
        num_stack = tf.keras.layers.LayerNormalization()(num_stack)
        num_stack = tf.keras.layers.GaussianNoise(0.01)(num_stack)
    else:
        num_stack = None

    emb_vectors = []
    for col in cat_cols:
        vocab = vocab_sizes[col]
        dim = emb_dim(vocab)
        emb = tf.keras.layers.Embedding(
            input_dim=vocab + 1,
            output_dim=dim,
            embeddings_regularizer=reg,
            name=f"emb_{col}",
        )(inputs[col])
        emb = tf.keras.layers.Reshape((dim,))(emb)
        emb_vectors.append(emb)

    feature_blocks = [v for v in emb_vectors]
    if num_stack is not None:
        feature_blocks.append(num_stack)

    x = tf.keras.layers.Concatenate(name="feature_concat")(feature_blocks)

    x = tf.keras.layers.Dense(cfg.hidden_per_group, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = tf.keras.layers.Dropout(cfg.dropout_group)(x)

    x = tf.keras.layers.Dense(cfg.signal_per_group, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = tf.keras.layers.Dropout(cfg.dropout_head)(x)

    out = tf.keras.layers.Dense(1, activation="sigmoid", dtype=tf.float32)(x)
    return tf.keras.Model(inputs=inputs, outputs=out, name="tpu_diabetes")


def make_tf_dataset(
    data: Dict[str, np.ndarray],
    y: np.ndarray | None,
    sample_weight: np.ndarray | None,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(data)
    elif sample_weight is None:
        ds = tf.data.Dataset.from_tensor_slices((data, y))
    else:
        ds = tf.data.Dataset.from_tensor_slices((data, y, sample_weight))

    if training:
        ds = ds.shuffle(200_000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=training).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    p_test: np.ndarray,
    weights: np.ndarray,
    vocab_sizes: Dict[str, int],
    cfg: CFG,
    strategy: tf.distribute.Strategy,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    y = train_df[cfg.target].values.astype("float32")
    strat_labels = train_df[cfg.target].astype(str) + "_" + train_df["domain_bin"].astype(str)

    # Prepare numpy dict inputs (int32 cats, float numerics)
    train_inputs = {col: train_df[col].values.reshape(-1, 1).astype("float32") for col in num_cols}
    test_inputs = {col: test_df[col].values.reshape(-1, 1).astype("float32") for col in num_cols}
    for col in cat_cols:
        train_inputs[col] = train_df[col].values.reshape(-1, 1).astype("int32")
        test_inputs[col] = test_df[col].values.reshape(-1, 1).astype("int32")

    oof = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, strat_labels), 1):
        print(f"\n========== Fold {fold}/{cfg.n_splits} ==========")
        X_tr = {k: v[tr_idx] for k, v in train_inputs.items()}
        X_va = {k: v[va_idx] for k, v in train_inputs.items()}
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr, w_va = weights[tr_idx], weights[va_idx]

        with strategy.scope():
            model = build_model(num_cols, cat_cols, vocab_sizes, cfg)
            opt = tf.keras.optimizers.experimental.AdamW(
                learning_rate=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
            model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["AUC"])

        train_ds = make_tf_dataset(X_tr, y_tr, w_tr, batch_size, training=True)
        val_ds = make_tf_dataset(X_va, y_va, w_va, batch_size, training=False)
        test_ds = make_tf_dataset(test_inputs, None, None, batch_size, training=False)

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", patience=3, factor=0.5, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=6, restore_best_weights=True, verbose=1
            ),
        ]

        model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks, verbose=2)

        va_pred = model.predict(val_ds, verbose=0).reshape(-1)
        oof[va_idx] = va_pred
        test_pred += model.predict(test_ds, verbose=0).reshape(-1) / cfg.n_splits

        fold_auc = roc_auc_score(y_va, va_pred, sample_weight=w_va)
        print(f"Fold {fold} weighted AUC: {fold_auc:.5f}")
        tf.keras.backend.clear_session()

    overall_auc = roc_auc_score(y, oof, sample_weight=weights)
    print(f"\nOOF weighted AUC: {overall_auc:.5f}")
    return oof, test_pred, overall_auc


def run_hgb_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    weights: np.ndarray,
    cfg: CFG,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Tabular baseline: HistGradientBoosting with importance weighting and test-like stratification."""

    y = train_df[cfg.target].values.astype("float32")
    strat_labels = train_df[cfg.target].astype(str) + "_" + train_df["domain_bin"].astype(str)

    X = train_df[feature_cols]
    X_test = test_df[feature_cols]

    oof = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=SEED)
    base_params = dict(
        loss="log_loss",
        learning_rate=cfg.hgb_lr,
        max_depth=cfg.hgb_max_depth,
        min_samples_leaf=cfg.hgb_min_leaf,
        max_bins=255,
        max_iter=cfg.hgb_estimators,
        random_state=SEED,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_labels), 1):
        print(f"\n[HGB] Fold {fold}/{cfg.n_splits}")
        model = HistGradientBoostingClassifier(**base_params)
        model.fit(X.iloc[tr_idx], y[tr_idx], sample_weight=weights[tr_idx])

        va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = va_pred
        test_pred += model.predict_proba(X_test)[:, 1] / cfg.n_splits

        fold_auc = roc_auc_score(y[va_idx], va_pred, sample_weight=weights[va_idx])
        print(f"[HGB] Fold weighted AUC: {fold_auc:.5f}")

    overall_auc = roc_auc_score(y, oof, sample_weight=weights)
    print(f"[HGB] OOF weighted AUC: {overall_auc:.5f}")
    return oof.astype("float32"), test_pred.astype("float32"), overall_auc


def stack_predictions(
    oof_dict: Dict[str, np.ndarray],
    test_dict: Dict[str, np.ndarray],
    y: np.ndarray,
    weights: np.ndarray,
    p_test: np.ndarray,
    cfg: CFG,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Meta-learner: logistic regression blending model outputs and p_test for shift-awareness."""

    feature_names = list(oof_dict.keys())
    X_stack = np.column_stack([oof_dict[name] for name in feature_names] + [p_test])
    X_test_stack = np.column_stack([test_dict[name] for name in feature_names] + [test_dict[feature_names[0]] * 0 + 0])

    # p_test for test rows is unknown; approximate with equal prior 0.5 for neutral influence
    X_test_stack[:, -1] = 0.5

    meta = LogisticRegression(
        C=cfg.stack_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=200,
    )
    meta.fit(X_stack, y, sample_weight=weights)

    oof_meta = meta.predict_proba(X_stack)[:, 1]
    test_meta = meta.predict_proba(X_test_stack)[:, 1]

    meta_auc = roc_auc_score(y, oof_meta, sample_weight=weights)
    print(f"[STACK] Weighted AUC: {meta_auc:.5f}")
    return oof_meta.astype("float32"), test_meta.astype("float32"), meta_auc


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: CFG) -> None:
    train, test, sample = load_data(cfg)
    num_cols, cat_cols = split_feature_types(train, cfg.target, cfg.id_col)
    print(f"Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

    # Align categories and augment numeric features with percentiles
    cat_map = align_categories(train, test, cat_cols)
    train_pct, test_pct = add_percentile_features(train, test, num_cols)

    # Use percentile features as numerics; keep categorical codes
    pct_cols = [f"{c}__pct" for c in num_cols]
    train_aug = pd.concat([train_pct, train[cat_cols]], axis=1)
    test_aug = pd.concat([test_pct, test[cat_cols]], axis=1)

    # Adversarial weighting to approximate the test distribution
    p_test, weights = adversarial_weights(train_aug, test_aug, pct_cols, cat_cols, cfg)
    train_aug["p_test"] = p_test
    train_aug["domain_weight"] = weights
    train_aug["domain_bin"] = pd.qcut(
        train_aug["p_test"].rank(method="first"), q=cfg.domain_bins, labels=False
    ).astype(int)

    # Encode categoricals to integer ids for TF model
    train_enc, test_enc, vocab_sizes = encode_categories(train_aug, test_aug, cat_cols)

    # Prepare strategy and batch size
    strategy, batch_size = detect_tpu_strategy(cfg.batch_per_replica)

    # Train with CV on TPU/CPU/GPU
    oof_nn, test_nn, oof_auc_nn = run_cv(
        train_enc,
        test_enc,
        num_cols=pct_cols,
        cat_cols=cat_cols,
        p_test=p_test,
        weights=weights,
        vocab_sizes=vocab_sizes,
        cfg=cfg,
        strategy=strategy,
        batch_size=batch_size,
    )

    # Add a tree-based baseline that excels on tabular numeric features
    hgb_features = pct_cols + cat_cols + ["p_test"]
    oof_hgb, test_hgb, oof_auc_hgb = run_hgb_cv(
        train_enc,
        test_enc,
        feature_cols=hgb_features,
        weights=weights,
        cfg=cfg,
    )

    # Stack models for a shift-aware blend
    oof_stack, test_stack, oof_auc_stack = stack_predictions(
        {"nn": oof_nn, "hgb": oof_hgb},
        {"nn": test_nn, "hgb": test_hgb},
        y=train_enc[cfg.target].values,
        weights=weights,
        p_test=p_test,
        cfg=cfg,
    )

    # Save artefacts
    np.save(cfg.output_oof, oof_stack)
    meta = {
        "oof_weighted_auc_stack": float(oof_auc_stack),
        "oof_weighted_auc_nn": float(oof_auc_nn),
        "oof_weighted_auc_hgb": float(oof_auc_hgb),
        "config": asdict(cfg),
        "cat_map": cat_map,
        "vocab_sizes": vocab_sizes,
    }
    with open(cfg.metadata_json, "w") as f:
        json.dump(meta, f, indent=2)

    sample[cfg.target] = test_stack
    sample.to_csv(cfg.output_submission, index=False)
    print(f"Saved submission to {cfg.output_submission}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TPU-ready training for diabetes competition")
    parser.add_argument("command", choices=["train"], help="Run the training pipeline")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = CFG()
    if args.command == "train":
        run_pipeline(cfg)


if __name__ == "__main__":
    main()
