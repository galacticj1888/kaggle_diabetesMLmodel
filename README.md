# Kaggle Diabetes (S5E12) – Multi-Model Stacking Baseline

This repository delivers a **fast, shift-resilient ensemble baseline** for the Playground Series S5E12 (Diabetes Prediction). It centers on three diverse tree boosters (LightGBM, XGBoost, CatBoost) trained on shared folds, plus rank/mean blends and a logistic-regression stacker. Everything runs cleanly inside a Kaggle notebook with a single command.

## Current anchor
- Naive LightGBM CV AUC ≈ 0.727, Public LB ≈ 0.697 (5-fold target-only stratification).
- Shift-aware weighting was less stable on LB, so the default stack **does not** use domain weights.

## Quickstart (Kaggle)
```bash
# Run the full stack (all base models + blends + stacker)
!python run_kaggle.py --mode stack

# Fast smoke check (2 folds, tiny iterations)
!python run_kaggle.py --mode stack --smoke

# Individual models if you want to iterate
!python baseline_lgbm.py
!python baseline_xgboost.py
!python baseline_catboost.py
```
Outputs land in:
- `outputs/`: `oof_*.npy`, `test_*.npy`, stacked outputs
- `submissions/`: submission CSVs for each model/blend/stack
- `artifacts/`: shared fold assignments and cached utilities

## Components
- `src/utils.py` – shared Kaggle/local path detection, preprocessing (median/UNKNOWN), fold persistence (int8 fold assignments), and saving helpers.
- `baseline_lgbm.py` – LightGBM baseline with native categoricals, early stopping, verbose per-fold AUC, cached OOF/test arrays.
- `baseline_xgboost.py` – XGBoost baseline with GPU autodetect + safe CPU fallback, categorical support with ordinal fallback.
- `baseline_catboost.py` – CatBoost baseline with GPU/CPU fallback, string categoricals, early stopping.
- `ensemble.py` – mean blend, rank-mean blend, and logistic regression stacker (honest meta-OOF on shared folds) with coefficient reporting.
- `run_full_stack.py` – orchestrator to run all base models then ensemble, path-stable via absolute script resolution.
- `run_kaggle.py` – notebook-friendly entrypoint with modes: `naive` (LGBM), `shift/full` (legacy shift-aware), `stack` (full ensemble).
- `init.py` – environment bootstrap with device detection, dependency/version report, and seed helper.

## Design choices
- **Shared folds**: persisted once (`artifacts/folds.npy`) to keep OOFs aligned for stacking.
- **No shift weighting by default**: adversarial weights are optional diagnostics only.
- **Caching**: each script skips retrain unless `--force` is passed.
- **Reproducibility**: global seed=42, float32 outputs, row order preserved.
- **Logging**: verbose fold logs and OOF metrics for every model and blend.

## Paths
- Kaggle: `/kaggle/input/playground-series-s5e12/{train.csv,test.csv,sample_submission.csv}`
- Local (fallback): `./data/{train.csv,test.csv,sample_submission.csv}`

Happy stacking—this baseline is built to iterate quickly and push LB AUC beyond the naive anchor.
