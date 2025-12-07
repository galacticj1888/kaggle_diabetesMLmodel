# Shift-aware Kaggle Diabetes Baselines (S5E12)

This repository provides a clean, fast baseline stack for the **Playground Series S5E12 – Diabetes Prediction** competition. It focuses on two runs you can execute directly inside a Kaggle notebook in under 30 minutes:

- **Naive baseline**: plain LightGBM, no shift handling — a reality check.
- **Shift-aware baseline**: LightGBM with adversarial weights, p_test feature, and shift-aware stratification.

Artifacts are cached to accelerate iteration and to make the effect of distribution-shift mitigation measurable.

## Quickstart (Kaggle)

```bash
# Option A: Naive baseline (no shift awareness, ~10 min)
!python run_kaggle.py --mode naive

# Option B: Shift-aware baseline (compute adversarial artifacts + train)
!python run_kaggle.py --mode full

# Option C: Shift-aware baseline when artifacts already exist
!python run_kaggle.py --mode shift --use_weights --use_p_test_feature --stratify_domain
```

Outputs are written to the working directory:
- `submission_naive.csv`, `oof_naive.npy`, `test_naive.npy`
- `submission_shift.csv`, `oof_shift.npy`, `test_shift.npy`
- `artifacts/adversarial_mapping.parquet`, `artifacts/adversarial_metadata.json`

## Components

### `baseline_naive.py`
- 5-fold StratifiedKFold on the target only.
- Native LightGBM categorical handling with simple missing-value fills.
- Early stopping (100 rounds, max 2000 trees) and verbose per-fold logging.
- Saves OOF/test predictions and a submission CSV.

### `adversarial_mapping.py`
- Trains a 5-fold adversarial classifier (train=0 vs test=1).
- Computes `p_test`, `domain_weight` (clipped/normalized), and `domain_bin` (10-quantile bins).
- Saves `artifacts/adversarial_mapping.parquet` and metadata JSON; skips recomputation unless `--force` is passed.

### `baseline_shift_aware.py`
- Loads adversarial artifacts and optionally:
  - Applies `domain_weight` as `sample_weight`.
  - Adds `p_test` as a feature.
  - Stratifies CV by `label × domain_bin`.
- Reports weighted and unweighted AUC per fold plus OOF.
- Saves OOF/test predictions and a submission CSV.

### `run_kaggle.py`
- Single entrypoint for Kaggle notebooks:
  - `--mode naive`: run the naive baseline.
  - `--mode shift`: run shift-aware baseline (assumes artifacts exist).
  - `--mode full`: recompute adversarial artifacts then train shift-aware baseline with all ablations on.

## Paths
- Kaggle data: `/kaggle/input/playground-series-s5e12/{train.csv,test.csv,sample_submission.csv}`
- Local default: `./data/{train.csv,test.csv,sample_submission.csv}`

## Notes
- Categorical values are aligned across train/test with an explicit `UNKNOWN` bucket.
- All outputs are float32 and preserve row order relative to the input CSVs.
- Logging is intentionally verbose so you can follow training progress from notebook cells.
