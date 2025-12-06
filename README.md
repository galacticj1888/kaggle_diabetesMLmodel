# TPU-first diabetes classifier

This repository hosts a TensorFlow pipeline that replatforms the Playground Series S5E12 diabetes competition workflow onto **Cloud TPU v5e-8** (with graceful CPU/GPU fallback). The script keeps feature engineering light but adds domain-adaptation tricks so the model is trained on a distribution closer to the public test set.

## Highlights
- **TPU-aware training** – automatic TPU detection, mixed bfloat16, and batched `tf.data` input.
- **Domain alignment** – adversarial validation estimates `p(test|x)`, importance weights `p/(1-p)`, and `domain_bin` strata for shift-aware CV.
- **Test-stable features** – global percentile transforms for numerics plus embedding-based categoricals (no GPU-only encoders).
- **Ensembled for AUC** – combines a TPU NN, a tabular HistGradientBoosting model, and a logistic stacker that blends both with `p_test`.
- **Single entry point** – run `python tpu_diabetes_pipeline.py train` to train and emit `submission_tpu.csv` and `oof_tpu.npy`.

## Quickstart
1. Update file paths in `CFG` if needed (defaults match the Kaggle competition input layout).
2. Launch a TPU notebook or a CPU/GPU runtime.
3. Execute:
   ```bash
   python tpu_diabetes_pipeline.py train
   ```
4. Submit the generated `submission_tpu.csv` to Kaggle.

## Files
- `tpu_diabetes_pipeline.py` – end-to-end training script with CV, weighting, and inference.
- `README.md` – this guide.

## Notes
- The pipeline avoids GPU-only libraries (CUDA XGBoost/cuML) so it can run end-to-end on TPU hosts.
- Validation uses **weighted AUC** with domain weights to mimic the leaderboard distribution.
