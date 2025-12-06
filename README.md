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

### Upload a model directory to Kaggle Models (optional)
If you have a SavedModel or other artefacts you want to publish to Kaggle Models, call the new
`upload-model` command (wrap in a notebook cell with `!` when running on Kaggle):

```bash
python tpu_diabetes_pipeline.py upload-model \
  --local_model_dir /kaggle/working/export_dir \
  --owner sicomaccapital \
  --model_slug diabetes-tpu \
  --variation_slug default \
  --framework keras \
  --version_notes "Update 2025-12-06"
```

The `local_model_dir` should point to the directory containing your model files (e.g., a TensorFlow
SavedModel or any export you want versioned). The command handles `kagglehub.login()` and publishes
to the handle `<owner>/<model_slug>/<framework>/<variation_slug>`.

## Running inside a Kaggle notebook
You do **not** need to edit the script to adjust paths—pass them from a cell instead:

```python
# Cell 1: (optional) inspect data
!ls /kaggle/input/playground-series-s5e12

# Cell 2: train and write submission_tpu.csv to the working directory
!python tpu_diabetes_pipeline.py train \
    --train_csv /kaggle/input/playground-series-s5e12/train.csv \
    --test_csv /kaggle/input/playground-series-s5e12/test.csv \
    --sample_submission_csv /kaggle/input/playground-series-s5e12/sample_submission.csv \
    --output_submission /kaggle/working/submission_tpu.csv

# Cell 3: download or view the submission
!head /kaggle/working/submission_tpu.csv
```

If you want faster experiments in the notebook, you can also override training parameters without
modifying the file:

```python
!python tpu_diabetes_pipeline.py train --epochs 10 --batch_per_replica 256
```

## Files
- `tpu_diabetes_pipeline.py` – end-to-end training script with CV, weighting, and inference.
- `README.md` – this guide.

## Notes
- The pipeline avoids GPU-only libraries (CUDA XGBoost/cuML) so it can run end-to-end on TPU hosts.
- Validation uses **weighted AUC** with domain weights to mimic the leaderboard distribution.
