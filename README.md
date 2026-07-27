# EPICS ROP Classification (RetCam + Neo)

Binary classification of retinal fundus images into:

- `ROP` (Retinopathy of Prematurity)
- `Normal`

The pipeline fine-tunes an ImageNet-pretrained **EfficientNet-B3** in two phases (classifier head warmup, then full fine-tuning), and evaluates with **stratified k-fold cross-validation plus a held-out test set** — see "Why k-fold + a held-out set" below.

## Project Structure

- `config.yaml` / `config.py`: all paths and hyperparameters in one place
- `dataset.py`: `ROPDataset` (indexes images/labels) + stratified split / k-fold helpers
- `model.py`: EfficientNet-B3 model factory
- `engine.py`: shared train/validate loops and the two-phase fine-tuning routine
- `train.py`: orchestrates held-out split → k-fold CV → final fit → test evaluation
- `predict.py`: single-image inference with confidence score
- `tests/`: split-integrity and model-shape tests (`pytest tests/`)
- `Neo_Normal/`, `Neo_ROP/`, `RetCam_Normal/`, `RetCam_ROP/`: image data folders
- `legacy/`: superseded experiments kept for reference (custom CuPy CNN, full-resolution training scripts, ad hoc GPU sanity checks) — not part of the active pipeline
- `ensemble/`: a separate, standalone pipeline implementing a 4-model voting-classifier ensemble (InceptionV3 + DenseNet201 + MobileNetV2 + Xception, combined via Hard/Soft/Weighted/Mix Voting) — see `ensemble/README.md`

## Requirements

```bash
pip install -r requirements.txt
```

If you're on an RTX 50-series (Blackwell) GPU, the stable PyTorch build may not support your GPU's compute capability (sm_120) — install the PyTorch nightly build with CUDA 12.8 support instead. See `legacy/old_Models/fix_pytorch_rtx5070ti.py` for the exact commands.

## Dataset Layout

`dataset.py` expects these four folders directly under `data_root` (set in `config.yaml`):

```text
Neo_Normal/
Neo_ROP/
RetCam_Normal/
RetCam_ROP/
```

`Neo_ROP`/`RetCam_ROP` are labeled class `1` (ROP), the `Normal` folders are class `0`.

## Why k-fold + a held-out set

With only ~185 images, a single train/val split can make a model look far better (or worse) than it really is, and picking the "best" checkpoint using the same split you report accuracy on is data leakage. So `train.py` does:

1. **Held-out test set** — a stratified slice (`test_size` in `config.yaml`, default 15%) is carved off once at the start and touched exactly once, at the very end.
2. **Stratified k-fold CV** (`n_folds`, default 5) on the remaining pool — the model is trained and evaluated fold-by-fold, and the mean ± std across folds is the actual performance estimate to trust. Results land in `results/cv/fold_*/` and `results/cv_summary.json`.
3. **Final model** — retrained on the full train+val pool (same hyperparameters CV already validated), evaluated once on the held-out test set for the headline numbers, saved to `results/best_rop_model.pth`.

> The numbers in `FINAL_PROJECT_REPORT.md` / `ADVANCED_REPORT.md` predate this fix (they came from a val set that was also used for model selection). Once you run the pipeline, regenerate those figures from `results/cv_summary.json` and `results/test_classification_report.txt`.

## Train

```bash
python train.py --config config.yaml
```

Update `data_root` in `config.yaml` first if your dataset lives somewhere other than the repo root. A full run trains `n_folds` CV models plus one final model (each doing `epochs_phase1 + epochs_phase2` epochs), so expect it to take a while.

For a fast end-to-end smoke test (tiny images, 1 epoch per phase, 2 folds — just to confirm the pipeline runs, not to produce a real model), use:

```bash
python train.py --config config.smoke.yaml
```

## Output

Everything lands under `output_dir` (default `results/`, gitignored):

- `cv/fold_*/classification_report.txt`, `cv/fold_*/confusion_matrix.png`
- `cv_summary.json` — mean ± std accuracy/F1/AUC across folds
- `best_rop_model.pth` — final model weights
- `test_classification_report.txt`, `confusion_matrix_test.png` — held-out test metrics
- `training_curves_final.png`

## Predict

```bash
python predict.py path/to/image.jpg --checkpoint results/best_rop_model.pth
```

## Tests

```bash
pytest tests/ -v
```

Covers split-integrity (no leakage between CV folds/test set, class balance preserved) and a model forward-pass shape check. No GPU or dataset required.
