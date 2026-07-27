# Voting-Classifier Ensemble (binary ROP classification)

Implements the approach from *"Early Detection of Retinopathy of Prematurity Using Voting
Classifier-Based Ensemble Deep Learning Models"* (Suresh Krishna & Sofana Reka, 2025) against
our own dataset: four transfer-learning base models combined with four voting strategies.

This is a **separate, standalone pipeline** from the one in the repo root (`train.py` /
EfficientNet-B3 + k-fold CV). It's an independent experiment, not wired into that pipeline's
config or evaluation code, though it follows the same "don't leak the test set" discipline:
a stratified train/val/test split is made once, the val split is used for checkpoint
selection and voting weights, and the test split is touched only in `evaluate.py`.

## What's implemented

- **Base models** (`models.py`): InceptionV3, DenseNet201, MobileNetV2, Xception — each
  ImageNet-pretrained with the head replaced by a single sigmoid output (binary classification,
  `BCEWithLogitsLoss`), matching the paper's Section 3.2/3.3. Xception isn't in torchvision, so
  it's pulled from `timm`.
- **Voting strategies** (`voting.py`): Hard, Soft, Weighted Average (weighted by each model's
  validation accuracy), and Mix Voting (Hard Voting, with ties broken by Soft Voting — the
  paper's Algorithm 1).

## Deviations from the paper (and why)

- **Binary only.** The paper's multi-class task needs per-image ROP stage labels; our dataset
  (`Neo_Normal`/`Neo_ROP`/`RetCam_Normal`/`RetCam_ROP`) only has Normal/ROP.
- **MobileNetV2**, not the unspecified "MobileNet" in the paper (torchvision doesn't ship a v1).
- **Single train/val/test split**, not the paper's separate 5-fold CV robustness check (Section
  4.1.3) — this is what "standalone/simpler" means in practice here. If you want CV-grade
  robustness, look at how `train.py` in the repo root does it and port the same pattern.
- **Voting weights** come from each model's held-out validation accuracy (the paper says
  "accuracy or F1 score" without pinning down the exact source split).

## Usage

Run everything from the repo root (so relative paths in `ensemble/config.yaml` resolve
correctly):

```bash
# 1. Train all 4 base models (saves checkpoints + val_summary.json + split.json)
python ensemble/train.py --config ensemble/config.yaml

# 2. Evaluate all 4 voting strategies + each base model on the held-out test set
python ensemble/evaluate.py --config ensemble/config.yaml

# 3. Predict a single image (default: Mix Voting)
python ensemble/predict.py path/to/image.jpg --method mix
```

Outputs land in `ensemble_results/` (gitignored): one `.pth` checkpoint per base model,
`split.json`, `val_summary.json`, `test_results.json` (individual + ensemble metrics), plus
ROC curves and confusion matrices per voting method.

Update `epochs`/`batch_size`/`image_sizes` in `ensemble/config.yaml` to fit your hardware —
training 4 full backbones (2 of them at 299×299) is considerably heavier than the single
EfficientNet-B3 in the root pipeline.
