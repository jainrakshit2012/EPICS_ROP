# EPICS ROP Screening System

A complete clinical screening tool for **Retinopathy of Prematurity (RoP)** combining:

- **Batch Classification** — 5-fold EfficientNet-V2-S ensemble, predicts ROP / NoRoP per image and gives a single patient-level diagnosis
- **Tri-Modal Segmentation** — three independent segmentation models (Demarcation Ridge, Optic Disc, Blood Vessels) applied on-demand to any ROP-positive image
- **Web Interface** — Flask-based UI for uploading up to 20 images per patient session, with patient ID fields and one-click segmentation

---

## Repository Structure

```
EPICS_ROP-main/
├── app.py                      ← Flask web application (entry point)
├── templates/
│   └── index.html              ← Single-page UI
├── model.py                    ← EfficientNet-B3 model factory (single-model pipeline)
├── dataset.py                  ← Dataset loader + stratified split helpers
├── engine.py                   ← Train/validate loops + two-phase fine-tuning
├── train.py                    ← EfficientNet-B3 k-fold training pipeline
├── predict.py                  ← CLI single-image inference
├── rop_ensemble_voting.py      ← 4-model ensemble trainer (ResNet50, DenseNet201, MobileNet, EfficientNet-B3)
├── config.yaml                 ← Training hyperparameters
├── requirements.txt
└── ensemble/                   ← Modular InceptionV3+DenseNet+MobileNet+Xception ensemble

# Sibling projects (required for the web app)
../Installer/EPICS_ROP_MSI_INSTALLER-main/
│   └── model/
│       ├── efficientnet_v2_s_fold1.pth   ← Classification weights (fold 1–5)
│       ├── efficientnet_v2_s_fold2.pth
│       ├── efficientnet_v2_s_fold3.pth
│       ├── efficientnet_v2_s_fold4.pth
│       ├── efficientnet_v2_s_fold5.pth
│       └── model_metadata.json

../Segmentation/SEGMENTATION-OD-BV-RIDGE/
│   ├── app/                              ← Segmentation inference package
│   └── model/
│       ├── best_model_RIDGE_manet.pth    ← Demarcation Ridge (MAnet + EfficientNet-B4)
│       ├── best_model_OD.pth             ← Optic Disc (UNet++ + EfficientNet-B4)
│       └── optimized_best_model_BV.pth   ← Blood Vessels (UNet++ + EfficientNet-B4)
```

---

## Prerequisites

Python 3.10+ with the following packages:

```bash
pip install -r requirements.txt
```

The full dependency list:

```
torch torchvision
flask
opencv-python>=5.0.0
segmentation-models-pytorch
albumentations
scikit-image
numpy pillow scikit-learn tqdm pyyaml timm
```

> **RTX 50-series (Blackwell) GPU users:** install PyTorch nightly with CUDA 12.8.
> See `legacy/old_Models/fix_pytorch_rtx5070ti.py` for the exact commands.

---

## Running the Web App

```bash
cd EPICS_ROP-main
python app.py
```

Open **http://localhost:5000** in your browser.

### What you can do

1. Fill in **Patient Name**, **Patient ID**, and **Session Reference**
2. Drag-and-drop or click to upload **up to 20 retinal images** (PNG, JPG, BMP, TIFF)
3. All images appear immediately — white background, no black padding
4. Click **Run Diagnosis**
   - Each image gets a **ROP** or **NoRoP** badge
   - If **any** image is ROP → patient diagnosis = **ROP**; all Normal → **NoRoP**
5. On any ROP-flagged image click **Run Segmentation** to get 4 overlay tabs:
   - **Combined** — all three modalities overlaid together
   - **Ridge** — Demarcation Ridge (red)
   - **Optic Disc** — OD geometry + Zone I/II boundaries (yellow)
   - **Blood Vessels** — vessel network (green)

---

## Model Details

### Classification — 5-Fold EfficientNet-V2-S Ensemble

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-V2-S |
| Training strategy | 5-fold cross-validation |
| Input size | 224 × 224 |
| Output | Sigmoid probability, threshold 0.5 |
| Decision rule | Average of 5 fold probabilities ≥ 0.5 → ROP |

### Segmentation — Tri-Modal

| Modality | Architecture | Input size | Model file |
|----------|-------------|------------|------------|
| Demarcation Ridge | MAnet + EfficientNet-B4 | 512 × 512 | `best_model_RIDGE_manet.pth` |
| Optic Disc | UNet++ + EfficientNet-B4 | 384 × 384 | `best_model_OD.pth` |
| Blood Vessels | UNet++ + EfficientNet-B4 | 384 × 384 | `optimized_best_model_BV.pth` |

---

## Changing / Swapping Models

### Swap the Classification Models (drop-in replacement)

The app looks for exactly these 5 files:

```
Installer/EPICS_ROP_MSI_INSTALLER-main/model/
    efficientnet_v2_s_fold1.pth
    efficientnet_v2_s_fold2.pth
    efficientnet_v2_s_fold3.pth
    efficientnet_v2_s_fold4.pth
    efficientnet_v2_s_fold5.pth
    model_metadata.json
```

To use your own trained weights, simply replace these `.pth` files (keep the same filenames).
The model architecture must remain **EfficientNet-V2-S with a single sigmoid output**.
Update `model_metadata.json` if you change the input size or threshold.

### Use a Different Number of Folds

In `app.py`, find the `_load_clf_ensemble()` function and change the range:

```python
for i in range(1, 6):   # ← change 6 to (n_folds + 1)
    path = os.path.join(_CLASSIFIER_MODEL_DIR, f"efficientnet_v2_s_fold{i}.pth")
```

### Use a Completely Different Classification Architecture

Replace the model instantiation block in `_load_clf_ensemble()` in `app.py`:

```python
# Default: EfficientNet-V2-S with binary head
model = tv_models.efficientnet_v2_s(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
```

For example, to use ResNet-50:

```python
import torchvision.models as tv_models
model = tv_models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
```

The `predict_image()` function expects the model to output a single sigmoid logit.
If your model outputs softmax probabilities (2-class), change the prediction line:

```python
# For softmax 2-class output (index 1 = ROP probability):
prob = torch.softmax(model(tensor), dim=1)[0][1].item()
```

### Change the Classification Model Directory

Edit the path at the top of `app.py`:

```python
_CLASSIFIER_MODEL_DIR = os.path.join(_INSTALLER_ROOT, "model")
# → change to any absolute or relative path, e.g.:
_CLASSIFIER_MODEL_DIR = r"C:\path\to\your\models"
```

---

### Swap the Segmentation Models (drop-in replacement)

The app looks for these files in:

```
Segmentation/SEGMENTATION-OD-BV-RIDGE/model/
    best_model_RIDGE_manet.pth
    best_model_OD.pth
    optimized_best_model_BV.pth
```

Replace any of these files with your own trained weights.
Architecture must match what is defined in `Segmentation/app/model_loader.py`:

| File | Architecture |
|------|-------------|
| `best_model_RIDGE_manet.pth` | `smp.MAnet(encoder_name="efficientnet-b4", in_channels=3, classes=1)` |
| `best_model_OD.pth` | `smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1)` |
| `optimized_best_model_BV.pth` | `smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1)` |

### Change the Segmentation Model Directory

Edit the path at the top of `app.py`:

```python
_SEG_MODEL_DIR = os.path.join(_SEG_ROOT, "model")
# → change to any absolute or relative path, e.g.:
_SEG_MODEL_DIR = r"C:\path\to\your\seg_models"
```

### Use a Different Segmentation Encoder

Edit `Segmentation/app/model_loader.py`, changing the `encoder_name` parameter:

```python
# e.g., switch Ridge from EfficientNet-B4 to ResNet-34:
model = smp.MAnet(
    encoder_name="resnet34",   # ← change encoder here
    encoder_weights=None,
    in_channels=3,
    classes=1,
)
```

Retrain the model and replace the checkpoint file.

---

## Training the Classification Model

### Ensemble (recommended — used by the web app)

```bash
python rop_ensemble_voting.py
```

Trains ResNet-50, DenseNet-201, MobileNet-V2, and EfficientNet-B3 in sequence.
Saved weights land in `ensemble_results/{model_name}.pth`.

> **Note:** The web app uses the 5-fold EfficientNet-V2-S from the `Installer/` folder.
> To train that model, see `Installer/EPICS_ROP_MSI_INSTALLER-main/`.

### Single EfficientNet-B3 Pipeline

```bash
python train.py --config config.yaml
```

Outputs `results/best_rop_model.pth`. Edit `config.yaml` to change dataset path, epochs, or learning rate.

Smoke test (fast, 1 epoch, to verify the pipeline runs):

```bash
python train.py --config config.smoke.yaml
```

### CLI Predict (single image)

```bash
python predict.py path/to/image.jpg --checkpoint results/best_rop_model.pth
```

---

## Dataset Layout

All four folders must exist under `data_root` (set in `config.yaml`, default is the repo root):

```
Neo_Normal/       Normal images (NeoNatal camera)
Neo_ROP/          ROP images    (NeoNatal camera)
RetCam_Normal/    Normal images (RetCam)
RetCam_ROP/       ROP images    (RetCam)
```

---

## Tests

```bash
pytest tests/ -v
```

Covers split integrity (no data leakage between folds and test set) and model forward-pass shape.
No GPU or dataset required.

---

## Patient Diagnosis Logic

For a single patient session (typically 12–20 images):

> If **any single image** is classified as ROP → the patient diagnosis is **ROP**.
> Only if **all images** are Normal → the patient diagnosis is **NoRoP**.

This conservative rule prioritises sensitivity — a missed ROP case is clinically more dangerous than a false positive.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Flask |
| Classification | PyTorch · EfficientNet-V2-S (5-fold ensemble) |
| Segmentation | segmentation-models-pytorch · MAnet / UNet++ · EfficientNet-B4 |
| Image processing | OpenCV 5, Albumentations, scikit-image |
| Frontend | Vanilla HTML/CSS/JS (no build step) |
