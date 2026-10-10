# EPICS ROP Screening System

A clinical screening tool for **Retinopathy of Prematurity (RoP)** built for the EPICS / HVD programme.

- **Batch Classification** — EfficientNet-B3 (`BEST MODEL/best_model.pth`), 2-class softmax, predicts ROP / NoRoP per image and gives a single patient-level diagnosis
- **Tri-Modal Segmentation** — three segmentation models (Demarcation Ridge, Optic Disc, Blood Vessels) on-demand for any ROP-positive image
- **Web Interface** — Flask UI, up to 20 images per patient, patient ID fields, one-click segmentation

---

## Repository Structure

```
EPICS_ROP-main/          ← this folder (run app.py from here)
├── app.py               ← Flask entry point
├── model.py             ← EfficientNet-B3 factory  (get_rop_model)
├── dataset.py           ← Transforms / dataset helpers
├── engine.py            ← Train / validate loops
├── train.py             ← k-fold training pipeline
├── predict.py           ← CLI single-image inference
├── rop_ensemble_voting.py
├── config.yaml
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    ├── epics.jpeg       ← EPICS logo (header)
    └── hvd.jpeg         ← HVD logo   (header)

# Sibling folders required by the web app
../BEST MODEL/
│   └── best_model.pth   ← Classification weights (EfficientNet-B3, 2-class)

../Segmentation/SEGMENTATION-OD-BV-RIDGE/
│   ├── app/             ← Segmentation inference package
│   └── model/
│       ├── best_model_RIDGE_manet.pth      ← Demarcation Ridge
│       ├── best_model_OD.pth               ← Optic Disc
│       └── optimized_best_model_BV.pth     ← Blood Vessels
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r EPICS_ROP-main/requirements.txt
```

> **PyTorch** must be installed separately if it is not already present — see https://pytorch.org/get-started/locally/
>
> **RTX 50-series GPU (Blackwell):** install PyTorch nightly with CUDA 12.8.

### 2. Run the web app

```bash
cd EPICS_ROP-main
python app.py
```

Open **http://localhost:5000** in your browser.

### 3. Use the UI

1. Enter **Patient Name**, **Patient ID**, and **Session Reference**
2. Drag-and-drop or click to upload **up to 20 retinal images** (PNG, JPG, BMP, TIFF)
3. Images appear immediately with a white background
4. Click **Run Diagnosis**
   - Each image gets a **ROP** or **NoRoP** badge
   - If **any** image is ROP → patient = **ROP**; all normal → **NoRoP**
5. On any ROP image click **Run Segmentation** — four overlay tabs appear:
   - **Combined** · **Ridge** · **Optic Disc** · **Blood Vessels**

---

## Patient Diagnosis Logic

> If **any single image** is ROP → patient = **ROP**.  
> Only if **all images** are Normal → patient = **NoRoP**.

This conservative rule prioritises sensitivity — a missed ROP case is clinically more dangerous than a false positive.

---

## Model Details

### Classification — EfficientNet-B3

| Property | Value |
|----------|-------|
| File | `BEST MODEL/best_model.pth` |
| Architecture | EfficientNet-B3 + custom head |
| Input size | 512 × 512 |
| Output | 2-class softmax (index 0 = Normal, index 1 = ROP) |

Custom head (defined in `model.py`):

```python
nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, 2),   # 2 classes
)
```

### Segmentation — Tri-Modal

| Modality | Architecture | Model file |
|----------|-------------|------------|
| Demarcation Ridge | MAnet + EfficientNet-B4 | `best_model_RIDGE_manet.pth` |
| Optic Disc | UNet++ + EfficientNet-B4 | `best_model_OD.pth` |
| Blood Vessels | UNet++ + EfficientNet-B4 | `optimized_best_model_BV.pth` |

---

## Swapping / Changing Models

### Replace the classification model (drop-in)

The app loads one file:

```
BEST MODEL/best_model.pth
```

Steps:
1. Train your model (see `train.py` or `rop_ensemble_voting.py`)
2. Save weights: `torch.save(model.state_dict(), "best_model.pth")`
3. Drop the new file into `BEST MODEL/` (keep the filename `best_model.pth`)

The architecture must match `get_rop_model(num_classes=2)` from `model.py` (EfficientNet-B3 + custom head).

### Use a different architecture

Edit `app.py` — find `_load_clf()` and change the model instantiation:

```python
# current — EfficientNet-B3 with 2-class softmax
m = get_rop_model(num_classes=2, pretrained=False)
m.load_state_dict(torch.load(_BEST_MODEL, ...))
```

For example, to use a ResNet-50 with sigmoid output:

```python
import torchvision.models as tv_models
m = tv_models.resnet50(weights=None)
m.fc = torch.nn.Linear(m.fc.in_features, 1)
m.load_state_dict(torch.load(_BEST_MODEL, ...))
```

Then update `predict_image()` to match:

```python
# sigmoid single-output model:
prob  = torch.sigmoid(output)[0].item()
is_rop = prob >= 0.5
```

### Change the model file path

Edit the constant near the top of `app.py`:

```python
_BEST_MODEL = os.path.normpath(os.path.join(_THIS_DIR, "..", "BEST MODEL", "best_model.pth"))
# change to any absolute path, e.g.:
_BEST_MODEL = r"C:\models\my_custom_model.pth"
```

### Swap segmentation models

Replace any file in `Segmentation/SEGMENTATION-OD-BV-RIDGE/model/`:

| File | Required architecture |
|------|----------------------|
| `best_model_RIDGE_manet.pth` | `smp.MAnet(encoder_name="efficientnet-b4", in_channels=3, classes=1)` |
| `best_model_OD.pth` | `smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1)` |
| `optimized_best_model_BV.pth` | `smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1)` |

To change the segmentation model directory:

```python
# app.py
_SEG_MODEL_DIR = os.path.join(_SEG_ROOT, "model")
# → change to:
_SEG_MODEL_DIR = r"C:\path\to\seg_models"
```

---

## Training

### Train EfficientNet-B3 (single model)

```bash
python train.py --config config.yaml
```

Output: `results/best_rop_model.pth`

Smoke test (1 epoch):

```bash
python train.py --config config.smoke.yaml
```

### Train 4-model ensemble

```bash
python rop_ensemble_voting.py
```

Weights saved to `ensemble_results/{model_name}.pth`.

### CLI single-image inference

```bash
python predict.py path/to/image.jpg --checkpoint "BEST MODEL/best_model.pth"
```

---

## Dataset Layout

Set `data_root` in `config.yaml`. Four sub-folders must exist:

```
Neo_Normal/      Normal images (NeoNatal camera)
Neo_ROP/         ROP images    (NeoNatal camera)
RetCam_Normal/   Normal images (RetCam)
RetCam_ROP/      ROP images    (RetCam)
```

---

## Tests

```bash
pytest tests/ -v
```

Covers split integrity and model forward-pass shape. No GPU or dataset required.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Flask |
| Classification | PyTorch · EfficientNet-B3 |
| Segmentation | segmentation-models-pytorch · MAnet / UNet++ · EfficientNet-B4 |
| Image processing | OpenCV ≥ 5.0, Albumentations, scikit-image |
| Frontend | Vanilla HTML / CSS / JS (no build step) |
