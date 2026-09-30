"""
RoP Screening Web App
Classification : 5-fold EfficientNet-V2-S ensemble
                (Installer/EPICS_ROP_MSI_INSTALLER-main/model/)
Segmentation  : Tri-Modal – Demarcation Ridge, Optic Disc, Blood Vessels
                (Segmentation/SEGMENTATION-OD-BV-RIDGE/model/)
"""
import base64
import io
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths to sibling projects
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECTS = os.path.dirname(os.path.dirname(_THIS_DIR))           # …/Projects
_INSTALLER_ROOT = os.path.join(_PROJECTS, "Installer", "EPICS_ROP_MSI_INSTALLER-main")
_SEG_ROOT       = os.path.join(_PROJECTS, "Segmentation", "SEGMENTATION-OD-BV-RIDGE")
_CLASSIFIER_MODEL_DIR = os.path.join(_INSTALLER_ROOT, "model")
_SEG_MODEL_DIR        = os.path.join(_SEG_ROOT, "model")

# Add segmentation package to path (its 'app' package has the predictor)
if _SEG_ROOT not in sys.path:
    sys.path.insert(0, _SEG_ROOT)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# ---------------------------------------------------------------------------
# Classification – 5-fold EfficientNet-V2-S ensemble (inlined, no import)
# ---------------------------------------------------------------------------
_clf_ensemble = None
_clf_device   = None


def _clf_models_available():
    return all(
        os.path.exists(os.path.join(_CLASSIFIER_MODEL_DIR, f"efficientnet_v2_s_fold{i}.pth"))
        for i in range(1, 6)
    )


def _load_clf_ensemble():
    global _clf_ensemble, _clf_device
    if _clf_ensemble is not None:
        return _clf_ensemble, _clf_device

    _clf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not _clf_models_available():
        return None, _clf_device

    ensemble = []
    for i in range(1, 6):
        path = os.path.join(_CLASSIFIER_MODEL_DIR, f"efficientnet_v2_s_fold{i}.pth")
        model = tv_models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model.load_state_dict(torch.load(path, map_location=_clf_device))
        model.to(_clf_device)
        model.eval()
        ensemble.append(model)

    _clf_ensemble = ensemble
    return _clf_ensemble, _clf_device


_clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image(pil_image):
    """Returns ('ROP'|'NoRoP', is_rop bool)."""
    ensemble, device = _load_clf_ensemble()
    if ensemble is None:
        return "NoRoP", False

    tensor = _clf_transform(pil_image).unsqueeze(0).to(device)
    probs = []
    with torch.no_grad():
        for model in ensemble:
            prob = torch.sigmoid(model(tensor)).item()
            probs.append(prob)

    avg = float(np.mean(probs))
    is_rop = avg >= 0.5
    return ("ROP" if is_rop else "NoRoP"), is_rop


# ---------------------------------------------------------------------------
# Segmentation – Tri-Modal (Ridge + OD + BV)
# ---------------------------------------------------------------------------
_seg_predictor = None


def _seg_models_available():
    needed = ["best_model_RIDGE_manet.pth", "best_model_OD.pth", "optimized_best_model_BV.pth"]
    return all(os.path.exists(os.path.join(_SEG_MODEL_DIR, n)) for n in needed)


def _load_seg_predictor():
    global _seg_predictor
    if _seg_predictor is not None:
        return _seg_predictor
    if not _seg_models_available():
        return None
    try:
        from app.predictor import ROPPredictor as SegPredictor
        _seg_predictor = SegPredictor(model_dir=_SEG_MODEL_DIR, lazy_load=False)
        return _seg_predictor
    except Exception as e:
        print(f"[Segmentation] Failed to load: {e}")
        return None


def run_segmentation(pil_image):
    """
    Returns dict with base64 JPEG for 4 overlays:
      combined, ridge, optic_disc, blood_vessels
    Falls back to GradCAM if segmentation models aren't available.
    """
    predictor = _load_seg_predictor()

    if predictor is None:
        b64 = _gradcam_fallback(pil_image)
        return {
            "combined":      b64,
            "ridge":         b64,
            "optic_disc":    b64,
            "blood_vessels": b64,
            "fallback": True,
        }

    img_np = np.array(pil_image.convert("RGB"))
    result = predictor.predict_all(img_np)
    overlays = result["overlays"]

    def _to_b64(rgb_np):
        buf = io.BytesIO()
        Image.fromarray(rgb_np.astype(np.uint8)).save(buf, format="JPEG", quality=90)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    return {
        "combined":      _to_b64(overlays["combined"]),
        "ridge":         _to_b64(overlays["ridge"]),
        "optic_disc":    _to_b64(overlays["optic_disc"]),
        "blood_vessels": _to_b64(overlays["blood_vessels"]),
        "fallback": False,
        "ridge_detected":   result["results"]["ridge"]["detected"],
        "od_detected":      result["results"]["optic_disc"]["detected"],
        "bv_detected":      result["results"]["blood_vessels"]["detected"],
    }


# ---------------------------------------------------------------------------
# GradCAM fallback (used if segmentation models not found)
# ---------------------------------------------------------------------------
def _gradcam_fallback(pil_image):
    ensemble, device = _load_clf_ensemble()
    if ensemble is None:
        return _clahe_fallback(pil_image)

    import torch.nn.functional as F
    from torchvision import models as _tv

    model = ensemble[-1]  # use fold 5 as representative
    tensor = _clf_transform(pil_image).unsqueeze(0).to(device).requires_grad_(True)

    acts, grads = {}, {}
    fh = model.features[-1].register_forward_hook(lambda m, i, o: acts.update({"v": o.detach()}))
    bh = model.features[-1].register_full_backward_hook(lambda m, gi, go: grads.update({"v": go[0].detach()}))

    out = model(tensor)
    model.zero_grad()
    out[0, 0].backward()
    fh.remove(); bh.remove()

    a, g = acts["v"].squeeze(0), grads["v"].squeeze(0)
    cam = F.relu((g.mean(dim=(1,2))[:, None, None] * a).sum(0))
    cam = (cam - cam.min()) / max((cam.max() - cam.min()).item(), 1e-6)
    cam_np = cam.cpu().numpy()
    cam_resized = np.array(Image.fromarray((cam_np * 255).astype(np.uint8)).resize(pil_image.size, Image.BILINEAR))
    return _heatmap_overlay(pil_image, cam_resized)


def _clahe_fallback(pil_image):
    try:
        import cv2
        img = np.array(pil_image.convert("RGB"))
        clahe = cv2.createCLAHE(3.0, (8, 8))
        enhanced = clahe.apply(img[:, :, 1])
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1)
        mask = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        return _heatmap_overlay(pil_image, mask)
    except ImportError:
        img = np.array(pil_image.convert("RGB")).astype(float)
        g = img[:, :, 1]
        mask = ((g - g.min()) / max(g.max() - g.min(), 1) * 255).astype(np.uint8)
        return _heatmap_overlay(pil_image, mask)


def _heatmap_overlay(pil_image, mask_np):
    base = np.array(pil_image.convert("RGB")).astype(np.float32)
    h, w = base.shape[:2]
    mask = np.array(Image.fromarray(mask_np.astype(np.uint8)).resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0
    ov = base.copy()
    ov[:, :, 0] = np.clip(base[:, :, 0] + mask * 120, 0, 255)
    ov[:, :, 1] = np.clip(base[:, :, 1] * (1 - mask * 0.4), 0, 255)
    ov[:, :, 2] = np.clip(base[:, :, 2] * (1 - mask * 0.4), 0, 255)
    blended = (base * 0.45 + ov * 0.55).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _image_to_b64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html",
        clf_available=_clf_models_available(),
        seg_available=_seg_models_available(),
    )


@app.route("/predict", methods=["POST"])
def predict():
    files        = request.files.getlist("images")
    patient_name = request.form.get("patient_name", "").strip()
    patient_id   = request.form.get("patient_id", "").strip()
    session_ref  = request.form.get("session_ref", "").strip()

    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    results, any_rop = [], False

    for f in files:
        filename = f.filename or "unknown"
        try:
            img   = Image.open(f.stream).convert("RGB")
            b64   = _image_to_b64(img)
            label, is_rop = predict_image(img)
            if is_rop:
                any_rop = True
            results.append({"filename": filename, "label": label, "is_rop": is_rop, "image_b64": b64})
        except Exception as e:
            results.append({"filename": filename, "label": "Error", "is_rop": False,
                            "image_b64": None, "error": str(e)})

    return jsonify({
        "patient_name":      patient_name,
        "patient_id":        patient_id,
        "session_ref":       session_ref,
        "patient_diagnosis": "ROP" if any_rop else "NoRoP",
        "images":            results,
    })


@app.route("/segment", methods=["POST"])
def segment():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image"}), 400
    try:
        img = Image.open(f.stream).convert("RGB")
        data = run_segmentation(img)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
