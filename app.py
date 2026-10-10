"""
RoP Screening Web App
Classification : EfficientNet-B3 (BEST MODEL/best_model.pth)
Segmentation  : Tri-Modal – Demarcation Ridge, Optic Disc, Blood Vessels
                (Segmentation/SEGMENTATION-OD-BV-RIDGE/model/)
"""
import base64
import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_BEST_MODEL  = os.path.join(_THIS_DIR, "..", "BEST MODEL", "best_model.pth")
_BEST_MODEL  = os.path.normpath(_BEST_MODEL)

_PROJECTS    = os.path.dirname(os.path.dirname(_THIS_DIR))
_SEG_ROOT    = os.path.join(_PROJECTS, "Segmentation", "SEGMENTATION-OD-BV-RIDGE")
_SEG_MODEL_DIR = os.path.join(_SEG_ROOT, "model")

# Add segmentation package to path
if _SEG_ROOT not in sys.path:
    sys.path.insert(0, _SEG_ROOT)

# Add this dir so we can import model.py / dataset.py
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model import get_rop_model
from dataset import get_transforms

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# ---------------------------------------------------------------------------
# Classification – EfficientNet-B3 best_model.pth
# ---------------------------------------------------------------------------
_clf_model  = None
_clf_device = None


def _clf_available():
    return os.path.exists(_BEST_MODEL)


def _load_clf():
    global _clf_model, _clf_device
    if _clf_model is not None:
        return _clf_model, _clf_device
    _clf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not _clf_available():
        return None, _clf_device
    m = get_rop_model(num_classes=2, pretrained=False)
    m.load_state_dict(torch.load(_BEST_MODEL, map_location=_clf_device))
    m.to(_clf_device)
    m.eval()
    _clf_model = m
    return _clf_model, _clf_device


def predict_image(pil_image):
    """Returns ('ROP' | 'NoRoP', is_rop bool)."""
    model, device = _load_clf()
    if model is None:
        return "NoRoP", False

    transform = get_transforms("val", 512)
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(1).item()   # 0 = Normal, 1 = ROP

    is_rop = pred == 1
    return ("ROP" if is_rop else "NoRoP"), is_rop


# ---------------------------------------------------------------------------
# Segmentation – Tri-Modal (Ridge + OD + BV)
# ---------------------------------------------------------------------------
_seg_predictor = None


def _seg_available():
    needed = ["best_model_RIDGE_manet.pth", "best_model_OD.pth", "optimized_best_model_BV.pth"]
    return all(os.path.exists(os.path.join(_SEG_MODEL_DIR, n)) for n in needed)


def _load_seg():
    global _seg_predictor
    if _seg_predictor is not None:
        return _seg_predictor
    if not _seg_available():
        return None
    try:
        from app.predictor import ROPPredictor as SegPredictor
        _seg_predictor = SegPredictor(model_dir=_SEG_MODEL_DIR, lazy_load=False)
        return _seg_predictor
    except Exception as e:
        print(f"[Segmentation] Load failed: {e}")
        return None


def run_segmentation(pil_image):
    predictor = _load_seg()

    if predictor is None:
        b64 = _gradcam_fallback(pil_image)
        return {"combined": b64, "ridge": b64, "optic_disc": b64, "blood_vessels": b64, "fallback": True}

    img_np = np.array(pil_image.convert("RGB"))
    result = predictor.predict_all(img_np)
    overlays = result["overlays"]

    def _enc(rgb):
        buf = io.BytesIO()
        Image.fromarray(rgb.astype(np.uint8)).save(buf, format="JPEG", quality=92)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    return {
        "combined":      _enc(overlays["combined"]),
        "ridge":         _enc(overlays["ridge"]),
        "optic_disc":    _enc(overlays["optic_disc"]),
        "blood_vessels": _enc(overlays["blood_vessels"]),
        "fallback": False,
        "ridge_detected": result["results"]["ridge"]["detected"],
        "od_detected":    result["results"]["optic_disc"]["detected"],
        "bv_detected":    result["results"]["blood_vessels"]["detected"],
    }


# ---------------------------------------------------------------------------
# GradCAM fallback
# ---------------------------------------------------------------------------
def _gradcam_fallback(pil_image):
    model, device = _load_clf()
    if model is None:
        return _clahe_fallback(pil_image)

    transform = get_transforms("val", 512)
    tensor = transform(pil_image).unsqueeze(0).to(device).requires_grad_(True)
    acts, grads = {}, {}

    fh = model.features[-1].register_forward_hook(lambda m, i, o: acts.update({"v": o.detach()}))
    bh = model.features[-1].register_full_backward_hook(lambda m, gi, go: grads.update({"v": go[0].detach()}))

    out = model(tensor)
    model.zero_grad()
    out[0, out.argmax(1).item()].backward()
    fh.remove(); bh.remove()

    a, g = acts["v"].squeeze(0), grads["v"].squeeze(0)
    cam = F.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
    cam = (cam - cam.min()) / max((cam.max() - cam.min()).item(), 1e-6)
    cam_np = cam.cpu().numpy()
    cam_resized = np.array(
        Image.fromarray((cam_np * 255).astype(np.uint8)).resize(pil_image.size, Image.BILINEAR)
    )
    return _heatmap_b64(pil_image, cam_resized)


def _clahe_fallback(pil_image):
    try:
        import cv2
        img = np.array(pil_image.convert("RGB"))
        clahe = cv2.createCLAHE(3.0, (8, 8))
        enhanced = clahe.apply(img[:, :, 1])
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1)
        return _heatmap_b64(pil_image, cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0))
    except Exception:
        g = np.array(pil_image.convert("RGB")).astype(float)[:, :, 1]
        mask = ((g - g.min()) / max(g.max() - g.min(), 1) * 255).astype(np.uint8)
        return _heatmap_b64(pil_image, mask)


def _heatmap_b64(pil_image, mask_np):
    base = np.array(pil_image.convert("RGB")).astype(np.float32)
    h, w = base.shape[:2]
    mask = np.array(
        Image.fromarray(mask_np.astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    ov = base.copy()
    ov[:, :, 0] = np.clip(base[:, :, 0] + mask * 120, 0, 255)
    ov[:, :, 1] = np.clip(base[:, :, 1] * (1 - mask * 0.4), 0, 255)
    ov[:, :, 2] = np.clip(base[:, :, 2] * (1 - mask * 0.4), 0, 255)
    blended = (base * 0.45 + ov * 0.55).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_b64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html",
        clf_available=_clf_available(),
        seg_available=_seg_available(),
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
            b64   = _img_b64(img)
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
        img  = Image.open(f.stream).convert("RGB")
        data = run_segmentation(img)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
