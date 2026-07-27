import argparse
import json
import os

import numpy as np
import torch
import yaml
from PIL import Image

from dataset import get_transform
from models import MODEL_NAMES, build_model
from voting import hard_voting, mix_voting, soft_voting, weighted_average_voting


def predict(image_path, cfg, method="mix"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(cfg["output_dir"], "val_summary.json")) as f:
        val_summary = json.load(f)

    image = Image.open(image_path).convert("RGB")
    probs = []
    for name in MODEL_NAMES:
        size = cfg["image_sizes"][name]
        tensor = get_transform(size, "val")(image).unsqueeze(0).to(device)

        model = build_model(name, pretrained=False).to(device)
        model.load_state_dict(torch.load(os.path.join(cfg["output_dir"], f"{name}.pth"), map_location=device))
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()
        probs.append(prob)
        print(f"{name:12s} P(ROP)={prob:.4f}")

    probs_matrix = np.array(probs).reshape(-1, 1)
    weights = [val_summary[name]["acc"] for name in MODEL_NAMES]

    if method == "hard":
        pred, conf = hard_voting(probs_matrix)[0], None
    elif method == "soft":
        pred_arr, prob_arr = soft_voting(probs_matrix)
        pred, conf = pred_arr[0], prob_arr[0]
    elif method == "weighted":
        pred_arr, prob_arr = weighted_average_voting(probs_matrix, weights)
        pred, conf = pred_arr[0], prob_arr[0]
    else:  # mix
        pred, conf = mix_voting(probs_matrix)[0], probs_matrix.mean()

    label = "ROP" if pred == 1 else "Normal"
    conf_str = f" (P(ROP)={conf:.4f})" if conf is not None else ""
    print(f"\nEnsemble ({method} voting) prediction: {label}{conf_str}")
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--method", default="mix", choices=["hard", "soft", "weighted", "mix"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    predict(args.image, cfg, args.method)
