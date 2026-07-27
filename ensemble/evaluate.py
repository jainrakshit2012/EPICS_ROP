import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import auc, confusion_matrix, roc_curve

from dataset import ROPDataset, make_loader
from models import MODEL_NAMES, build_model
from voting import compute_metrics, hard_voting, mix_voting, soft_voting, weighted_average_voting


@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().ravel()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().ravel().tolist())
    return np.array(all_labels), np.array(all_probs)


def save_confusion_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "ROP"], yticklabels=["Normal", "ROP"])
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_roc(y_true, y_prob, path, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def print_table(title, rows):
    print(f"\n=== {title} ===")
    header = f"{'model':24s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'spec':>7s} {'f1':>7s} {'auc':>7s}"
    print(header)
    for name, m in rows:
        print(
            f"{name:24s} {m['accuracy']:.4f} {m['precision']:.4f} "
            f"{m['recall']:.4f} {m['specificity']:.4f} {m['f1']:.4f} {m['auc']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = ROPDataset(cfg["data_root"])

    with open(os.path.join(cfg["output_dir"], "split.json")) as f:
        split = json.load(f)
    test_idx = split["test"]

    with open(os.path.join(cfg["output_dir"], "val_summary.json")) as f:
        val_summary = json.load(f)

    probs_matrix = []
    y_true = None
    individual_results = {}

    for name in MODEL_NAMES:
        size = cfg["image_sizes"][name]
        test_loader = make_loader(full_dataset, test_idx, size, "val", cfg["batch_size"], False, cfg["num_workers"])

        model = build_model(name, pretrained=False).to(device)
        model.load_state_dict(torch.load(os.path.join(cfg["output_dir"], f"{name}.pth"), map_location=device))

        labels, probs = get_probs(model, test_loader, device)
        y_true = labels
        probs_matrix.append(probs)

        preds = (probs > 0.5).astype(int)
        individual_results[name] = compute_metrics(labels, preds, probs)
        save_roc(labels, probs, os.path.join(cfg["output_dir"], f"roc_{name}.png"), f"ROC - {name}")

    probs_matrix = np.array(probs_matrix)  # (n_models, n_test)
    weights = [val_summary[name]["acc"] for name in MODEL_NAMES]

    hard_pred = hard_voting(probs_matrix)
    soft_pred, soft_prob = soft_voting(probs_matrix)
    wav_pred, wav_prob = weighted_average_voting(probs_matrix, weights)
    mix_pred = mix_voting(probs_matrix)

    ensemble_results = {
        "hard_voting": compute_metrics(y_true, hard_pred, soft_prob),
        "soft_voting": compute_metrics(y_true, soft_pred, soft_prob),
        "weighted_average_voting": compute_metrics(y_true, wav_pred, wav_prob),
        "mix_voting": compute_metrics(y_true, mix_pred, soft_prob),
    }

    for method, pred in [
        ("hard_voting", hard_pred),
        ("soft_voting", soft_pred),
        ("weighted_average_voting", wav_pred),
        ("mix_voting", mix_pred),
    ]:
        save_confusion_matrix(
            y_true, pred,
            os.path.join(cfg["output_dir"], f"confusion_matrix_{method}.png"),
            method.replace("_", " ").title(),
        )

    print_table("Individual base models (held-out test set)", individual_results.items())
    print_table("Ensemble voting methods (held-out test set)", ensemble_results.items())

    with open(os.path.join(cfg["output_dir"], "test_results.json"), "w") as f:
        json.dump(
            {
                "individual": individual_results,
                "ensemble": ensemble_results,
                "weights": dict(zip(MODEL_NAMES, weights)),
                "n_test": len(test_idx),
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {cfg['output_dir']}/test_results.json")


if __name__ == "__main__":
    main()
