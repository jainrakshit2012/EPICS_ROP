import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from config import Config
from dataset import ROPDataset, get_transforms, make_loader, stratified_kfold_splits, stratified_split
from engine import fit_two_phase, validate
from model import get_rop_model

CLASS_NAMES = ["Normal", "ROP"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_confusion_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_training_curves(history, path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_cross_validation(full_dataset, trainval_idx, device, cfg):
    """Stratified k-fold CV over the trainval pool. This is the robust performance estimate."""
    cv_dir = os.path.join(cfg.output_dir, "cv")
    os.makedirs(cv_dir, exist_ok=True)

    folds = stratified_kfold_splits(trainval_idx, full_dataset.labels, cfg.n_folds, cfg.seed)
    fold_metrics = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        fold_dir = os.path.join(cv_dir, f"fold_{fold_i}")
        os.makedirs(fold_dir, exist_ok=True)

        train_loader = make_loader(
            full_dataset, train_idx, get_transforms("train", cfg.image_size),
            cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        )
        val_loader = make_loader(
            full_dataset, val_idx, get_transforms("val", cfg.image_size),
            cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        )

        model = get_rop_model(num_classes=2, pretrained=True).to(device)
        desc = f"[fold {fold_i + 1}/{cfg.n_folds}]"
        _, metrics, _ = fit_two_phase(model, train_loader, val_loader, device, cfg, desc=desc)

        report = classification_report(
            metrics["y_true"], metrics["y_pred"], target_names=CLASS_NAMES, zero_division=0
        )
        with open(os.path.join(fold_dir, "classification_report.txt"), "w") as f:
            f.write(report)
        save_confusion_matrix(
            metrics["y_true"], metrics["y_pred"],
            os.path.join(fold_dir, "confusion_matrix.png"),
            f"Confusion Matrix - Fold {fold_i}",
        )

        fold_metrics.append({"fold": fold_i, "acc": metrics["acc"], "f1": metrics["f1"], "auc": metrics["auc"]})
        print(f"{desc} acc={metrics['acc']:.2f}% f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}\n{report}")

    accs = [m["acc"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]
    aucs = [m["auc"] for m in fold_metrics]

    summary = {
        "folds": fold_metrics,
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs)),
    }
    with open(os.path.join(cfg.output_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(
        "\nCross-validation summary "
        f"(n_folds={cfg.n_folds}): acc={summary['acc_mean']:.2f}%±{summary['acc_std']:.2f} "
        f"f1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f} "
        f"auc={summary['auc_mean']:.4f}±{summary['auc_std']:.4f}"
    )
    return summary


def run_final_fit_and_test(full_dataset, trainval_idx, test_idx, device, cfg):
    """
    Retrains on the full trainval pool (same hyperparameters CV already validated - no further
    tuning here to avoid double-dipping), then evaluates exactly once on the held-out test set.
    """
    final_train_idx, final_val_idx = stratified_split(
        trainval_idx, full_dataset.labels, cfg.final_val_size, cfg.seed
    )

    train_loader = make_loader(
        full_dataset, final_train_idx, get_transforms("train", cfg.image_size),
        cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
    )
    val_loader = make_loader(
        full_dataset, final_val_idx, get_transforms("val", cfg.image_size),
        cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )
    test_loader = make_loader(
        full_dataset, test_idx, get_transforms("val", cfg.image_size),
        cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )

    model = get_rop_model(num_classes=2, pretrained=True).to(device)
    model, _, history = fit_two_phase(model, train_loader, val_loader, device, cfg, desc="[final]")

    save_training_curves(history, os.path.join(cfg.output_dir, "training_curves_final.png"))

    criterion = nn.CrossEntropyLoss()
    test_metrics = validate(model, test_loader, criterion, device)

    report = classification_report(
        test_metrics["y_true"], test_metrics["y_pred"], target_names=CLASS_NAMES, zero_division=0
    )
    with open(os.path.join(cfg.output_dir, "test_classification_report.txt"), "w") as f:
        f.write(report)
    save_confusion_matrix(
        test_metrics["y_true"], test_metrics["y_pred"],
        os.path.join(cfg.output_dir, "confusion_matrix_test.png"),
        "Confusion Matrix - Held-out Test Set",
    )

    checkpoint_path = os.path.join(cfg.output_dir, "best_rop_model.pth")
    torch.save(model.state_dict(), checkpoint_path)

    print(
        f"\nHeld-out test set (n={len(test_idx)}, never used for training/model selection): "
        f"acc={test_metrics['acc']:.2f}% f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f}\n{report}"
    )
    print(f"Saved final model to {checkpoint_path}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = ROPDataset(cfg.data_root)
    all_indices = list(range(len(full_dataset)))
    trainval_idx, test_idx = stratified_split(all_indices, full_dataset.labels, cfg.test_size, cfg.seed)
    print(f"Held-out test set: {len(test_idx)} images (untouched until final evaluation)")
    print(f"Train+val pool: {len(trainval_idx)} images ({cfg.n_folds}-fold CV)")

    run_cross_validation(full_dataset, trainval_idx, device, cfg)
    run_final_fit_and_test(full_dataset, trainval_idx, test_idx, device, cfg)


if __name__ == "__main__":
    main()
