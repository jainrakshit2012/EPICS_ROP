import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import ROPDataset, make_loader, stratified_split
from models import MODEL_NAMES, build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().ravel()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().ravel().tolist())

    preds = [1 if p > 0.5 else 0 for p in all_probs]
    return {
        "acc": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = ROPDataset(cfg["data_root"])
    all_indices = list(range(len(full_dataset)))
    trainval_idx, test_idx = stratified_split(all_indices, full_dataset.labels, cfg["test_size"], cfg["seed"])
    train_idx, val_idx = stratified_split(trainval_idx, full_dataset.labels, cfg["val_size"], cfg["seed"])
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} (test set untouched until evaluate.py)")

    with open(os.path.join(cfg["output_dir"], "split.json"), "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)

    val_summary = {}
    for name in MODEL_NAMES:
        print(f"\n=== Training {name} ===")
        size = cfg["image_sizes"][name]
        train_loader = make_loader(full_dataset, train_idx, size, "train", cfg["batch_size"], True, cfg["num_workers"])
        val_loader = make_loader(full_dataset, val_idx, size, "val", cfg["batch_size"], False, cfg["num_workers"])

        model = build_model(name, pretrained=True).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

        best_f1 = -1.0
        best_state = None
        best_val_metrics = None

        for epoch in range(cfg["epochs"]):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"[{name}] epoch {epoch + 1}/{cfg['epochs']} "
                f"train_loss={train_loss:.4f} val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f}"
            )
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val_metrics = val_metrics

        model.load_state_dict(best_state)
        checkpoint_path = os.path.join(cfg["output_dir"], f"{name}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        val_summary[name] = best_val_metrics
        print(f"Saved {checkpoint_path} (best val_acc={best_val_metrics['acc']:.4f}, val_f1={best_val_metrics['f1']:.4f})")

    with open(os.path.join(cfg["output_dir"], "val_summary.json"), "w") as f:
        json.dump(val_summary, f, indent=2)
    print(f"\nSaved val_summary.json (used as Weighted Average Voting weights in evaluate.py)")


if __name__ == "__main__":
    main()
