import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy()[:, 1].tolist())

    avg_loss = running_loss / len(loader)
    acc = 100.0 * float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # only one class present in this split/batch
        auc = float("nan")

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "y_true": all_labels,
        "y_pred": all_preds,
        "probs": all_probs,
    }


def fit_two_phase(model, train_loader, val_loader, device, cfg, desc=""):
    """
    Phase 1: warm up the classifier head with the backbone frozen.
    Phase 2: fine-tune the full network at a low LR, tracking the best-AUC checkpoint on val_loader.

    Returns the model loaded with its best-AUC weights, that checkpoint's val metrics, and training history.
    """
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.classifier.parameters(), lr=cfg.lr_phase1)
    for epoch in range(cfg.epochs_phase1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v = validate(model, val_loader, criterion, device)
        print(
            f"{desc} [phase1 {epoch + 1}/{cfg.epochs_phase1}] "
            f"train_loss={t_loss:.4f} acc={t_acc:.2f}% | "
            f"val_loss={v['loss']:.4f} acc={v['acc']:.2f}% f1={v['f1']:.4f}"
        )

    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr_phase2, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_auc = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = None

    for epoch in range(cfg.epochs_phase2):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v = validate(model, val_loader, criterion, device)
        scheduler.step(v["loss"])

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v["loss"])
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v["acc"])

        print(
            f"{desc} [phase2 {epoch + 1}/{cfg.epochs_phase2}] "
            f"train_loss={t_loss:.4f} acc={t_acc:.2f}% | "
            f"val_loss={v['loss']:.4f} acc={v['acc']:.2f}% auc={v['auc']:.4f}"
        )

        if v["auc"] > best_auc:
            best_auc = v["auc"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = v

    if best_metrics is None:
        # AUC was undefined (nan) every epoch, e.g. a degenerate split; fall back to the last epoch.
        best_metrics = v
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_metrics, history
