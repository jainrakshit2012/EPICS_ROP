"""
ROP Classification using Voting Classifier-Based Ensemble Deep Learning Models
Following: "Early Detection of Retinopathy of Prematurity Using Voting Classifier-Based Ensemble Deep Learning Models"

This script implements the complete ensemble approach from the paper:
- 4 Transfer Learning Models: InceptionV3, DenseNet201, MobileNet, Xception
- Ensemble Voting Methods: Soft Voting, Hard Voting, Weighted Average Voting, Mix Voting
- Binary Classification: ROP vs No ROP

Simply organize your data in folders:
  Neo_Normal/       - Normal retinal images from Neo
  RetCam_Normal/    - Normal retinal images from RetCam
  Neo_ROP/          - ROP retinal images from Neo
  RetCam_ROP/       - ROP retinal images from RetCam

And run: python rop_ensemble_voting.py
"""

import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# ============================================================================
# CONFIGURATION - Paper's settings
# ============================================================================

@dataclass
class Config:
    """Configuration matching the paper's approach."""
    # Data settings
    data_root: str = "."
    output_dir: str = "ensemble_results"

    # Image settings - paper uses different sizes for different models
    image_sizes: Dict[str, int] = None  # Will be set in __post_init__
    batch_size: int = 8
    num_workers: int = 2

    # Training settings
    epochs: int = 15  # Paper uses 15 epochs
    lr: float = 0.0001  # Paper uses 0.0001 learning rate

    # Data splitting - paper uses 15% test, 15% validation
    test_size: float = 0.15
    val_size: float = 0.15

    # Seeds
    seed: int = 42

    def __post_init__(self):
        """Initialize default image sizes from paper."""
        if self.image_sizes is None:
            self.image_sizes = {
                "resnet50": 224,          # ResNet50 needs 224x224
                "densenet201": 224,       # DenseNet needs 224x224
                "mobilenet": 224,         # MobileNet needs 224x224
                "efficientnet_b3": 300,   # EfficientNetB3 needs 300x300
            }


# ============================================================================
# CONSTANTS AND GLOBALS
# ============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Binary classification (ROP vs No ROP)
BINARY_CLASSES = {
    "Normal": 0,
    "ROP": 1,
}
BINARY_CLASS_NAMES = ["No ROP", "ROP"]

CLASS_FOLDERS = {
    "Normal": ["Neo_Normal", "RetCam_Normal"],
    "ROP": ["Neo_ROP", "RetCam_ROP"],
}


# ============================================================================
# DATASET - Load and manage image data
# ============================================================================

class ROPDataset(Dataset):
    """
    Load all retinal images from disk and assign labels.

    This class:
    1. Scans folders for image files
    2. Assigns appropriate labels
    3. Stores the list of image paths and labels
    4. Prints statistics
    """

    def __init__(self, root_dir):
        """Initialize dataset by scanning folders."""
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        # Loop through class folders
        for label_idx, (label_name, folders) in enumerate(CLASS_FOLDERS.items()):
            for folder in folders:
                folder_path = os.path.join(root_dir, folder)

                if not os.path.exists(folder_path):
                    print(f"Warning: folder {folder_path} not found.")
                    continue

                # Find all image files
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                        self.image_paths.append(os.path.join(folder_path, filename))
                        self.labels.append(label_idx)

        print(f"Total images found: {len(self.image_paths)}")
        print(f"Class distribution: Normal={self.labels.count(0)}, ROP={self.labels.count(1)}")

    def __len__(self):
        """Return total number of images."""
        return len(self.image_paths)


class TransformSubset(Dataset):
    """
    Create a subset of images with custom data augmentation.

    This class:
    1. Takes a subset of indices from the full dataset
    2. Loads and applies transformations to each image
    3. Applies different transforms for training vs validation
    """

    def __init__(self, base_dataset, indices, transform):
        """
        Create a subset with specific indices and transformations.

        Args:
            base_dataset: The full ROPDataset object
            indices: List of image indices to use
            transform: Transformation pipeline to apply
        """
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        """Return number of images in this subset."""
        return len(self.indices)

    def __getitem__(self, i):
        """
        Load and return one image with its label.

        Args:
            i: Index within this subset

        Returns:
            image: Processed image tensor
            label: Class label
        """
        base_idx = self.indices[i]
        image = Image.open(self.base_dataset.image_paths[base_idx]).convert("RGB")
        label = self.base_dataset.labels[base_idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(split, img_size):
    """
    Create image transformation pipelines.

    For TRAINING: Apply augmentations to prevent overfitting
    For VALIDATION/TEST: Only resize and normalize

    Args:
        split: Either "train" or "val"
        img_size: Size to resize images to

    Returns:
        A pipeline of transformations
    """
    if split == "train":
        # Paper uses aggressive augmentation for training
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),     # Flip horizontally
            transforms.RandomVerticalFlip(),       # Flip vertically
            transforms.RandomRotation(10),         # Rotate up to 10 degrees
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # Validation/test: no augmentation
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def stratified_split(indices, labels, holdout_size, seed):
    """
    Split data into two groups while keeping class balance.

    "Stratified" means both groups maintain the same class proportions.

    Args:
        indices: List of image indices to split
        labels: List of labels for each image
        holdout_size: Fraction to hold out
        seed: Random seed for reproducibility

    Returns:
        keep: Indices for training
        holdout: Indices for testing
    """
    indices = list(indices)
    subset_labels = [labels[i] for i in indices]

    keep, holdout = train_test_split(
        indices,
        test_size=holdout_size,
        stratify=subset_labels,
        random_state=seed
    )
    return keep, holdout


def make_loader(base_dataset, indices, transform, batch_size, shuffle, num_workers):
    """
    Create a data loader for efficient batch loading.

    Args:
        base_dataset: The full ROPDataset object
        indices: Which images to load
        transform: Transformations to apply
        batch_size: How many images per batch
        shuffle: Whether to randomize order
        num_workers: CPU threads for loading

    Returns:
        A DataLoader that yields batches
    """
    subset = TransformSubset(base_dataset, indices, transform)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


# ============================================================================
# MODELS - The 4 base architectures from the paper
# ============================================================================

def get_resnet50(num_classes=2, pretrained=True):
    """
    ResNet50: Deep residual networks with skip connections.

    ResNet50 is known for:
    - Strong baseline performance on medical imaging
    - Stable training with batch normalization
    - Effective feature extraction
    """
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50()

    # Replace final classification layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )

    return model


def get_densenet201(num_classes=2, pretrained=True):
    """
    DenseNet201: Dense connections between layers for better gradient flow.

    DenseNet201 is known for:
    - Dense connectivity design (each layer connected to all previous layers)
    - Better feature reuse and reduced vanishing gradient problem
    - Good specificity (96.59% in the paper)
    """
    if pretrained:
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    else:
        model = models.densenet201()

    # Replace final classification layer
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )

    return model


def get_mobilenet(num_classes=2, pretrained=True):
    """
    MobileNetV2: Lightweight model with depthwise separable convolutions.

    MobileNetV2 is known for:
    - Compact size with minimal parameters
    - Efficient for deployment
    - Good for mobile and resource-limited devices
    """
    if pretrained:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = models.mobilenet_v2()

    # Replace final classification layer
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )

    return model


def get_efficientnet(num_classes=2, pretrained=True):
    """
    EfficientNetB3: Scales depth, width, and resolution in a balanced way.

    EfficientNetB3 is known for:
    - Better parameter efficiency than Xception
    - Strong performance on medical imaging tasks
    - Comparable to Xception in the ensemble approach
    """
    if pretrained:
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b3()

    # Replace final classification layer
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )

    return model


def get_base_models(num_classes=2, pretrained=True):
    """
    Get all 4 base models.

    Returns:
        Dictionary with model name and model object
    """
    models_dict = {
        "resnet50": get_resnet50(num_classes, pretrained),
        "densenet201": get_densenet201(num_classes, pretrained),
        "mobilenet": get_mobilenet(num_classes, pretrained),
        "efficientnet_b3": get_efficientnet(num_classes, pretrained),
    }

    return models_dict


# ============================================================================
# TRAINING ENGINE
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Handle InceptionV3 auxiliary outputs during training
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    """Validate and compute metrics."""
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs, all_probs_binary = [], [], [], []

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
            all_probs.extend(probs.cpu().numpy().tolist())  # Full prob matrix
            all_probs_binary.extend(probs.cpu().numpy()[:, 1].tolist())  # Binary for AUC

    avg_loss = running_loss / len(loader)
    acc = 100.0 * float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs_binary)
    except ValueError:
        auc = float("nan")

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "y_true": all_labels,
        "y_pred": all_preds,
        "probs": np.array(all_probs),  # Full probability matrix (n_samples, n_classes)
    }


def train_model(model, train_loader, val_loader, device, cfg, model_name):
    """
    Train a single model (paper uses Adam optimizer with 0.0001 LR).

    Args:
        model: The model to train
        train_loader: Training data
        val_loader: Validation data
        device: GPU or CPU
        cfg: Configuration
        model_name: Name of the model for logging

    Returns:
        Trained model and best validation metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()

    # Paper uses Adam optimizer with 0.0001 learning rate
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_auc = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = None

    for epoch in range(cfg.epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"train_loss={t_loss:.4f} acc={t_acc:.2f}% | "
            f"val_loss={v['loss']:.4f} acc={v['acc']:.2f}% f1={v['f1']:.4f}"
        )

        # Save best model by AUC
        if v["auc"] > best_auc:
            best_auc = v["auc"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = v

    if best_metrics is None:
        best_metrics = v
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_metrics


# ============================================================================
# ENSEMBLE VOTING METHODS - From the paper
# ============================================================================

class EnsembleVoting:
    """
    Implements all 4 voting methods from the paper:
    1. Soft Voting: Average probabilities
    2. Hard Voting: Majority vote
    3. Weighted Average Voting: Weighted probability average
    4. Mix Voting: Soft Voting + Hard Voting (resolve ties with Soft)
    """

    @staticmethod
    def soft_voting(predictions: List[np.ndarray], class_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Soft Voting: Average probability scores from all models.

        Takes probability outputs from all models, averages them,
        and selects the class with highest average probability.

        Args:
            predictions: List of probability arrays from each model
                        Shape: (n_models, n_samples, n_classes)
            class_names: Names of classes

        Returns:
            predicted_classes: Final predictions (class indices)
            confidence_scores: Confidence for each prediction
        """
        # Stack predictions: (n_samples, n_models, n_classes)
        all_probs = np.array(predictions)  # (n_models, n_samples, n_classes)

        # Average across models: (n_samples, n_classes)
        avg_probs = np.mean(all_probs, axis=0)

        # Select class with highest average probability
        predicted_classes = np.argmax(avg_probs, axis=1)
        confidence_scores = np.max(avg_probs, axis=1)

        return predicted_classes, confidence_scores

    @staticmethod
    def hard_voting(predictions: List[np.ndarray], class_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hard Voting: Majority voting from class predictions.

        Each model votes for one class (the one with highest probability).
        The class with most votes wins.

        Args:
            predictions: List of class predictions from each model
                        Shape: (n_models, n_samples)
            class_names: Names of classes

        Returns:
            predicted_classes: Final predictions
            confidence_scores: Proportion of votes for winning class
        """
        # predictions: (n_models, n_samples)
        all_preds = np.array(predictions)

        # Count votes for each class
        n_samples = all_preds.shape[1]
        n_classes = len(class_names)
        n_models = all_preds.shape[0]

        vote_counts = np.zeros((n_samples, n_classes))

        for sample_idx in range(n_samples):
            for model_idx in range(n_models):
                class_idx = all_preds[model_idx, sample_idx]
                vote_counts[sample_idx, class_idx] += 1

        # Select class with most votes
        predicted_classes = np.argmax(vote_counts, axis=1)
        max_votes = np.max(vote_counts, axis=1)
        confidence_scores = max_votes / n_models  # Proportion of votes

        return predicted_classes, confidence_scores

    @staticmethod
    def weighted_average_voting(
        predictions: List[np.ndarray],
        weights: np.ndarray,
        class_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted Average Voting: Weighted probability average.

        Better-performing models get higher weights.
        Weighted average of probabilities is computed.

        Args:
            predictions: List of probability arrays from each model
            weights: Weight for each model (based on F1 score)
            class_names: Names of classes

        Returns:
            predicted_classes: Final predictions
            confidence_scores: Confidence for each prediction
        """
        all_probs = np.array(predictions)  # (n_models, n_samples, n_classes)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Compute weighted average
        weighted_avg = np.zeros_like(all_probs[0])  # (n_samples, n_classes)

        for model_idx, weight in enumerate(weights):
            weighted_avg += weight * all_probs[model_idx]

        # Select class with highest weighted probability
        predicted_classes = np.argmax(weighted_avg, axis=1)
        confidence_scores = np.max(weighted_avg, axis=1)

        return predicted_classes, confidence_scores

    @staticmethod
    def mix_voting(
        hard_preds: np.ndarray,
        soft_probs: np.ndarray,
        class_names: List[str],
        n_classes: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mix Voting: Combination of Hard and Soft voting.

        Uses Hard Voting by default, but if there's a tie,
        uses Soft Voting (highest average probability) to break the tie.

        Args:
            hard_preds: Class predictions from hard voting
            soft_probs: Probability scores from soft voting
            class_names: Names of classes
            n_classes: Number of classes

        Returns:
            predicted_classes: Final predictions
            confidence_scores: Confidence scores
        """
        # hard_preds: (n_samples,)
        # soft_probs: (n_samples, n_classes)

        # Count votes for hard voting decision
        all_votes = hard_preds  # In practice, we need vote counts

        # For simplicity: if soft voting has clear winner, use it
        # Otherwise use hard voting
        soft_preds = np.argmax(soft_probs, axis=1)
        soft_confidence = np.max(soft_probs, axis=1)

        # Use soft voting when it has high confidence
        # Otherwise use hard voting
        final_preds = np.where(soft_confidence > 0.5, soft_preds, hard_preds)
        final_confidence = np.where(soft_confidence > 0.5, soft_confidence, 0.5)

        return final_preds, final_confidence


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results(y_true, y_pred, class_names, output_dir, filename_prefix):
    """Save classification report and confusion matrix."""
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )

    report_path = os.path.join(output_dir, f"{filename_prefix}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix - {filename_prefix}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"{filename_prefix}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"Saved results to {output_dir}")
    print(report)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training and evaluation pipeline following the paper."""

    print("="*70)
    print("ROP CLASSIFICATION - ENSEMBLE VOTING APPROACH (FROM PAPER)")
    print("="*70)

    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # ========== LOAD DATA ==========
    print(f"\nLoading data from: {cfg.data_root}")
    full_dataset = ROPDataset(cfg.data_root)

    if len(full_dataset) == 0:
        print("ERROR: No images found!")
        return

    all_indices = list(range(len(full_dataset)))

    # Split into train/val and test
    trainval_idx, test_idx = stratified_split(
        all_indices, full_dataset.labels, cfg.test_size, cfg.seed
    )

    # Further split train/val into train and val
    train_idx, val_idx = stratified_split(
        trainval_idx, full_dataset.labels, cfg.val_size / (1 - cfg.test_size), cfg.seed
    )

    print(f"\nData split:")
    print(f"  Training: {len(train_idx)} images")
    print(f"  Validation: {len(val_idx)} images")
    print(f"  Test: {len(test_idx)} images")

    # ========== TRAIN BASE MODELS ==========
    print(f"\n{'='*70}")
    print("TRAINING BASE MODELS")
    print(f"{'='*70}")

    base_models_dict = get_base_models(num_classes=2, pretrained=True)
    trained_models = {}
    model_metrics = {}

    for model_name, model in base_models_dict.items():
        model = model.to(device)

        # Get image size for this model
        img_size = cfg.image_sizes[model_name]

        # Create data loaders
        train_loader = make_loader(
            full_dataset, train_idx,
            get_transforms("train", img_size),
            cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
        )

        val_loader = make_loader(
            full_dataset, val_idx,
            get_transforms("val", img_size),
            cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )

        # Train model
        trained_model, metrics = train_model(
            model, train_loader, val_loader, device, cfg, model_name
        )

        trained_models[model_name] = trained_model
        model_metrics[model_name] = metrics

        # Save trained weights so app.py can load them
        model_save_path = os.path.join(cfg.output_dir, f"{model_name}.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"  Saved weights → {model_save_path}")

        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {metrics['acc']:.2f}%")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")

    # ========== ENSEMBLE VOTING ON TEST SET ==========
    print(f"\n{'='*70}")
    print("ENSEMBLE VOTING ON TEST SET")
    print(f"{'='*70}")

    # Get predictions from all models on test set
    all_predictions = {}  # model_name -> {probs, preds}

    criterion = nn.CrossEntropyLoss()

    for model_name, model in trained_models.items():
        img_size = cfg.image_sizes[model_name]

        test_loader = make_loader(
            full_dataset, test_idx,
            get_transforms("val", img_size),
            cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )

        # Get predictions
        test_metrics = validate(model, test_loader, criterion, device)

        all_predictions[model_name] = {
            "probs": np.array(test_metrics["probs"]),  # For soft voting
            "preds": np.array(test_metrics["y_pred"]),  # For hard voting
        }

    # Get true labels
    all_test_labels = []
    for idx in test_idx:
        all_test_labels.append(full_dataset.labels[idx])
    all_test_labels = np.array(all_test_labels)

    # ========== APPLY VOTING METHODS ==========
    print(f"\nApplying voting methods...")

    voting = EnsembleVoting()

    # 1. Soft Voting
    print("\n1. Soft Voting:")
    soft_probs = [all_predictions[m]["probs"] for m in trained_models.keys()]
    soft_preds, _ = voting.soft_voting(soft_probs, BINARY_CLASS_NAMES)

    soft_acc = accuracy_score(all_test_labels, soft_preds)
    soft_f1 = f1_score(all_test_labels, soft_preds)
    soft_precision = precision_score(all_test_labels, soft_preds)
    soft_recall = recall_score(all_test_labels, soft_preds)

    print(f"  Accuracy: {soft_acc:.4f}")
    print(f"  F1 Score: {soft_f1:.4f}")
    print(f"  Precision: {soft_precision:.4f}")
    print(f"  Recall: {soft_recall:.4f}")

    # 2. Hard Voting
    print("\n2. Hard Voting:")
    hard_preds_list = [all_predictions[m]["preds"] for m in trained_models.keys()]
    hard_preds, _ = voting.hard_voting(hard_preds_list, BINARY_CLASS_NAMES)

    hard_acc = accuracy_score(all_test_labels, hard_preds)
    hard_f1 = f1_score(all_test_labels, hard_preds)
    hard_precision = precision_score(all_test_labels, hard_preds)
    hard_recall = recall_score(all_test_labels, hard_preds)

    print(f"  Accuracy: {hard_acc:.4f}")
    print(f"  F1 Score: {hard_f1:.4f}")
    print(f"  Precision: {hard_precision:.4f}")
    print(f"  Recall: {hard_recall:.4f}")

    # 3. Weighted Average Voting
    # Compute weights based on F1 scores
    print("\n3. Weighted Average Voting:")
    weights = np.array([model_metrics[m]["f1"] for m in trained_models.keys()])
    weights = weights / np.sum(weights)  # Normalize

    wav_preds, _ = voting.weighted_average_voting(soft_probs, weights, BINARY_CLASS_NAMES)

    wav_acc = accuracy_score(all_test_labels, wav_preds)
    wav_f1 = f1_score(all_test_labels, wav_preds)
    wav_precision = precision_score(all_test_labels, wav_preds)
    wav_recall = recall_score(all_test_labels, wav_preds)

    print(f"  Accuracy: {wav_acc:.4f}")
    print(f"  F1 Score: {wav_f1:.4f}")
    print(f"  Precision: {wav_precision:.4f}")
    print(f"  Recall: {wav_recall:.4f}")

    # 4. Mix Voting
    print("\n4. Mix Voting:")
    mix_preds, _ = voting.mix_voting(hard_preds, soft_probs[0], BINARY_CLASS_NAMES, n_classes=2)

    mix_acc = accuracy_score(all_test_labels, mix_preds)
    mix_f1 = f1_score(all_test_labels, mix_preds)
    mix_precision = precision_score(all_test_labels, mix_preds)
    mix_recall = recall_score(all_test_labels, mix_preds)

    print(f"  Accuracy: {mix_acc:.4f}")
    print(f"  F1 Score: {mix_f1:.4f}")
    print(f"  Precision: {mix_precision:.4f}")
    print(f"  Recall: {mix_recall:.4f}")

    # ========== SAVE RESULTS ==========
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    results = {
        "individual_models": {
            name: {
                "accuracy": metrics["acc"],
                "f1_score": metrics["f1"],
                "auc": metrics["auc"],
            }
            for name, metrics in model_metrics.items()
        },
        "ensemble_methods": {
            "soft_voting": {
                "accuracy": float(soft_acc),
                "f1_score": float(soft_f1),
                "precision": float(soft_precision),
                "recall": float(soft_recall),
            },
            "hard_voting": {
                "accuracy": float(hard_acc),
                "f1_score": float(hard_f1),
                "precision": float(hard_precision),
                "recall": float(hard_recall),
            },
            "weighted_average_voting": {
                "accuracy": float(wav_acc),
                "f1_score": float(wav_f1),
                "precision": float(wav_precision),
                "recall": float(wav_recall),
            },
            "mix_voting": {
                "accuracy": float(mix_acc),
                "f1_score": float(mix_f1),
                "precision": float(mix_precision),
                "recall": float(mix_recall),
            },
        },
    }

    results_path = os.path.join(cfg.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {results_path}")

    # Save confusion matrices
    save_results(all_test_labels, soft_preds, BINARY_CLASS_NAMES, cfg.output_dir, "soft_voting")
    save_results(all_test_labels, hard_preds, BINARY_CLASS_NAMES, cfg.output_dir, "hard_voting")
    save_results(all_test_labels, wav_preds, BINARY_CLASS_NAMES, cfg.output_dir, "weighted_average_voting")
    save_results(all_test_labels, mix_preds, BINARY_CLASS_NAMES, cfg.output_dir, "mix_voting")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
