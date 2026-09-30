"""
Standalone ROP Classification Pipeline
Consolidated from the full modular setup into a single file for easy deployment.

This script trains a deep learning model to classify retinal images into:
- Normal (healthy retinas)
- ROP (Retinopathy of Prematurity - diseased retinas)

The pipeline:
1. Loads images from organized folders (Neo_Normal, RetCam_Normal, Neo_ROP, RetCam_ROP)
2. Splits data into train/validation/test sets (stratified to keep class balance)
3. Trains an EfficientNet-B3 model using 5-fold cross-validation
4. Evaluates performance on a held-out test set
5. Saves the trained model and performance reports

Simply organize your data in the folder structure above and run this script!
"""

import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

# Data handling and visualization
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Deep learning with PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# Machine learning utilities
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


# ============================================================================
# CONFIGURATION - All the settings we can tweak
# ============================================================================

@dataclass
class Config:
    """
    Configuration object that holds all hyperparameters and settings.
    Adjust these values to change training behavior.
    """
    # Data settings - where to find images and save results
    data_root: str = "."  # Directory containing Neo_Normal/, Neo_ROP/, etc.
    output_dir: str = "results"  # Where to save trained models and reports

    # Image settings - how to process images
    image_size: int = 512  # Resize all images to 512x512 pixels
    batch_size: int = 8  # Process 8 images at a time (increase if you have more GPU memory)
    num_workers: int = 2  # Use 2 CPU threads to load images in parallel

    # Training settings - Phase 1: warm up the model head
    epochs_phase1: int = 5  # Train only the classification head for 5 epochs
    lr_phase1: float = 0.001  # Learning rate (step size for weight updates)

    # Training settings - Phase 2: fine-tune the entire model
    epochs_phase2: int = 25  # Train the whole model for 25 epochs
    lr_phase2: float = 0.00001  # Lower learning rate to avoid ruining pre-trained weights
    weight_decay: float = 0.0001  # Regularization to prevent overfitting

    # Data splitting - how to divide data
    test_size: float = 0.15  # Hold out 15% of data for final testing (never touch during training)
    n_folds: int = 5  # Use 5-fold cross-validation (train 5 different models on different splits)
    final_val_size: float = 0.1  # Use 10% of training data just for validation during final training

    # Reproducibility - use same seed for consistent results
    seed: int = 42


# ============================================================================
# DATASET - Load and manage image data
# ============================================================================

# These are the mean and standard deviation values from ImageNet (a large image dataset).
# We use these to normalize our images so they match what the pre-trained model expects.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Define the folder structure we expect
# Images should be organized as:
#   Neo_Normal/       - Normal retinal images from Neo camera
#   RetCam_Normal/    - Normal retinal images from RetCam camera
#   Neo_ROP/          - ROP retinal images from Neo camera
#   RetCam_ROP/       - ROP retinal images from RetCam camera
CLASS_FOLDERS = {
    "Normal": ["Neo_Normal", "RetCam_Normal"],  # Label 0 = Normal
    "ROP": ["Neo_ROP", "RetCam_ROP"],          # Label 1 = ROP (disease)
}

# Simple names for our two classes
CLASS_NAMES = ["Normal", "ROP"]


class ROPDataset(Dataset):
    """
    Load all retinal images from disk and assign labels.

    This class:
    1. Scans folders for image files
    2. Assigns label 0 (Normal) or 1 (ROP) based on folder name
    3. Stores the list of image paths and labels
    4. Prints statistics about how many images we found
    """

    def __init__(self, root_dir):
        """
        Initialize the dataset by scanning folders and finding all images.

        Args:
            root_dir: Directory containing the four class folders
        """
        self.root_dir = root_dir
        self.image_paths = []  # List to store full path of each image
        self.labels = []       # List to store label (0=Normal, 1=ROP) for each image

        # Loop through each class (Normal and ROP)
        for label_idx, (label_name, folders) in enumerate(CLASS_FOLDERS.items()):
            # Loop through each folder within that class
            for folder in folders:
                folder_path = os.path.join(root_dir, folder)

                # Check if folder exists
                if not os.path.exists(folder_path):
                    print(f"Warning: folder {folder_path} not found.")
                    continue

                # Look at each file in the folder
                for filename in os.listdir(folder_path):
                    # Only accept image files (ignore other file types)
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                        # Add the full path to the image and its label
                        self.image_paths.append(os.path.join(folder_path, filename))
                        self.labels.append(label_idx)

        # Print summary of what we found
        print(f"Total images found: {len(self.image_paths)}")
        print(f"Class distribution: Normal={self.labels.count(0)}, ROP={self.labels.count(1)}")

    def __len__(self):
        """Return total number of images in the dataset."""
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
            indices: List of image indices to use for this subset
            transform: Transformation pipeline to apply to each image
        """
        self.base_dataset = base_dataset
        self.indices = list(indices)  # Which images to use from the full dataset
        self.transform = transform     # How to augment/process each image

    def __len__(self):
        """Return number of images in this subset."""
        return len(self.indices)

    def __getitem__(self, i):
        """
        Load and return one image with its label.

        Args:
            i: Index within this subset (0 to len(self)-1)

        Returns:
            image: Processed image tensor
            label: Class label (0=Normal, 1=ROP)
        """
        # Convert subset index to full dataset index
        base_idx = self.indices[i]

        # Load the image from disk and convert to RGB (in case it's grayscale)
        image = Image.open(self.base_dataset.image_paths[base_idx]).convert("RGB")

        # Get the label for this image
        label = self.base_dataset.labels[base_idx]

        # Apply the transformation pipeline (resize, augment, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(split, img_size):
    """
    Create image transformation pipelines.

    For TRAINING: Apply aggressive augmentations to prevent overfitting
    For VALIDATION/TEST: Only resize and normalize (no random changes)

    Args:
        split: Either "train" or "val" (or anything else = validation mode)
        img_size: Size to resize images to (e.g., 512x512)

    Returns:
        A pipeline of transformations to apply to images
    """
    if split == "train":
        # Training transformations: make the image look different each time
        # This teaches the model to recognize ROP even with variations
        return transforms.Compose([
            # Resize to fixed size
            transforms.Resize((img_size, img_size)),
            # Randomly flip left-right (retinas can appear mirrored)
            transforms.RandomHorizontalFlip(),
            # Randomly flip up-down
            transforms.RandomVerticalFlip(),
            # Randomly rotate up to 20 degrees (camera angle variation)
            transforms.RandomRotation(20),
            # Randomly adjust brightness, contrast, saturation (lighting changes)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            # Convert from PIL image to PyTorch tensor (array of numbers)
            transforms.ToTensor(),
            # Normalize using ImageNet statistics so the model recognizes the images
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # Validation/test transformations: no augmentation, just standardization
    return transforms.Compose([
        # Resize to fixed size
        transforms.Resize((img_size, img_size)),
        # Convert from PIL image to PyTorch tensor
        transforms.ToTensor(),
        # Normalize using ImageNet statistics
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def stratified_split(indices, labels, holdout_size, seed):
    """
    Split data into two groups while keeping class balance.

    "Stratified" means: if your data is 60% Normal and 40% ROP,
    both the kept and holdout groups will also be 60% Normal and 40% ROP.
    This is important for fair training!

    Args:
        indices: List of image indices to split
        labels: List of labels for each image (0=Normal, 1=ROP)
        holdout_size: Fraction to hold out (e.g., 0.15 = 15%)
        seed: Random seed for reproducibility

    Returns:
        keep: Indices for training
        holdout: Indices for testing/validation
    """
    indices = list(indices)
    # Get the labels for only the indices we're splitting
    subset_labels = [labels[i] for i in indices]

    # Use sklearn's train_test_split with stratify to maintain class balance
    keep, holdout = train_test_split(
        indices,
        test_size=holdout_size,  # Fraction to hold out
        stratify=subset_labels,  # Keep class proportions equal
        random_state=seed        # Use seed for reproducible splits
    )
    return keep, holdout


def stratified_kfold_splits(indices, labels, n_folds, seed):
    """
    Split data into k folds for cross-validation.

    Cross-validation = train k different models on k different splits.
    This gives a more robust performance estimate than training just once.

    Example with n_folds=5:
    - Fold 1: train on folds 2-5, validate on fold 1
    - Fold 2: train on folds 1,3-5, validate on fold 2
    - ... and so on

    Args:
        indices: List of image indices to split
        labels: List of labels for each image (0=Normal, 1=ROP)
        n_folds: Number of folds (e.g., 5)
        seed: Random seed for reproducibility

    Returns:
        List of (train_indices, val_indices) tuples, one for each fold
    """
    indices = np.array(indices)
    # Get the labels for only the indices we're splitting
    subset_labels = [labels[i] for i in indices]

    # Create a stratified k-fold splitter
    # shuffle=True means randomize the data before splitting
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = []
    # For each fold, get the training and validation indices
    for train_pos, val_pos in skf.split(indices, subset_labels):
        # train_pos and val_pos are positions within the indices array
        # Convert them to actual indices
        splits.append((indices[train_pos].tolist(), indices[val_pos].tolist()))

    return splits


def make_loader(base_dataset, indices, transform, batch_size, shuffle, num_workers):
    """
    Create a data loader for efficient batch loading.

    A DataLoader:
    1. Takes the specified indices from the dataset
    2. Applies transformations to each image
    3. Groups them into batches
    4. Loads them in parallel using multiple CPU threads

    Args:
        base_dataset: The full ROPDataset object
        indices: Which images to load
        transform: Transformations to apply to each image
        batch_size: How many images per batch (e.g., 8)
        shuffle: Whether to randomize order (True for training, False for testing)
        num_workers: How many CPU threads to use for loading

    Returns:
        A DataLoader that yields batches of images and labels
    """
    # Create a subset with the given indices and transformations
    subset = TransformSubset(base_dataset, indices, transform)

    # Wrap it in a DataLoader for batch processing
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


# ============================================================================
# MODEL - Deep Learning Architecture
# ============================================================================

def get_rop_model(num_classes=2, pretrained=True):
    """
    Create an EfficientNet-B3 model for ROP classification.

    Architecture:
    1. EfficientNet-B3 backbone: A powerful pre-trained model that can recognize
       patterns in images (trained on millions of images from ImageNet)
    2. Custom classification head: Adapted layers to output 2 classes (Normal or ROP)

    Why EfficientNet-B3?
    - It's balanced: powerful enough to learn complex patterns, but not so large
      that it needs tons of data or GPU memory
    - It's pre-trained: we use weights learned from ImageNet, so we start with
      knowledge about general image features (edges, textures, shapes)
    - This is called "transfer learning" and speeds up training significantly

    Args:
        num_classes: Number of output classes (2 = Normal or ROP)
        pretrained: Whether to use ImageNet pre-training (almost always True)

    Returns:
        A PyTorch model ready for training
    """
    if pretrained:
        # Load EfficientNet-B3 with ImageNet weights
        # These weights are from a model trained on millions of images
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        model = models.efficientnet_b3(weights=weights)
    else:
        # Create a model with random weights (rarely used)
        model = models.efficientnet_b3()

    # The original EfficientNet was trained for 1000 ImageNet classes
    # We need to replace the last layer for our 2-class problem (Normal vs ROP)

    # Get the number of features from the second-to-last layer
    in_features = model.classifier[1].in_features

    # Replace the classification head with our custom head
    model.classifier = nn.Sequential(
        # Dropout: randomly ignore 30% of neurons to reduce overfitting
        nn.Dropout(p=0.3, inplace=True),
        # Linear layer: transform features to 512 dimensions
        nn.Linear(in_features, 512),
        # ReLU: activation function (adds non-linearity)
        nn.ReLU(),
        # Dropout: randomly ignore 20% of neurons again
        nn.Dropout(p=0.2),
        # Final Linear layer: output 2 scores (one for Normal, one for ROP)
        nn.Linear(512, num_classes),
    )

    return model


# ============================================================================
# TRAINING ENGINE - Functions to train and evaluate the model
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch (one pass through all training data).

    One epoch:
    1. Go through all training images in batches
    2. For each batch: predict, calculate error, update weights
    3. Accumulate overall loss and accuracy

    Args:
        model: The neural network to train
        loader: DataLoader with training images and labels
        criterion: Loss function (measures prediction error)
        optimizer: Optimization algorithm (updates weights)
        device: "cuda" for GPU or "cpu" for CPU

    Returns:
        average_loss: Average error across all batches
        accuracy: Percentage of correct predictions
    """
    # Set model to training mode (enables dropout and other regularizations)
    model.train()

    running_loss = 0.0  # Accumulate total loss
    correct = 0         # Count correct predictions
    total = 0           # Count total predictions

    # Process each batch of images
    for images, labels in tqdm(loader, desc="train", leave=False):
        # Move images and labels to GPU (if available) for faster processing
        images, labels = images.to(device), labels.to(device)

        # Step 1: Forward pass - feed images through the model to get predictions
        # First, clear any old gradients from previous batches
        optimizer.zero_grad()

        # Run images through the model
        outputs = model(images)

        # Step 2: Calculate loss (how wrong our predictions are)
        loss = criterion(outputs, labels)

        # Step 3: Backward pass - calculate gradients (direction to update weights)
        loss.backward()

        # Step 4: Update weights - move in the direction that reduces loss
        optimizer.step()

        # Accumulate metrics for this batch
        running_loss += loss.item()  # Add to total loss

        # Get predicted class (which is higher: Normal or ROP?)
        _, predicted = outputs.max(1)

        # Count correct predictions
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Return average loss and accuracy for this epoch
    return running_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    """
    Evaluate the model on validation/test data.

    Unlike training, this doesn't update weights. We just check how well
    the model performs on data it hasn't seen before.

    Args:
        model: The neural network to evaluate
        loader: DataLoader with validation/test images and labels
        criterion: Loss function
        device: "cuda" for GPU or "cpu" for CPU

    Returns:
        Dictionary with metrics:
        - loss: Average error
        - acc: Accuracy percentage
        - f1: F1 score (balance between precision and recall)
        - auc: Area Under the Curve (ROC-AUC, good for imbalanced data)
        - y_true: True labels for all images
        - y_pred: Predicted labels for all images
        - probs: Probability scores for the ROP class
    """
    # Set model to evaluation mode (disables dropout, uses batch norm statistics)
    model.eval()

    running_loss = 0.0
    all_labels = []  # Collect all true labels
    all_preds = []   # Collect all predictions
    all_probs = []   # Collect confidence scores for ROP class

    # Don't calculate gradients during validation (saves memory and time)
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val", leave=False):
            # Move to device
            images, labels = images.to(device), labels.to(device)

            # Forward pass (only prediction, no weight update)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Accumulate loss
            running_loss += loss.item()

            # Convert raw scores to probabilities (0 to 1)
            # Softmax normalizes the two output scores to sum to 1
            probs = torch.softmax(outputs, dim=1)

            # Get predicted class (which score is highest?)
            _, predicted = outputs.max(1)

            # Store results for this batch
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())
            # Store probability of class 1 (ROP)
            all_probs.extend(probs.cpu().numpy()[:, 1].tolist())

    # Calculate overall metrics
    avg_loss = running_loss / len(loader)

    # Accuracy: what fraction of predictions were correct?
    acc = 100.0 * float(np.mean(np.array(all_preds) == np.array(all_labels)))

    # F1 Score: balance between correctly finding ROP and not false-alarming
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # AUC Score: how well does the model rank ROP images higher than Normal?
    # (Good metric even if data is imbalanced)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Only one class present in this batch - can't calculate AUC
        auc = float("nan")

    # Return all metrics in a dictionary
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
    Two-phase training strategy for transfer learning.

    Why two phases?
    The model starts with ImageNet knowledge. We want to adapt it to ROP classification
    without ruining the pre-trained features.

    PHASE 1: Warm up the classification head (5 epochs)
    - Freeze the backbone (don't change pre-trained weights)
    - Only train the new classification layers we added
    - Use higher learning rate (we're learning from scratch here)
    - Goal: adapt the pre-trained features to our ROP classes

    PHASE 2: Fine-tune the full network (25 epochs)
    - Unfreeze the backbone (now allow all weights to change)
    - Train everything, but with very low learning rate
    - Low LR is important: we don't want to destroy the pre-trained knowledge
    - Goal: carefully adjust all weights to specialize on ROP detection
    - Save the best model (by AUC score) during this phase

    Args:
        model: The EfficientNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: GPU or CPU
        cfg: Configuration object with hyperparameters
        desc: Description string for logging (e.g., "[fold 1/5]")

    Returns:
        model: The trained model (with best weights loaded)
        best_metrics: Validation metrics at best checkpoint
        history: Training curves (loss and accuracy per epoch)
    """
    # Loss function: measures error between predictions and ground truth
    criterion = nn.CrossEntropyLoss()

    # Track training history for plotting later
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ========== PHASE 1: Warm up the classification head ==========
    print(f"{desc} Starting Phase 1: Train classification head only")

    # Freeze all backbone parameters (don't update them)
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only the classifier head (these are new, untrained layers)
    for p in model.classifier.parameters():
        p.requires_grad = True

    # Optimizer: only updates the unfrozen classifier parameters
    # Adam is a good choice: adapts learning rate per parameter
    optimizer = optim.Adam(model.classifier.parameters(), lr=cfg.lr_phase1)

    # Phase 1: Train for several epochs
    for epoch in range(cfg.epochs_phase1):
        # Train one epoch on training data
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on validation data
        v = validate(model, val_loader, criterion, device)

        # Print progress
        print(
            f"{desc} [phase1 {epoch + 1}/{cfg.epochs_phase1}] "
            f"train_loss={t_loss:.4f} acc={t_acc:.2f}% | "
            f"val_loss={v['loss']:.4f} acc={v['acc']:.2f}% f1={v['f1']:.4f}"
        )

    # ========== PHASE 2: Fine-tune the full network ==========
    print(f"{desc} Starting Phase 2: Fine-tune full network")

    # Unfreeze all parameters (now we can update the backbone too)
    for p in model.parameters():
        p.requires_grad = True

    # Optimizer: updates all parameters with very low learning rate
    # AdamW is like Adam but better for weight decay (regularization)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr_phase2,  # Much lower than phase1!
        weight_decay=cfg.weight_decay  # Regularization to prevent overfitting
    )

    # Learning rate scheduler: reduce LR if validation doesn't improve
    # (if we're stuck, lower LR to make smaller, more careful updates)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",      # Reduce when loss stops decreasing
        factor=0.5,      # Multiply LR by 0.5
        patience=3       # Wait 3 epochs before reducing
    )

    # Track the best model so far
    best_auc = float("-inf")  # Start with worst possible AUC
    best_state = copy.deepcopy(model.state_dict())  # Save initial weights
    best_metrics = None

    # Phase 2: Train for many epochs, saving the best model
    for epoch in range(cfg.epochs_phase2):
        # Train one epoch
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on validation data
        v = validate(model, val_loader, criterion, device)

        # Potentially reduce learning rate if loss plateaued
        scheduler.step(v["loss"])

        # Record metrics for plotting
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v["loss"])
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v["acc"])

        # Print progress
        print(
            f"{desc} [phase2 {epoch + 1}/{cfg.epochs_phase2}] "
            f"train_loss={t_loss:.4f} acc={t_acc:.2f}% | "
            f"val_loss={v['loss']:.4f} acc={v['acc']:.2f}% auc={v['auc']:.4f}"
        )

        # Save model if it's the best so far (by AUC score)
        if v["auc"] > best_auc:
            best_auc = v["auc"]
            # Deep copy the weights (not just a reference)
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = v
            print(f"{desc}     -> New best AUC: {best_auc:.4f}")

    # If AUC was always NaN (one class only), just use final metrics
    if best_metrics is None:
        best_metrics = v
        best_state = copy.deepcopy(model.state_dict())

    # Load the best model weights back into the model
    model.load_state_dict(best_state)

    return model, best_metrics, history


# ============================================================================
# UTILITY FUNCTIONS - Helpers for saving results and reproducibility
# ============================================================================

def set_seed(seed):
    """
    Set random seeds for reproducibility.

    Deep learning uses randomness in multiple places:
    - Random weight initialization
    - Shuffling training data
    - Dropout randomness
    - CUDA operations on GPU

    Setting all these seeds ensures identical results when running again.

    Args:
        seed: Random seed value (e.g., 42)
    """
    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy random
    np.random.seed(seed)

    # Set seed for PyTorch CPU operations
    torch.manual_seed(seed)

    # Set seed for PyTorch GPU operations (if GPU available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_confusion_matrix(y_true, y_pred, path, title):
    """
    Create and save a confusion matrix visualization.

    Confusion Matrix shows:
    - Top-left: Normal images correctly classified as Normal (True Negatives)
    - Top-right: Normal images incorrectly classified as ROP (False Positives)
    - Bottom-left: ROP images incorrectly classified as Normal (False Negatives)
    - Bottom-right: ROP images correctly classified as ROP (True Positives)

    Args:
        y_true: True labels (0=Normal, 1=ROP)
        y_pred: Predicted labels (0=Normal, 1=ROP)
        path: Where to save the PNG file
        title: Title for the plot
    """
    # Calculate the confusion matrix
    # (how many of each type of error/correct prediction)
    cm = confusion_matrix(y_true, y_pred)

    # Create a figure
    plt.figure(figsize=(6, 5))

    # Draw the heatmap (colored grid showing the matrix)
    sns.heatmap(
        cm,
        annot=True,  # Show numbers in cells
        fmt="d",     # Format as integers
        cmap="Blues", # Blue color scheme
        xticklabels=CLASS_NAMES,  # Label columns (Predicted)
        yticklabels=CLASS_NAMES   # Label rows (Actual)
    )

    # Labels and title
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Adjust layout to fit everything
    plt.tight_layout()

    # Save as PNG
    plt.savefig(path)
    print(f"Saved confusion matrix to {path}")

    # Close the figure to free memory
    plt.close()


def save_training_curves(history, path):
    """
    Create and save plots of training progress over epochs.

    Shows:
    - Left plot: Training loss vs Validation loss (how error changes)
    - Right plot: Training accuracy vs Validation accuracy (how performance changes)

    Good signs:
    - Both curves should decrease/increase smoothly
    - Training and validation should track together
    - If they diverge too much, model might be overfitting

    Args:
        history: Dictionary with keys: "train_loss", "val_loss", "train_acc", "val_acc"
        path: Where to save the PNG file
    """
    # Create a figure with 2 subplots (side by side)
    plt.figure(figsize=(12, 5))

    # Left plot: Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["val_loss"], label="Val Loss", marker='o')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Right plot: Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", marker='o')
    plt.plot(history["val_acc"], label="Val Acc", marker='o')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save as PNG
    plt.savefig(path)
    print(f"Saved training curves to {path}")

    # Close figure
    plt.close()


# ============================================================================
# MAIN PIPELINE - Orchestration of training and evaluation
# ============================================================================

def run_cross_validation(full_dataset, trainval_idx, device, cfg):
    """
    Run k-fold cross-validation to estimate model performance.

    What's cross-validation?
    - Split data into k groups
    - For each group: train on k-1 groups, test on the remaining group
    - This gives k different performance estimates
    - Average them for robust, unbiased performance estimate

    Example with 5 folds:
    - Fold 1: train on folds 2-5, test on fold 1
    - Fold 2: train on folds 1,3-5, test on fold 2
    - Fold 3: train on folds 1-2,4-5, test on fold 3
    - Fold 4: train on folds 1-3,5, test on fold 4
    - Fold 5: train on folds 1-4, test on fold 5

    Why?
    - Tests how well model generalizes to unseen data
    - More reliable than single train/test split
    - Doesn't waste data (everything is used for training and testing)

    Args:
        full_dataset: Full ROPDataset object
        trainval_idx: Indices to use for CV (everything except held-out test)
        device: GPU or CPU
        cfg: Configuration object

    Returns:
        summary: Dictionary with average metrics across folds
    """
    # Create directory for CV results
    cv_dir = os.path.join(cfg.output_dir, "cv")
    os.makedirs(cv_dir, exist_ok=True)

    # Generate k-fold splits
    print(f"\nStarting {cfg.n_folds}-fold cross-validation...")
    folds = stratified_kfold_splits(trainval_idx, full_dataset.labels, cfg.n_folds, cfg.seed)
    fold_metrics = []

    # Train and evaluate on each fold
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_i + 1} / {cfg.n_folds}")
        print(f"{'='*60}")

        # Create folder for this fold's results
        fold_dir = os.path.join(cv_dir, f"fold_{fold_i}")
        os.makedirs(fold_dir, exist_ok=True)

        # Create data loaders for this fold
        # Training loader: shuffled, with augmentation
        train_loader = make_loader(
            full_dataset, train_idx, get_transforms("train", cfg.image_size),
            cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        )

        # Validation loader: not shuffled, no augmentation
        val_loader = make_loader(
            full_dataset, val_idx, get_transforms("val", cfg.image_size),
            cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        )

        # Create a fresh model for this fold (start from pre-trained weights)
        model = get_rop_model(num_classes=2, pretrained=True).to(device)

        # Train the model on this fold
        desc = f"[fold {fold_i + 1}/{cfg.n_folds}]"
        _, metrics, _ = fit_two_phase(model, train_loader, val_loader, device, cfg, desc=desc)

        # Generate and save classification report (precision, recall, f1, etc.)
        report = classification_report(
            metrics["y_true"], metrics["y_pred"],
            target_names=CLASS_NAMES,
            zero_division=0
        )
        with open(os.path.join(fold_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        # Generate and save confusion matrix visualization
        save_confusion_matrix(
            metrics["y_true"], metrics["y_pred"],
            os.path.join(fold_dir, "confusion_matrix.png"),
            f"Confusion Matrix - Fold {fold_i}",
        )

        # Record metrics for this fold
        fold_metrics.append({
            "fold": fold_i,
            "acc": metrics["acc"],
            "f1": metrics["f1"],
            "auc": metrics["auc"]
        })

        # Print fold results
        print(f"{desc} RESULTS:")
        print(f"  Accuracy: {metrics['acc']:.2f}%")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC:      {metrics['auc']:.4f}")
        print(report)

    # Calculate aggregate statistics across all folds
    accs = [m["acc"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]
    aucs = [m["auc"] for m in fold_metrics]

    summary = {
        "folds": fold_metrics,
        # Mean and std dev of accuracy across folds
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        # Mean and std dev of F1 across folds
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        # Mean and std dev of AUC across folds (use nanmean to ignore NaN values)
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
    }

    # Save summary to JSON file
    with open(os.path.join(cfg.output_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(
        f"Accuracy:  {summary['acc_mean']:.2f}% ± {summary['acc_std']:.2f}%\n"
        f"F1 Score:  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}\n"
        f"AUC:       {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}"
    )

    return summary


def run_final_fit_and_test(full_dataset, trainval_idx, test_idx, device, cfg):
    """
    Train the final model on all available training data, then evaluate on held-out test set.

    Why this extra step?
    - Cross-validation gave us a good estimate of performance
    - Now we train one final model using all trainval data (more data = better model)
    - We evaluate this final model only on the held-out test set (which it has never seen)
    - This gives us the most accurate estimate of real-world performance

    Data flow:
    1. Take the trainval_idx (everything except held-out test)
    2. Split it into: training (90%) and validation (10%) for LR scheduling
    3. Train the model with these splits
    4. Evaluate on the held-out test set (never used for training)

    Args:
        full_dataset: Full ROPDataset object
        trainval_idx: Indices for training/validation (not held-out test)
        test_idx: Indices for final evaluation (held out from everything)
        device: GPU or CPU
        cfg: Configuration object

    Returns:
        test_metrics: Metrics on the held-out test set
    """
    print(f"\n{'='*60}")
    print("FINAL TRAINING AND TESTING")
    print(f"{'='*60}")

    # Split trainval data into training and validation
    # This validation set is only used for:
    # - Deciding when to save the best model checkpoint
    # - Adjusting learning rate if needed
    # It's NOT used for final performance reporting
    final_train_idx, final_val_idx = stratified_split(
        trainval_idx, full_dataset.labels, cfg.final_val_size, cfg.seed
    )

    print(f"Final training set size: {len(final_train_idx)}")
    print(f"Final validation set size (for LR scheduling): {len(final_val_idx)}")
    print(f"Held-out test set size (final evaluation): {len(test_idx)}")

    # Create data loaders
    # Training loader: shuffled, with augmentation
    train_loader = make_loader(
        full_dataset, final_train_idx, get_transforms("train", cfg.image_size),
        cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
    )

    # Validation loader: for LR scheduling only
    val_loader = make_loader(
        full_dataset, final_val_idx, get_transforms("val", cfg.image_size),
        cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )

    # Test loader: final evaluation (no shuffling needed)
    test_loader = make_loader(
        full_dataset, test_idx, get_transforms("val", cfg.image_size),
        cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )

    # Create a fresh model
    print("\nCreating model and training...")
    model = get_rop_model(num_classes=2, pretrained=True).to(device)

    # Train the model using the two-phase approach
    model, _, history = fit_two_phase(
        model, train_loader, val_loader, device, cfg, desc="[final]"
    )

    # Save training curves (loss and accuracy over epochs)
    save_training_curves(history, os.path.join(cfg.output_dir, "training_curves_final.png"))

    # ========== FINAL EVALUATION ON HELD-OUT TEST SET ==========
    print("\nEvaluating on held-out test set...")

    criterion = nn.CrossEntropyLoss()
    test_metrics = validate(model, test_loader, criterion, device)

    # Generate classification report (precision, recall, f1, support)
    report = classification_report(
        test_metrics["y_true"], test_metrics["y_pred"],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    # Save classification report to file
    with open(os.path.join(cfg.output_dir, "test_classification_report.txt"), "w") as f:
        f.write(report)

    # Save confusion matrix visualization
    save_confusion_matrix(
        test_metrics["y_true"], test_metrics["y_pred"],
        os.path.join(cfg.output_dir, "confusion_matrix_test.png"),
        "Confusion Matrix - Held-out Test Set",
    )

    # Save the trained model weights to disk
    checkpoint_path = os.path.join(cfg.output_dir, "best_rop_model.pth")
    torch.save(model.state_dict(), checkpoint_path)

    # Print final results
    print(f"\n{'='*60}")
    print("HELD-OUT TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"Sample count: {len(test_idx)} (never used during training)")
    print(f"\nMetrics:")
    print(f"  Accuracy: {test_metrics['acc']:.2f}%")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  AUC:      {test_metrics['auc']:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Saved model weights to: {checkpoint_path}")
    print(f"Saved test report to: {os.path.join(cfg.output_dir, 'test_classification_report.txt')}")

    return test_metrics


def main():
    """
    Main entry point for the ROP classification pipeline.

    This orchestrates the entire training workflow:
    1. Load configuration
    2. Set up reproducibility
    3. Load all images
    4. Split into train/val/test sets
    5. Run cross-validation for robust performance estimate
    6. Train final model and evaluate on held-out test set
    """
    # ========== SETUP ==========
    print("="*60)
    print("ROP CLASSIFICATION PIPELINE")
    print("="*60)

    # Load configuration (hyperparameters, data paths, etc.)
    cfg = Config()

    # Set random seeds for reproducibility
    set_seed(cfg.seed)

    # Create output directory if it doesn't exist
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Detect and use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========== DATA LOADING AND SPLITTING ==========
    print(f"\nLoading images from: {cfg.data_root}")
    full_dataset = ROPDataset(cfg.data_root)

    if len(full_dataset) == 0:
        print("ERROR: No images found! Please check your data folder structure:")
        print("  Neo_Normal/")
        print("  RetCam_Normal/")
        print("  Neo_ROP/")
        print("  RetCam_ROP/")
        return

    # Get all indices
    all_indices = list(range(len(full_dataset)))

    # Split into train/val pool (85%) and held-out test set (15%)
    # Stratified: maintains class balance in both groups
    trainval_idx, test_idx = stratified_split(
        all_indices, full_dataset.labels, cfg.test_size, cfg.seed
    )

    print(f"\nData split:")
    print(f"  Held-out test set: {len(test_idx)} images")
    print(f"    -> Will be evaluated only at the very end")
    print(f"  Train+val pool:    {len(trainval_idx)} images")
    print(f"    -> Will be used for {cfg.n_folds}-fold cross-validation")

    # ========== CROSS-VALIDATION ==========
    # This is the main evaluation step
    # Trains multiple models to get robust performance estimate
    run_cross_validation(full_dataset, trainval_idx, device, cfg)

    # ========== FINAL TRAINING AND TESTING ==========
    # Now train one final model using all trainval data
    # and evaluate it on the held-out test set
    run_final_fit_and_test(full_dataset, trainval_idx, test_idx, device, cfg)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {cfg.output_dir}")
    print(f"  - cv/                         : Cross-validation results per fold")
    print(f"  - cv_summary.json             : Cross-validation summary statistics")
    print(f"  - best_rop_model.pth          : Final trained model weights")
    print(f"  - test_classification_report.txt : Detailed test metrics")
    print(f"  - confusion_matrix_test.png   : Test set confusion matrix")
    print(f"  - training_curves_final.png   : Training progress plots")


if __name__ == "__main__":
    # Run the main pipeline when script is executed
    main()
