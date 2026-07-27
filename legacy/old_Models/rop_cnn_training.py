#!/usr/bin/env python3
"""
Complete CNN Training Script for ROP Dataset (2048×2048 images)
Optimized for RTX 5070Ti 12GB with Mixed Precision Training

Dataset: HVDROPDB - Retinopathy of Prematurity
Hardware: Legion 7i Pro - Intel Ultra 9 275HX, 32GB RAM, RTX 5070Ti 12GB
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset paths - UPDATE THESE!
    # After running prepare_dataset.py, use the prepared dataset path:
    DATASET_PATH = r"C:\Users\vishw\Downloads\EPICS\HVDROPDB_prepared"  # Change this
    TRAIN_DIR = "train"  # Subfolder with training images
    VAL_DIR = "val"      # Subfolder with validation images
    TEST_DIR = "test"    # Subfolder with test images
    
    # Image settings
    IMAGE_SIZE = (2048, 2048)  # Full resolution as requested
    
    # Model settings
    MODEL_NAME = "efficientnet_b0"  # Options: efficientnet_b0, mobilenet_v2, resnet50
    NUM_CLASSES = 2  # ROP vs Normal - UPDATE if different
    PRETRAINED = True
    
    # Training settings
    BATCH_SIZE = 2  # Conservative for 2048×2048 images
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 2 * 4 = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Optimization settings
    MIXED_PRECISION = True  # CRITICAL for memory savings!
    NUM_WORKERS = 4  # Data loading workers (adjust based on CPU cores)
    PIN_MEMORY = True
    
    # Save settings
    CHECKPOINT_DIR = "checkpoints"
    SAVE_BEST_ONLY = True
    SAVE_FREQUENCY = 5  # Save every N epochs
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Random seed for reproducibility
    SEED = 42

# ============================================================================
# DATASET CLASS
# ============================================================================

class ROPDataset(Dataset):
    """
    Dataset class for ROP classification
    Assumes folder structure:
        DATASET_PATH/
            train/
                ROP/
                    image1.jpg
                    image2.jpg
                Normal/
                    image1.jpg
                    image2.jpg
            val/
                ROP/
                Normal/
            test/
                ROP/
                Normal/
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Get class folders
        class_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls.name: idx for idx, cls in enumerate(class_folders)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load all image paths and labels
        for class_folder in class_folders:
            label = self.class_to_idx[class_folder.name]
            for img_path in class_folder.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(str(img_path))
                    self.labels.append(label)
        
        print(f"[{split.upper()}] Loaded {len(self.images)} images")
        print(f"Classes: {self.class_to_idx}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# DATA AUGMENTATION & TRANSFORMS
# ============================================================================

def get_transforms(split='train', image_size=(2048, 2048)):
    """
    Get data transforms for training/validation
    For medical images, use conservative augmentation
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:  # val/test
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model(model_name, num_classes, pretrained=True):
    """
    Create model with pretrained weights
    Supports: efficientnet_b0, mobilenet_v2, resnet50, densenet121
    """
    print(f"Creating {model_name} model...")
    
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, config, epoch):
    """Train for one epoch with mixed precision and gradient accumulation"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [TRAIN]")
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        
        # Mixed precision training
        with autocast(enabled=config.MIXED_PRECISION):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights every GRADIENT_ACCUMULATION_STEPS
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Statistics
        running_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate(model, dataloader, criterion, config, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [VAL]")
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            # Mixed precision inference
            with autocast(enabled=config.MIXED_PRECISION):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# ============================================================================
# SAVE CHECKPOINT
# ============================================================================

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, config, is_best=False):
    """Save model checkpoint"""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'config': vars(config)
    }
    
    if is_best:
        path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
        torch.save(checkpoint, path)
        print(f"💾 Saved BEST model to {path}")
    else:
        path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint to {path}")

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================

def plot_training_history(history, config):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(config.CHECKPOINT_DIR, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved training history plot to {save_path}")
    plt.close()

# ============================================================================
# PLOT CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, config, split='validation'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {split.capitalize()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path = os.path.join(config.CHECKPOINT_DIR, f'confusion_matrix_{split}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved confusion matrix to {save_path}")
    plt.close()

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
    
    print("="*80)
    print("ROP CNN TRAINING - OPTIMIZED FOR 2048×2048 IMAGES")
    print("="*80)
    print(f"\n🖥️  Device: {Config.DEVICE}")
    print(f"🎨 Model: {Config.MODEL_NAME}")
    print(f"📐 Image Size: {Config.IMAGE_SIZE}")
    print(f"🔢 Batch Size: {Config.BATCH_SIZE} (effective: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS})")
    print(f"⚡ Mixed Precision: {Config.MIXED_PRECISION}")
    print(f"🔄 Gradient Accumulation Steps: {Config.GRADIENT_ACCUMULATION_STEPS}")
    
    if torch.cuda.is_available():
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create datasets
    print("\n📂 Loading datasets...")
    train_dataset = ROPDataset(
        Config.DATASET_PATH, 
        split=Config.TRAIN_DIR,
        transform=get_transforms('train', Config.IMAGE_SIZE)
    )
    val_dataset = ROPDataset(
        Config.DATASET_PATH,
        split=Config.VAL_DIR,
        transform=get_transforms('val', Config.IMAGE_SIZE)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # Create model
    model = create_model(Config.MODEL_NAME, Config.NUM_CLASSES, Config.PRETRAINED)
    model = model.to(Config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.NUM_EPOCHS,
        eta_min=1e-6
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=Config.MIXED_PRECISION)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    start_time = datetime.now()
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, Config, epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, Config, epoch
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, Config, is_best=True)
        
        # Save periodic checkpoint
        if (epoch + 1) % Config.SAVE_FREQUENCY == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, Config, is_best=False)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"⏱️  Total Training Time: {training_time}")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    plot_training_history(history, Config)
    
    # Plot final confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds, 
        list(train_dataset.idx_to_class.values()),
        Config, 'validation'
    )
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(
        val_labels, val_preds,
        target_names=list(train_dataset.idx_to_class.values())
    ))
    
    # Save history to JSON
    history_path = os.path.join(Config.CHECKPOINT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\n💾 Saved training history to {history_path}")
    
    print("\n✅ All done! Check the 'checkpoints' folder for results.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
