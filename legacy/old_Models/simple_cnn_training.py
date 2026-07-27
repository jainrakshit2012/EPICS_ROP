#!/usr/bin/env python3
"""
Simplified CNN Training Script - Using Older/Simpler Models
Better CUDA compatibility for RTX 5070Ti
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# SIMPLE CNN FROM SCRATCH (Most Compatible)
# ============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN built from scratch - maximum CUDA compatibility
    No pretrained weights, basic architecture
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2048 -> 1024
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1024 -> 512
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 512 -> 256
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256 -> 128
            
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 -> 64
        )
        
        # Adaptive pooling to handle any input size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset paths
    DATASET_PATH = r"C:\Users\vishw\Downloads\EPICS\HVDROPDB_RetCam_Neo_Classification"
    
    # Image settings
    IMAGE_SIZE = (2048, 2048)  # Full resolution as requested
    
    # Model settings
    NUM_CLASSES = 2  # ROP vs Normal
    
    # Training settings
    BATCH_SIZE = 1  # Very conservative for 2048×2048
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2
    
    # Save settings
    CHECKPOINT_DIR = "checkpoints_simple"
    
    SEED = 42

# ============================================================================
# DATASET CLASS
# ============================================================================

class ROPDataset(Dataset):
    def __init__(self, root_dir, folders, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Map folders to labels
        # ROP = 1, Normal = 0
        folder_to_label = {
            'RetCam_ROP': 1,
            'Neo_ROP': 1,
            'RetCam_Normal': 0,
            'Neo_Normal': 0
        }
        
        for folder in folders:
            folder_path = Path(root_dir) / folder
            if not folder_path.exists():
                print(f"Warning: {folder_path} not found")
                continue
            
            label = folder_to_label[folder]
            
            for img_path in folder_path.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(str(img_path))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images")
        print(f"  ROP: {sum(self.labels)}")
        print(f"  Normal: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# TRANSFORMS
# ============================================================================

def get_transforms(image_size=(512, 512)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, gradient_accumulation_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * gradient_accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("SIMPLIFIED CNN TRAINING - 2048×2048 RESOLUTION")
    print("="*80)
    
    print(f"\nDevice: {Config.DEVICE}")
    print(f"Image Size: {Config.IMAGE_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Gradient Accumulation Steps: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective Batch Size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create dataset
    print("\nLoading dataset...")
    
    # All folders
    all_folders = ['RetCam_ROP', 'Neo_ROP', 'RetCam_Normal', 'Neo_Normal']
    
    full_dataset = ROPDataset(
        Config.DATASET_PATH,
        all_folders,
        transform=get_transforms(Config.IMAGE_SIZE)
    )
    
    # Split dataset 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"\nTrain: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\nCreating SimpleCNN model...")
    model = SimpleCNN(num_classes=Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Create checkpoint dir
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE,
            Config.GRADIENT_ACCUMULATION_STEPS
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, Config.DEVICE
        )
        
        # Update learning rate
        scheduler.step(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"  ✅ Saved best model (val_acc: {val_acc:.2f}%)")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {Config.CHECKPOINT_DIR}/best_model.pth")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        if "CUDA" in str(e):
            print("\n" + "="*80)
            print("❌ CUDA ERROR DETECTED")
            print("="*80)
            print(f"\nError: {e}")
            print("\nYour RTX 5070Ti is too new for current PyTorch.")
            print("\nOptions:")
            print("1. Edit this file, change IMAGE_SIZE to (256, 256)")
            print("2. Change BATCH_SIZE to 2")
            print("3. Or set DEVICE = torch.device('cpu') on line 166")
        else:
            raise

