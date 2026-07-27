#!/usr/bin/env python3
"""
Dataset Preparation Script for HVDROPDB
Reorganizes the dataset into train/val/test splits for CNN training

Original structure:
    HVDROPDB_RetCam_Neo_Classification/
    ├── RetCam_ROP/
    ├── Neo_Normal/
    ├── Neo_ROP/
    └── RetCam_Normal/

Output structure:
    HVDROPDB_prepared/
    ├── train/
    │   ├── ROP/
    │   └── Normal/
    ├── val/
    │   ├── ROP/
    │   └── Normal/
    └── test/
        ├── ROP/
        └── Normal/
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import numpy as np

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # SOURCE: Your original dataset path
    SOURCE_DIR = r"C:\Users\vishw\Downloads\EPICS\HVDROPDB_RetCam_Neo_Classification"
    
    # DESTINATION: Where to create the organized dataset
    OUTPUT_DIR = r"C:\Users\vishw\Downloads\EPICS\HVDROPDB_prepared"
    
    # Split ratios
    TRAIN_RATIO = 0.70  # 70% for training
    VAL_RATIO = 0.15    # 15% for validation
    TEST_RATIO = 0.15   # 15% for testing
    
    # Source folders
    SOURCE_FOLDERS = {
        'ROP': ['RetCam_ROP', 'Neo_ROP'],
        'Normal': ['RetCam_Normal', 'Neo_Normal']
    }
    
    # Image extensions to look for
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_image_files(folder_path):
    """Get all image files from a folder"""
    folder = Path(folder_path)
    image_files = []
    
    if not folder.exists():
        print(f"⚠️  Warning: Folder not found: {folder_path}")
        return image_files
    
    for ext in Config.IMAGE_EXTENSIONS:
        image_files.extend(list(folder.glob(f'*{ext}')))
        image_files.extend(list(folder.glob(f'*{ext.upper()}')))
    
    return image_files

def create_directory_structure(output_dir):
    """Create train/val/test directory structure"""
    splits = ['train', 'val', 'test']
    classes = ['ROP', 'Normal']
    
    for split in splits:
        for class_name in classes:
            dir_path = Path(output_dir) / split / class_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory structure at: {output_dir}")

def copy_files(file_list, destination_dir, class_name):
    """Copy files to destination with progress"""
    copied = 0
    
    for file_path in file_list:
        try:
            # Create unique filename: classname_originalfolder_filename
            # e.g., ROP_RetCam_32.png
            source_folder = file_path.parent.name
            new_filename = f"{class_name}_{source_folder}_{file_path.name}"
            dest_path = Path(destination_dir) / new_filename
            
            shutil.copy2(file_path, dest_path)
            copied += 1
        except Exception as e:
            print(f"⚠️  Error copying {file_path}: {e}")
    
    return copied

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def prepare_dataset():
    """Main function to prepare the dataset"""
    
    print("="*80)
    print("HVDROPDB DATASET PREPARATION")
    print("="*80)
    
    source_dir = Path(Config.SOURCE_DIR)
    output_dir = Path(Config.OUTPUT_DIR)
    
    # Verify source directory exists
    if not source_dir.exists():
        print(f"❌ ERROR: Source directory not found: {source_dir}")
        print(f"Please update SOURCE_DIR in the Config class")
        return
    
    # Create output directory structure
    create_directory_structure(output_dir)
    
    print(f"\n📊 Processing dataset...")
    print(f"Source: {source_dir}")
    print(f"Destination: {output_dir}")
    
    # Process each class
    for class_name, source_folders in Config.SOURCE_FOLDERS.items():
        print(f"\n{'='*80}")
        print(f"Processing class: {class_name}")
        print(f"{'='*80}")
        
        # Collect all images for this class
        all_images = []
        for folder_name in source_folders:
            folder_path = source_dir / folder_name
            images = get_image_files(folder_path)
            all_images.extend(images)
            print(f"  {folder_name}: {len(images)} images")
        
        total_images = len(all_images)
        print(f"\n  Total {class_name} images: {total_images}")
        
        if total_images == 0:
            print(f"  ⚠️  Warning: No images found for {class_name}")
            continue
        
        # Shuffle images
        random.shuffle(all_images)
        
        # Calculate split sizes
        train_size = int(total_images * Config.TRAIN_RATIO)
        val_size = int(total_images * Config.VAL_RATIO)
        test_size = total_images - train_size - val_size
        
        print(f"\n  Split sizes:")
        print(f"    Train: {train_size} ({Config.TRAIN_RATIO*100:.0f}%)")
        print(f"    Val:   {val_size} ({Config.VAL_RATIO*100:.0f}%)")
        print(f"    Test:  {test_size} ({Config.TEST_RATIO*100:.0f}%)")
        
        # Split the data
        train_images = all_images[:train_size]
        val_images = all_images[train_size:train_size+val_size]
        test_images = all_images[train_size+val_size:]
        
        # Copy files to respective directories
        print(f"\n  Copying files...")
        
        train_copied = copy_files(
            train_images, 
            output_dir / 'train' / class_name,
            class_name
        )
        print(f"    Train: {train_copied}/{len(train_images)} copied")
        
        val_copied = copy_files(
            val_images,
            output_dir / 'val' / class_name,
            class_name
        )
        print(f"    Val:   {val_copied}/{len(val_images)} copied")
        
        test_copied = copy_files(
            test_images,
            output_dir / 'test' / class_name,
            class_name
        )
        print(f"    Test:  {test_copied}/{len(test_images)} copied")
    
    # Print final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        for class_name in ['ROP', 'Normal']:
            class_dir = output_dir / split / class_name
            count = len(list(class_dir.glob('*.*')))
            print(f"  {class_name}: {count} images")
    
    print(f"\n{'='*80}")
    print("✅ DATASET PREPARATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nPrepared dataset location:")
    print(f"  {output_dir}")
    print(f"\n📝 Next steps:")
    print(f"  1. Update DATASET_PATH in rop_cnn_training.py to:")
    print(f"     DATASET_PATH = r'{output_dir}'")
    print(f"  2. Run: python rop_cnn_training.py")

# ============================================================================
# VERIFICATION FUNCTION
# ============================================================================

def verify_dataset():
    """Verify the prepared dataset"""
    output_dir = Path(Config.OUTPUT_DIR)
    
    if not output_dir.exists():
        print("❌ Prepared dataset not found. Run prepare_dataset() first.")
        return
    
    print("="*80)
    print("DATASET VERIFICATION")
    print("="*80)
    
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        split_total = 0
        
        print(f"\n{split.upper()}:")
        for class_name in ['ROP', 'Normal']:
            class_dir = split_dir / class_name
            
            if class_dir.exists():
                images = list(class_dir.glob('*.*'))
                count = len(images)
                split_total += count
                total_images += count
                
                # Check image dimensions (sample first 3 images)
                print(f"  {class_name}: {count} images")
                
                if images:
                    from PIL import Image
                    sample_images = images[:3]
                    sizes = set()
                    
                    for img_path in sample_images:
                        try:
                            with Image.open(img_path) as img:
                                sizes.add(img.size)
                        except:
                            pass
                    
                    if sizes:
                        print(f"    Sample sizes: {sizes}")
            else:
                print(f"  {class_name}: ⚠️  Directory not found")
        
        print(f"  Subtotal: {split_total} images")
    
    print(f"\n{'='*80}")
    print(f"Total images in prepared dataset: {total_images}")
    print(f"{'='*80}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("HVDROPDB Dataset Preparation Tool")
    print("="*80)
    print("\nThis script will:")
    print("  1. Read images from RetCam_ROP, Neo_ROP, RetCam_Normal, Neo_Normal")
    print("  2. Combine them into ROP and Normal classes")
    print("  3. Split into 70% train, 15% val, 15% test")
    print("  4. Create organized train/val/test folder structure")
    print("\n" + "="*80)
    
    # Ask for confirmation
    response = input("\nProceed with dataset preparation? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        prepare_dataset()
        
        # Verify the prepared dataset
        print("\n")
        verify_response = input("Verify the prepared dataset? (yes/no): ").lower().strip()
        if verify_response in ['yes', 'y']:
            verify_dataset()
    else:
        print("\n❌ Dataset preparation cancelled.")
        print("\n💡 To run later, execute: python prepare_dataset.py")
