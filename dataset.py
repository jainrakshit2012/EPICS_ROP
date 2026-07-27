import os

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_FOLDERS = {
    "Normal": ["Neo_Normal", "RetCam_Normal"],
    "ROP": ["Neo_ROP", "RetCam_ROP"],
}


class ROPDataset(Dataset):
    """
    Indexes ROP classification images without applying any transform.
    Maps Neo_Normal/RetCam_Normal -> 0, Neo_ROP/RetCam_ROP -> 1.
    Use TransformSubset to attach a (possibly different) transform per split.
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        for label_idx, (label_name, folders) in enumerate(CLASS_FOLDERS.items()):
            for folder in folders:
                folder_path = os.path.join(root_dir, folder)
                if not os.path.exists(folder_path):
                    print(f"Warning: folder {folder_path} not found.")
                    continue
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                        self.image_paths.append(os.path.join(folder_path, filename))
                        self.labels.append(label_idx)

        print(f"Total images found: {len(self.image_paths)}")
        print(f"Class distribution: Normal={self.labels.count(0)}, ROP={self.labels.count(1)}")

    def __len__(self):
        return len(self.image_paths)


class TransformSubset(Dataset):
    """A subset of a ROPDataset, addressed by index, with its own transform."""

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        image = Image.open(self.base_dataset.image_paths[base_idx]).convert("RGB")
        label = self.base_dataset.labels[base_idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(split, img_size):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def stratified_split(indices, labels, holdout_size, seed):
    """Splits `indices` into (keep, holdout), stratified on the labels at those indices."""
    indices = list(indices)
    subset_labels = [labels[i] for i in indices]
    keep, holdout = train_test_split(
        indices, test_size=holdout_size, stratify=subset_labels, random_state=seed
    )
    return keep, holdout


def stratified_kfold_splits(indices, labels, n_folds, seed):
    """Yields (train_idx, val_idx) pairs of original-dataset indices for each fold."""
    indices = np.array(indices)
    subset_labels = [labels[i] for i in indices]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_pos, val_pos in skf.split(indices, subset_labels):
        splits.append((indices[train_pos].tolist(), indices[val_pos].tolist()))
    return splits


def make_loader(base_dataset, indices, transform, batch_size, shuffle, num_workers):
    subset = TransformSubset(base_dataset, indices, transform)
    return DataLoader(
        subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
