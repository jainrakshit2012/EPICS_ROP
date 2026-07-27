import os

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_FOLDERS = {
    "Normal": ["Neo_Normal", "RetCam_Normal"],
    "ROP": ["Neo_ROP", "RetCam_ROP"],
}


class ROPDataset(Dataset):
    """Indexes ROP classification images without applying any transform."""

    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []

        for label_idx, (_, folders) in enumerate(CLASS_FOLDERS.items()):
            for folder in folders:
                folder_path = os.path.join(root_dir, folder)
                if not os.path.exists(folder_path):
                    print(f"Warning: folder {folder_path} not found.")
                    continue
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                        self.image_paths.append(os.path.join(folder_path, filename))
                        self.labels.append(label_idx)

        print(f"Total images found: {len(self.image_paths)} (Normal={self.labels.count(0)}, ROP={self.labels.count(1)})")

    def __len__(self):
        return len(self.image_paths)


class TransformSubset(Dataset):
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
        return image, float(label)


def get_transform(image_size, split):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
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


def make_loader(base_dataset, indices, image_size, split, batch_size, shuffle, num_workers):
    subset = TransformSubset(base_dataset, indices, get_transform(image_size, split))
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
