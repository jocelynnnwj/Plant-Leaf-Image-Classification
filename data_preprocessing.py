"""
Load and preprocess PlantVillage dataset
"""
import os
from pathlib import Path
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import random

class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(class_path, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def split_dataset(dataset, val_ratio=0.1, test_ratio=0.1, seed=42):
    random.seed(seed)
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    test_size = int(test_ratio * n)
    val_size = int(val_ratio * n)
    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size+val_size]
    train_idx = indices[test_size+val_size:]
    from torch.utils.data import Subset
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

def preprocess_image(img: Image.Image, size: int):
    """Resize and normalize image"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img) 