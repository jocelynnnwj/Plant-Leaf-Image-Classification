"""
Load and preprocess PlantVillage dataset
"""
import os
from pathlib import Path
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split

# Mapping from original folder names to readable category names
label_map = {
    "Tomato_Early_blight": "Tomato Early Blight",
    "Pepper__bell___Bacterial_spot": "Pepper Bell Bacterial Spot",
    "Pepper__bell___healthy": "Pepper Bell Healthy",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___healthy": "Potato Healthy",
    "Potato___Late_blight": "Potato Late Blight",
    "Tomato_Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato_Leaf_Mold": "Tomato Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Spider Mites",
    "Tomato__Target_Spot": "Tomato Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato_healthy": "Tomato Healthy",
    "Tomato_Late_blight": "Tomato Late Blight"
}

def load_image_paths_and_labels(root_dir):
    """
    Load image paths and mapped labels from a directory of class subfolders.
    Returns: image_paths, labels (both lists)
    """
    image_paths = []
    labels = []
    for category_folder in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category_folder)
        if os.path.isdir(category_path):
            label = label_map.get(category_folder, category_folder)
            for image_file in os.listdir(category_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(category_path, image_file))
                    labels.append(label)
    return image_paths, labels

def get_categories(labels):
    """Return sorted list of unique categories from a list of labels."""
    return sorted(list(set(labels)))

def split_paths_and_labels(image_paths, labels, test_size=0.2, random_state=42):
    """
    Split image paths and labels into train and test sets with stratification.
    Returns: train_paths, test_paths, train_labels, test_labels
    """
    return train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

class PlantDataset(Dataset):
    def __init__(self, image_paths, labels, categories, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.categories = categories  # List of unique category names
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.categories.index(self.labels[idx])  # Convert label to index
        if self.transform:
            image = self.transform(image)
        return image, label

def preprocess_image(img: Image.Image, size: int):
    """Resize and normalize image"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

def match_test_set(original_test_paths, segmented_image_paths):
    """
    Given a list of original test image paths and segmented image paths,
    return the subset of segmented_image_paths that match the test set identifiers.
    """
    original_test_ids = set([os.path.basename(p).rsplit('.', 1)[0] for p in original_test_paths])
    matched_segmented = [p for p in segmented_image_paths if os.path.basename(p).rsplit('.', 1)[0] in original_test_ids]
    return matched_segmented

def load_segmented_image_paths_and_labels(segmented_dir):
    """
    Load segmented image paths and mapped labels from a flat folder of segmented images.
    Returns: image_paths, labels (both lists)
    """
    image_paths = []
    labels = []
    for image_file in os.listdir(segmented_dir):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(segmented_dir, image_file)
            # Extract the category key (remove last underscore and number)
            category_key = '_'.join(image_file.split('_')[:-1])
            label = label_map.get(category_key, category_key)
            image_paths.append(image_path)
            labels.append(label)
    return image_paths, labels

def split_segmented_by_test_ids(seg_image_paths, seg_labels, test_identifiers):
    """
    Split segmented images into test and train sets using a set of test identifiers.
    Returns: seg_train_paths, seg_test_paths, seg_train_labels, seg_test_labels
    """
    seg_test_paths = []
    seg_test_labels = []
    seg_train_paths = []
    seg_train_labels = []
    for path, label in zip(seg_image_paths, seg_labels):
        identifier = os.path.basename(path).rsplit('.', 1)[0]
        if identifier in test_identifiers:
            seg_test_paths.append(path)
            seg_test_labels.append(label)
        else:
            seg_train_paths.append(path)
            seg_train_labels.append(label)
    return seg_train_paths, seg_test_paths, seg_train_labels, seg_test_labels 