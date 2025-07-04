import os
import random
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import ViTForImageClassification, ViTFeatureExtractor
import matplotlib.pyplot as plt
import joblib

# --- CONFIG ---
ORIG_DATA_DIR = 'PlantVillage_example'
SAM_DATA_DIR = 'sam_images'
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 5
LEARNING_RATE = 5e-5
SEED = 350
IMG_SIZE = 224  # ViT/ResNet50 default

# --- SET SEED ---
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- DATASET ---
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

# --- AUGMENTATION ---
def get_transforms(aug=True):
    t = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if aug:
        t += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1))) if random.random() < 0.5 else img),
        ]
    t += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(t)

# --- DATA LOADERS ---
def get_loaders(data_dir, batch_size=BATCH_SIZE, aug=True):
    dataset = PlantDataset(data_dir, transform=get_transforms(aug))
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, test_loader, dataset.class_to_idx

# --- MODEL LOADING ---
def get_vit_model(num_classes):
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes)
    return model

def get_resnet_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# --- TRAINING ---
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)[0] if hasattr(model, 'classifier') else model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

# --- EVALUATION ---
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)[0] if hasattr(model, 'classifier') else model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            pred = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(labels.numpy())
    return np.array(preds), np.array(trues)

# --- METRICS ---
def print_metrics(y_true, y_pred, average='macro'):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average=average):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average=average):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average=average):.4f}")

def add_gaussian_noise(images, mean=0.0, std=0.1):
    noisy_imgs = images + torch.randn_like(images) * std + mean
    return torch.clamp(noisy_imgs, 0., 1.)

# --- PLOTTING ---
def plot_confusion_matrix(y_true, y_pred, class_names, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_roc_curve(y_true, y_score, n_classes, filename):
    # y_true: (N,) int labels, y_score: (N, n_classes) probabilities
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true_bin = np.eye(n_classes)[y_true]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- ZERO-SHOT EVALUATION ---
def zero_shot_eval(model, loader, device, model_type='vit'):
    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)[0] if model_type == 'vit' else model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            prob = torch.softmax(outputs, dim=1).cpu().numpy()
            pred = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(labels.numpy())
            probs.extend(prob)
    return np.array(preds), np.array(trues), np.array(probs)

# --- MODEL SAVE/LOAD ---
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

# --- MAIN ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading original dataset...')
    train_loader, test_loader, class_to_idx = get_loaders(ORIG_DATA_DIR)
    num_classes = len(class_to_idx)
    class_names = list(class_to_idx.keys())

    # --- Zero-shot evaluation ---
    print('Zero-shot evaluation (ViT, original)...')
    vit = get_vit_model(num_classes).to(device)
    y_pred, y_true, y_prob = zero_shot_eval(vit, test_loader, device, model_type='vit')
    print('ViT Zero-Shot Results (Original):')
    print_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, class_names, 'vit_zeroshot_confmat.png')
    plot_roc_curve(y_true, y_prob, num_classes, 'vit_zeroshot_roc.png')

    print('Zero-shot evaluation (ResNet50, original)...')
    resnet = get_resnet_model(num_classes).to(device)
    y_pred, y_true, y_prob = zero_shot_eval(resnet, test_loader, device, model_type='resnet')
    print('ResNet50 Zero-Shot Results (Original):')
    print_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, class_names, 'resnet_zeroshot_confmat.png')
    plot_roc_curve(y_true, y_prob, num_classes, 'resnet_zeroshot_roc.png')

    # --- Fine-tuning and evaluation ---
    print('Training ViT on original dataset...')
    vit = get_vit_model(num_classes).to(device)
    optimizer = optim.AdamW(vit.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        loss = train_one_epoch(vit, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}')
    save_model(vit, 'vit_original.pth')
    y_pred, y_true, y_prob = zero_shot_eval(vit, test_loader, device, model_type='vit')
    print('ViT Results (Original):')
    print_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, class_names, 'vit_original_confmat.png')
    plot_roc_curve(y_true, y_prob, num_classes, 'vit_original_roc.png')

    print('Training ResNet50 on original dataset...')
    resnet = get_resnet_model(num_classes).to(device)
    optimizer = optim.AdamW(resnet.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        loss = train_one_epoch(resnet, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}')
    save_model(resnet, 'resnet_original.pth')
    y_pred, y_true, y_prob = zero_shot_eval(resnet, test_loader, device, model_type='resnet')
    print('ResNet50 Results (Original):')
    print_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, class_names, 'resnet_original_confmat.png')
    plot_roc_curve(y_true, y_prob, num_classes, 'resnet_original_roc.png')

    print('Loading SAM-segmented dataset...')
    train_loader, test_loader, _ = get_loaders(SAM_DATA_DIR)

    print('Training ViT on SAM-segmented dataset...')
    vit = get_vit_model(num_classes).to(device)
    optimizer = optim.AdamW(vit.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        loss = train_one_epoch(vit, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}')
    save_model(vit, 'vit_sam.pth')
    y_pred, y_true, y_prob = zero_shot_eval(vit, test_loader, device, model_type='vit')
    print('ViT Results (SAM):')
    print_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, class_names, 'vit_sam_confmat.png')
    plot_roc_curve(y_true, y_prob, num_classes, 'vit_sam_roc.png')

    print('Training ResNet50 on SAM-segmented dataset...')
    resnet = get_resnet_model(num_classes).to(device)
    optimizer = optim.AdamW(resnet.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        loss = train_one_epoch(resnet, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}')
    save_model(resnet, 'resnet_sam.pth')
    y_pred, y_true, y_prob = zero_shot_eval(resnet, test_loader, device, model_type='resnet')
    print('ResNet50 Results (SAM):')
    print_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, class_names, 'resnet_sam_confmat.png')
    plot_roc_curve(y_true, y_prob, num_classes, 'resnet_sam_roc.png')

    # --- Robustness testing (add noise) ---
    print('Robustness testing (ViT, original, noisy)...')
    vit = get_vit_model(num_classes).to(device)
    vit = load_model(vit, 'vit_original.pth')
    noisy_preds, noisy_trues, _ = [], [], []
    for images, labels in test_loader:
        noisy_images = add_gaussian_noise(images)
        noisy_images = noisy_images.to(device)
        outputs = vit(noisy_images)[0]
        pred = outputs.argmax(dim=1).cpu().numpy()
        noisy_preds.extend(pred)
        noisy_trues.extend(labels.numpy())
    print('ViT Robustness (Original, Noisy):')
    print_metrics(noisy_trues, noisy_preds)
    plot_confusion_matrix(noisy_trues, noisy_preds, class_names, 'vit_original_noisy_confmat.png')

    print('Robustness testing (ResNet50, original, noisy)...')
    resnet = get_resnet_model(num_classes).to(device)
    resnet = load_model(resnet, 'resnet_original.pth')
    noisy_preds, noisy_trues, _ = [], [], []
    for images, labels in test_loader:
        noisy_images = add_gaussian_noise(images)
        noisy_images = noisy_images.to(device)
        outputs = resnet(noisy_images)
        pred = outputs.argmax(dim=1).cpu().numpy()
        noisy_preds.extend(pred)
        noisy_trues.extend(labels.numpy())
    print('ResNet50 Robustness (Original, Noisy):')
    print_metrics(noisy_trues, noisy_preds)
    plot_confusion_matrix(noisy_trues, noisy_preds, class_names, 'resnet_original_noisy_confmat.png')

if __name__ == '__main__':
    main() 