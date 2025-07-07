"""
Evaluate model performance: metrics and plots
"""
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro')
    }

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

def zero_shot_evaluate(model, image_paths, labels, label_map, model_type='vit', num_images=300):
    """
    Perform zero-shot evaluation on a list of image paths and labels.
    model_type: 'vit' for HuggingFace ViT, 'resnet' for torchvision ResNet50.
    label_map: dict mapping folder/category names to readable labels.
    Returns: accuracy (float)
    """
    correct = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    if model_type == 'vit':
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        for idx, (image_path, true_label) in enumerate(zip(image_paths[:num_images], labels[:num_images])):
            image = Image.open(image_path).convert('RGB')
            inputs = processor(images=image, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_index = logits.argmax(-1).item()
                predicted_label = model.config.id2label[predicted_index]
            if predicted_label == label_map.get(true_label, true_label):
                correct += 1
            total += 1
            if idx % 50 == 0:
                print(f"Processed {idx + 1}/{num_images} images.")
    elif model_type == 'resnet':
        # Download ImageNet class labels
        import json, urllib.request
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        imagenet_classes = json.loads(urllib.request.urlopen(url).read().decode())
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for idx, (image_path, true_label) in enumerate(zip(image_paths[:num_images], labels[:num_images])):
            image = Image.open(image_path).convert('RGB')
            inputs = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(inputs)
                predicted_index = outputs.argmax(-1).item()
                predicted_label = imagenet_classes[predicted_index]
            if predicted_label == label_map.get(true_label, true_label):
                correct += 1
            total += 1
            if idx % 50 == 0:
                print(f"Processed {idx + 1}/{num_images} images.")
    else:
        raise ValueError("model_type must be 'vit' or 'resnet'")
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"Zero-Shot Classification Accuracy on {total} images: {accuracy:.2f}%")
    return accuracy

# TODO: add ROC curve plotting function if needed 