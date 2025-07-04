"""
Orchestration script to run pipeline
"""
from config import *
from data_preprocessing import PlantDataset, split_dataset
from sam_segmentation import batch_segment_folder
from train import train_model
from evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve
from vit_model import ViTClassifier
from resnet_model import ResNetClassifier
from utils import set_seed, log_metrics
import os
import torch


def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Preprocess data (assume data is already organized in folders)
    print('Loading original dataset...')
    orig_dataset = PlantDataset(DATA_DIR)
    train_set, val_set, test_set = split_dataset(orig_dataset)
    class_names = list(orig_dataset.class_to_idx.keys())
    num_classes = len(class_names)

    # 2. Segment images (if needed)
    # batch_segment_folder(DATA_DIR, SAM_OUTPUT_DIR)  # Uncomment to run segmentation

    # 3. Train models
    print('Training ViT...')
    vit = ViTClassifier(num_classes, device=device)
    vit = train_model(vit, train_set, val_set, config=globals(), epochs=EPOCHS_ORIGINAL, lr=LR, batch_size=BATCH_SIZE)
    vit.save('vit_model.pth')

    print('Training ResNet...')
    resnet = ResNetClassifier(num_classes, device=device)
    resnet = train_model(resnet, train_set, val_set, config=globals(), epochs=EPOCHS_ORIGINAL, lr=LR, batch_size=BATCH_SIZE)
    resnet.save('resnet_model.pth')

    # 4. Evaluate and save results
    print('Evaluating ViT...')
    vit_preds, vit_trues = vit.predict(test_set)
    vit_metrics = compute_metrics(vit_trues, vit_preds)
    log_metrics(vit_metrics, 'vit_metrics.json')
    plot_confusion_matrix(vit_trues, vit_preds, class_names, 'vit_confmat.png')

    print('Evaluating ResNet...')
    resnet_preds, resnet_trues = resnet.predict(test_set)
    resnet_metrics = compute_metrics(resnet_trues, resnet_preds)
    log_metrics(resnet_metrics, 'resnet_metrics.json')
    plot_confusion_matrix(resnet_trues, resnet_preds, class_names, 'resnet_confmat.png')

if __name__ == '__main__':
    main() 