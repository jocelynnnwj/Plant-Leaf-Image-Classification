"""
Orchestration script to run pipeline
"""
from config import *
from data_preprocessing import (
    PlantDataset, split_dataset, label_map, match_test_set,
    load_image_paths_and_labels, get_categories, split_paths_and_labels,
    load_segmented_image_paths_and_labels, split_segmented_by_test_ids
)
from sam_segmentation import batch_segment_folder
from train import train_model
from evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve, zero_shot_evaluate
from vit_model import ViTClassifier
from resnet_model import ResNetClassifier
from utils import set_seed, log_metrics
import os
import torch


def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. First round: Original data
    print('Loading original dataset...')
    orig_image_paths, orig_labels = load_image_paths_and_labels(DATA_DIR)
    categories = get_categories(orig_labels)
    train_paths, test_paths, train_labels, test_labels = split_paths_and_labels(orig_image_paths, orig_labels, test_size=0.2)

    # 2. Train ViT on original data
    print('Training ViT on original data...')
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    vit = ViTClassifier(categories, device=device)
    vit = train_model(vit, train_paths, train_labels, categories, transform, epochs=4, lr=5e-5, batch_size=32, save_path="vit_original_model")

    # 3. Second round: SAM-segmented data
    print('Loading SAM-segmented dataset...')
    seg_image_paths, seg_labels = load_segmented_image_paths_and_labels(SAM_OUTPUT_DIR)
    # Get test identifiers from original test set
    test_identifiers = set([os.path.basename(p).rsplit('.', 1)[0] for p in test_paths])
    seg_train_paths, seg_test_paths, seg_train_labels, seg_test_labels = split_segmented_by_test_ids(seg_image_paths, seg_labels, test_identifiers)

    # 4. Train ViT on SAM-segmented data
    print('Training ViT on SAM-segmented data...')
    vit_sam = ViTClassifier(categories, device=device)
    vit_sam = train_model(vit_sam, seg_train_paths, seg_train_labels, categories, transform, epochs=4, lr=5e-5, batch_size=32, save_path="vit_sam_model")

    # 5. Evaluate on SAM-segmented test set
    print('Evaluating ViT on SAM-segmented test set...')
    seg_test_dataset = PlantDataset(seg_test_paths, seg_test_labels, categories, transform=transform)
    from torch.utils.data import DataLoader
    seg_test_loader = DataLoader(seg_test_dataset, batch_size=32, shuffle=False)
    preds, trues = vit_sam.predict(seg_test_loader)
    metrics = compute_metrics(trues, preds)
    print("SAM-segmented test set metrics:", metrics)

    # 6. Zero-shot evaluation (ViT)
    print('Running zero-shot evaluation with ViT...')
    from transformers import ViTForImageClassification
    vit_zs = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    zero_shot_evaluate(vit_zs, test_image_paths, test_labels, label_map, model_type='vit', num_images=300)

    # 7. Zero-shot evaluation (ResNet)
    print('Running zero-shot evaluation with ResNet50...')
    from torchvision import models
    resnet_zs = models.resnet50(pretrained=True)
    zero_shot_evaluate(resnet_zs, test_image_paths, test_labels, label_map, model_type='resnet', num_images=300)

    # 8. Train models
    print('Training ResNet...')
    resnet = ResNetClassifier(categories, device=device)
    resnet = train_model(resnet, train_paths, train_labels, categories, transform, epochs=EPOCHS_ORIGINAL, lr=LR, batch_size=BATCH_SIZE)
    resnet.save('resnet_model.pth')

    # 9. Evaluate and save results
    print('Evaluating ViT...')
    vit_preds, vit_trues = vit.predict(test_paths)
    vit_metrics = compute_metrics(vit_trues, vit_preds)
    log_metrics(vit_metrics, 'vit_metrics.json')
    plot_confusion_matrix(vit_trues, vit_preds, categories, 'vit_confmat.png')

    print('Evaluating ResNet...')
    resnet_preds, resnet_trues = resnet.predict(test_paths)
    resnet_metrics = compute_metrics(resnet_trues, resnet_preds)
    log_metrics(resnet_metrics, 'resnet_metrics.json')
    plot_confusion_matrix(resnet_trues, resnet_preds, categories, 'resnet_confmat.png')

if __name__ == '__main__':
    main() 