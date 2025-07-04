# Plant Leaf Disease Classification Pipeline

This repository implements the pipeline described in **CSCI_566_Final_Project_Report.pdf**, combining the Segment Anything Model (SAM) with Vision Transformer (ViT) and ResNet50 to detect plant leaf diseases.

## Features
- **SAM Segmentation**: Isolates disease-relevant regions to remove background noise.
- **Dual-Model Training**: Fine-tunes both ViT and ResNet50 on original and SAM-segmented datasets.
- **Data Augmentation**: Applies rotations, flips, and Gaussian blur to improve robustness.
- **Evaluation Metrics**: Reports accuracy, F1-score, precision, recall, ROC-AUC, and robustness under noise.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**
   - Place the PlantVillage dataset under `data/plant_village/`, organized into class subfolders.

3. **Run the pipeline**
   ```bash
   python main.py
   ```
   This will:
   - Preprocess images and split into train/test sets
   - Generate SAM-segmented images in `data/sam_segmented/`
   - Fine-tune ViT and ResNet50 in two stages
   - Evaluate and save metrics and ROC plots under `outputs/`

## Configuration
Modify `config.py` to adjust:
- Paths (`DATA_DIR`, `SAM_OUTPUT_DIR`)
- Hyperparameters (`BATCH_SIZE`, `IMG_SIZE`, `LR`, `EPOCHS`)

## Results & Report
See **CSCI_566_Final_Project_Report.pdf** for detailed methodology, quantitative results, and discussion.

## License & Citation
If you use this work, please cite:
> Shen, R., Bai, X., & Huang, W. (2025). *Plant Leaf Image Classification Using SAM-Based Segmentation with Vision Transformers and ResNet Architectures*. CSCI 566 Final Project Report. 