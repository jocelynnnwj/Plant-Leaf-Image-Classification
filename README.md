# Plant Leaf Image Classification Using SAM and Vision Transformers

This repository provides a robust, modular pipeline for plant leaf disease detection using state-of-the-art deep learning and segmentation models. It combines the Segment Anything Model (SAM) for precise leaf segmentation with Vision Transformer (ViT) and ResNet50 for accurate disease classification.

## Features
- **Automatic Segmentation:** Uses SAM to isolate plant leaves and remove background noise.
- **Dual-Model Training:** Fine-tunes both ViT and ResNet50 on original and segmented datasets.
- **Data Augmentation:** Applies rotations, flips, and Gaussian blur for improved robustness.
- **Comprehensive Evaluation:** Reports accuracy, F1-score, precision, recall, ROC-AUC, and robustness under noise.

## About
This project is designed for real-world agricultural applications and research. It is modular, extensible, and ready for production or further research in plant disease detection and computer vision.

## Authors
- **Rongyi Shen** ([rongyish@usc.edu](mailto:rongyish@usc.edu))
- **Xiao Bai** ([xiaobai@usc.edu](mailto:xiaobai@usc.edu))
- **Wenjing Huang** ([whuang08@usc.edu](mailto:whuang08@usc.edu))

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**
   - Place your PlantVillage dataset under `data/plant_village/`, organized into class subfolders.

3. **Run the pipeline**
   ```bash
   python main.py
   ```

## Configuration
Edit `config.py` to adjust:
- Paths (`DATA_DIR`, `SAM_OUTPUT_DIR`)
- Hyperparameters (`BATCH_SIZE`, `IMG_SIZE`, `LR`, `EPOCHS`)

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team
- Rongyi Shen
- Xiao Bai
- Wenjing Huang

## 🙏 Acknowledgments
- The PlantVillage dataset and the open-source community for providing valuable resources.
- The developers of [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), [HuggingFace Transformers](https://huggingface.co/docs/transformers/index), and [PyTorch](https://pytorch.org/).
- Inspiration from recent advances in computer vision and plant pathology research.

## 📞 Support
For questions or support, please contact the development team via the emails above or create an issue in the repository.

---

Built with ❤️ for real-world impact in plant disease detection and computer vision. 