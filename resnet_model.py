"""
Load and fine-tune ResNet50
"""
import torchvision.models as models
import torch.nn as nn
import torch
import os
import json

class ResNetClassifier:
    def __init__(self, categories, device='cpu'):
        """
        Initialize ResNet50 for classification.
        categories: list of class names (used for num_classes)
        """
        self.categories = categories
        self.num_classes = len(categories)
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.device = device
        self.model.to(self.device)
        # Processor config for normalization (harmonized with ViT)
        self.processor = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }

    def predict(self, loader):
        """Predict class indices for a DataLoader."""
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                pred = outputs.argmax(dim=1).cpu().numpy()
                preds.extend(pred)
                trues.extend(labels.numpy())
        return preds, trues

    def save(self, path):
        """Save the full model object and processor config to a directory."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.model, os.path.join(path, 'resnet_model.pt'))
        with open(os.path.join(path, 'resnet_processor.json'), 'w') as f:
            json.dump(self.processor, f, indent=2)

    def load(self, path):
        """Load the full model object and processor config from a directory."""
        self.model = torch.load(os.path.join(path, 'resnet_model.pt'), map_location=self.device)
        self.model.to(self.device)
        processor_path = os.path.join(path, 'resnet_processor.json')
        if os.path.exists(processor_path):
            with open(processor_path, 'r') as f:
                self.processor = json.load(f)

    def forward(self, x):
        return self.model(x) 