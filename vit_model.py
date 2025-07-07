"""
Load and fine-tune Vision Transformer
"""
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import os

class ViTClassifier:
    def __init__(self, categories, device='cpu'):
        self.categories = categories
        self.num_classes = len(categories)
        self.id2label = {i: label for i, label in enumerate(categories)}
        self.label2id = {label: i for i, label in enumerate(categories)}
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.device = device
        self.model.to(self.device)

    def fit(self, train_loader, val_loader, epochs, lr=5e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).logits
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"[ViT] Epoch {epoch+1}/{epochs} complete.")

    def predict(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.model(images).logits
                pred = outputs.argmax(dim=1).cpu().numpy()
                preds.extend(pred)
                trues.extend(labels.numpy())
        return preds, trues

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    def load(self, path):
        self.model = ViTForImageClassification.from_pretrained(path)
        self.processor = ViTImageProcessor.from_pretrained(path)
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x) 