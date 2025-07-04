"""
Load and fine-tune Vision Transformer
"""
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch

class ViTClassifier:
    def __init__(self, num_classes, device='cpu'):
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224', num_labels=num_classes)
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
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def forward(self, x):
        return self.model(x) 