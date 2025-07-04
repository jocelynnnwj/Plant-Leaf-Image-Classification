"""
Load and fine-tune ResNet50
"""
import torchvision.models as models
import torch.nn as nn
import torch

class ResNetClassifier:
    def __init__(self, num_classes, device='cpu'):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.device = device
        self.model.to(self.device)

    def fit(self, train_loader, val_loader, epochs, lr=5e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"[ResNet] Epoch {epoch+1}/{epochs} complete.")

    def predict(self, loader):
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
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def forward(self, x):
        return self.model(x) 