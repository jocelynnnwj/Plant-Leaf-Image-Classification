"""
Unified training loop for both models
"""
import torch
from torch.utils.data import DataLoader
from config import *
from data_preprocessing import PlantDataset
from transformers import get_scheduler
from tqdm import tqdm
import os

# Accepts either ViTClassifier or ResNetClassifier

def train_model(model, image_paths, labels, categories, transform, epochs=4, lr=5e-5, batch_size=32, save_path=None):
    """
    Train the model using the provided image paths, labels, and categories.
    Uses AdamW, CrossEntropyLoss, and a linear scheduler. Prints loss and accuracy per epoch.
    Saves the model and processor if save_path is provided.
    """
    train_dataset = PlantDataset(image_paths, labels, categories, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    num_training_steps = len(train_loader) * epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    device = model.device
    model.model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model.model(images)
            loss = criterion(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
        print(f"Epoch {epoch + 1} completed. Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    if save_path:
        model.save(save_path)
    return model 