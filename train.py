"""
Unified training loop for both models
"""
import torch
from torch.utils.data import DataLoader
from config import *
from data_preprocessing import PlantDataset, split_dataset

# Accepts either ViTClassifier or ResNetClassifier

def train_model(model, train_set, val_set, config, epochs=5, lr=5e-5, batch_size=32):
    """Train the model and return the best model and training history."""
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    model.fit(train_loader, val_loader, epochs=epochs, lr=lr)
    return model 