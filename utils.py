"""
Utility functions: seeding, logging, visualization
"""
import random
import numpy as np
import torch
import json

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def log_metrics(metrics, filename=None):
    """Print and optionally save metrics as JSON."""
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    if filename:
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)

# TODO: add logging and plotting helpers 