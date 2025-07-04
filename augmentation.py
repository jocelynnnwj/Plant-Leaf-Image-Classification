"""
Implement data augmentation routines
"""
import albumentations as A
import numpy as np

def get_train_augmenter(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Rotate(limit=45, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def get_val_augmenter(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def augment(image, augmenter):
    """Apply albumentations augmenter to a numpy image (HWC, RGB)"""
    return augmenter(image=np.array(image))['image'] 