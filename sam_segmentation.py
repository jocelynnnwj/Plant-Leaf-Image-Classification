"""
Apply SAM to segment plant leaf images
"""
# from sam_model import SamPredictor  # placeholder import
import os

def segment_image(image_path: str, output_path: str):
    """Generate segmentation mask and save result (stub)"""
    # TODO: implement SAM segmentation logic or call your existing code
    # For now, just copy the image as a placeholder
    from shutil import copyfile
    copyfile(image_path, output_path)


def batch_segment_folder(input_folder: str, output_folder: str):
    """Apply segment_image to all images in a folder (stub)"""
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            segment_image(os.path.join(input_folder, fname), os.path.join(output_folder, fname)) 