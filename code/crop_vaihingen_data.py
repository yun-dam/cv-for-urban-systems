import os
import sys
import tifffile
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))
from config import *

# ==============================================================================
# Configuration - 使用config.py中的配置
# ==============================================================================
# 使用config.py中定义的路径
ORIGINAL_IMAGE_DIR = str(VAIHINGEN_DIR / "top")
ORIGINAL_LABEL_DIR = str(VAIHINGEN_DIR / "ground_truth")

# 输出目录
CROPPED_IMAGE_DIR = str(VAIHINGEN_DIR / "top_cropped_512")
CROPPED_LABEL_DIR = str(VAIHINGEN_LABELS_DIR)

# 从config.py获取图像处理参数
PATCH_SIZE = CROP_SIZE
STRIDE = CROP_SIZE  # No overlap if equal to PATCH_SIZE

# ==============================================================================
# Main Function
# ==============================================================================
def crop_dataset():
    """
    Traverse original images and labels, crop them into patches of specified size, and save.
    """
    print("Start cropping dataset...")
    
    # Create output directories
    os.makedirs(CROPPED_IMAGE_DIR, exist_ok=True)
    os.makedirs(CROPPED_LABEL_DIR, exist_ok=True)

    image_files = sorted([f for f in os.listdir(ORIGINAL_IMAGE_DIR) if f.endswith(".tif")])

    total_patches = 0
    for image_file in tqdm(image_files, desc="Cropping files"):
        # Build paths
        image_path = os.path.join(ORIGINAL_IMAGE_DIR, image_file)
        # Assume label filename corresponds to image filename
        label_file = image_file
        label_path = os.path.join(ORIGINAL_LABEL_DIR, label_file)

        if not os.path.exists(label_path):
            print(f"Warning: Label file {label_path} not found, skipping {image_file}")
            continue

        # Read image and label
        try:
            image = tifffile.imread(image_path)
            label = tifffile.imread(label_path)
        except Exception as e:
            print(f"Error: Failed to read {image_file} or {label_file}: {e}")
            continue
        
        # Check if image and label sizes match
        if image.shape[:2] != label.shape[:2]:
            print(f"Warning: Image and label size mismatch for {image_file}, skipping.")
            continue

        img_h, img_w, _ = image.shape

        # Crop using sliding window
        for y in range(0, img_h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, img_w - PATCH_SIZE + 1, STRIDE):
                # Crop image and label
                image_patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :]
                label_patch = label[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :]

                # Build output filename
                base_name = Path(image_file).stem
                patch_filename = f"{base_name}_patch_{y}_{x}.tif"

                # Save patches
                tifffile.imwrite(os.path.join(CROPPED_IMAGE_DIR, patch_filename), image_patch)
                tifffile.imwrite(os.path.join(CROPPED_LABEL_DIR, patch_filename), label_patch)
                total_patches += 1

    print(f"\nCropping finished! Generated {total_patches} patches in total.")
    print(f"Image patches saved in: {CROPPED_IMAGE_DIR}")
    print(f"Label patches saved in: {CROPPED_LABEL_DIR}")

if __name__ == "__main__":
    crop_dataset()