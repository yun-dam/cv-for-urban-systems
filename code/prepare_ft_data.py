import os
import sys
import shutil
import random
import tifffile
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

# ==============================================================================
# Configuration - Using configuration from config.py
# ==============================================================================
# Use paths defined in config.py
CROPPED_IMAGE_DIR = str(ORIGINAL_DIR / "top_cropped_512")
CROPPED_LABEL_DIR = str(ORIGINAL_DIR / "ground_truth_cropped_512")

# Output directories - Using configuration from config.py
BASE_OUTPUT_DIR = str(FINETUNE_DATA_DIR)
FINETUNE_POOL_IMG_DIR = str(FINETUNE_DATA_DIR / "finetune_pool" / "images")
FINETUNE_POOL_MASK_DIR = str(FINETUNE_DATA_DIR / "finetune_pool" / "masks")
TEST_IMG_DIR = str(FINETUNE_DATA_DIR / "test/images")
TEST_LABEL_DIR = str(FINETUNE_DATA_DIR / "test/labels")
TEST_MASK_DIR = str(FINETUNE_DATA_DIR / "test/masks")

# Dataset split parameters - Using configuration from config.py
NUM_FINETUNE_POOL = DATASET_CONFIG['finetune_pool_size']
RANDOM_SEED = DATASET_CONFIG['random_seed']

# Class and color definitions - Using configuration from config.py
GT_COLOR_VALUES = CLASS_COLORS_RGB
CLASSES = URBAN_CLASSES

# ==============================================================================
# Main Function
# ==============================================================================
def prepare_data():
    """
    1. Splits cropped data into a 'finetune_pool' (for later train/val split) and a 'test' set.
    2. For the 'finetune_pool', it generates binary masks for each class.
    3. For the 'test' set, it copies the original images and labels, and also generates binary masks for each class.
    """
    print("Start preparing fine-tuning data...")
    
    # Create all output directories
    for path in [FINETUNE_POOL_IMG_DIR, FINETUNE_POOL_MASK_DIR, TEST_IMG_DIR, TEST_LABEL_DIR, TEST_MASK_DIR]:
        os.makedirs(path, exist_ok=True)

    all_files = sorted([f for f in os.listdir(CROPPED_IMAGE_DIR) if f.endswith(".tif")])
    random.seed(RANDOM_SEED)
    random.shuffle(all_files)

    # --- MODIFIED ---
    # Split file lists into a finetune pool and a test set
    finetune_pool_files = all_files[:NUM_FINETUNE_POOL]
    test_files = all_files[NUM_FINETUNE_POOL:]

    print(f"Dataset split: {len(finetune_pool_files)} for finetune pool, {len(test_files)} for test set.")

    # --- Process finetune pool ---
    print(f"\nProcessing Finetune Pool...")
    for filename in tqdm(finetune_pool_files, desc="Processing Finetune Pool"):
        # 1. Copy image file
        shutil.copy(os.path.join(CROPPED_IMAGE_DIR, filename), os.path.join(FINETUNE_POOL_IMG_DIR, filename))

        # 2. Create and save binary masks
        label_path = os.path.join(CROPPED_LABEL_DIR, filename)
        label_img = tifffile.imread(label_path)
        
        base_name = Path(filename).stem
        for class_name, color in GT_COLOR_VALUES.items():
            mask = np.all(label_img == np.array(color), axis=-1)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            
            # Replace spaces in class name with underscores for valid filenames
            safe_class_name = class_name.replace(" ", "_").replace("/", "_")
            mask_filename = f"{base_name}_{safe_class_name}.png"
            mask_img.save(os.path.join(FINETUNE_POOL_MASK_DIR, mask_filename))

    # --- Process test set ---
    print("\nProcessing Test Set (copying files and generating masks)...")
    for filename in tqdm(test_files, desc="Processing Test Set"):
        # 1. Copy image file
        shutil.copy(os.path.join(CROPPED_IMAGE_DIR, filename), os.path.join(TEST_IMG_DIR, filename))
        
        # 2. Copy label file
        shutil.copy(os.path.join(CROPPED_LABEL_DIR, filename), os.path.join(TEST_LABEL_DIR, filename))
        
        # 3. Create and save binary masks
        label_path = os.path.join(CROPPED_LABEL_DIR, filename)
        label_img = tifffile.imread(label_path)
        
        base_name = Path(filename).stem
        for class_name, color in GT_COLOR_VALUES.items():
            mask = np.all(label_img == np.array(color), axis=-1)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            
            # Replace spaces in class name with underscores for valid filenames
            safe_class_name = class_name.replace(" ", "_").replace("/", "_")
            mask_filename = f"{base_name}_{safe_class_name}.png"
            mask_img.save(os.path.join(TEST_MASK_DIR, mask_filename))

    print("\n✅ Data preparation completed!")
    print(f"Finetune pool data (images and masks) created in: {Path(FINETUNE_POOL_IMG_DIR).parent}")
    print(f"Test data (images, labels and masks) created in: {Path(TEST_IMG_DIR).parent}")

if __name__ == "__main__":
    prepare_data()
