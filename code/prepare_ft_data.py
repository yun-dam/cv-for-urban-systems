import os
import shutil
import random
import tifffile
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# ==============================================================================
# Configuration
# ==============================================================================
# Input directories (cropped data)
CROPPED_IMAGE_DIR = "./data/Vaihingen/top_cropped_512"
CROPPED_LABEL_DIR = "./data/Vaihingen/ground_truth_cropped_512"

# Output directories (for fine-tuning data)
BASE_OUTPUT_DIR = "./data/Vaihingen/finetune_data"
TRAIN_IMG_DIR = os.path.join(BASE_OUTPUT_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(BASE_OUTPUT_DIR, "train/masks")
VAL_IMG_DIR = os.path.join(BASE_OUTPUT_DIR, "val/images")
VAL_MASK_DIR = os.path.join(BASE_OUTPUT_DIR, "val/masks")
TEST_IMG_DIR = os.path.join(BASE_OUTPUT_DIR, "test/images")
TEST_LABEL_DIR = os.path.join(BASE_OUTPUT_DIR, "test/labels") # Test set labels remain unchanged

# Dataset split parameters
NUM_TRAIN = 20
NUM_VAL = 50
RANDOM_SEED = 42 # For reproducibility

# Class and color definitions (same as before)
GT_COLOR_VALUES = {
    'impervious surface': (255, 255, 255),
    'building': (0, 0, 255),
    'low vegetation': (0, 255, 255),
    'tree': (0, 255, 0),
    'car': (255, 255, 0),
    'background': (255, 0, 0)
}
CLASSES = list(GT_COLOR_VALUES.keys())

# ==============================================================================
# Main Function
# ==============================================================================
def prepare_data():
    """
    1. Split dataset into train, val, test
    2. Generate binary masks for train and val sets
    """
    print("Start preparing fine-tuning data...")
    
    # Create all output directories
    for path in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, TEST_IMG_DIR, TEST_LABEL_DIR]:
        os.makedirs(path, exist_ok=True)

    all_files = sorted([f for f in os.listdir(CROPPED_IMAGE_DIR) if f.endswith(".tif")])
    random.seed(RANDOM_SEED)
    random.shuffle(all_files)

    # Split file lists
    train_files = all_files[:NUM_TRAIN]
    val_files = all_files[NUM_TRAIN : NUM_TRAIN + NUM_VAL]
    test_files = all_files[NUM_TRAIN + NUM_VAL:]

    print(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # --- Process train and val sets ---
    for split_name, file_list, out_img_dir, out_mask_dir in [
        ("Train Set", train_files, TRAIN_IMG_DIR, TRAIN_MASK_DIR),
        ("Validation Set", val_files, VAL_IMG_DIR, VAL_MASK_DIR)
    ]:
        print(f"\nProcessing {split_name}...")
        for filename in tqdm(file_list, desc=f"Processing {split_name}"):
            # 1. Copy image file
            shutil.copy(os.path.join(CROPPED_IMAGE_DIR, filename), os.path.join(out_img_dir, filename))

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
                mask_img.save(os.path.join(out_mask_dir, mask_filename))

    # --- Process test set ---
    print("\nProcessing Test Set (copying files only)...")
    for filename in tqdm(test_files, desc="Processing Test Set"):
        shutil.copy(os.path.join(CROPPED_IMAGE_DIR, filename), os.path.join(TEST_IMG_DIR, filename))
        shutil.copy(os.path.join(CROPPED_LABEL_DIR, filename), os.path.join(TEST_LABEL_DIR, filename))

    print("\n✅ Data preparation completed!")
    print(f"Data saved in: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_data()