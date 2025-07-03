import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from PIL import Image
import random
import shutil
from sklearn.model_selection import KFold

# ----------------------------
# Configuration
# ----------------------------

# Input from the pool created by prepare_ft_data.py
BASE_INPUT_DIR = "./data/Vaihingen/finetune_data/"
POOL_IMG_DIR = os.path.join(BASE_INPUT_DIR, "finetune_pool/images")
POOL_MASK_DIR = os.path.join(BASE_INPUT_DIR, "finetune_pool/masks")

# Base output directory for all generated data
BASE_OUTPUT_DIR = os.path.join(BASE_INPUT_DIR, "cv_prepared_data")

# Sub-directories for CV folds and the final full training set
CV_FOLDS_DIR = os.path.join(BASE_OUTPUT_DIR, "cv_folds")
ALL_AUGMENTED_DIR = os.path.join(BASE_OUTPUT_DIR, "all_data_augmented")

# Class list
CLASSES = [
    "impervious_surface",
    "building", 
    "low_vegetation",
    "tree",
    "car",
    "background"
]

# Augmentation and Cross-validation settings
N_AUG_PER_IMAGE = 5  # 1 original + 4 augmented = 5 total per image
N_SPLITS = 5         # 5-fold cross-validation
RANDOM_SEED = 42

# ----------------------------
# Augmentation Pipeline
# ----------------------------

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=25,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
        p=0.7
    ),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
], additional_targets={f"mask{i}": "mask" for i in range(len(CLASSES))})

# ----------------------------
# Utility Functions
# ----------------------------

def load_image(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def save_image(path, img_rgb):
    Image.fromarray(img_rgb).save(path)

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Mask not found {path}, creating empty mask")
        return np.zeros((512, 512), dtype=np.uint8)
    return mask

def save_mask(path, mask):
    Image.fromarray(mask).save(path)

def augment_and_save_set(file_list, input_img_dir, input_mask_dir, output_img_dir, output_mask_dir, desc):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    for img_name in tqdm(file_list, desc=desc):
        img_base = os.path.splitext(img_name)[0]
        img_path = os.path.join(input_img_dir, img_name)
        
        image_np = load_image(img_path)
        h, w = image_np.shape[:2]
        
        mask_list = []
        for cls in CLASSES:
            mask_path = os.path.join(input_mask_dir, f"{img_base}_{cls}.png")
            mask_np = load_mask(mask_path)
            if mask_np.shape[:2] != (h, w):
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_list.append(mask_np)

        # Save original version
        save_image(os.path.join(output_img_dir, f"{img_base}_orig.tif"), image_np)
        for i, cls in enumerate(CLASSES):
            save_mask(os.path.join(output_mask_dir, f"{img_base}_orig_{cls}.png"), mask_list[i])

        # Save augmented versions
        for aug_idx in range(N_AUG_PER_IMAGE - 1):
            transformed = transform(image=image_np, **{f"mask{i}": mask_list[i] for i in range(len(CLASSES))})
            aug_image = transformed["image"]
            aug_masks = [transformed[f"mask{i}"] for i in range(len(CLASSES))]
            
            save_image(os.path.join(output_img_dir, f"{img_base}_aug{aug_idx:02d}.tif"), aug_image)
            for i, cls in enumerate(CLASSES):
                save_mask(os.path.join(output_mask_dir, f"{img_base}_aug{aug_idx:02d}_{cls}.png"), aug_masks[i])

# ----------------------------
# Main Logic
# ----------------------------

def main():
    print("🚀 Starting data preparation for Cross-Validation and Final Training...")
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    if os.path.exists(BASE_OUTPUT_DIR):
        print(f"Removing old directory: {BASE_OUTPUT_DIR}")
        shutil.rmtree(BASE_OUTPUT_DIR)
    
    image_files = np.array(sorted([f for f in os.listdir(POOL_IMG_DIR) if f.lower().endswith(".tif")]))
    
    if len(image_files) == 0:
        raise RuntimeError(f"No image files found in: {POOL_IMG_DIR}")
    
    print(f"Found {len(image_files)} original images in the pool.")
    
    # --- Task 1: Create 5-Fold Cross-Validation Data ---
    print("\n--- Task 1: Creating 5-Fold CV Datasets ---")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(image_files)):
        fold_num = fold_idx + 1
        print(f"\n===== Processing Fold {fold_num}/{N_SPLITS} =====")
        
        train_files = image_files[train_indices]
        val_files = image_files[val_indices]
        
        fold_dir = os.path.join(CV_FOLDS_DIR, f"fold_{fold_num}")
        train_output_img_dir = os.path.join(fold_dir, "train_augmented/images")
        train_output_mask_dir = os.path.join(fold_dir, "train_augmented/masks")
        val_output_img_dir = os.path.join(fold_dir, "val_augmented/images")
        val_output_mask_dir = os.path.join(fold_dir, "val_augmented/masks")
        
        augment_and_save_set(train_files, POOL_IMG_DIR, POOL_MASK_DIR, train_output_img_dir, train_output_mask_dir, desc=f"Fold {fold_num} [Train]")
        augment_and_save_set(val_files, POOL_IMG_DIR, POOL_MASK_DIR, val_output_img_dir, val_output_mask_dir, desc=f"Fold {fold_num} [Val]")

    # --- Task 2: Create Full Augmented Dataset for Final Training ---
    print("\n--- Task 2: Creating Full Augmented Dataset for Final Training ---")
    final_train_img_dir = os.path.join(ALL_AUGMENTED_DIR, 'images')
    final_train_mask_dir = os.path.join(ALL_AUGMENTED_DIR, 'masks')
    
    augment_and_save_set(image_files, POOL_IMG_DIR, POOL_MASK_DIR, final_train_img_dir, final_train_mask_dir, desc="Augmenting all data")

    print("\n✅ All data preparation completed!")
    print(f"📁 5-Fold CV data saved in: {CV_FOLDS_DIR}")
    print(f"📁 Full augmented training data saved in: {ALL_AUGMENTED_DIR}")

if __name__ == "__main__":
    main()