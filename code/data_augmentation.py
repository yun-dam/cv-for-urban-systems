import os
import sys
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from pathlib import Path
from PIL import Image
import albumentations as A

# ==============================================================================
# Core Script Configuration
# ==============================================================================
# Control whether to apply data augmentation to validation sets in cross-validation
# Warning: For unbiased evaluation, it is strongly recommended to keep this as False.
# Validation sets should reflect model performance on real, unmodified data.
AUGMENT_VALIDATION_SET = False

# ==============================================================================
# Environment Setup and Imports
# ==============================================================================
# Add project root directory to Python path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *

# ==============================================================================
# Load Configuration from config.py
# ==============================================================================
# Input directories
FINETUNE_POOL_IMG_DIR = FINETUNE_DATA_DIR / "finetune_pool" / "images"
FINETUNE_POOL_MASK_DIR = FINETUNE_DATA_DIR / "finetune_pool" / "masks"
# POOL_IMG_DIR = ORIGINAL_IMG_DIR
# POOL_MASK_DIR = ORIGINAL_MASK_DIR

# Output directories
BASE_OUTPUT_DIR = FINETUNE_DATA_DIR / "cv_prepared_data"
CV_FOLDS_DIR = BASE_OUTPUT_DIR / "cv_folds"
ALL_AUGMENTED_DIR = BASE_OUTPUT_DIR / "all_data_for_final_train"
FINAL_TRAIN_DIR = ALL_AUGMENTED_DIR / "train"
FINAL_VAL_DIR = ALL_AUGMENTED_DIR / "val"

# Parameters
N_AUG_PER_IMAGE = AUGMENTATION_CONFIG['num_augmentations_per_image']
N_SPLITS = DATASET_CONFIG['n_cv_folds']
RANDOM_SEED = DATASET_CONFIG['random_seed']
FINAL_VAL_SPLIT = DATASET_CONFIG['final_val_split']
CLASSES = [cls.replace(' ', '_').replace('/', '_') for cls in URBAN_CLASSES]

# ==============================================================================
# Data Augmentation Pipeline (Albumentations)
# ==============================================================================
# 1) Geometric transformations
geom_transforms = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.1,      # Maximum shift: 10%
        scale_limit=0.2,      # Maximum scale: 20%
        rotate_limit=25,      # Maximum rotation: +/- 25 degrees
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
        p=0.7
    ),
    # Apply random crop if image size >= 512
    A.RandomCrop(height=512, width=512, p=0.5),
]

# 2) Photometric transformations
photo_transforms = [
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.3),
]

# Composition: geometric transformations first, then photometric transformations
transform = A.Compose(
    geom_transforms + photo_transforms,
    additional_targets={f"mask{i}": "mask" for i in range(len(CLASSES))}
)

# ==============================================================================
# Core Processing Functions
# ==============================================================================
def process_image_set(file_list: list, input_img_dir: Path, input_mask_dir: Path,
                      output_img_dir: Path, output_mask_dir: Path,
                      should_augment: bool, desc: str):
    """
    Process a set of images. Decide whether to perform data augmentation or just copy original files based on should_augment flag.

    Args:
        file_list (list): List of filenames to process.
        input_img_dir (Path): Input directory for original images.
        input_mask_dir (Path): Input directory for original masks.
        output_img_dir (Path): Output directory for processed images.
        output_mask_dir (Path): Output directory for processed masks.
        should_augment (bool): Whether to perform data augmentation.
        desc (str): Description text for tqdm progress bar.
    """
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    for img_name in tqdm(file_list, desc=desc):
        img_path = input_img_dir / img_name
        img_stem = img_path.stem

        # --- Process image ---
        image_np = np.array(Image.open(img_path).convert("RGB"))
        h, w = image_np.shape[:2]

        # --- Load masks for all classes ---
        masks_np = []
        for cls in CLASSES:
            mask_path = input_mask_dir / f"{img_stem}_{cls}.png"
            if mask_path.exists():
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                masks_np.append(mask)
            else:
                # If mask doesn't exist, create an empty one
                masks_np.append(np.zeros((h, w), dtype=np.uint8))

        # --- Save original version (always executed) ---
        Image.fromarray(image_np).save(output_img_dir / f"{img_stem}_orig.tif")
        for i, cls in enumerate(CLASSES):
            Image.fromarray(masks_np[i]).save(output_mask_dir / f"{img_stem}_orig_{cls}.png")

        # --- If needed, execute and save augmented versions ---
        if should_augment:
            # N_AUG_PER_IMAGE includes original image, so we generate N-1 augmented versions
            for aug_idx in range(N_AUG_PER_IMAGE - 1):
                # Apply augmentation
                transformed = transform(
                    image=image_np,
                    **{f"mask{i}": masks_np[i] for i in range(len(CLASSES))}
                )
                
                aug_image = transformed["image"]
                aug_masks = [transformed[f"mask{i}"] for i in range(len(CLASSES))]

                # Save augmented images and masks
                Image.fromarray(aug_image).save(output_img_dir / f"{img_stem}_aug{aug_idx:02d}.tif")
                for i, cls in enumerate(CLASSES):
                    Image.fromarray(aug_masks[i]).save(output_mask_dir / f"{img_stem}_aug{aug_idx:02d}_{cls}.png")

# ==============================================================================
# Main Logic
# ==============================================================================
def main():
    """
    Execute complete data preparation workflow:
    1. Create data folds for cross-validation.
    2. Create fully augmented dataset for final training.
    """
    print("🚀 Starting unified data preparation workflow...")
    print(f"ℹ️  Augment validation set: {'Yes' if AUGMENT_VALIDATION_SET else 'No'}")

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Clean old output directory to ensure fresh start
    if BASE_OUTPUT_DIR.exists():
        print(f"🧹 Cleaning old directory: {BASE_OUTPUT_DIR}")
        shutil.rmtree(BASE_OUTPUT_DIR)

    # Load original image file list
    image_files = np.array(sorted([f.name for f in FINETUNE_POOL_IMG_DIR.glob('*.tif')]))
    if len(image_files) == 0:
        raise FileNotFoundError(f"No .tif image files found in {FINETUNE_POOL_IMG_DIR}.")
    print(f"🏞️ Found {len(image_files)} original images for processing.")

    # --- Task 1: Create K-Fold cross-validation dataset ---
    print("\n--- Task 1: Create K-Fold CV dataset ---")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(image_files)):
        fold_num = fold_idx + 1
        print(f"\n===== Processing Fold {fold_num}/{N_SPLITS} =====")

        train_files = image_files[train_indices]
        val_files = image_files[val_indices]

        # Define output paths for this fold
        fold_dir = CV_FOLDS_DIR / f"fold_{fold_num}"
        train_output_img_dir = fold_dir / "train/images"
        train_output_mask_dir = fold_dir / "train/masks"
        val_output_img_dir = fold_dir / "val/images"
        val_output_mask_dir = fold_dir / "val/masks"

        # Process training set (always augment)
        process_image_set(train_files, FINETUNE_POOL_IMG_DIR, FINETUNE_POOL_MASK_DIR,
                          train_output_img_dir, train_output_mask_dir,
                          should_augment=True, desc=f"Fold {fold_num} [Training]")

        # Process validation set (augment based on flag)
        process_image_set(val_files, FINETUNE_POOL_IMG_DIR, FINETUNE_POOL_MASK_DIR,
                          val_output_img_dir, val_output_mask_dir,
                          should_augment=AUGMENT_VALIDATION_SET, desc=f"Fold {fold_num} [Validation]")

    # --- Task 2: Create dataset for final training (train/validation split) ---
    print("\n--- Task 2: Create dataset for final training ---")
    
    # Split training and validation sets at original image level
    from sklearn.model_selection import train_test_split
    final_train_files, final_val_files = train_test_split(
        image_files, test_size=FINAL_VAL_SPLIT, random_state=RANDOM_SEED
    )
    
    print(f"Final training data split:")
    print(f"  - Training set: {len(final_train_files)} original images (will be augmented to {len(final_train_files) * N_AUG_PER_IMAGE} images)")
    print(f"  - Validation set: {len(final_val_files)} original images (kept original)")
    
    # Process final training set (with augmentation)
    final_train_img_dir = FINAL_TRAIN_DIR / 'images'
    final_train_mask_dir = FINAL_TRAIN_DIR / 'masks'
    process_image_set(final_train_files, FINETUNE_POOL_IMG_DIR, FINETUNE_POOL_MASK_DIR,
                      final_train_img_dir, final_train_mask_dir,
                      should_augment=True, desc="Final training set augmentation")
    
    # Process final validation set (no augmentation, copy original only)
    final_val_img_dir = FINAL_VAL_DIR / 'images'
    final_val_mask_dir = FINAL_VAL_DIR / 'masks'
    process_image_set(final_val_files, FINETUNE_POOL_IMG_DIR, FINETUNE_POOL_MASK_DIR,
                      final_val_img_dir, final_val_mask_dir,
                      should_augment=False, desc="Final validation set (no augmentation)")

    print("\n✅ All data preparation completed!")
    print(f"📁 K-Fold CV data saved in: {CV_FOLDS_DIR}")
    print(f"📁 Final training data saved in: {ALL_AUGMENTED_DIR}")
    print(f"    - Training set: {FINAL_TRAIN_DIR}")
    print(f"    - Validation set: {FINAL_VAL_DIR}")

if __name__ == "__main__":
    main()