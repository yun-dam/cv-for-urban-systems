import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from PIL import Image
import random
from sklearn.model_selection import train_test_split

# ----------------------------
# Configuration
# ----------------------------

# Input directories (all original data)
ORIGINAL_IMG_DIR = "./data/Vaihingen/finetune_data/train/images"
ORIGINAL_MASK_DIR = "./data/Vaihingen/finetune_data/train/masks"

# Output directories for augmented train/val split
TRAIN_IMG_DIR = "./data/Vaihingen/finetune_data/train_augmented/images"
TRAIN_MASK_DIR = "./data/Vaihingen/finetune_data/train_augmented/masks"
VAL_IMG_DIR = "./data/Vaihingen/finetune_data/val_augmented/images"
VAL_MASK_DIR = "./data/Vaihingen/finetune_data/val_augmented/masks"

# Class list
CLASSES = [
    "impervious_surface",
    "building", 
    "low_vegetation",
    "tree",
    "car",
    "background"
]

# Augmentation settings
N_AUG_PER_IMAGE = 5  # 1 original + 4 augmented = 5 total per image
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% val
RANDOM_SEED = 42

# ----------------------------
# Augmentation Pipeline
# ----------------------------

# Geometric and photometric transformations
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
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.3),
], additional_targets={f"mask{i}": "mask" for i in range(len(CLASSES))})

# ----------------------------
# Utility Functions
# ----------------------------

def load_image(path):
    """Load image as RGB using OpenCV (BGR to RGB)."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def save_image(path, img_rgb):
    """Save RGB image using PIL."""
    if path.endswith('.tif'):
        Image.fromarray(img_rgb).save(path)
    else:
        Image.fromarray(img_rgb).save(path)

def load_mask(path):
    """Load grayscale mask (0 or 255) using OpenCV."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Mask not found {path}, creating empty mask")
        return np.zeros((352, 352), dtype=np.uint8)
    return mask

def save_mask(path, mask):
    """Save binary mask as PNG using PIL."""
    Image.fromarray(mask).save(path)

def augment_single_image(image_np, mask_list, img_base, n_augmentations):
    """
    Generate augmented versions of a single image and its masks.
    Returns list of (image, masks, filename_suffix) tuples.
    """
    h, w = image_np.shape[:2]
    results = []
    
    # 1. Add original (no augmentation)
    results.append((image_np, mask_list, "orig"))
    
    # 2. Generate augmented versions
    for aug_idx in range(n_augmentations - 1):  # -1 because we already have original
        try:
            transformed = transform(
                image=image_np,
                **{f"mask{i}": mask_list[i] for i in range(len(CLASSES))}
            )
            
            aug_image = transformed["image"]
            aug_masks = [transformed[f"mask{i}"] for i in range(len(CLASSES))]
            aug_suffix = f"aug{aug_idx:02d}"
            
            results.append((aug_image, aug_masks, aug_suffix))
            
        except Exception as e:
            print(f"Warning: Augmentation failed for {img_base}, iteration {aug_idx}: {e}")
            # If augmentation fails, just duplicate the original
            results.append((image_np, mask_list, f"dup{aug_idx:02d}"))
    
    return results

def save_augmented_data(results, img_base, output_img_dir, output_mask_dir):
    """Save augmented images and masks to specified directories."""
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    for image, masks, suffix in results:
        # Save image
        img_filename = f"{img_base}_{suffix}.tif"
        img_path = os.path.join(output_img_dir, img_filename)
        save_image(img_path, image)
        
        # Save masks
        for i, cls in enumerate(CLASSES):
            mask_filename = f"{img_base}_{suffix}_{cls}.png"
            mask_path = os.path.join(output_mask_dir, mask_filename)
            save_mask(mask_path, masks[i])

# ----------------------------
# Main Logic
# ----------------------------

def main():
    """
    Main function: Augment all images and split into train/val sets.
    """
    print("🚀 Starting augmentation and train/val split...")
    
    # Set random seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 1. Get all original images
    image_files = sorted([
        f for f in os.listdir(ORIGINAL_IMG_DIR)
        if f.lower().endswith((".tif", ".png", ".jpg", ".jpeg"))
    ])
    
    if len(image_files) == 0:
        raise RuntimeError(f"No image files found in: {ORIGINAL_IMG_DIR}")
    
    print(f"Found {len(image_files)} original images")
    
    # 2. Split original images into train/val (to avoid data leakage)
    train_files, val_files = train_test_split(
        image_files, 
        train_size=TRAIN_VAL_SPLIT, 
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    print(f"Train images: {len(train_files)} -> Will generate {len(train_files) * N_AUG_PER_IMAGE} augmented")
    print(f"Val images: {len(val_files)} -> Will generate {len(val_files) * N_AUG_PER_IMAGE} augmented")
    
    # 3. Process training images
    print("\\n📈 Processing training images...")
    for img_name in tqdm(train_files, desc="Augmenting training data"):
        img_base = os.path.splitext(img_name)[0]
        img_path = os.path.join(ORIGINAL_IMG_DIR, img_name)
        
        # Load image
        image_np = load_image(img_path)
        h, w = image_np.shape[:2]
        
        # Load all class masks
        mask_list = []
        for cls in CLASSES:
            mask_path = os.path.join(ORIGINAL_MASK_DIR, f"{img_base}_{cls}.png")
            mask_np = load_mask(mask_path)
            if mask_np.shape[0] != h or mask_np.shape[1] != w:
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_list.append(mask_np)
        
        # Generate augmented versions
        augmented_results = augment_single_image(image_np, mask_list, img_base, N_AUG_PER_IMAGE)
        
        # Save to training directory
        save_augmented_data(augmented_results, img_base, TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    
    # 4. Process validation images
    print("\\n📊 Processing validation images...")
    for img_name in tqdm(val_files, desc="Augmenting validation data"):
        img_base = os.path.splitext(img_name)[0]
        img_path = os.path.join(ORIGINAL_IMG_DIR, img_name)
        
        # Load image
        image_np = load_image(img_path)
        h, w = image_np.shape[:2]
        
        # Load all class masks
        mask_list = []
        for cls in CLASSES:
            mask_path = os.path.join(ORIGINAL_MASK_DIR, f"{img_base}_{cls}.png")
            mask_np = load_mask(mask_path)
            if mask_np.shape[0] != h or mask_np.shape[1] != w:
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_list.append(mask_np)
        
        # Generate augmented versions
        augmented_results = augment_single_image(image_np, mask_list, img_base, N_AUG_PER_IMAGE)
        
        # Save to validation directory
        save_augmented_data(augmented_results, img_base, VAL_IMG_DIR, VAL_MASK_DIR)
    
    # 5. Summary
    total_train_images = len(os.listdir(TRAIN_IMG_DIR))
    total_val_images = len(os.listdir(VAL_IMG_DIR))
    total_train_masks = len(os.listdir(TRAIN_MASK_DIR))
    total_val_masks = len(os.listdir(VAL_MASK_DIR))
    
    print("\\n✅ Augmentation and split completed!")
    print("📁 Results:")
    print(f"   Training images: {total_train_images}")
    print(f"   Training masks: {total_train_masks}")
    print(f"   Validation images: {total_val_images}")
    print(f"   Validation masks: {total_val_masks}")
    print(f"\\n📋 Original split:")
    print(f"   Train files: {train_files}")
    print(f"   Val files: {val_files}")

if __name__ == "__main__":
    main()