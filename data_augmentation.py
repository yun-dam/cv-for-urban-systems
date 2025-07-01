import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from PIL import Image

# ----------------------------
# User Configurations
# ----------------------------

# Directories for original RGB images and class-wise binary masks
IMAGE_DIR      = "./raw_images"   # e.g., "tile_0001.png", ...
MASK_DIR       = "./augmented_tiles/masks"  # e.g., "tile_0001_tree.png", ...

# Output directories for augmented images and masks
AUG_IMAGE_DIR  = "./augmented_tiles/images"
AUG_MASK_DIR   = "./augmented_tiles/masks_augmented"

# Class list (should match the class name in mask filenames)
CLASSES = [
    "tree",
    "road",
    "built_area",
    "building"
]

# Number of augmented samples per image (original + augmented = total)
N_AUG_PER_IMAGE = 10

# ----------------------------
# Augmentation Pipeline (Albumentations)
# ----------------------------

# 1) Geometric Transformations
geom_transforms = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.1,      # max shift: 10%
        scale_limit=0.2,      # max scale: 20%
        rotate_limit=25,      # max rotation: +/- 25 degrees
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
        p=0.7
    ),
    # Apply RandomCrop if image size >= 512
    A.RandomCrop(height=512, width=512, p=0.5),
]

# 2) Photometric Transformations
photo_transforms = [
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.3),
]

# Compose: Geometric first, then Photometric
transform = A.Compose(
    geom_transforms + photo_transforms,
    additional_targets={f"mask{i}": "mask" for i in range(len(CLASSES))}
)

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
    Image.fromarray(img_rgb).save(path)

def load_mask(path):
    """Load grayscale mask (0 or 255) using OpenCV."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return mask

def save_mask(path, mask):
    """Save binary mask as PNG using PIL."""
    Image.fromarray(mask).save(path)

# ----------------------------
# Main Logic
# ----------------------------

def main():
    # Create output directories
    os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
    os.makedirs(AUG_MASK_DIR, exist_ok=True)

    # 1) List original images
    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if len(image_files) == 0:
        raise RuntimeError(f"No image files found in: {IMAGE_DIR}")

    # 2) Perform augmentation for each image
    for img_name in tqdm(image_files, desc="Augmenting images"):
        img_base, _ = os.path.splitext(img_name)
        img_path = os.path.join(IMAGE_DIR, img_name)

        image_np = load_image(img_path)
        h, w = image_np.shape[:2]

        # Load class-wise masks
        mask_list = []
        for cls in CLASSES:
            mask_fname = f"{img_base}_{cls.replace(' ', '_')}.png"
            mask_path = os.path.join(MASK_DIR, mask_fname)
            mask_np = load_mask(mask_path)
            if mask_np.shape[0] != h or mask_np.shape[1] != w:
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_list.append(mask_np)

        # 3) Save original image and masks
        orig_img_save = os.path.join(AUG_IMAGE_DIR, f"{img_base}_orig.png")
        save_image(orig_img_save, image_np)

        for cls, mask_np in zip(CLASSES, mask_list):
            orig_mask_save = os.path.join(
                AUG_MASK_DIR,
                f"{img_base}_orig_{cls.replace(' ', '_')}.png"
            )
            save_mask(orig_mask_save, mask_np)

        # 4) Perform augmentations (excluding the original)
        for aug_idx in range(N_AUG_PER_IMAGE - 1):
            transformed = transform(
                image=image_np,
                **{f"mask{i}": mask_list[i] for i in range(len(CLASSES))}
            )

            aug_image = transformed["image"]
            aug_masks = [transformed[f"mask{i}"] for i in range(len(CLASSES))]

            # 5) Save augmented image and masks
            aug_tag = f"aug{aug_idx:02d}"
            aug_img_name = f"{img_base}_{aug_tag}.png"
            save_image(os.path.join(AUG_IMAGE_DIR, aug_img_name), aug_image)

            for i, cls in enumerate(CLASSES):
                aug_mask_name = f"{img_base}_{aug_tag}_{cls.replace(' ', '_')}.png"
                save_mask(
                    os.path.join(AUG_MASK_DIR, aug_mask_name),
                    aug_masks[i]
                )

    print("\n✅ Augmentation complete!")
    print(f"Total images generated: {len(os.listdir(AUG_IMAGE_DIR))}")
    print(f"Total masks generated: {len(os.listdir(AUG_MASK_DIR))}")


if __name__ == "__main__":
    main()
