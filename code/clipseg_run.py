# -*- coding: utf-8 -*-
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import random
from pathlib import Path
from tqdm import tqdm
import tifffile

# ==============================================================================
# 1. Model Loading
# ==============================================================================
# Load both pretrained and finetuned models for comparison
PRETRAINED_MODEL_PATH = "CIDAS/clipseg-rd64-refined"
FINETUNED_MODEL_PATH = "./clipseg_finetuned_model/best_model"

# Load pretrained model
processor_pretrained = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL_PATH)
model_pretrained = CLIPSegForImageSegmentation.from_pretrained(PRETRAINED_MODEL_PATH)
model_pretrained.eval()

# Load finetuned model
processor_finetuned = CLIPSegProcessor.from_pretrained(FINETUNED_MODEL_PATH)
model_finetuned = CLIPSegForImageSegmentation.from_pretrained(FINETUNED_MODEL_PATH)
model_finetuned.eval()

matplotlib.rcParams['font.family'] = 'Arial'

# ==============================================================================
# 2. Configuration
# ==============================================================================
# IMAGE_DIR = "./data/Vaihingen/top_cropped_512"
IMAGE_DIR = "data/Vaihingen/finetune_data/test/images"
# LABEL_DIR = "./data/Vaihingen/ground_truth_cropped_512"
LABEL_DIR = "data/Vaihingen/finetune_data/test/labels"
# OUTPUT_DIR = "./vaihingen_cropped_visualizations"
OUTPUT_DIR = "data/Vaihingen/finetune_data/test/outputs"
# IMAGE_DIR = "./data/Potsdam/top_cropped_512"
# LABEL_DIR = "./data/Potsdam/ground_truth_cropped_512"
# OUTPUT_DIR = "./potsdam_cropped_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES_TO_TEST = 16
RANDOM_SEED = 42  # Set seed for reproducible results

CLASSES_COLORS_BGR = {
    'Impervious surfaces': (255, 255, 255),  # White
    'Building': (0, 0, 255),                 # Blue (BGR format)
    'Low vegetation': (0, 255, 255),         # Cyan (BGR format)
    'Tree': (0, 255, 0),                     # Green (BGR format)
    'Car': (255, 255, 0),                    # Yellow (BGR format)
    'Clutter/background': (255, 0, 0)        # Red (BGR format)
}

PROMPTS = ["impervious surface", "building", "low vegetation", "tree", "car", "background"]

SEGMENTATION_COLORS_RGB = {
    "impervious surface": (255, 255, 255),  # White
    "building":           (0, 0, 255),      # Blue - fixed: consistent with GT building color
    "low vegetation":     (0, 255, 255),    # Cyan - fixed: consistent with GT low vegetation color
    "tree":               (0, 255, 0),      # Green
    "car":                (255, 255, 0),    # Yellow
    "background":         (255, 0, 0)       # Red
}

# ==============================================================================
# 3. Helper Functions
# ==============================================================================
def compute_iou(pred_mask, gt_mask, threshold=0.5):
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return intersection / union if union > 0 else float('nan')

def potsdam_labels_to_mask_dict(label_path, class_color_map_bgr, prompts_list):
    label_img_bgr = tifffile.imread(label_path)
    if label_img_bgr is None:
        raise FileNotFoundError(f"Label file not found: {label_path}")

    h, w, _ = label_img_bgr.shape
    mask_dict = {prompt: np.zeros((h, w), dtype=np.uint8) for prompt in prompts_list}

    for i, prompt in enumerate(prompts_list):
        official_class_name = list(class_color_map_bgr.keys())[i]
        bgr_color = np.array(class_color_map_bgr[official_class_name])
        mask = np.all(label_img_bgr == bgr_color, axis=-1)
        mask_dict[prompt] = (mask * 255).astype(np.uint8)

    return mask_dict

# ==============================================================================
# 4. Main Execution
# ==============================================================================
def main():
    # Set random seed for reproducible results
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    try:
        all_patch_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".tif")])
        if len(all_patch_files) < NUM_SAMPLES_TO_TEST:
            print(f"Warning: Number of patches in directory ({len(all_patch_files)}) is less than the required number for testing ({NUM_SAMPLES_TO_TEST}). Will test all available patches.")
            selected_files = all_patch_files
        else:
            # Shuffle and select NUM_SAMPLES_TO_TEST samples
            random.shuffle(all_patch_files)
            selected_files = all_patch_files[:NUM_SAMPLES_TO_TEST]
        print(f"Randomly selected {len(selected_files)} samples from {len(all_patch_files)} available patches for testing.")
    except FileNotFoundError:
        print(f"Error: Could not find image directory {IMAGE_DIR}. Please run the `crop_data.py` script first.")
        return

    # Iterate through the selected files
    for tile_file in tqdm(selected_files, desc="Processing Cropped Patches"):
        image_path = os.path.join(IMAGE_DIR, tile_file)
        # Cropped image and label filenames are exactly the same
        label_path = os.path.join(LABEL_DIR, tile_file)

        if not os.path.exists(label_path):
            print(f"Warning: Label for {tile_file} not found at {label_path}. Skipping.")
            continue

        # --- Model Inference ---
        image = Image.open(image_path).convert("RGB")
        images = [image for _ in PROMPTS]
        
        # Pretrained model inference
        inputs_pretrained = processor_pretrained(images=images, text=PROMPTS, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs_pretrained = model_pretrained(**inputs_pretrained)
        masks_pretrained = outputs_pretrained.logits.sigmoid().squeeze().cpu().numpy()
        resized_masks_pretrained = [
            np.array(Image.fromarray(m).resize(image.size, resample=Image.BILINEAR))
            for m in masks_pretrained
        ]
        
        # Finetuned model inference
        inputs_finetuned = processor_finetuned(images=images, text=PROMPTS, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs_finetuned = model_finetuned(**inputs_finetuned)
        masks_finetuned = outputs_finetuned.logits.sigmoid().squeeze().cpu().numpy()
        resized_masks_finetuned = [
            np.array(Image.fromarray(m).resize(image.size, resample=Image.BILINEAR))
            for m in masks_finetuned
        ]

        # --- Generate Segmentation Maps ---
        color_map_rgb = np.array([SEGMENTATION_COLORS_RGB[prompt] for prompt in PROMPTS], dtype=np.uint8)
        
        # Pretrained segmentation map
        stacked_masks_pretrained = np.stack(resized_masks_pretrained, axis=0)
        pred_labels_pretrained = np.argmax(stacked_masks_pretrained, axis=0)
        segmentation_map_pretrained = color_map_rgb[pred_labels_pretrained]
        
        # Finetuned segmentation map
        stacked_masks_finetuned = np.stack(resized_masks_finetuned, axis=0)
        pred_labels_finetuned = np.argmax(stacked_masks_finetuned, axis=0)
        segmentation_map_finetuned = color_map_rgb[pred_labels_finetuned]

        gt_img_bgr = tifffile.imread(label_path)

        # --- 4-Panel Visualization ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=120)
        
        # Original image
        axes[0, 0].imshow(np.array(image))
        axes[0, 0].set_title(f"Original Image\n{tile_file}", fontsize=12)
        axes[0, 0].axis("off")
        
        # Pretrained model output
        axes[0, 1].imshow(segmentation_map_pretrained)
        axes[0, 1].set_title("Pretrained CLIPSeg Output", fontsize=12)
        axes[0, 1].axis("off")
        
        # Finetuned model output
        axes[1, 0].imshow(segmentation_map_finetuned)
        axes[1, 0].set_title("Finetuned CLIPSeg Output", fontsize=12)
        axes[1, 0].axis("off")
        
        # Ground truth
        axes[1, 1].imshow(gt_img_bgr)
        axes[1, 1].set_title("Ground Truth", fontsize=12)
        axes[1, 1].axis("off")

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"{Path(tile_file).stem}_4panel_comparison.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # --- IoU Calculation ---
        try:
            mask_gt_dict = potsdam_labels_to_mask_dict(label_path, CLASSES_COLORS_BGR, PROMPTS)

            print(f"\n📊 IoU for Patch: {tile_file}")
            print("  Pretrained Model:")
            for i, prompt in enumerate(PROMPTS):
                iou = compute_iou(resized_masks_pretrained[i], mask_gt_dict[prompt])
                print(f"    - {prompt:20s}: IoU = {iou:.4f}")
            
            print("  Finetuned Model:")
            for i, prompt in enumerate(PROMPTS):
                iou = compute_iou(resized_masks_finetuned[i], mask_gt_dict[prompt])
                print(f"    - {prompt:20s}: IoU = {iou:.4f}")
            print("-" * 50)
        except Exception as e:
            print(f"An error occurred during IoU calculation for {tile_file}: {e}")

if __name__ == "__main__":
    main()