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
import json
import csv
from datetime import datetime

# ==============================================================================
# 1. Model Loading
# ==============================================================================
# Load both pretrained and finetuned models for comparison
PRETRAINED_MODEL_PATH = "CIDAS/clipseg-rd64-refined"
FINETUNED_MODEL_PATH = "./clipseg_finetuned_model_searched/best_model"

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
IMAGE_DIR = "data/Vaihingen/finetune_data/test/images"
LABEL_DIR = "data/Vaihingen/finetune_data/test/labels"
OUTPUT_DIR = "output/vaihingen_model_comparison_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES_TO_TEST = 100
RANDOM_SEED = 42  # Set seed for reproducible results
RUN_FULL_EVALUATION = True  # Set to True to evaluate on full test set

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

def evaluate_full_test_set(model_pretrained, processor_pretrained, model_finetuned, processor_finetuned, 
                          image_dir, label_dir, prompts, class_color_map, output_dir):
    """
    Evaluate both models on the full test set and save results.
    """
    print("\n🔍 Starting full test set evaluation...")
    
    # Get all test images
    all_test_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    print(f"Found {len(all_test_files)} test images")
    
    # Initialize results storage
    results = {
        'pretrained': {prompt: [] for prompt in prompts},
        'finetuned': {prompt: [] for prompt in prompts},
        'metadata': {
            'total_images': len(all_test_files),
            'evaluation_date': datetime.now().isoformat(),
            'classes': prompts
        }
    }
    
    # Process all test images
    for tile_file in tqdm(all_test_files, desc="Evaluating full test set"):
        image_path = os.path.join(image_dir, tile_file)
        label_path = os.path.join(label_dir, tile_file)
        
        if not os.path.exists(label_path):
            print(f"Warning: Label for {tile_file} not found. Skipping.")
            continue
            
        try:
            # Load image and run inference
            image = Image.open(image_path).convert("RGB")
            images = [image for _ in prompts]
            
            # Pretrained model inference
            inputs_pretrained = processor_pretrained(images=images, text=prompts, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs_pretrained = model_pretrained(**inputs_pretrained)
            masks_pretrained = outputs_pretrained.logits.sigmoid().squeeze().cpu().numpy()
            resized_masks_pretrained = [
                np.array(Image.fromarray(m).resize(image.size, resample=Image.BILINEAR))
                for m in masks_pretrained
            ]
            
            # Finetuned model inference
            inputs_finetuned = processor_finetuned(images=images, text=prompts, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs_finetuned = model_finetuned(**inputs_finetuned)
            masks_finetuned = outputs_finetuned.logits.sigmoid().squeeze().cpu().numpy()
            resized_masks_finetuned = [
                np.array(Image.fromarray(m).resize(image.size, resample=Image.BILINEAR))
                for m in masks_finetuned
            ]
            
            # Get ground truth masks
            mask_gt_dict = potsdam_labels_to_mask_dict(label_path, class_color_map, prompts)
            
            # Calculate IoU for each class
            for i, prompt in enumerate(prompts):
                iou_pretrained = compute_iou(resized_masks_pretrained[i], mask_gt_dict[prompt])
                iou_finetuned = compute_iou(resized_masks_finetuned[i], mask_gt_dict[prompt])
                
                if not np.isnan(iou_pretrained):
                    results['pretrained'][prompt].append(iou_pretrained)
                if not np.isnan(iou_finetuned):
                    results['finetuned'][prompt].append(iou_finetuned)
                    
        except Exception as e:
            print(f"Error processing {tile_file}: {e}")
            continue
    
    # Calculate average IoU scores
    avg_results = {
        'pretrained': {},
        'finetuned': {},
        'improvement': {}  # Improvement from pretrained to finetuned
    }
    
    print("\n📊 Full Test Set Evaluation Results:")
    print("=" * 60)
    
    for prompt in prompts:
        # Calculate averages
        pretrained_scores = results['pretrained'][prompt]
        finetuned_scores = results['finetuned'][prompt]
        
        avg_pretrained = np.mean(pretrained_scores) if pretrained_scores else 0.0
        avg_finetuned = np.mean(finetuned_scores) if finetuned_scores else 0.0
        improvement = avg_finetuned - avg_pretrained
        
        avg_results['pretrained'][prompt] = avg_pretrained
        avg_results['finetuned'][prompt] = avg_finetuned
        avg_results['improvement'][prompt] = improvement
        
        print(f"{prompt:20s}: Pretrained={avg_pretrained:.4f}, Finetuned={avg_finetuned:.4f}, Δ={improvement:+.4f}")
    
    # Calculate overall averages
    overall_pretrained = np.mean(list(avg_results['pretrained'].values()))
    overall_finetuned = np.mean(list(avg_results['finetuned'].values()))
    overall_improvement = overall_finetuned - overall_pretrained
    
    print("=" * 60)
    print(f"{'Overall Average':20s}: Pretrained={overall_pretrained:.4f}, Finetuned={overall_finetuned:.4f}, Δ={overall_improvement:+.4f}")
    
    # Save detailed results to JSON
    results_with_averages = {
        'detailed_results': results,
        'average_results': avg_results,
        'overall_averages': {
            'pretrained': overall_pretrained,
            'finetuned': overall_finetuned,
            'improvement': overall_improvement
        }
    }
    
    json_path = os.path.join(output_dir, 'full_evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_with_averages, f, indent=2)
    
    # Save summary to CSV for easy analysis
    csv_path = os.path.join(output_dir, 'evaluation_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Pretrained_IoU', 'Finetuned_IoU', 'Improvement', 'Num_Samples'])
        
        for prompt in prompts:
            writer.writerow([
                prompt,
                f"{avg_results['pretrained'][prompt]:.4f}",
                f"{avg_results['finetuned'][prompt]:.4f}",
                f"{avg_results['improvement'][prompt]:+.4f}",
                len(results['pretrained'][prompt])
            ])
        
        # Add overall row
        writer.writerow([
            'OVERALL',
            f"{overall_pretrained:.4f}",
            f"{overall_finetuned:.4f}",
            f"{overall_improvement:+.4f}",
            len(all_test_files)
        ])
    
    print(f"\n💾 Results saved:")
    print(f"   - Detailed results: {json_path}")
    print(f"   - Summary CSV: {csv_path}")
    
    return avg_results

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
        print(f"Found {len(all_patch_files)} total test images")
    except FileNotFoundError:
        print(f"Error: Could not find image directory {IMAGE_DIR}. Please run the `crop_data.py` script first.")
        return
    
    # Run full evaluation if requested
    if RUN_FULL_EVALUATION:
        evaluate_full_test_set(
            model_pretrained, processor_pretrained,
            model_finetuned, processor_finetuned,
            IMAGE_DIR, LABEL_DIR, PROMPTS, CLASSES_COLORS_BGR, OUTPUT_DIR
        )
        print("\n" + "=" * 60)
        print("Full evaluation completed. Now running visualization on sample images...")
        print("=" * 60)
    
    # Select samples for visualization
    if len(all_patch_files) < NUM_SAMPLES_TO_TEST:
        print(f"Warning: Number of patches in directory ({len(all_patch_files)}) is less than the required number for testing ({NUM_SAMPLES_TO_TEST}). Will test all available patches.")
        selected_files = all_patch_files
    else:
        # Shuffle and select NUM_SAMPLES_TO_TEST samples
        random.shuffle(all_patch_files)
        selected_files = all_patch_files[:NUM_SAMPLES_TO_TEST]
    print(f"Selected {len(selected_files)} samples for visualization from {len(all_patch_files)} available patches.")

    # Iterate through the selected files for visualization
    for tile_file in tqdm(selected_files, desc="Creating Visualizations"):
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

        # --- IoU Calculation for Visualization ---
        iou_info = {'pretrained': {}, 'finetuned': {}}
        try:
            mask_gt_dict = potsdam_labels_to_mask_dict(label_path, CLASSES_COLORS_BGR, PROMPTS)
            for i, prompt in enumerate(PROMPTS):
                iou_info['pretrained'][prompt] = compute_iou(resized_masks_pretrained[i], mask_gt_dict[prompt])
                iou_info['finetuned'][prompt] = compute_iou(resized_masks_finetuned[i], mask_gt_dict[prompt])
        except Exception as e:
            print(f"An error occurred during IoU calculation for {tile_file}: {e}")

        # --- 4-Panel Visualization ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=120)
        
        # Original image
        axes[0, 0].imshow(np.array(image))
        axes[0, 0].set_title(f"Original Image\n{tile_file}", fontsize=12)
        axes[0, 0].axis("off")
        
        # Pretrained model output
        iou_text_pretrained = "IoU Scores:\n" + "\n".join([
            f"  - {p}: {iou_info['pretrained'].get(p, float('nan')):.3f}" for p in PROMPTS
        ])
        axes[0, 1].imshow(segmentation_map_pretrained)
        axes[0, 1].set_title(f"Pretrained CLIPSeg Output\n\n{iou_text_pretrained}", fontsize=10, loc='left')
        axes[0, 1].axis("off")
        
        # Finetuned model output
        iou_text_finetuned = "IoU Scores:\n" + "\n".join([
            f"  - {p}: {iou_info['finetuned'].get(p, float('nan')):.3f}" for p in PROMPTS
        ])
        axes[1, 0].imshow(segmentation_map_finetuned)
        axes[1, 0].set_title(f"Finetuned CLIPSeg Output\n\n{iou_text_finetuned}", fontsize=10, loc='left')
        axes[1, 0].axis("off")
        
        # Ground truth
        axes[1, 1].imshow(gt_img_bgr)
        axes[1, 1].set_title("Ground Truth", fontsize=12)
        axes[1, 1].axis("off")

        plt.tight_layout(pad=2.0)
        save_path = os.path.join(OUTPUT_DIR, f"{Path(tile_file).stem}_4panel_comparison.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        # --- IoU Calculation ---
        # This part is now for printing to console, using the calculated values
        print(f"\n📊 IoU for Patch: {tile_file}")
        if iou_info.get('pretrained'):
            print("  Pretrained Model:")
            for prompt, iou in iou_info['pretrained'].items():
                print(f"    - {prompt:20s}: IoU = {iou:.4f}")
        
        if iou_info.get('finetuned'):
            print("  Finetuned Model:")
            for prompt, iou in iou_info['finetuned'].items():
                print(f"    - {prompt:20s}: IoU = {iou:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()