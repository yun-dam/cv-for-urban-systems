#!/usr/bin/env python3
"""
Unified inference script for generating segmentation masks.
Supports both pretrained and finetuned CLIPSeg models.
"""

# ==============================================================================
# CONFIGURATION - Modify these settings as needed
# ==============================================================================

# Model selection
USE_FINETUNED_MODEL = True      # True: use finetuned model, False: use pretrained model
SAVE_PROBABILITY_MAPS = True    # True: save probability maps (.npy), False: only save binary masks

# Path configuration
TEST_IMAGES_DIR = None           # None: use default from config, or specify path like "data/test/images"
OUTPUT_BASE_DIR = None           # None: use default (OUTPUT_DIR/inference), or specify custom path

# ==============================================================================
# END OF CONFIGURATION
# ==============================================================================

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import tifffile
from tqdm import tqdm
import json

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from utils import get_device, load_model

def run_inference(model, processor, image_path, classes, device):
    """
    Run inference on a single image.
    
    Args:
        model: CLIPSeg model
        processor: CLIPSeg processor
        image_path: Path to input image
        classes: List of class names
        device: Computation device
        
    Returns:
        pred_masks: Dictionary of class_name -> binary mask (0 or 1)
        pred_probs: Dictionary of class_name -> probability map (0-1)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare inputs for all classes
    images = [image] * len(classes)
    inputs = processor(images=images, text=classes, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process predictions
    masks = outputs.logits.sigmoid().cpu()
    resized_masks = torch.nn.functional.interpolate(
        masks.unsqueeze(0), 
        size=image.size[::-1],  # (H, W)
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    # Convert to numpy array (C, H, W) -> individual masks
    pred_masks = {}
    pred_probs = {}
    
    for i, class_name in enumerate(classes):
        prob_map = resized_masks[i].numpy()
        binary_mask = (prob_map > 0.5).astype(np.uint8)
        
        pred_probs[class_name] = prob_map
        pred_masks[class_name] = binary_mask
    
    return pred_masks, pred_probs

def save_results(image_name, pred_masks, pred_probs, output_dir, save_probs=False):
    """
    Save inference results.
    
    Args:
        image_name: Name of the image
        pred_masks: Dictionary of binary masks
        pred_probs: Dictionary of probability maps
        output_dir: Output directory path
        save_probs: Whether to save probability maps
    """
    # Create subdirectories
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    if save_probs:
        probs_dir = output_dir / "probabilities"
        probs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual class masks
    for class_name, mask in pred_masks.items():
        safe_class_name = class_name.replace(' ', '_')
        mask_path = masks_dir / f"{image_name}_{safe_class_name}.tif"
        tifffile.imwrite(str(mask_path), mask)
    
    # Save probability maps if requested
    if save_probs:
        for class_name, prob_map in pred_probs.items():
            safe_class_name = class_name.replace(' ', '_')
            prob_path = probs_dir / f"{image_name}_{safe_class_name}_prob.npy"
            np.save(prob_path, prob_map)

def main():
    # Setup paths based on configuration
    if TEST_IMAGES_DIR:
        test_dir = Path(TEST_IMAGES_DIR)
    else:
        test_dir = Path(FINETUNE_DATA_DIR) / "test" / "images"
    
    if OUTPUT_BASE_DIR:
        output_dir = Path(OUTPUT_BASE_DIR)
    else:
        output_dir = Path(OUTPUT_DIR) / "inference"
    
    # Determine model type
    model_type = 'finetuned' if USE_FINETUNED_MODEL else 'pretrained'
    
    # Create output directory structure
    model_output_dir = output_dir / model_type
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Model: {model_type}")
    print(f"  Test directory: {test_dir}")
    print(f"  Output directory: {model_output_dir}")
    print(f"  Save probability maps: {SAVE_PROBABILITY_MAPS}")
    print()
    
    # Load model
    if USE_FINETUNED_MODEL:
        model_path = Path(FINETUNED_MODEL_DIR) / "best_model"
        print(f"Loading finetuned model from: {model_path}")
        model, processor, _ = load_model(model_path, device)
    else:
        print(f"Loading pretrained model: {PRETRAINED_MODEL}")
        processor = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL)
        model = CLIPSegForImageSegmentation.from_pretrained(PRETRAINED_MODEL)
        model.to(device)
        model.eval()
    
    # Get all test images
    test_images = list(test_dir.glob("*.tif"))
    print(f"Found {len(test_images)} test images")
    
    # Run inference on all images
    print(f"\nRunning inference with {model_type} model...")
    
    for image_path in tqdm(test_images, desc="Processing images"):
        image_name = image_path.stem
        
        # Run inference
        pred_masks, pred_probs = run_inference(
            model, processor, image_path, URBAN_CLASSES, device
        )
        
        # Save results
        save_results(
            image_name, pred_masks, pred_probs, 
            model_output_dir, save_probs=SAVE_PROBABILITY_MAPS
        )
    
    # Save metadata
    metadata = {
        'model': model_type,
        'model_path': str(model_path) if USE_FINETUNED_MODEL else PRETRAINED_MODEL,
        'test_dir': str(test_dir),
        'num_images': len(test_images),
        'classes': URBAN_CLASSES,
        'save_probabilities': SAVE_PROBABILITY_MAPS
    }
    
    with open(model_output_dir / 'inference_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Inference completed!")
    print(f"  Results saved to: {model_output_dir}")
    print(f"  - Binary masks: {model_output_dir}/masks/")
    if SAVE_PROBABILITY_MAPS:
        print(f"  - Probability maps: {model_output_dir}/probabilities/")

if __name__ == "__main__":
    main()