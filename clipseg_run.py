from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import json
import cv2  
from pathlib import Path

# 1. Load the model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-refined")
model.eval()

# Set Arial font globally
matplotlib.rcParams['font.family'] = 'Arial'

# Output directory for visualizations
output_dir = "./clipseg_visualizations"
os.makedirs(output_dir, exist_ok=True)

# 2. Input images and prompts
image_path = "./tiles/"  # Change to your target image directory
tile_files = [f for f in os.listdir(image_path) if f.endswith(".png")]
tile_files = ['tile_0019.png', 'tile_0039.png']  # Use two sample files for testing

def compute_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Compute IoU based on binary masks
    """
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)  # Labelme masks: 0 or 255

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return float('nan')  # avoid division by zero
    return intersection / union

def labelme_json_to_mask_dict(json_path, image_size, class_list):
    """
    Generate class-wise binary mask dictionary from Labelme JSON file
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    h, w = image_size
    mask_dict = {cls: np.zeros((h, w), dtype=np.uint8) for cls in class_list}

    for shape in data['shapes']:
        label = shape['label']
        if label in class_list:
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask_dict[label], [points], 255)
    return mask_dict

classes = ["water", "green space", "road", "building"]
for k in range(len(tile_files)):

    image = Image.open(image_path+tile_files[k]).convert("RGB")
    prompts = classes

    # 3. Duplicate image to match number of prompts
    images = [image for _ in prompts]

    # 4. Preprocess for model input
    inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True)

    # 5. Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 6. Extract masks (in probability format)
    masks = outputs.logits.sigmoid().squeeze().numpy()  # shape: (N, H, W)

    # 7. Resize masks to original image size
    resized_masks = []
    for i in range(len(prompts)):
        mask = masks[i]
        resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)
        resized_masks.append(np.array(resized) / 255.0)  # Normalize back to 0~1

    # 8. Set overlay colors
    overlay_colors = {
        "water": [0, 0, 255],            # Blue
        "green space": [0, 200, 0],      # Dark green
        "road with trees": [0, 255, 0],  # Light green
        "road": [128, 128, 128],         # Gray
        "building": [255, 0, 0],         # Red            
    }

    # 9. Create overlay image
    image_np = np.array(image)
    overlay_result = image_np.copy().astype(float)

    for i, prompt in enumerate(prompts):
        mask = resized_masks[i]
        color = np.array(overlay_colors[prompt])  # RGB
        alpha = 0.7  # Overlay transparency
        for c in range(3):
            overlay_result[:, :, c] = (
                overlay_result[:, :, c] * (1 - mask * alpha) + color[c] * mask * alpha
            )

    overlay_result = np.clip(overlay_result, 0, 255).astype(np.uint8)

    # 10. Visualization
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Original", fontsize=20)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_result)
    plt.title("CLIPSeg Overlay", fontsize=20)
    plt.axis("off")

    plt.tight_layout()

    # Save result
    save_path = os.path.join(output_dir, f"{tile_files[k][:-4]}_overlay_k{k}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {save_path}")

    # === Load GT mask ===
    json_path = os.path.join("./milestone", tile_files[k].replace(".png", "_3.json"))
    mask_gt_dict = labelme_json_to_mask_dict(json_path, image.size[::-1], classes)

    # === Compute IoU ===
    print(f"\n📊 Tile: {tile_files[k]}")
    for i, cls in enumerate(classes):
        pred_mask = resized_masks[i]
        gt_mask = mask_gt_dict[cls]
        iou = compute_iou(pred_mask, gt_mask)
        print(f"  - {cls:15s}: IoU = {iou:.3f}")
