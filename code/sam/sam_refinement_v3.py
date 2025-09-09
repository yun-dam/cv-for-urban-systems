import os
import sys
import numpy as np
from pathlib import Path
import torch
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import URBAN_CLASSES

# Import SAM
from segment_anything import sam_model_registry, SamPredictor

class SAMRefinementV3:
    """Uses SAM to refine segmentation masks on a per-instance basis."""

    def __init__(self, sam_checkpoint: str, model_type: str = "vit_b", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Using device: {self.device}")

        print(f"Loading SAM {model_type} model...")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

        self.class_priorities = {
            'car': 6, 'building': 5, 'tree': 4,
            'low vegetation': 3, 'impervious surface': 2, 'background': 1
        }

    def refine_class_by_instance(self, prob_map: np.ndarray, config: dict) -> np.ndarray:
        """
        Refines the mask for a single class by processing each instance separately.
        """
        # 1. Get initial binary mask and find contours
        binary_mask = (prob_map > config['prob_threshold']).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 2. Filter contours by area
        filtered_contours = [c for c in contours if cv2.contourArea(c) > config['min_area']]

        if not filtered_contours:
            return np.zeros_like(prob_map, dtype=np.uint8)

        # 3. Process each contour (instance) with SAM
        final_mask = np.zeros_like(prob_map, dtype=np.uint8)
        for contour in filtered_contours:
            # a. Create a tight bounding box for the instance
            x, y, w, h = cv2.boundingRect(contour)
            box = np.array([x, y, x + w, y + h])

            # b. Predict with SAM using the box as a prompt
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],  # SAM expects a batch of boxes
                multimask_output=False # We want the single best mask for the box
            )

            # c. Add the refined mask to our final class mask
            # The returned mask is a boolean array, convert to uint8
            instance_mask = masks[0].astype(np.uint8)
            final_mask = np.maximum(final_mask, instance_mask)

        return final_mask

    def resolve_overlaps(self, refined_masks: dict) -> dict:
        """Resolves overlaps between class masks based on defined priority."""
        if not refined_masks:
            return refined_masks

        h, w = next(iter(refined_masks.values())).shape
        final_resolved_map = np.zeros((h, w), dtype=np.int32)

        sorted_classes = sorted(
            refined_masks.keys(),
            key=lambda x: self.class_priorities.get(x, 0),
            reverse=True # Higher priority first
        )

        for class_name in sorted_classes:
            mask = refined_masks[class_name]
            # Place the mask pixels where no higher-priority class has been placed yet
            final_resolved_map[mask > 0] = self.class_priorities.get(class_name, 0)

        # Create final masks from the resolved map
        resolved_masks = {}
        for class_name in refined_masks.keys():
            priority = self.class_priorities.get(class_name, 0)
            resolved_masks[class_name] = (final_resolved_map == priority).astype(np.uint8)

        return resolved_masks

    def process_image(self, image_path: str, prob_maps: dict, class_configs: dict) -> dict:
        """Processes a single image, refining all class masks."""
        image = np.array(Image.open(image_path).convert("RGB"))
        self.predictor.set_image(image)

        refined_masks = {}
        for class_name, prob_map in prob_maps.items():
            if class_name in class_configs:
                print(f"  Refining {class_name}...")
                config = class_configs[class_name]
                refined_mask = self.refine_class_by_instance(prob_map, config)
                refined_masks[class_name] = refined_mask

        # Resolve overlaps between the newly refined masks
        resolved_masks = self.resolve_overlaps(refined_masks)
        return resolved_masks

def create_colored_mask(masks_dict: dict, colors: dict, h: int, w: int) -> np.ndarray:
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_name, mask in masks_dict.items():
        if class_name in colors:
            colored_mask[mask > 0] = colors[class_name]
    return colored_mask

def main():
    """Main function to run the V3 refinement process."""
    # --- Configuration ---
    SAM_CHECKPOINT = "models/sam_vit_h_4b8939.pth"  # Using ViT-H model
    CLIPSEG_OUTPUT_DIR = "output/inference/finetuned"
    OUTPUT_DIR = "output/sam_refined_v3"
    TEST_IMAGES_DIR = "data/Vaihingen/finetune_data/test/images"

    CLASS_CONFIGS = {
        'building': {'prob_threshold': 0.5, 'min_area': 50},
        # Add other classes here if needed, e.g.:
        # 'tree': {'prob_threshold': 0.4, 'min_area': 100},
        # 'car': {'prob_threshold': 0.6, 'min_area': 20},
    }

    CLASS_COLORS = {
        'building': [0, 0, 255], 'tree': [0, 255, 0],
        'car': [255, 255, 0], 'low vegetation': [0, 255, 255],
        'impervious surface': [255, 255, 255], 'background': [255, 0, 0]
    }

    # --- Setup ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/masks", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)

    refiner = SAMRefinementV3(sam_checkpoint=SAM_CHECKPOINT, model_type="vit_h")
    test_images = sorted(Path(TEST_IMAGES_DIR).glob("*.tif"))

    # --- Processing Loop ---
    for img_path in tqdm(test_images, desc="Refining with SAM v3"):
        img_name = img_path.stem
        print(f"\nProcessing {img_name}...")

        # Load all necessary probability and mask files for this image
        prob_maps = {}
        clipseg_masks = {}
        for class_name in CLASS_CONFIGS.keys():
            safe_class_name = class_name.replace(' ', '_')
            prob_path = Path(CLIPSEG_OUTPUT_DIR) / f"probabilities/{img_name}_{safe_class_name}_prob.npy"
            mask_path = Path(CLIPSEG_OUTPUT_DIR) / f"masks/{img_name}_{safe_class_name}.tif"

            if prob_path.exists() and mask_path.exists():
                prob_maps[class_name] = np.load(prob_path)
                clipseg_masks[class_name] = np.array(Image.open(mask_path))

        if not prob_maps:
            print(f"  No probability maps found for {img_name}, skipping...")
            continue

        # Run the refinement process
        refined_masks = refiner.process_image(str(img_path), prob_maps, CLASS_CONFIGS)

        # Save refined masks
        for class_name, mask in refined_masks.items():
            safe_name = class_name.replace(' ', '_')
            Image.fromarray(mask * 255).save(f"{OUTPUT_DIR}/masks/{img_name}_{safe_name}_refined.tif")

        # Create and save 3-panel visualization
        original_image = np.array(Image.open(img_path).convert("RGB"))
        h, w, _ = original_image.shape

        clipseg_color = create_colored_mask(clipseg_masks, CLASS_COLORS, h, w)
        refined_color = create_colored_mask(refined_masks, CLASS_COLORS, h, w)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(original_image); axes[0].set_title('Original'); axes[0].axis('off')
        axes[1].imshow(clipseg_color); axes[1].set_title('CLIPSeg Original'); axes[1].axis('off')
        axes[2].imshow(refined_color); axes[2].set_title('SAM Refined (v3)'); axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/visualizations/{img_name}_comparison.png", dpi=150)
        plt.close(fig)

    print(f"\n✓ SAM v3 refinement complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
