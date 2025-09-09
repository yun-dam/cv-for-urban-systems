import os
import sys
import numpy as np
from pathlib import Path
import torch
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation
import warnings
import random
warnings.filterwarnings('ignore')

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import URBAN_CLASSES

# Import SAM
from segment_anything import sam_model_registry, SamPredictor

class SAMRefinementV5:
    """Hybrid approach: combines CLIPSeg's high-confidence regions with SAM's boundary refinement."""

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

    def split_regions(self, prob_map: np.ndarray, config: dict):
        """
        Split the probability map into core, boundary, and background regions.
        """
        # Binary mask from probability
        binary_mask = (prob_map > config['prob_threshold']).astype(np.uint8)
        
        # Calculate distance transform
        dist_transform = distance_transform_edt(binary_mask)
        
        # Define regions based on confidence and distance
        core_threshold = config.get('core_threshold', 0.8)
        boundary_width = config.get('boundary_width', 10)
        
        # Core region: high confidence AND far from boundary
        core_region = (prob_map > core_threshold) & (dist_transform > boundary_width)
        
        # Boundary region: medium confidence OR close to edge
        boundary_region = (
            ((prob_map > config['prob_threshold']) & (prob_map <= core_threshold)) |
            ((prob_map > config['prob_threshold']) & (dist_transform <= boundary_width))
        )
        
        # Background region: low confidence
        background_region = prob_map <= config['prob_threshold']
        
        return core_region, boundary_region, background_region, binary_mask

    def get_adaptive_points(self, core_region, boundary_region, background_region, contour):
        """
        Generate adaptive point prompts based on region analysis.
        """
        points = []
        labels = []
        
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # 1. Dense positive points in core region
        core_pixels = np.argwhere(core_region[y:y+h, x:x+w])
        if len(core_pixels) > 0:
            # Sample up to 5 core points
            num_core_points = min(5, len(core_pixels))
            core_indices = np.random.choice(len(core_pixels), num_core_points, replace=False)
            for idx in core_indices:
                py, px = core_pixels[idx]
                points.append((x + px, y + py))
                labels.append(1)  # Positive
        
        # 2. Sparse positive points in boundary region
        boundary_pixels = np.argwhere(boundary_region[y:y+h, x:x+w])
        if len(boundary_pixels) > 0:
            # Sample fewer boundary points
            num_boundary_points = min(3, len(boundary_pixels))
            boundary_indices = np.random.choice(len(boundary_pixels), num_boundary_points, replace=False)
            for idx in boundary_indices:
                py, px = boundary_pixels[idx]
                points.append((x + px, y + py))
                labels.append(1)  # Still positive but with less certainty
        
        # 3. Negative points in nearby background
        # Expand search area for negative points
        margin = 20
        y_start = max(0, y - margin)
        y_end = min(background_region.shape[0], y + h + margin)
        x_start = max(0, x - margin)
        x_end = min(background_region.shape[1], x + w + margin)
        
        bg_pixels = np.argwhere(background_region[y_start:y_end, x_start:x_end])
        if len(bg_pixels) > 0:
            # Sample negative points
            num_neg_points = min(5, len(bg_pixels))
            neg_indices = np.random.choice(len(bg_pixels), num_neg_points, replace=False)
            for idx in neg_indices:
                py, px = bg_pixels[idx]
                points.append((x_start + px, y_start + py))
                labels.append(0)  # Negative
        
        return points, labels

    def intelligent_fusion(self, core_region, sam_mask, clipseg_mask, min_area_ratio=0.9):
        """
        Intelligently fuse CLIPSeg and SAM results.
        """
        # Calculate areas
        core_area = np.sum(core_region)
        sam_area = np.sum(sam_mask)
        clipseg_area = np.sum(clipseg_mask)
        
        # Check if SAM result is valid
        if sam_area == 0:
            # SAM failed, return CLIPSeg result
            return clipseg_mask
        
        # Check if SAM preserves core region
        core_preserved = np.sum(sam_mask & core_region) / max(core_area, 1)
        
        if core_preserved > 0.95:
            # SAM preserved core well, use SAM result
            return sam_mask
        elif core_preserved > 0.7:
            # Partial preservation, combine results
            # Keep core region and add SAM's extensions
            combined_mask = core_region.astype(np.uint8)
            sam_extension = sam_mask & (~core_region)
            combined_mask = np.maximum(combined_mask, sam_extension)
            return combined_mask
        else:
            # SAM damaged core region, prefer CLIPSeg
            # But still try to use SAM for boundary refinement
            clipseg_eroded = binary_erosion(clipseg_mask, iterations=3)
            sam_boundary = sam_mask & (~clipseg_eroded)
            refined_mask = np.maximum(clipseg_mask, sam_boundary)
            return refined_mask

    def refine_class_hybrid(self, prob_map: np.ndarray, config: dict) -> np.ndarray:
        """
        Hybrid refinement combining CLIPSeg confidence with SAM boundary optimization.
        """
        # 1. Split regions
        core_region, boundary_region, background_region, binary_mask = self.split_regions(prob_map, config)
        
        # 2. Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > config['min_area']]
        
        if not filtered_contours:
            return np.zeros_like(prob_map, dtype=np.uint8)
        
        # 3. Process each instance
        final_mask = np.zeros_like(prob_map, dtype=np.uint8)
        
        for contour in filtered_contours:
            # Get instance-specific regions
            x, y, w, h = cv2.boundingRect(contour)
            instance_mask = np.zeros_like(binary_mask)
            cv2.drawContours(instance_mask, [contour], -1, 1, -1)
            
            # Extract regions for this instance
            instance_core = core_region & instance_mask.astype(bool)
            instance_boundary = boundary_region & instance_mask.astype(bool)
            
            # Generate adaptive points
            points, labels = self.get_adaptive_points(
                instance_core, instance_boundary, background_region, contour
            )
            
            # Prepare SAM inputs
            box = np.array([x, y, x + w, y + h])
            
            if len(points) > 0:
                point_coords = np.array(points)
                point_labels = np.array(labels)
            else:
                point_coords = None
                point_labels = None
            
            # Get SAM prediction
            try:
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box[None, :],
                    multimask_output=True
                )
                
                # Select best mask
                best_idx = np.argmax(scores)
                sam_mask = masks[best_idx].astype(np.uint8)
                
                # Intelligent fusion
                refined_instance = self.intelligent_fusion(
                    instance_core, sam_mask, instance_mask, 
                    min_area_ratio=config.get('min_area_ratio', 0.9)
                )
                
                final_mask = np.maximum(final_mask, refined_instance)
                
            except Exception as e:
                # If SAM fails, keep CLIPSeg result
                print(f"  SAM failed for an instance: {e}")
                final_mask = np.maximum(final_mask, instance_mask)
        
        # 4. Post-processing
        # Fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
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
            reverse=True
        )

        for class_name in sorted_classes:
            mask = refined_masks[class_name]
            final_resolved_map[mask > 0] = self.class_priorities.get(class_name, 0)

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
                print(f"  Hybrid refinement for {class_name}...")
                config = class_configs[class_name]
                refined_mask = self.refine_class_hybrid(prob_map, config)
                refined_masks[class_name] = refined_mask

        resolved_masks = self.resolve_overlaps(refined_masks)
        return resolved_masks

def create_colored_mask(masks_dict: dict, colors: dict, h: int, w: int) -> np.ndarray:
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_name, mask in masks_dict.items():
        if class_name in colors:
            colored_mask[mask > 0] = colors[class_name]
    return colored_mask

def main():
    """Main function to run the V5 hybrid refinement process."""
    # --- Configuration ---
    SAM_CHECKPOINT = "models/sam_vit_h_4b8939.pth"
    CLIPSEG_OUTPUT_DIR = "output/inference/finetuned"
    OUTPUT_DIR = "output/sam_refined_v5"
    TEST_IMAGES_DIR = "data/Vaihingen/finetune_data/test/images"

    CLASS_CONFIGS = {
        'building': {
            'prob_threshold': 0.5,      # Initial threshold
            'core_threshold': 0.8,      # High confidence core
            'boundary_width': 10,       # Pixels from edge
            'min_area': 50,
            'min_area_ratio': 0.9      # Minimum area preservation
        },
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

    refiner = SAMRefinementV5(sam_checkpoint=SAM_CHECKPOINT, model_type="vit_h")
    test_images = sorted(Path(TEST_IMAGES_DIR).glob("*.tif"))
    
    # Randomly select a subset of test images
    random.seed(42)
    num_samples = 439
    num_samples = min(num_samples, len(test_images))
    print(f"Total test images found: {len(test_images)}")
    print(f"Using {num_samples} images for testing")
    if len(test_images) > num_samples:
        test_images = random.sample(test_images, num_samples)
        print(f"Randomly selected {len(test_images)} images for testing")

    # --- Processing Loop ---
    for img_path in tqdm(test_images, desc="Hybrid refinement with SAM v5"):
        img_name = img_path.stem
        print(f"\nProcessing {img_name}...")

        # Load probability and mask files
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

        # Run hybrid refinement
        refined_masks = refiner.process_image(str(img_path), prob_maps, CLASS_CONFIGS)

        # Save refined masks
        for class_name, mask in refined_masks.items():
            safe_name = class_name.replace(' ', '_')
            Image.fromarray(mask * 255).save(f"{OUTPUT_DIR}/masks/{img_name}_{safe_name}_refined.tif")

        # Create 4-panel visualization including confidence regions
        original_image = np.array(Image.open(img_path).convert("RGB"))
        h, w, _ = original_image.shape

        # Create confidence visualization
        if 'building' in prob_maps:
            prob_map = prob_maps['building']
            config = CLASS_CONFIGS['building']
            core_region, boundary_region, _, _ = refiner.split_regions(prob_map, config)
            
            # Create confidence map visualization
            confidence_viz = np.zeros((h, w, 3), dtype=np.uint8)
            confidence_viz[core_region] = [0, 255, 0]  # Green for core
            confidence_viz[boundary_region] = [255, 255, 0]  # Yellow for boundary
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(confidence_viz)
            axes[0, 1].set_title('Confidence Regions (Green=Core, Yellow=Boundary)')
            axes[0, 1].axis('off')
            
            clipseg_color = create_colored_mask(clipseg_masks, CLASS_COLORS, h, w)
            axes[1, 0].imshow(clipseg_color)
            axes[1, 0].set_title('CLIPSeg Original')
            axes[1, 0].axis('off')
            
            refined_color = create_colored_mask(refined_masks, CLASS_COLORS, h, w)
            axes[1, 1].imshow(refined_color)
            axes[1, 1].set_title('SAM Hybrid Refinement (v5)')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/visualizations/{img_name}_comparison.png", dpi=150)
            plt.close(fig)

    print(f"\n✓ SAM v5 hybrid refinement complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()