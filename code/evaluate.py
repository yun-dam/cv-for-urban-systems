

import os
import sys
import json
import argparse
import tifffile
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from utils import (
    create_data_loader, 
    load_model, 
    calculate_iou,
    FineTuneDataset
)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='CLIPSeg Model Evaluation Script')
    
    parser.add_argument('--model_path', type=str, 
                        default=str(EVALUATION_CONFIG['default_model_path']),
                        help='Path to the fine-tuned model directory.')
    parser.add_argument('--test_data_dir', type=str, 
                        default=str(EVALUATION_CONFIG['default_test_data']),
                        help='Path to the test data directory.')
    parser.add_argument('--output_dir', type=str, 
                        default=str(EVALUATION_CONFIG['default_output_dir']),
                        help='Directory to save evaluation results.')
    parser.add_argument('--batch_size', type=int, 
                        default=EVALUATION_CONFIG['batch_size'],
                        help='Batch size for evaluation.')
    parser.add_argument('--num_samples', type=int, 
                        default=EVALUATION_CONFIG['num_visualization_samples'],
                        help='Number of samples to visualize.')
    parser.add_argument('--device', type=str, 
                        default=EVALUATION_CONFIG['device'],
                        help='Computation device (auto/cpu/cuda/mps).')
    
    return parser.parse_args()

def setup_device(device_arg: str) -> torch.device:
    """Sets up the computation device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device

def get_ground_truth_masks(label_path: Path, classes: List[str], colors: Dict[str, Tuple[int, int, int]]) -> Dict[str, np.ndarray]:
    """Generates a dictionary of ground truth masks from a label file."""
    label_img_bgr = tifffile.imread(label_path)
    h, w, _ = label_img_bgr.shape
    mask_dict = {cls: np.zeros((h, w), dtype=np.uint8) for cls in classes}

    for class_name, bgr_color in colors.items():
        # Find the prompt name corresponding to the official class name
        prompt_name = class_name.lower().replace('_', ' ')
        if prompt_name in mask_dict:
            mask = np.all(label_img_bgr == np.array(bgr_color), axis=-1)
            mask_dict[prompt_name] = (mask * 255).astype(np.uint8)
            
    return mask_dict

def evaluate_model_performance(model, processor, test_loader, device, classes) -> Dict:
    """Evaluates a single model's performance and returns detailed results."""
    model.eval()
    class_ious = {cls: [] for cls in classes}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {Path(model.name_or_path).name}", leave=False):
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            logits = outputs.logits.unsqueeze(1)
            labels = batch["labels"].to(device)
            class_indices = batch["class_indices"].cpu().numpy()

            predictions = torch.sigmoid(logits).cpu().numpy()
            targets = labels.cpu().numpy()

            for i in range(predictions.shape[0]):
                class_idx = class_indices[i]
                class_name = classes[class_idx]
                iou = calculate_iou(predictions[i, 0], targets[i, 0])
                class_ious[class_name].append(iou)

    mean_ious = {cls: np.mean(ious) if ious else 0.0 for cls, ious in class_ious.items()}
    mean_iou = np.mean(list(mean_ious.values()))
    
    return {'class_ious': mean_ious, 'mean_iou': mean_iou}

def generate_comparison_report(results: Dict, output_dir: Path):
    """Generates and saves a comparative performance report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a DataFrame for easy comparison
    df_data = []
    all_classes = sorted(results['finetuned']['class_ious'].keys())
    
    for cls in all_classes:
        ft_iou = results['finetuned']['class_ious'].get(cls, 0.0)
        pt_iou = results['pretrained']['class_ious'].get(cls, 0.0)
        improvement = ft_iou - pt_iou
        df_data.append({'Class': cls, 'Finetuned_IoU': ft_iou, 'Pretrained_IoU': pt_iou, 'Improvement': improvement})
        
    # Add mean IoU
    ft_miou = results['finetuned']['mean_iou']
    pt_miou = results['pretrained']['mean_iou']
    mean_improvement = ft_miou - pt_miou
    df_data.append({'Class': 'Mean', 'Finetuned_IoU': ft_miou, 'Pretrained_IoU': pt_miou, 'Improvement': mean_improvement})
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    csv_path = output_dir / 'evaluation_summary.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Comparative report saved to: {csv_path}")

    # Save full results to JSON
    json_path = output_dir / 'full_evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to: {json_path}")

def visualize_comparison(
    finetuned_model, ft_processor, 
    pretrained_model, pt_processor, 
    test_data_dir: str, device, classes, 
    output_dir: Path, num_samples: int
):
    """Visualizes a 4-panel comparison for a number of random samples."""
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    test_images = sorted(list(Path(test_data_dir).glob('*.tif')))
    if len(test_images) > num_samples:
        random.seed(RANDOM_SEED)
        test_images = random.sample(test_images, num_samples)

    print(f"Generating {len(test_images)} visualization samples...")

    for image_path in tqdm(test_images, desc="Creating visualizations"):
        label_path = FINETUNE_DATA_DIR / "test/labels" / image_path.name
        if not label_path.exists():
            print(f"Warning: Label for {image_path.name} not found. Skipping.")
            continue

        image = Image.open(image_path).convert("RGB")
        images = [image] * len(classes)
        
        # Get ground truth masks
        gt_masks = get_ground_truth_masks(label_path, classes, CLASS_COLORS_BGR)

        # --- Inference and IoU calculation for both models ---
        models = {
            "Finetuned": (finetuned_model, ft_processor),
            "Pretrained": (pretrained_model, pt_processor)
        }
        
        segmentation_maps = {}
        iou_texts = {}

        for model_name, (model, processor) in models.items():
            inputs = processor(images=images, text=classes, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            masks = outputs.logits.sigmoid().cpu()
            resized_masks = torch.nn.functional.interpolate(masks.unsqueeze(0), size=image.size[::-1], mode='bilinear', align_corners=False).squeeze(0)
            
            # Create segmentation map
            pred_labels = resized_masks.argmax(dim=0)
            color_map_rgb = np.array([SEGMENTATION_COLORS_RGB[c] for c in classes], dtype=np.uint8)
            segmentation_maps[model_name] = color_map_rgb[pred_labels.numpy()]

            # Calculate and format IoU text
            iou_scores = {cls: calculate_iou(resized_masks[i].numpy(), gt_masks[cls]) for i, cls in enumerate(classes)}
            iou_text = f"{model_name} IoU Scores:\n" + "\n".join([f"  - {c}: {s:.3f}" for c, s in iou_scores.items()])
            iou_texts[model_name] = iou_text

        # --- Create 4-panel plot ---
        fig, axes = plt.subplots(2, 2, figsize=EVALUATION_CONFIG['figure_size'], dpi=EVALUATION_CONFIG['visualization_dpi'])
        
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f"Original Image\n{image_path.name}", fontsize=10)
        
        axes[0, 1].imshow(segmentation_maps['Pretrained'])
        axes[0, 1].set_title(iou_texts['Pretrained'], fontsize=8, loc='left')

        axes[1, 0].imshow(segmentation_maps['Finetuned'])
        axes[1, 0].set_title(iou_texts['Finetuned'], fontsize=8, loc='left')

        axes[1, 1].imshow(tifffile.imread(label_path))
        axes[1, 1].set_title("Ground Truth", fontsize=10)

        for ax in axes.flat:
            ax.axis("off")
            
        plt.tight_layout(pad=1.0)
        save_path = vis_dir / f"{image_path.stem}_4panel_comparison.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    print(f"Visualizations saved to: {vis_dir}")

def main():
    """Main function to run the evaluation."""
    args = parse_args()
    
    print("🔧 Evaluation Configuration:")
    print(f"  - Fine-tuned Model: {args.model_path}")
    print(f"  - Pre-trained Model: {PRETRAINED_MODEL}")
    print(f"  - Test Data: {args.test_data_dir}")
    print(f"  - Output Dir: {args.output_dir}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Visualization Samples: {args.num_samples}")
    
    device = setup_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # --- Load Models ---
        print("\nLoading models...")
        finetuned_model, ft_processor, _ = load_model(Path(args.model_path), device)
        pretrained_model, pt_processor, _ = load_model(Path(PRETRAINED_MODEL), device)

        # --- Load Data ---
        print("Loading test data...")
        # We use the fine-tuned processor for data loading, as it's compatible.
        test_dataset = FineTuneDataset(
            image_paths=sorted([str(p) for p in Path(args.test_data_dir).glob('*.tif')]),
            mask_dir=str(FINETUNE_DATA_DIR / "test/masks"),
            classes=URBAN_CLASSES,
            processor=ft_processor
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=lambda b: ft_processor.collate_fn(b, ft_processor)
        )
        print(f"Loaded {len(test_dataset)} test samples.")

        # --- Run Evaluation ---
        print("\nStarting model performance evaluation...")
        results = {}
        results['finetuned'] = evaluate_model_performance(finetuned_model, ft_processor, test_loader, device, URBAN_CLASSES)
        results['pretrained'] = evaluate_model_performance(pretrained_model, pt_processor, test_loader, device, URBAN_CLASSES)
        
        # --- Generate Reports and Visuals ---
        print("\nGenerating reports and visualizations...")
        generate_comparison_report(results, output_dir)
        visualize_comparison(
            finetuned_model, ft_processor,
            pretrained_model, pt_processor,
            args.test_data_dir, device, URBAN_CLASSES,
            output_dir, args.num_samples
        )
        
        print("\n" + "="*50)
        print("✅ Evaluation Summary (mIoU):")
        print(f"  - Pretrained: {results['pretrained']['mean_iou']:.4f}")
        print(f"  - Finetuned:  {results['finetuned']['mean_iou']:.4f}")
        print(f"  - Improvement: {results['finetuned']['mean_iou'] - results['pretrained']['mean_iou']:+.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ An error occurred during evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # A simple trick to make the collate_fn accessible to the DataLoader
    CLIPSegProcessor.collate_fn = staticmethod(
        lambda batch, processor: {
            'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'input_ids': torch.nn.utils.rnn.pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id),
            'attention_mask': torch.nn.utils.rnn.pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0),
            'class_indices': torch.tensor([item['class_idx'] for item in batch], dtype=torch.long)
        }
    )
    exit(main())