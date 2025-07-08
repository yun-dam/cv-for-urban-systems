

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
    FineTuneDataset,
    get_device
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
                        default=DEVICE,
                        help='Computation device (auto/cpu/cuda/mps).')
    
    return parser.parse_args()

def setup_device(device_arg: str) -> torch.device:
    """Sets up the computation device."""
    # Use get_device function from utils.py
    # If user didn't specify device, use global configuration
    if device_arg == DEVICE:
        device = get_device()
    else:
        # If user specified a specific device, respect user's choice
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

def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    """Calculate multiple evaluation metrics for a single prediction."""
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = (true_mask > 0).astype(np.uint8)
    
    # True Positives, False Positives, False Negatives, True Negatives
    tp = np.sum(pred_binary & true_binary)
    fp = np.sum(pred_binary & ~true_binary)
    fn = np.sum(~pred_binary & true_binary)
    tn = np.sum(~pred_binary & ~true_binary)
    
    # IoU
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Pixel Accuracy
    pixel_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pixel_accuracy': pixel_acc
    }

def evaluate_model_performance(model, processor, test_loader, device, classes) -> Dict:
    """Evaluates a single model's performance and returns detailed results."""
    model.eval()
    # Initialize metric storage for each class
    class_metrics = {cls: {'iou': [], 'precision': [], 'recall': [], 'f1': [], 'pixel_accuracy': []} 
                     for cls in classes}
    
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
                
                # Calculate all metrics
                metrics = calculate_metrics(predictions[i, 0], targets[i, 0])
                
                # Store metrics
                for metric_name, value in metrics.items():
                    class_metrics[class_name][metric_name].append(value)

    # Calculate mean values for each metric and class
    results = {}
    for metric_name in ['iou', 'precision', 'recall', 'f1', 'pixel_accuracy']:
        class_values = {}
        for cls in classes:
            values = class_metrics[cls][metric_name]
            class_values[cls] = np.mean(values) if values else 0.0
        
        results[f'class_{metric_name}s'] = class_values
        results[f'mean_{metric_name}'] = np.mean(list(class_values.values()))
    
    return results

def generate_comparison_report(results: Dict, output_dir: Path):
    """Generates and saves a comparative performance report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrames for each metric
    metrics = ['iou', 'precision', 'recall', 'f1', 'pixel_accuracy']
    all_classes = sorted(results['finetuned']['class_ious'].keys())
    
    # Summary DataFrame with all metrics
    summary_data = []
    
    for cls in all_classes:
        row = {'Class': cls}
        for metric in metrics:
            ft_value = results['finetuned'][f'class_{metric}s'].get(cls, 0.0)
            pt_value = results['pretrained'][f'class_{metric}s'].get(cls, 0.0)
            improvement = ft_value - pt_value
            
            row[f'FT_{metric.upper()}'] = ft_value
            row[f'PT_{metric.upper()}'] = pt_value
            row[f'Δ_{metric.upper()}'] = improvement
        
        summary_data.append(row)
    
    # Add mean values
    mean_row = {'Class': 'Mean'}
    for metric in metrics:
        ft_mean = results['finetuned'][f'mean_{metric}']
        pt_mean = results['pretrained'][f'mean_{metric}']
        mean_improvement = ft_mean - pt_mean
        
        mean_row[f'FT_{metric.upper()}'] = ft_mean
        mean_row[f'PT_{metric.upper()}'] = pt_mean
        mean_row[f'Δ_{metric.upper()}'] = mean_improvement
    
    summary_data.append(mean_row)
    
    # Create and save comprehensive summary
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = output_dir / 'evaluation_summary_all_metrics.csv'
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
    print(f"Comprehensive summary saved to: {summary_csv_path}")
    
    # Create metric-specific comparison tables
    for metric in metrics:
        metric_data = []
        for cls in all_classes:
            ft_value = results['finetuned'][f'class_{metric}s'].get(cls, 0.0)
            pt_value = results['pretrained'][f'class_{metric}s'].get(cls, 0.0)
            improvement = ft_value - pt_value
            metric_data.append({
                'Class': cls,
                f'Finetuned_{metric.upper()}': ft_value,
                f'Pretrained_{metric.upper()}': pt_value,
                'Improvement': improvement
            })
        
        # Add mean
        ft_mean = results['finetuned'][f'mean_{metric}']
        pt_mean = results['pretrained'][f'mean_{metric}']
        metric_data.append({
            'Class': 'Mean',
            f'Finetuned_{metric.upper()}': ft_mean,
            f'Pretrained_{metric.upper()}': pt_mean,
            'Improvement': ft_mean - pt_mean
        })
        
        metric_df = pd.DataFrame(metric_data)
        metric_csv_path = output_dir / f'{metric}_comparison.csv'
        metric_df.to_csv(metric_csv_path, index=False, float_format='%.4f')
    
    # Save full results to JSON
    json_path = output_dir / 'full_evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to: {json_path}")
    
    # Create visualization plots for metrics
    create_metric_comparison_plots(results, all_classes, metrics, output_dir)

def create_metric_comparison_plots(results: Dict, classes: List[str], metrics: List[str], output_dir: Path):
    """Create bar plots comparing metrics between pretrained and finetuned models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data
        ft_values = [results['finetuned'][f'class_{metric}s'][cls] for cls in classes]
        pt_values = [results['pretrained'][f'class_{metric}s'][cls] for cls in classes]
        
        # Add mean values
        ft_values.append(results['finetuned'][f'mean_{metric}'])
        pt_values.append(results['pretrained'][f'mean_{metric}'])
        class_labels = classes + ['Mean']
        
        # Create grouped bar plot
        x = np.arange(len(class_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pt_values, width, label='Pretrained', alpha=0.8)
        bars2 = ax.bar(x + width/2, ft_values, width, label='Finetuned', alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Classes')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
    
    # Remove empty subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plot_path = output_dir / 'metrics_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison plot saved to: {plot_path}")

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

            # Calculate all metrics for each class
            all_metrics = {}
            for i, cls in enumerate(classes):
                metrics = calculate_metrics(resized_masks[i].numpy(), gt_masks[cls])
                all_metrics[cls] = metrics
            
            # Format text with key metrics
            metric_text = f"{model_name} Metrics:\n"
            for cls in classes:
                m = all_metrics[cls]
                metric_text += f"  {cls}: IoU={m['iou']:.3f}, F1={m['f1']:.3f}\n"
            iou_texts[model_name] = metric_text

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
        
        print("\n" + "="*70)
        print("✅ Evaluation Summary:")
        print("="*70)
        
        # Display all metrics
        metrics = ['IoU', 'Precision', 'Recall', 'F1', 'Pixel_Accuracy']
        print(f"{'Metric':<15} {'Pretrained':<12} {'Finetuned':<12} {'Improvement':<12}")
        print("-" * 70)
        
        for metric in metrics:
            metric_key = metric.lower()
            pt_value = results['pretrained'][f'mean_{metric_key}']
            ft_value = results['finetuned'][f'mean_{metric_key}']
            improvement = ft_value - pt_value
            print(f"{metric:<15} {pt_value:<12.4f} {ft_value:<12.4f} {improvement:+12.4f}")
        
        print("="*70)
        
        # Per-class IoU improvements
        print("\nPer-class IoU improvements:")
        for cls in URBAN_CLASSES:
            pt_iou = results['pretrained']['class_ious'][cls]
            ft_iou = results['finetuned']['class_ious'][cls]
            improvement = ft_iou - pt_iou
            print(f"  {cls:<20}: {pt_iou:.4f} → {ft_iou:.4f} ({improvement:+.4f})")
        
        print(f"\nDetailed results saved to: {output_dir}")
        
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