#!/usr/bin/env python3
"""
评估每个样本大小训练的模型并生成详细的可视化报告 (V2版本)

功能：
1. 对每个训练好的模型（5, 10, 20, 40, 80, 160, 320）进行独立评估
2. 生成与evaluate.py相同格式的评估报告和可视化
3. 与预训练模型进行对比
4. 为每个模型创建独立的输出目录

V2版本修改：
- 使用与主pipeline对齐的扁平化mask结构
- 路径调整为sample_size_experiment_v2
- 数据加载逻辑与主pipeline完全一致
"""

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
from tqdm import tqdm
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))

from config import *
from utils import (
    load_model, 
    get_device,
    create_data_loader  # 使用主pipeline的数据加载器
)

# Disable warnings
import warnings
warnings.filterwarnings("ignore", message="The following named arguments are not valid")
warnings.filterwarnings("ignore", message="Using a slow image processor")

# Disable tokenizers parallelism to avoid fork warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 实验配置 - V2路径
EXPERIMENT_DATA_DIR = DATA_DIR / "Vaihingen" / "sample_size_experiment_v2"
EXPERIMENT_MODEL_DIR = MODELS_DIR / "sample_size_experiment_v2"
EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "sample_size_experiment_v2"
SAMPLE_SIZES = [5, 10, 20, 40, 80, 160, 320]



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

def evaluate_model_performance(model, test_loader, device, classes, desc_prefix="") -> Dict:
    """Evaluates a single model's performance and returns detailed results."""
    model.eval()
    # Initialize metric storage for each class
    class_metrics = {cls: {'iou': [], 'precision': [], 'recall': [], 'f1': [], 'pixel_accuracy': []} 
                     for cls in classes}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"{desc_prefix}Evaluating", leave=False):
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

def generate_comparison_report(results: Dict, output_dir: Path, sample_size: int):
    """Generates and saves a comparative performance report for a specific sample size."""
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
    summary_csv_path = output_dir / f'evaluation_summary_{sample_size}_samples.csv'
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
    print(f"  Summary saved to: {summary_csv_path}")
    
    # Save full results to JSON
    json_path = output_dir / f'full_evaluation_results_{sample_size}_samples.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization plots for metrics
    create_metric_comparison_plots(results, all_classes, metrics, output_dir, sample_size)

def create_metric_comparison_plots(results: Dict, classes: List[str], metrics: List[str], output_dir: Path, sample_size: int):
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
        bars2 = ax.bar(x + width/2, ft_values, width, label=f'Finetuned ({sample_size})', alpha=0.8)
        
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
    
    plt.suptitle(f'Model Performance Comparison - {sample_size} Training Samples', fontsize=16)
    plt.tight_layout()
    plot_path = output_dir / f'metrics_comparison_{sample_size}_samples.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_comparison(
    finetuned_model, ft_processor, 
    pretrained_model, pt_processor, 
    test_data_dir: str, device, classes, 
    output_dir: Path, num_samples: int, sample_size: int
):
    """Visualizes a 4-panel comparison for a number of random samples."""
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    test_images = sorted(list(Path(test_data_dir).glob('*.tif')))
    if len(test_images) > num_samples:
        random.seed(RANDOM_SEED)
        test_images = random.sample(test_images, num_samples)

    print(f"  Generating {len(test_images)} visualization samples...")

    for idx, image_path in enumerate(tqdm(test_images, desc="  Creating visualizations", leave=False)):
        image = Image.open(image_path).convert("RGB")
        images = [image] * len(classes)
        
        # Get ground truth masks - V2版本：使用扁平化mask结构
        gt_masks = {}
        mask_dir = EXPERIMENT_DATA_DIR / "fixed_test_set" / "masks"
        
        for class_name in classes:
            safe_class_name = class_name.replace(' ', '_')
            # V2版本：扁平化结构，尝试不同的文件扩展名
            mask_path_tif = mask_dir / f"{image_path.stem}_{safe_class_name}.tif"  
            mask_path_png = mask_dir / f"{image_path.stem}_{safe_class_name}.png"
            
            mask_loaded = False
            if mask_path_tif.exists():
                mask = np.array(Image.open(mask_path_tif).convert("L"))
                mask_loaded = True
            elif mask_path_png.exists():
                mask = np.array(Image.open(mask_path_png).convert("L"))
                mask_loaded = True
            else:
                mask = np.zeros(image.size[::-1], dtype=np.uint8)
                
            # 确保mask值在正确的范围内
            if mask_loaded:
                # 如果mask值在0-255范围，转换为0-1范围用于可视化
                if mask.max() > 1:
                    mask = (mask > 127).astype(np.uint8) * 255
                else:
                    mask = mask * 255
                    
            gt_masks[class_name] = mask

        # --- Inference and IoU calculation for both models ---
        models = {
            f"Finetuned-{sample_size}": (finetuned_model, ft_processor),
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
            color_map_rgb = np.array([CLASS_COLORS_RGB[c] for c in classes], dtype=np.uint8)
            segmentation_maps[model_name] = color_map_rgb[pred_labels.numpy()]

            # Calculate all metrics for each class
            all_metrics = {}
            for i, cls in enumerate(classes):
                metrics = calculate_metrics(resized_masks[i].numpy(), gt_masks[cls])
                all_metrics[cls] = metrics
            
            # Format text with key metrics - compact format to fit all classes
            metric_text = f"{model_name}:\n"
            # Show all classes in a more compact format
            for i, cls in enumerate(classes):
                m = all_metrics[cls]
                # Use abbreviated class names
                if cls == 'impervious surface':
                    abbr = 'Imperv'
                elif cls == 'low vegetation':
                    abbr = 'LowVeg'
                elif cls == 'background':
                    abbr = 'Bkgnd'
                else:
                    abbr = cls.capitalize()[:5]  # First 5 chars
                
                metric_text += f"{abbr}:{m['iou']:.2f} "
                if i == 2:  # Break line after 3 classes
                    metric_text += "\n"
            
            mean_iou = np.mean([m['iou'] for m in all_metrics.values()])
            metric_text += f"\nMean IoU: {mean_iou:.3f}"
            iou_texts[model_name] = metric_text

        # --- Create 4-panel plot ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f"Original Image\n{image_path.name}", fontsize=10)
        
        axes[0, 1].imshow(segmentation_maps['Pretrained'])
        axes[0, 1].set_title(iou_texts['Pretrained'], fontsize=8, loc='left')

        axes[1, 0].imshow(segmentation_maps[f'Finetuned-{sample_size}'])
        axes[1, 0].set_title(iou_texts[f'Finetuned-{sample_size}'], fontsize=8, loc='left')

        # Create composite ground truth from individual masks
        gt_composite = np.zeros((*image.size[::-1], 3), dtype=np.uint8)
        for i, (class_name, mask) in enumerate(gt_masks.items()):
            # masks are 0/255, so use > 127 threshold
            mask_bool = mask > 127
            gt_composite[mask_bool] = CLASS_COLORS_RGB[class_name]
        
        # 调试信息：检查ground truth是否有内容
        gt_pixels = np.sum(gt_composite > 0)
        if gt_pixels == 0:
            print(f"    Warning: Ground truth for {image_path.name} appears to be empty")
            # 如果所有mask都是空的，创建一个调试用的ground truth显示
            debug_msg = "No GT masks found\nCheck mask files:"
            for class_name in classes:
                safe_class_name = class_name.replace(' ', '_')
                mask_path_tif = mask_dir / f"{image_path.stem}_{safe_class_name}.tif"
                mask_path_png = mask_dir / f"{image_path.stem}_{safe_class_name}.png"
                if mask_path_tif.exists():
                    debug_msg += f"\n✓ {safe_class_name}.tif"
                elif mask_path_png.exists():
                    debug_msg += f"\n✓ {safe_class_name}.png"
                else:
                    debug_msg += f"\n✗ {safe_class_name}"
        
        axes[1, 1].imshow(gt_composite)
        axes[1, 1].set_title("Ground Truth", fontsize=10)

        for ax in axes.flat:
            ax.axis("off")
            
        plt.tight_layout(pad=1.0)
        save_path = vis_dir / f"sample_{idx+1}_comparison.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

def evaluate_single_model(sample_size: int, device, output_base_dir: Path):
    """评估单个样本大小的模型"""
    print(f"\n{'='*60}")
    print(f"Evaluating model trained with {sample_size} samples")
    print(f"{'='*60}")
    
    # 模型路径
    model_path = EXPERIMENT_MODEL_DIR / f"model_{sample_size}" / "best_model"
    
    # 检查模型是否存在
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return None
    
    # 输出目录
    output_dir = output_base_dir / f"model_{sample_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 加载模型
        print(f"Loading finetuned model from: {model_path}")
        finetuned_model, ft_processor, metadata = load_model(model_path, device)
        
        print("Loading pretrained model for comparison...")
        pretrained_model, pt_processor, _ = load_model(Path(PRETRAINED_MODEL), device)
        
        # 加载测试数据 - V2版本：使用主pipeline的数据加载器
        test_images_dir = EXPERIMENT_DATA_DIR / "fixed_test_set" / "images"
        test_masks_dir = EXPERIMENT_DATA_DIR / "fixed_test_set" / "masks"
        
        print(f"Loading test data from: {test_images_dir}")
        
        # 获取测试图像路径列表（create_data_loader需要图像路径列表，不是目录路径）
        test_image_paths = sorted([str(p) for p in test_images_dir.glob('*.tif')])
        
        if not test_image_paths:
            raise FileNotFoundError(f"No .tif images found in {test_images_dir}")
        
        # 使用正确的参数调用create_data_loader
        test_loader = create_data_loader(
            test_image_paths,      # 第一个参数是图像路径列表
            str(test_masks_dir),   # 第二个参数是masks目录
            URBAN_CLASSES,         # 类别列表
            ft_processor,          # processor
            batch_size=EVALUATION_CONFIG['batch_size'],
            shuffle=False
        )
        
        # 计算测试样本数量
        total_samples = len(test_image_paths) * len(URBAN_CLASSES)
        print(f"Loaded {len(test_image_paths)} images × {len(URBAN_CLASSES)} classes = {total_samples} test samples")
        
        # 运行评估
        print("\nEvaluating model performance...")
        results = {}
        results['finetuned'] = evaluate_model_performance(
            finetuned_model, test_loader, device, URBAN_CLASSES, 
            desc_prefix=f"Model-{sample_size} "
        )
        results['pretrained'] = evaluate_model_performance(
            pretrained_model, test_loader, device, URBAN_CLASSES,
            desc_prefix="Pretrained "
        )
        
        # 生成报告和可视化
        print("\nGenerating reports and visualizations...")
        generate_comparison_report(results, output_dir, sample_size)
        
        # 生成样本可视化
        visualize_comparison(
            finetuned_model, ft_processor,
            pretrained_model, pt_processor,
            str(test_images_dir), device, URBAN_CLASSES,
            output_dir, num_samples=EVALUATION_CONFIG['num_visualization_samples'], sample_size=sample_size
        )
        
        # 打印总结
        print(f"\n{'='*50}")
        print(f"Model-{sample_size} Evaluation Summary:")
        print(f"{'='*50}")
        
        metrics = ['IoU', 'Precision', 'Recall', 'F1', 'Pixel_Accuracy']
        print(f"{'Metric':<15} {'Pretrained':<12} {'Finetuned':<12} {'Improvement':<12}")
        print("-" * 50)
        
        for metric in metrics:
            metric_key = metric.lower()
            pt_value = results['pretrained'][f'mean_{metric_key}']
            ft_value = results['finetuned'][f'mean_{metric_key}']
            improvement = ft_value - pt_value
            print(f"{metric:<15} {pt_value:<12.4f} {ft_value:<12.4f} {improvement:+12.4f}")
        
        # 保存到summary
        results['sample_size'] = sample_size
        results['model_path'] = str(model_path)
        
        return results
        
    except Exception as e:
        print(f"❌ Error evaluating model-{sample_size}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数：评估所有样本大小的模型"""
    print("🚀 Sample Size Experiment - Individual Model Evaluation (V2)")
    print("=" * 60)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_base_dir = EXPERIMENT_OUTPUT_DIR / "individual_evaluations"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 评估每个模型
    all_results = {}
    successful_evaluations = []
    failed_evaluations = []
    
    for sample_size in SAMPLE_SIZES:
        results = evaluate_single_model(sample_size, device, output_base_dir)
        if results:
            all_results[f'model_{sample_size}'] = results
            successful_evaluations.append(sample_size)
        else:
            failed_evaluations.append(sample_size)
    
    # 创建汇总报告
    if all_results:
        print("\n" + "="*60)
        print("Creating summary report across all models...")
        print("="*60)
        
        # 创建汇总表
        summary_data = []
        for model_name, results in all_results.items():
            sample_size = results['sample_size']
            row = {
                'Sample_Size': sample_size,
                'Mean_IoU': results['finetuned']['mean_iou'],
                'Mean_Precision': results['finetuned']['mean_precision'],
                'Mean_Recall': results['finetuned']['mean_recall'],
                'Mean_F1': results['finetuned']['mean_f1'],
                'Mean_Pixel_Acc': results['finetuned']['mean_pixel_accuracy'],
                'IoU_vs_Pretrained': results['finetuned']['mean_iou'] - results['pretrained']['mean_iou']
            }
            summary_data.append(row)
        
        # 添加预训练模型基准
        if all_results:
            first_result = list(all_results.values())[0]
            pretrained_row = {
                'Sample_Size': 0,  # 0表示预训练模型
                'Mean_IoU': first_result['pretrained']['mean_iou'],
                'Mean_Precision': first_result['pretrained']['mean_precision'],
                'Mean_Recall': first_result['pretrained']['mean_recall'],
                'Mean_F1': first_result['pretrained']['mean_f1'],
                'Mean_Pixel_Acc': first_result['pretrained']['mean_pixel_accuracy'],
                'IoU_vs_Pretrained': 0.0
            }
            summary_data.insert(0, pretrained_row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Sample_Size')
        
        # 保存汇总表
        summary_path = output_base_dir / 'all_models_summary.csv'
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        
        print("\nOverall Performance Summary:")
        print(summary_df.to_string(index=False))
        
        # 保存完整结果
        with open(output_base_dir / 'all_models_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # 打印最终总结
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"✅ Successfully evaluated: {successful_evaluations}")
    if failed_evaluations:
        print(f"❌ Failed: {failed_evaluations}")
    print(f"\nResults saved to: {output_base_dir}")
    print("\n🎉 Individual model evaluation completed! (V2)")

if __name__ == "__main__":
    main()