#!/usr/bin/env python3
"""
评估所有模型的性能并生成可视化图表 (V2版本)

功能：
1. 计算每个模型在测试集上的IoU、Precision、Recall、F1等指标
2. 生成性能随训练样本数变化的曲线图
3. 创建详细的评估报告
4. 生成示例预测的可视化对比

V2版本修改：
- 使用与主pipeline对齐的扁平化mask结构
- 路径调整为sample_size_experiment_v2
- Mask加载逻辑与主pipeline完全一致
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tifffile
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))

from config import *

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 实验配置 - V2路径
EXPERIMENT_DATA_DIR = DATA_DIR / "Vaihingen" / "sample_size_experiment_v2"
EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "sample_size_experiment_v2"
SAMPLE_SIZES = [5, 10, 20, 40, 80, 160, 320]


def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    """
    计算评估指标
    
    Args:
        pred_mask: 预测mask (0-1之间的概率或0/255的二值)
        true_mask: 真实mask (0或255)
        threshold: 二值化阈值
    
    Returns:
        metrics: 包含IoU、Precision、Recall、F1、Accuracy的字典
    """
    # 确保是numpy数组
    if not isinstance(pred_mask, np.ndarray):
        pred_mask = np.array(pred_mask)
    if not isinstance(true_mask, np.ndarray):
        true_mask = np.array(true_mask)
    
    # 二值化
    if pred_mask.max() > 1:
        pred_binary = (pred_mask > 127).astype(np.uint8)
    else:
        pred_binary = (pred_mask > threshold).astype(np.uint8)
    
    # 真实mask可能是0/1或0/255
    if true_mask.max() > 1:
        true_binary = (true_mask > 127).astype(np.uint8)
    else:
        true_binary = (true_mask > 0).astype(np.uint8)
    
    # 计算TP, FP, FN, TN
    tp = np.sum(pred_binary & true_binary)
    fp = np.sum(pred_binary & ~true_binary)
    fn = np.sum(~pred_binary & true_binary)
    tn = np.sum(~pred_binary & ~true_binary)
    
    # 计算指标
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def evaluate_model_predictions(pred_dir, gt_dir):
    """
    评估一个模型的所有预测结果
    
    Args:
        pred_dir: 预测结果目录
        gt_dir: 真实标签目录
    
    Returns:
        results: 评估结果字典
    """
    results = {
        'per_class': {},
        'per_image': {},
        'overall': {}
    }
    
    # 初始化每个类别的指标列表
    for class_name in URBAN_CLASSES:
        results['per_class'][class_name] = {
            'iou': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': []
        }
    
    # 获取所有测试图像
    test_images = sorted(gt_dir.glob("*.tif"))
    
    print(f"Evaluating {len(test_images)} images...")
    
    for img_path in tqdm(test_images):
        img_name = img_path.stem
        results['per_image'][img_name] = {}
        
        # 对每个类别计算指标
        for class_name in URBAN_CLASSES:
            safe_class_name = class_name.replace(' ', '_')
            
            # 加载预测mask - V2版本：修正文件格式和类别名处理
            # 预测结果也是.tif格式，且类别名中的空格已被下划线替代
            pred_path = pred_dir / "masks" / f"{img_name}_{safe_class_name}.tif"
            if not pred_path.exists():
                # 尝试png格式作为备选
                pred_path = pred_dir / "masks" / f"{img_name}_{safe_class_name}.png"
                if not pred_path.exists():
                    print(f"Warning: Prediction not found for {img_name} - {class_name}")
                    continue
            
            pred_mask = np.array(Image.open(pred_path).convert("L"))
            
            # V2版本：加载真实mask - 使用扁平化结构
            gt_path = gt_dir.parent / "masks" / f"{img_name}_{safe_class_name}.tif"
            if not gt_path.exists():
                # 尝试png格式
                gt_path = gt_dir.parent / "masks" / f"{img_name}_{safe_class_name}.png"
                if not gt_path.exists():
                    print(f"Warning: Ground truth not found for {img_name} - {class_name}")
                    continue
            
            true_mask = np.array(Image.open(gt_path).convert("L"))
            
            # 计算指标
            metrics = calculate_metrics(pred_mask, true_mask)
            
            # 保存结果
            results['per_image'][img_name][class_name] = metrics
            
            # 累积到类别级别
            for metric_name, value in metrics.items():
                if metric_name in results['per_class'][class_name]:
                    results['per_class'][class_name][metric_name].append(value)
    
    # 计算每个类别的平均指标
    for class_name in URBAN_CLASSES:
        class_metrics = results['per_class'][class_name]
        for metric_name in ['iou', 'precision', 'recall', 'f1', 'accuracy']:
            if class_metrics[metric_name]:
                class_metrics[f'mean_{metric_name}'] = np.mean(class_metrics[metric_name])
                class_metrics[f'std_{metric_name}'] = np.std(class_metrics[metric_name])
            else:
                class_metrics[f'mean_{metric_name}'] = 0.0
                class_metrics[f'std_{metric_name}'] = 0.0
    
    # 计算总体平均指标
    all_metrics = {
        'iou': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': []
    }
    
    for img_results in results['per_image'].values():
        for class_results in img_results.values():
            for metric_name in all_metrics.keys():
                if metric_name in class_results:
                    all_metrics[metric_name].append(class_results[metric_name])
    
    # 计算总体统计
    for metric_name, values in all_metrics.items():
        if values:
            results['overall'][f'mean_{metric_name}'] = np.mean(values)
            results['overall'][f'std_{metric_name}'] = np.std(values)
        else:
            results['overall'][f'mean_{metric_name}'] = 0.0
            results['overall'][f'std_{metric_name}'] = 0.0
    
    return results


def plot_performance_curves(all_results, output_dir):
    """
    绘制性能随训练样本数变化的曲线（使用对数刻度）
    """
    metrics_to_plot = ['iou', 'precision', 'recall', 'f1']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # 收集数据点
        x_values = []
        y_values = []
        y_errors = []
        
        # 预训练模型 (用1代表，因为对数坐标不能为0)
        if 'pretrained' in all_results:
            x_values.append(1)
            y_values.append(all_results['pretrained']['overall'][f'mean_{metric}'])
            y_errors.append(all_results['pretrained']['overall'][f'std_{metric}'])
        
        # 微调模型
        for size in SAMPLE_SIZES:
            model_name = f'finetuned_{size}'
            if model_name in all_results:
                x_values.append(size)
                y_values.append(all_results[model_name]['overall'][f'mean_{metric}'])
                y_errors.append(all_results[model_name]['overall'][f'std_{metric}'])
        
        # 绘制曲线
        ax.errorbar(x_values, y_values, yerr=y_errors, 
                   marker='o', markersize=8, linewidth=2, 
                   capsize=5, capthick=2, label=metric.upper())
        
        # 设置坐标轴
        ax.set_xlabel('Number of Training Images (log scale)', fontsize=12)
        ax.set_ylabel(f'Mean {metric.upper()}', fontsize=12)
        ax.set_title(f'{metric.upper()} vs Training Set Size', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 设置对数刻度
        ax.set_xscale('log')
        
        # 设置x轴刻度和标签
        all_x = [1] + SAMPLE_SIZES
        ax.set_xticks(all_x)
        ax.set_xticklabels(['Pretrained'] + [str(s) for s in SAMPLE_SIZES], rotation=45)
        
        # 设置y轴范围
        ax.set_ylim(0, 1.05)
        
        # 添加数值标签
        for x, y in zip(x_values, y_values):
            ax.annotate(f'{y:.3f}', 
                       xy=(x, y), 
                       xytext=(0, 5), 
                       textcoords='offset points',
                       ha='center',
                       fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'performance_curves.pdf', bbox_inches='tight')
    print(f"✓ Saved performance curves")


def plot_class_wise_comparison(all_results, output_dir):
    """
    绘制每个类别的性能对比（使用对数刻度）
    """
    # 为每个类别创建一个子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, class_name in enumerate(URBAN_CLASSES):
        ax = axes[idx]
        
        # 收集IoU数据
        x_values = []
        y_values = []
        
        # 预训练模型
        if 'pretrained' in all_results:
            x_values.append(1)  # 用1代表预训练模型
            y_values.append(all_results['pretrained']['per_class'][class_name]['mean_iou'])
        
        # 微调模型
        for size in SAMPLE_SIZES:
            model_name = f'finetuned_{size}'
            if model_name in all_results:
                x_values.append(size)
                y_values.append(all_results[model_name]['per_class'][class_name]['mean_iou'])
        
        # 绘制曲线
        ax.plot(x_values, y_values, marker='o', markersize=8, linewidth=2)
        
        # 设置坐标轴
        ax.set_xlabel('Number of Training Images (log scale)', fontsize=11)
        ax.set_ylabel('Mean IoU', fontsize=11)
        ax.set_title(f'{class_name}', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # 设置对数刻度
        ax.set_xscale('log')
        
        # 设置x轴刻度
        all_x = [1] + SAMPLE_SIZES
        ax.set_xticks(all_x)
        ax.set_xticklabels(['Pre'] + [str(s) for s in SAMPLE_SIZES], rotation=45)
        
        # 设置y轴范围
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_wise_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved class-wise performance comparison")


def create_summary_table(all_results, output_dir):
    """
    创建汇总表格
    
    Args:
        all_results: 所有模型的评估结果
        output_dir: 输出目录
    """
    # 准备数据
    rows = []
    
    # 添加预训练模型
    if 'pretrained' in all_results:
        row = {
            'Model': 'Pretrained',
            'Training Images': 0,
            'Mean IoU': all_results['pretrained']['overall']['mean_iou'],
            'Mean Precision': all_results['pretrained']['overall']['mean_precision'],
            'Mean Recall': all_results['pretrained']['overall']['mean_recall'],
            'Mean F1': all_results['pretrained']['overall']['mean_f1'],
            'Mean Accuracy': all_results['pretrained']['overall']['mean_accuracy']
        }
        rows.append(row)
    
    # 添加微调模型
    for size in SAMPLE_SIZES:
        model_name = f'finetuned_{size}'
        if model_name in all_results:
            row = {
                'Model': f'Finetuned-{size}',
                'Training Images': size,
                'Mean IoU': all_results[model_name]['overall']['mean_iou'],
                'Mean Precision': all_results[model_name]['overall']['mean_precision'],
                'Mean Recall': all_results[model_name]['overall']['mean_recall'],
                'Mean F1': all_results[model_name]['overall']['mean_f1'],
                'Mean Accuracy': all_results[model_name]['overall']['mean_accuracy']
            }
            rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    
    # 格式化数值
    for col in ['Mean IoU', 'Mean Precision', 'Mean Recall', 'Mean F1', 'Mean Accuracy']:
        df[col] = df[col].apply(lambda x: f'{x:.4f}')
    
    # 保存表格
    df.to_csv(output_dir / 'summary_table.csv', index=False)
    
    # 打印表格
    print("\n" + "="*80)
    print("Performance Summary")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


def visualize_sample_predictions(all_results, output_dir, num_samples=20):
    """
    可视化样本预测结果（类似evaluate_each_model_v2.py的风格）
    使用argmax来生成分割图，每个像素只属于一个类别
    """
    # 获取测试图像
    test_images_dir = EXPERIMENT_DATA_DIR / "fixed_test_set" / "images"
    all_test_images = sorted(test_images_dir.glob("*.tif"))
    
    # 随机选择样本
    import random
    random.seed(RANDOM_SEED)
    test_images = random.sample(all_test_images, min(num_samples, len(all_test_images)))
    
    inference_results_dir = EXPERIMENT_OUTPUT_DIR / "inference_results"
    
    # 包含所有样本大小的模型
    models = ['pretrained'] + [f'finetuned_{s}' for s in SAMPLE_SIZES]
    
    for img_idx, img_path in enumerate(test_images):
        img_name = img_path.stem
        
        # 加载原始图像
        original_img = np.array(Image.open(img_path))
        h, w = original_img.shape[:2]
        
        # 准备可视化的图像列表
        vis_images = []
        vis_titles = []
        
        # 1. 原始图像
        vis_images.append(original_img)
        vis_titles.append('Original')
        
        # 2. 所有模型的预测结果
        for model_name in models:
            # 加载所有类别的预测mask
            pred_probs = np.zeros((len(URBAN_CLASSES), h, w), dtype=np.float32)
            
            for i, class_name in enumerate(URBAN_CLASSES):
                safe_class_name = class_name.replace(' ', '_')
                pred_path = inference_results_dir / model_name / "masks" / f"{img_name}_{safe_class_name}.tif"
                if not pred_path.exists():
                    pred_path = inference_results_dir / model_name / "masks" / f"{img_name}_{safe_class_name}.png"
                
                if pred_path.exists():
                    pred_mask = np.array(Image.open(pred_path).convert("L"))
                    # 转换为0-1概率
                    if pred_mask.max() > 1:
                        pred_probs[i] = pred_mask / 255.0
                    else:
                        pred_probs[i] = pred_mask
            
            # 使用argmax选择每个像素的类别
            pred_labels = pred_probs.argmax(axis=0)
            
            # 创建彩色分割图
            color_map = np.zeros((h, w, 3), dtype=np.uint8)
            for i, class_name in enumerate(URBAN_CLASSES):
                color_map[pred_labels == i] = CLASS_COLORS_RGB[class_name]
            
            vis_images.append(color_map)
            
            # 设置标题
            if model_name == 'pretrained':
                vis_titles.append('Pretrained')
            else:
                size = int(model_name.split('_')[1])
                vis_titles.append(f'FT-{size}')
        
        # 3. Ground Truth
        gt_probs = np.zeros((len(URBAN_CLASSES), h, w), dtype=np.float32)
        
        for i, class_name in enumerate(URBAN_CLASSES):
            safe_class_name = class_name.replace(' ', '_')
            gt_path = EXPERIMENT_DATA_DIR / "fixed_test_set" / "masks" / f"{img_name}_{safe_class_name}.tif"
            if not gt_path.exists():
                gt_path = EXPERIMENT_DATA_DIR / "fixed_test_set" / "masks" / f"{img_name}_{safe_class_name}.png"
            
            if gt_path.exists():
                gt_mask = np.array(Image.open(gt_path).convert("L"))
                # 转换为0-1
                if gt_mask.max() > 1:
                    gt_probs[i] = gt_mask / 255.0
                else:
                    gt_probs[i] = gt_mask
        
        # 使用argmax选择每个像素的类别
        gt_labels = gt_probs.argmax(axis=0)
        
        # 创建彩色GT图
        gt_color_map = np.zeros((h, w, 3), dtype=np.uint8)
        for i, class_name in enumerate(URBAN_CLASSES):
            gt_color_map[gt_labels == i] = CLASS_COLORS_RGB[class_name]
        
        vis_images.append(gt_color_map)
        vis_titles.append('Ground Truth')
        
        # 创建可视化
        fig, axes = plt.subplots(1, len(vis_images), figsize=(3*len(vis_images), 3))
        
        for ax, img, title in zip(axes, vis_images, vis_titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'sample_predictions_{img_idx+1}.png', 
                   dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {num_samples} sample prediction visualizations")


def main():
    """主函数：评估所有模型并生成可视化"""
    print("🚀 Sample Size Experiment - Evaluation and Visualization (V2)")
    print("=" * 60)
    
    # 检查推理结果
    inference_results_dir = EXPERIMENT_OUTPUT_DIR / "inference_results"
    if not inference_results_dir.exists():
        print(f"❌ Error: Inference results not found at {inference_results_dir}")
        print("Please run inference_all_v2.py first.")
        return
    
    # 真实标签目录
    gt_dir = EXPERIMENT_DATA_DIR / "fixed_test_set" / "images"
    
    # 创建输出目录
    eval_output_dir = EXPERIMENT_OUTPUT_DIR / "evaluation_results"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 评估所有模型
    all_results = {}
    
    # 1. 评估预训练模型
    print("\n1. Evaluating pretrained model...")
    pretrained_dir = inference_results_dir / "pretrained"
    if pretrained_dir.exists():
        all_results['pretrained'] = evaluate_model_predictions(pretrained_dir, gt_dir)
    
    # 2. 评估微调模型
    print("\n2. Evaluating finetuned models...")
    for sample_size in SAMPLE_SIZES:
        model_name = f'finetuned_{sample_size}'
        model_dir = inference_results_dir / model_name
        
        if model_dir.exists():
            print(f"\nEvaluating {model_name}...")
            all_results[model_name] = evaluate_model_predictions(model_dir, gt_dir)
        else:
            print(f"⚠️ Results not found for {model_name}, skipping...")
    
    # 保存详细结果
    with open(eval_output_dir / 'detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 3. 生成可视化
    print("\n3. Generating visualizations...")
    
    # 性能曲线
    plot_performance_curves(all_results, eval_output_dir)
    
    # 类别级别对比
    plot_class_wise_comparison(all_results, eval_output_dir)
    
    # 汇总表格
    summary_df = create_summary_table(all_results, eval_output_dir)
    
    # 样本预测可视化
    visualize_sample_predictions(all_results, eval_output_dir, num_samples=20)
    
    # 4. 生成最终报告
    report = {
        'experiment': 'Sample Size Impact Analysis (V2)',
        'timestamp': datetime.now().isoformat(),
        'test_set_size': len(list(gt_dir.glob("*.tif"))),
        'models_evaluated': list(all_results.keys()),
        'key_findings': {
            'best_performing_model': max(all_results.items(), 
                                       key=lambda x: x[1]['overall']['mean_iou'])[0],
            'pretrained_baseline_iou': all_results.get('pretrained', {}).get('overall', {}).get('mean_iou', 0),
            'improvement_over_baseline': {}
        }
    }
    
    # 计算相对于基准的改进
    baseline_iou = all_results.get('pretrained', {}).get('overall', {}).get('mean_iou', 0)
    for model_name, results in all_results.items():
        if model_name != 'pretrained':
            model_iou = results['overall']['mean_iou']
            improvement = ((model_iou - baseline_iou) / baseline_iou) * 100 if baseline_iou > 0 else 0
            report['key_findings']['improvement_over_baseline'][model_name] = f"{improvement:.1f}%"
    
    with open(eval_output_dir / 'experiment_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"Results saved to: {eval_output_dir}")
    print("\nKey files generated:")
    print("  - performance_curves.png: Main visualization showing performance vs sample size")
    print("  - class_wise_performance.png: Per-class performance breakdown")
    print("  - summary_table.csv: Numerical results table")
    print("  - sample_predictions_*.png: Example predictions")
    print("  - detailed_results.json: Complete evaluation data")
    
    print("\n🎉 Sample size experiment completed successfully! (V2)")


if __name__ == "__main__":
    main()