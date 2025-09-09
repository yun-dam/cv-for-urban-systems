#!/usr/bin/env python3
"""
评估SAM优化对不同样本量模型的影响
比较CLIPSeg原始输出与SAM优化后的性能

功能：
1. 对每个模型计算CLIPSeg和CLIPSeg+SAM的指标
2. 计算SAM带来的性能提升
3. 分析训练样本量与SAM提升的关系
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))

from config import *

# 实验配置
EXPERIMENT_DATA_DIR = DATA_DIR / "Vaihingen" / "sample_size_experiment_v2"
EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "sample_size_experiment_sam"
SAMPLE_SIZES = [5, 10, 20, 40, 80, 160, 320]

# 测试集路径
FIXED_TEST_DIR = EXPERIMENT_DATA_DIR / "fixed_test_set"
GROUND_TRUTH_DIR = FIXED_TEST_DIR / "masks"


def calculate_metrics(pred_mask, true_mask):
    """计算评估指标"""
    pred_binary = pred_mask.astype(bool)
    true_binary = true_mask.astype(bool)
    
    # True Positives, False Positives, False Negatives
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


def evaluate_model(model_name: str, use_sam: bool = False) -> Dict:
    """评估单个模型的性能"""
    print(f"\nEvaluating {model_name} {'with SAM' if use_sam else 'without SAM'}...")
    
    # 确定预测结果路径
    if use_sam:
        pred_dir = EXPERIMENT_OUTPUT_DIR / "sam_refined" / model_name
    else:
        pred_dir = EXPERIMENT_OUTPUT_DIR / "inference" / model_name / "masks"
    
    if not pred_dir.exists():
        print(f"  WARNING: Predictions not found at {pred_dir}")
        return None
    
    # 获取测试图像列表
    test_images = sorted(GROUND_TRUTH_DIR.glob("*_building.png"))
    
    # 存储每张图像的指标
    all_metrics = []
    
    for gt_path in tqdm(test_images, desc=f"Evaluating {model_name}"):
        # 提取图像名称
        img_name = gt_path.stem.replace("_building", "")
        
        # 加载ground truth
        gt_mask = np.array(Image.open(gt_path).convert('L')) > 0
        
        # 加载预测结果
        if use_sam:
            pred_path = pred_dir / f"{img_name}_building_refined.tif"
        else:
            pred_path = pred_dir / f"{img_name}_building.tif"
        
        if pred_path.exists():
            pred_mask = tifffile.imread(pred_path) > 0
            
            # 计算指标
            metrics = calculate_metrics(pred_mask, gt_mask)
            all_metrics.append(metrics)
    
    if not all_metrics:
        return None
    
    # 计算平均指标
    avg_metrics = {}
    for metric_name in ['iou', 'precision', 'recall', 'f1', 'pixel_accuracy']:
        values = [m[metric_name] for m in all_metrics]
        avg_metrics[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return avg_metrics


def analyze_sam_impact():
    """分析SAM对不同样本量模型的影响"""
    print("Analyzing SAM impact across different sample sizes...")
    
    results = {
        'pretrained': {},
        'models': {}
    }
    
    # 评估预训练模型
    print("\n1. Evaluating pretrained model")
    results['pretrained']['without_sam'] = evaluate_model('pretrained', use_sam=False)
    results['pretrained']['with_sam'] = evaluate_model('pretrained', use_sam=True)
    
    # 评估各个样本量的模型
    print("\n2. Evaluating fine-tuned models")
    for sample_size in SAMPLE_SIZES:
        model_name = f"model_{sample_size}"
        results['models'][sample_size] = {
            'without_sam': evaluate_model(model_name, use_sam=False),
            'with_sam': evaluate_model(model_name, use_sam=True)
        }
    
    # 计算改进幅度
    print("\n3. Calculating improvements")
    improvements = {}
    
    # 预训练模型的改进
    if results['pretrained']['without_sam'] and results['pretrained']['with_sam']:
        improvements['pretrained'] = {}
        for metric in ['iou', 'precision', 'recall', 'f1', 'pixel_accuracy']:
            base = results['pretrained']['without_sam'][metric]['mean']
            enhanced = results['pretrained']['with_sam'][metric]['mean']
            improvements['pretrained'][metric] = {
                'absolute': enhanced - base,
                'relative': ((enhanced - base) / base * 100) if base > 0 else 0
            }
    
    # 微调模型的改进
    improvements['models'] = {}
    for sample_size in SAMPLE_SIZES:
        if (results['models'][sample_size]['without_sam'] and 
            results['models'][sample_size]['with_sam']):
            improvements['models'][sample_size] = {}
            for metric in ['iou', 'precision', 'recall', 'f1', 'pixel_accuracy']:
                base = results['models'][sample_size]['without_sam'][metric]['mean']
                enhanced = results['models'][sample_size]['with_sam'][metric]['mean']
                improvements['models'][sample_size][metric] = {
                    'absolute': enhanced - base,
                    'relative': ((enhanced - base) / base * 100) if base > 0 else 0
                }
    
    return results, improvements


def save_results_table(results: Dict, improvements: Dict):
    """保存详细结果表格"""
    print("\n4. Creating results tables")
    
    # 创建综合结果DataFrame
    data = []
    
    # 添加预训练模型
    if 'pretrained' in results:
        row = {'Model': 'Pretrained', 'Sample_Size': 0}
        if results['pretrained']['without_sam']:
            for metric in ['iou', 'precision', 'recall', 'f1']:
                row[f'{metric}_base'] = results['pretrained']['without_sam'][metric]['mean']
        if results['pretrained']['with_sam']:
            for metric in ['iou', 'precision', 'recall', 'f1']:
                row[f'{metric}_sam'] = results['pretrained']['with_sam'][metric]['mean']
        if 'pretrained' in improvements:
            for metric in ['iou', 'precision', 'recall', 'f1']:
                row[f'{metric}_improve'] = improvements['pretrained'][metric]['absolute']
                row[f'{metric}_improve_pct'] = improvements['pretrained'][metric]['relative']
        data.append(row)
    
    # 添加微调模型
    for sample_size in SAMPLE_SIZES:
        row = {'Model': f'Model_{sample_size}', 'Sample_Size': sample_size}
        if results['models'][sample_size]['without_sam']:
            for metric in ['iou', 'precision', 'recall', 'f1']:
                row[f'{metric}_base'] = results['models'][sample_size]['without_sam'][metric]['mean']
        if results['models'][sample_size]['with_sam']:
            for metric in ['iou', 'precision', 'recall', 'f1']:
                row[f'{metric}_sam'] = results['models'][sample_size]['with_sam'][metric]['mean']
        if sample_size in improvements['models']:
            for metric in ['iou', 'precision', 'recall', 'f1']:
                row[f'{metric}_improve'] = improvements['models'][sample_size][metric]['absolute']
                row[f'{metric}_improve_pct'] = improvements['models'][sample_size][metric]['relative']
        data.append(row)
    
    # 创建并保存DataFrame
    df = pd.DataFrame(data)
    output_path = EXPERIMENT_OUTPUT_DIR / "sam_impact_analysis.csv"
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"  Results table saved to: {output_path}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("SAM Impact Summary (IoU)")
    print("="*80)
    print(f"{'Model':<15} {'Base IoU':<10} {'SAM IoU':<10} {'Improvement':<12} {'Relative %':<10}")
    print("-"*80)
    
    for _, row in df.iterrows():
        if 'iou_base' in row and 'iou_sam' in row:
            print(f"{row['Model']:<15} {row['iou_base']:<10.4f} {row['iou_sam']:<10.4f} "
                  f"{row.get('iou_improve', 0):<12.4f} {row.get('iou_improve_pct', 0):<10.2f}%")
    
    return df


def create_improvement_plot(df: pd.DataFrame):
    """创建改进幅度可视化图"""
    print("\n5. Creating visualization plots")
    
    # 筛选微调模型数据
    finetuned_df = df[df['Sample_Size'] > 0].copy()
    
    if len(finetuned_df) == 0:
        print("  No fine-tuned model data available for plotting")
        return
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 图1：IoU绝对值对比
    x = finetuned_df['Sample_Size']
    ax1.plot(x, finetuned_df['iou_base'], 'o-', label='CLIPSeg', markersize=8, linewidth=2)
    ax1.plot(x, finetuned_df['iou_sam'], 's-', label='CLIPSeg + SAM', markersize=8, linewidth=2)
    ax1.set_xlabel('Training Sample Size')
    ax1.set_ylabel('IoU')
    ax1.set_title('IoU Performance: CLIPSeg vs CLIPSeg+SAM')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_xticks(SAMPLE_SIZES)
    ax1.set_xticklabels(SAMPLE_SIZES)
    
    # 图2：相对改进百分比
    ax2.bar(x, finetuned_df['iou_improve_pct'], width=0.4, alpha=0.7)
    ax2.set_xlabel('Training Sample Size')
    ax2.set_ylabel('Relative Improvement (%)')
    ax2.set_title('SAM Improvement Percentage by Sample Size')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xscale('log')
    ax2.set_xticks(SAMPLE_SIZES)
    ax2.set_xticklabels(SAMPLE_SIZES)
    
    # 添加数值标签
    for i, (size, pct) in enumerate(zip(x, finetuned_df['iou_improve_pct'])):
        ax2.text(size, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plot_path = EXPERIMENT_OUTPUT_DIR / "sam_improvement_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved to: {plot_path}")
    plt.close()


def main():
    """主函数"""
    print("SAM-Enhanced Sample Size Experiment - Step 3: Evaluation")
    print("="*80)
    
    # 创建输出目录
    EXPERIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 运行分析
    results, improvements = analyze_sam_impact()
    
    # 保存详细结果
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'sam_enhanced_sample_size',
        'results': results,
        'improvements': improvements
    }
    
    results_path = EXPERIMENT_OUTPUT_DIR / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    # 创建结果表格和图表
    df = save_results_table(results, improvements)
    create_improvement_plot(df)
    
    print("\n" + "="*80)
    print("✓ Evaluation completed!")
    print(f"All results saved to: {EXPERIMENT_OUTPUT_DIR}")


if __name__ == "__main__":
    # 需要先导入PIL来加载ground truth
    from PIL import Image
    main()