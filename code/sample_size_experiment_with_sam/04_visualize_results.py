#!/usr/bin/env python3
"""
生成SAM优化前后的可视化对比图
为不同样本量的模型创建视觉对比

功能：
1. 为每个样本量选择代表性图像
2. 创建4面板对比图：原图、GT、CLIPSeg、CLIPSeg+SAM
3. 生成整体对比矩阵
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tifffile
import random
from tqdm import tqdm
from typing import List

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))

from config import *

# 实验配置
EXPERIMENT_DATA_DIR = DATA_DIR / "Vaihingen" / "sample_size_experiment_v2"
EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "sample_size_experiment_sam"
SAMPLE_SIZES = [5, 10, 20, 40, 80, 160, 320]

# 路径配置
TEST_IMAGES_DIR = EXPERIMENT_DATA_DIR / "fixed_test_set" / "images"
GROUND_TRUTH_DIR = EXPERIMENT_DATA_DIR / "fixed_test_set" / "labels"


def create_single_comparison(img_name: str, model_name: str, output_path: Path):
    """为单个图像创建4面板对比图"""
    
    # 加载原始图像
    img_path = TEST_IMAGES_DIR / f"{img_name}.tif"
    if not img_path.exists():
        return False
    original = np.array(Image.open(img_path).convert("RGB"))
    
    # 加载ground truth
    gt_path = GROUND_TRUTH_DIR / f"{img_name}.tif"
    if gt_path.exists():
        gt_img = tifffile.imread(gt_path)
        # 创建building mask (蓝色 = [0, 0, 255])
        gt_mask = np.all(gt_img == np.array([0, 0, 255]), axis=-1)
    else:
        # 尝试从masks目录加载
        gt_mask_path = EXPERIMENT_DATA_DIR / "fixed_test_set" / "masks" / f"{img_name}_building.png"
        if gt_mask_path.exists():
            gt_mask = np.array(Image.open(gt_mask_path).convert('L')) > 0
        else:
            return False
    
    # 加载CLIPSeg预测
    clipseg_path = EXPERIMENT_OUTPUT_DIR / "inference" / model_name / "masks" / f"{img_name}_building.tif"
    if clipseg_path.exists():
        clipseg_mask = tifffile.imread(clipseg_path) > 0
    else:
        return False
    
    # 加载SAM优化结果
    sam_path = EXPERIMENT_OUTPUT_DIR / "sam_refined" / model_name / f"{img_name}_building_refined.tif"
    if sam_path.exists():
        sam_mask = tifffile.imread(sam_path) > 0
    else:
        return False
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原始图像
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # Ground Truth
    axes[0, 1].imshow(gt_mask, cmap='Blues')
    axes[0, 1].set_title('Ground Truth', fontsize=14)
    axes[0, 1].axis('off')
    
    # CLIPSeg预测
    axes[1, 0].imshow(clipseg_mask, cmap='Blues')
    axes[1, 0].set_title(f'CLIPSeg ({model_name})', fontsize=14)
    axes[1, 0].axis('off')
    
    # SAM优化结果
    axes[1, 1].imshow(sam_mask, cmap='Blues')
    axes[1, 1].set_title(f'CLIPSeg + SAM ({model_name})', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.suptitle(f'Building Segmentation Comparison - {img_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def create_sample_size_matrix(selected_images: List[str], output_path: Path):
    """创建展示所有样本量效果的大型对比矩阵"""
    
    n_images = len(selected_images)
    n_models = len(SAMPLE_SIZES) + 1  # +1 for pretrained
    
    # 创建图形
    fig = plt.figure(figsize=(4*n_images, 3*n_models))
    gs = gridspec.GridSpec(n_models, n_images, hspace=0.3, wspace=0.2)
    
    model_names = ['pretrained'] + [f'model_{s}' for s in SAMPLE_SIZES]
    
    for row, model_name in enumerate(model_names):
        for col, img_name in enumerate(selected_images):
            ax = fig.add_subplot(gs[row, col])
            
            # 加载SAM优化结果
            sam_path = EXPERIMENT_OUTPUT_DIR / "sam_refined" / model_name / f"{img_name}_building_refined.tif"
            if sam_path.exists():
                sam_mask = tifffile.imread(sam_path) > 0
                ax.imshow(sam_mask, cmap='Blues')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_facecolor('lightgray')
            
            # 设置标题
            if row == 0:
                ax.set_title(f'{img_name}', fontsize=10)
            if col == 0:
                if model_name == 'pretrained':
                    label = 'Pretrained'
                else:
                    sample_size = model_name.split('_')[1]
                    label = f'{sample_size} samples'
                ax.set_ylabel(label, fontsize=12, rotation=0, ha='right', va='center')
            
            ax.axis('off')
    
    plt.suptitle('SAM-Enhanced Building Segmentation Across Different Training Sample Sizes', 
                 fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def create_improvement_visualization(img_name: str, output_path: Path):
    """创建展示SAM改进效果的可视化"""
    
    models_to_show = ['model_5', 'model_20', 'model_80', 'model_320']
    
    fig, axes = plt.subplots(2, len(models_to_show), figsize=(16, 8))
    
    for col, model_name in enumerate(models_to_show):
        # CLIPSeg结果
        clipseg_path = EXPERIMENT_OUTPUT_DIR / "inference" / model_name / "masks" / f"{img_name}_building.tif"
        if clipseg_path.exists():
            clipseg_mask = tifffile.imread(clipseg_path) > 0
            axes[0, col].imshow(clipseg_mask, cmap='Blues')
        axes[0, col].set_title(f'CLIPSeg\n({model_name.split("_")[1]} samples)', fontsize=12)
        axes[0, col].axis('off')
        
        # SAM优化结果
        sam_path = EXPERIMENT_OUTPUT_DIR / "sam_refined" / model_name / f"{img_name}_building_refined.tif"
        if sam_path.exists():
            sam_mask = tifffile.imread(sam_path) > 0
            axes[1, col].imshow(sam_mask, cmap='Blues')
        axes[1, col].set_title(f'+ SAM', fontsize=12)
        axes[1, col].axis('off')
    
    # 添加行标签
    axes[0, 0].text(-0.15, 0.5, 'Original\nCLIPSeg', transform=axes[0, 0].transAxes,
                    fontsize=14, ha='right', va='center', weight='bold')
    axes[1, 0].text(-0.15, 0.5, 'SAM\nRefined', transform=axes[1, 0].transAxes,
                    fontsize=14, ha='right', va='center', weight='bold')
    
    plt.suptitle(f'SAM Refinement Impact Across Sample Sizes - {img_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    """主函数"""
    print("SAM-Enhanced Sample Size Experiment - Step 4: Visualization")
    print("="*80)
    
    # 创建输出目录
    vis_dir = EXPERIMENT_OUTPUT_DIR / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取测试图像列表
    test_images = sorted([p.stem for p in TEST_IMAGES_DIR.glob("*.tif")])
    
    # 随机选择代表性图像
    random.seed(42)
    n_samples = min(5, len(test_images))
    selected_images = random.sample(test_images, n_samples)
    
    print(f"Selected {n_samples} images for visualization")
    
    # 1. 为每个模型创建单独的对比图
    print("\n1. Creating individual comparison plots...")
    comparison_dir = vis_dir / "individual_comparisons"
    comparison_dir.mkdir(exist_ok=True)
    
    for model_name in ['pretrained'] + [f'model_{s}' for s in SAMPLE_SIZES]:
        print(f"  Processing {model_name}...")
        model_dir = comparison_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        for img_name in selected_images[:3]:  # 每个模型3张图
            output_path = model_dir / f"{img_name}_comparison.png"
            create_single_comparison(img_name, model_name, output_path)
    
    # 2. 创建样本量对比矩阵
    print("\n2. Creating sample size comparison matrix...")
    matrix_path = vis_dir / "sample_size_matrix.png"
    create_sample_size_matrix(selected_images[:4], matrix_path)
    
    # 3. 创建改进效果展示
    print("\n3. Creating improvement visualization...")
    for i, img_name in enumerate(selected_images[:2]):
        improvement_path = vis_dir / f"improvement_showcase_{i+1}.png"
        create_improvement_visualization(img_name, improvement_path)
    
    # 4. 创建最佳案例展示
    print("\n4. Creating best case examples...")
    best_cases_dir = vis_dir / "best_cases"
    best_cases_dir.mkdir(exist_ok=True)
    
    # 选择小样本模型的最佳改进案例
    for model_name in ['model_5', 'model_10', 'model_20']:
        best_path = best_cases_dir / f"{model_name}_best_improvement.png"
        if selected_images:
            create_single_comparison(selected_images[0], model_name, best_path)
    
    print("\n" + "="*80)
    print("✓ Visualization completed!")
    print(f"All visualizations saved to: {vis_dir}")
    print("\nGenerated visualizations:")
    print("  - Individual comparisons for each model")
    print("  - Sample size comparison matrix")
    print("  - Improvement showcase plots")
    print("  - Best case examples for small sample models")


if __name__ == "__main__":
    main()