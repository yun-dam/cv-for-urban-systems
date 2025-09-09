#!/usr/bin/env python3
"""
对所有训练好的模型在固定测试集上进行推理 - 修订版
与原始pipeline的inference.py保持格式一致

功能：
1. 加载每个样本大小训练的模型
2. 在相同的测试集上进行推理
3. 保存每个模型的预测结果（TIF格式，0/1值）
4. 同时运行预训练模型作为基准对比
"""

import os
import sys
from pathlib import Path
from glob import glob
import numpy as np
from PIL import Image
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import tifffile
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))

from config import *
from utils import get_device, load_model

# Disable warnings
import warnings
warnings.filterwarnings("ignore", message="The following named arguments are not valid")
warnings.filterwarnings("ignore", message="Using a slow image processor")

# 实验配置 - 使用v2路径
EXPERIMENT_DATA_DIR = DATA_DIR / "Vaihingen" / "sample_size_experiment_v2"
EXPERIMENT_MODEL_DIR = MODELS_DIR / "sample_size_experiment_v2"
EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "sample_size_experiment_v2"
SAMPLE_SIZES = [5, 10, 20, 40, 80, 160, 320]

# 固定测试集路径
FIXED_TEST_DIR = EXPERIMENT_DATA_DIR / "fixed_test_set"


def run_inference_on_image(model, processor, image_path, classes, device):
    """
    对单张图像运行推理
    与原始pipeline的inference.py中的run_inference函数保持一致
    
    Args:
        model: CLIPSeg模型
        processor: CLIPSeg处理器
        image_path: 图像路径
        classes: 类别列表
        device: 计算设备
        
    Returns:
        pred_masks: 字典，类别名到二值mask的映射（0或1）
        pred_probs: 字典，类别名到概率图的映射（0-1）
    """
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 准备输入
    images = [image] * len(classes)
    inputs = processor(images=images, text=classes, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 处理预测结果
    masks = outputs.logits.sigmoid().cpu()
    resized_masks = torch.nn.functional.interpolate(
        masks.unsqueeze(0), 
        size=image.size[::-1],  # (H, W)
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    # 转换为字典格式
    pred_masks = {}
    pred_probs = {}
    
    for i, class_name in enumerate(classes):
        prob_map = resized_masks[i].numpy()
        binary_mask = (prob_map > 0.5).astype(np.uint8)  # 保持0/1值
        
        pred_probs[class_name] = prob_map
        pred_masks[class_name] = binary_mask
    
    return pred_masks, pred_probs


def save_results(image_name, pred_masks, pred_probs, output_dir, save_probs=True):
    """
    保存推理结果
    与原始pipeline的inference.py中的save_results函数保持一致
    
    Args:
        image_name: 图像名称
        pred_masks: 二值mask字典（0/1值）
        pred_probs: 概率图字典
        output_dir: 输出目录
        save_probs: 是否保存概率图
    """
    # 创建子目录
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    if save_probs:
        probs_dir = output_dir / "probabilities"
        probs_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每个类别的mask（与原始inference.py第107-110行完全一致）
    for class_name, mask in pred_masks.items():
        safe_class_name = class_name.replace(' ', '_')
        mask_path = masks_dir / f"{image_name}_{safe_class_name}.tif"
        tifffile.imwrite(str(mask_path), mask)  # TIF格式，0/1值
    
    # 保存概率图（如果需要）
    if save_probs:
        for class_name, prob_map in pred_probs.items():
            safe_class_name = class_name.replace(' ', '_')
            prob_path = probs_dir / f"{image_name}_{safe_class_name}_prob.npy"
            np.save(prob_path, prob_map)


def inference_for_model(model_path, model_name, test_images, output_base_dir):
    """
    对一个模型运行完整的推理
    
    Args:
        model_path: 模型路径
        model_name: 模型名称（用于输出目录）
        test_images: 测试图像列表
        output_base_dir: 输出基础目录
    """
    print(f"\n{'='*60}")
    print(f"Running inference for: {model_name}")
    print(f"{'='*60}")
    
    device = get_device()
    
    # 创建输出目录
    model_output_dir = output_base_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型和处理器
    if model_name == "pretrained":
        # 预训练模型
        processor = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL)
        model = CLIPSegForImageSegmentation.from_pretrained(PRETRAINED_MODEL)
        model.to(device)
        model.eval()
        print(f"Loaded pretrained model: {PRETRAINED_MODEL}")
    else:
        # 微调模型
        if not model_path.exists():
            print(f"❌ Error: Model not found at {model_path}")
            return False
        
        model, processor, metadata = load_model(model_path, device)
        print(f"Loaded fine-tuned model from: {model_path}")
        if metadata:
            print(f"  - Training sample size: {metadata.get('sample_size', 'N/A')}")
            print(f"  - Best epoch: {metadata.get('epoch', 'N/A')}")
            print(f"  - Best val loss: {metadata.get('val_loss', 'N/A'):.4f}")
    
    # 运行推理
    print(f"\nProcessing {len(test_images)} test images...")
    
    success_count = 0
    for img_path in tqdm(test_images, desc="Inference"):
        try:
            # 运行推理
            pred_masks, pred_probs = run_inference_on_image(
                model, processor, img_path, URBAN_CLASSES, device
            )
            
            # 保存结果
            save_results(
                img_path.stem, 
                pred_masks, 
                pred_probs, 
                model_output_dir,
                save_probs=True  # 保存概率图用于分析
            )
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue
    
    print(f"✓ Completed inference for {model_name}: {success_count}/{len(test_images)} images")
    
    # 保存推理元数据
    metadata = {
        'model_name': model_name,
        'model_path': str(model_path) if model_name != "pretrained" else PRETRAINED_MODEL,
        'test_images': len(test_images),
        'successful_images': success_count,
        'timestamp': datetime.now().isoformat(),
        'output_format': 'TIF (0/1 values)',
        'classes': URBAN_CLASSES
    }
    
    metadata_path = model_output_dir / 'inference_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return True


def main():
    """主函数：对所有模型运行推理"""
    print("=" * 60)
    print("Sample Size Experiment - Inference V2")
    print("Using TIF format with 0/1 values (matching original pipeline)")
    print("=" * 60)
    
    # 检查测试集是否存在
    if not FIXED_TEST_DIR.exists():
        print(f"❌ Error: Test set not found at {FIXED_TEST_DIR}")
        print("Please run prepare_experiment_data_v2.py first.")
        return
    
    # 获取测试图像
    test_images = sorted(list((FIXED_TEST_DIR / "images").glob("*.tif")))
    if not test_images:
        print(f"❌ Error: No test images found in {FIXED_TEST_DIR / 'images'}")
        return
    
    print(f"\nFound {len(test_images)} test images")
    
    # 创建输出目录
    inference_output_dir = EXPERIMENT_OUTPUT_DIR / "inference_results"
    inference_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录所有模型的信息
    all_models_info = {
        'experiment': 'sample_size_experiment_v2',
        'test_set_size': len(test_images),
        'models': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # 1. 首先运行预训练模型作为基准
    print("\n" + "-" * 60)
    print("Running baseline (pretrained) model...")
    if inference_for_model(Path("pretrained"), "pretrained", test_images, inference_output_dir):
        all_models_info['models']['pretrained'] = {
            'type': 'baseline',
            'status': 'completed'
        }
    
    # 2. 运行所有微调模型
    for sample_size in SAMPLE_SIZES:
        print("\n" + "-" * 60)
        model_dir = EXPERIMENT_MODEL_DIR / f"model_{sample_size}" / "best_model"
        model_name = f"finetuned_{sample_size}"
        
        if model_dir.exists():
            if inference_for_model(model_dir, model_name, test_images, inference_output_dir):
                all_models_info['models'][model_name] = {
                    'type': 'finetuned',
                    'sample_size': sample_size,
                    'status': 'completed'
                }
        else:
            print(f"⚠️ Warning: Model not found for sample size {sample_size}")
            all_models_info['models'][model_name] = {
                'type': 'finetuned',
                'sample_size': sample_size,
                'status': 'not_found'
            }
    
    # 保存总体推理信息
    summary_path = inference_output_dir / 'inference_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_models_info, f, indent=2)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
    print(f"Results saved to: {inference_output_dir}")
    print("\nCompleted models:")
    for model_name, info in all_models_info['models'].items():
        if info['status'] == 'completed':
            print(f"  ✓ {model_name}")
    
    print("\nKey improvements in V2:")
    print("  ✓ TIF format output (matching original pipeline)")
    print("  ✓ Binary masks with 0/1 values")
    print("  ✓ Consistent with original inference.py logic")
    print("\nNext step: Run evaluate_and_visualize_v2.py")


if __name__ == "__main__":
    main()