#!/usr/bin/env python3
"""
对所有训练好的模型在固定测试集上进行推理，并保存概率图
基于 sample_size_experiment/inference_all_v2.py 修改

主要修改：
1. 保存概率图（.npy格式）用于SAM优化
2. 同时保存二值掩码以保持兼容性
"""

import os
import sys
from pathlib import Path
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
warnings.filterwarnings("ignore")

# 实验配置
EXPERIMENT_DATA_DIR = DATA_DIR / "Vaihingen" / "sample_size_experiment_v2"
EXPERIMENT_MODEL_DIR = MODELS_DIR / "sample_size_experiment_v2"
EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "sample_size_experiment_sam"
SAMPLE_SIZES = [5, 10, 20, 40, 80, 160, 320]

# 固定测试集路径
FIXED_TEST_DIR = EXPERIMENT_DATA_DIR / "fixed_test_set"


def run_inference_on_image_with_probs(model, processor, image_path, classes, device):
    """
    对单张图像运行推理，返回二值掩码和概率图
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
    
    # 转换为numpy数组
    pred_masks = {}
    pred_probs = {}
    
    for i, class_name in enumerate(classes):
        prob_map = resized_masks[i].numpy()
        binary_mask = (prob_map > 0.5).astype(np.uint8)
        
        pred_probs[class_name] = prob_map
        pred_masks[class_name] = binary_mask
    
    return pred_masks, pred_probs


def run_inference_for_model(model_name, model_path, test_images, device):
    """
    为单个模型运行推理
    """
    print(f"\n{'='*50}")
    print(f"Running inference for {model_name}")
    print(f"Model path: {model_path}")
    
    # 创建输出目录
    output_dir = EXPERIMENT_OUTPUT_DIR / "inference" / model_name
    masks_dir = output_dir / "masks"
    probs_dir = output_dir / "probabilities"
    
    masks_dir.mkdir(parents=True, exist_ok=True)
    probs_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    if model_path.exists():
        print(f"Loading fine-tuned model from {model_path}")
        model, processor, _ = load_model(model_path, device)
    else:
        print(f"Using pretrained model")
        processor = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL)
        model = CLIPSegForImageSegmentation.from_pretrained(PRETRAINED_MODEL)
        model.to(device)
        model.eval()
    
    # 推理统计
    inference_stats = {
        "model": model_name,
        "model_path": str(model_path),
        "num_images": len(test_images),
        "timestamp": datetime.now().isoformat(),
        "classes": URBAN_CLASSES
    }
    
    # 对每张图像进行推理
    for img_path in tqdm(test_images, desc=f"Inference for {model_name}"):
        img_name = img_path.stem
        
        # 运行推理
        pred_masks, pred_probs = run_inference_on_image_with_probs(
            model, processor, img_path, URBAN_CLASSES, device
        )
        
        # 保存结果
        for class_name in URBAN_CLASSES:
            safe_class_name = class_name.replace(' ', '_')
            
            # 保存二值掩码
            mask_path = masks_dir / f"{img_name}_{safe_class_name}.tif"
            tifffile.imwrite(str(mask_path), pred_masks[class_name])
            
            # 保存概率图
            prob_path = probs_dir / f"{img_name}_{safe_class_name}_prob.npy"
            np.save(prob_path, pred_probs[class_name])
    
    # 保存推理统计
    stats_path = output_dir / "inference_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(inference_stats, f, indent=2)
    
    print(f"Inference completed for {model_name}")
    print(f"Results saved to: {output_dir}")


def main():
    """主函数：对所有模型运行推理"""
    print("SAM-Enhanced Sample Size Experiment - Step 1: Inference with Probabilities")
    print(f"Fixed test set: {FIXED_TEST_DIR}")
    
    # 创建主输出目录
    EXPERIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")
    
    # 加载测试图像列表
    test_images = sorted(list((FIXED_TEST_DIR / "images").glob("*.tif")))
    print(f"Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("ERROR: No test images found!")
        return
    
    # 1. 首先运行预训练模型作为基准
    print("\n1. Running pretrained model as baseline")
    run_inference_for_model(
        "pretrained",
        Path("dummy_path"),  # 不会使用
        test_images,
        device
    )
    
    # 2. 对每个样本大小的模型运行推理
    for sample_size in SAMPLE_SIZES:
        model_name = f"model_{sample_size}"
        model_path = EXPERIMENT_MODEL_DIR / model_name / "best_model"
        
        if model_path.exists():
            run_inference_for_model(
                model_name,
                model_path,
                test_images,
                device
            )
        else:
            print(f"\nWARNING: Model not found for sample size {sample_size}")
            print(f"Expected path: {model_path}")
    
    # 保存实验元数据
    metadata = {
        "experiment": "sam_enhanced_sample_size",
        "step": "inference_with_probabilities",
        "timestamp": datetime.now().isoformat(),
        "sample_sizes": SAMPLE_SIZES,
        "test_set_size": len(test_images),
        "classes": URBAN_CLASSES,
        "output_dir": str(EXPERIMENT_OUTPUT_DIR)
    }
    
    metadata_path = EXPERIMENT_OUTPUT_DIR / "experiment_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*50)
    print("✓ Inference with probabilities completed!")
    print(f"All results saved to: {EXPERIMENT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()