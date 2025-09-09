#!/usr/bin/env python3
"""
批量应用SAM优化到所有模型的输出
使用SAM v5的混合策略（高置信度核心保护 + 边界优化）

功能：
1. 对每个模型（pretrained + 7个不同样本量）的输出进行SAM优化
2. 只处理building类别（可扩展到其他类别）
3. 保存优化后的掩码
"""

import os
import sys
import numpy as np
from pathlib import Path
import torch
from PIL import Image
import cv2
from tqdm import tqdm
import json
from datetime import datetime
from scipy.ndimage import distance_transform_edt, binary_erosion
import tifffile
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "code"))

from config import *

# Import SAM
from segment_anything import sam_model_registry, SamPredictor

# 实验配置
EXPERIMENT_OUTPUT_DIR = OUTPUT_DIR / "sample_size_experiment_sam"
SAMPLE_SIZES = [5, 10, 20, 40, 80, 160, 320]
SAM_CHECKPOINT = "models/sam_vit_h_4b8939.pth"
TEST_IMAGES_DIR = DATA_DIR / "Vaihingen" / "sample_size_experiment_v2" / "fixed_test_set" / "images"

# SAM优化配置
REFINEMENT_CONFIG = {
    'building': {
        'prob_threshold': 0.5,      # 初始阈值
        'core_threshold': 0.8,      # 高置信度核心
        'boundary_width': 10,       # 边界宽度（像素）
        'min_area': 50,            # 最小面积
        'min_area_ratio': 0.9      # 最小面积保留比例
    },
}


class SAMRefinementBatch:
    """批量SAM优化处理器，使用v5混合策略"""
    
    def __init__(self, sam_checkpoint: str, model_type: str = "vit_h", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        print(f"Loading SAM {model_type} model...")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
    
    def split_regions(self, prob_map: np.ndarray, config: dict):
        """将概率图分割为核心、边界和背景区域"""
        binary_mask = (prob_map > config['prob_threshold']).astype(np.uint8)
        dist_transform = distance_transform_edt(binary_mask)
        
        core_region = (prob_map > config['core_threshold']) & (dist_transform > config['boundary_width'])
        boundary_region = (
            ((prob_map > config['prob_threshold']) & (prob_map <= config['core_threshold'])) |
            ((prob_map > config['prob_threshold']) & (dist_transform <= config['boundary_width']))
        )
        background_region = prob_map <= config['prob_threshold']
        
        return core_region, boundary_region, background_region, binary_mask
    
    def get_adaptive_points(self, core_region, boundary_region, background_region, contour):
        """生成自适应点提示"""
        points = []
        labels = []
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # 核心区域密集正样本
        core_pixels = np.argwhere(core_region[y:y+h, x:x+w])
        if len(core_pixels) > 0:
            num_core_points = min(5, len(core_pixels))
            core_indices = np.random.choice(len(core_pixels), num_core_points, replace=False)
            for idx in core_indices:
                py, px = core_pixels[idx]
                points.append((x + px, y + py))
                labels.append(1)
        
        # 边界区域稀疏正样本
        boundary_pixels = np.argwhere(boundary_region[y:y+h, x:x+w])
        if len(boundary_pixels) > 0:
            num_boundary_points = min(3, len(boundary_pixels))
            boundary_indices = np.random.choice(len(boundary_pixels), num_boundary_points, replace=False)
            for idx in boundary_indices:
                py, px = boundary_pixels[idx]
                points.append((x + px, y + py))
                labels.append(1)
        
        # 背景负样本
        margin = 20
        y_start = max(0, y - margin)
        y_end = min(background_region.shape[0], y + h + margin)
        x_start = max(0, x - margin)
        x_end = min(background_region.shape[1], x + w + margin)
        
        bg_pixels = np.argwhere(background_region[y_start:y_end, x_start:x_end])
        if len(bg_pixels) > 0:
            num_neg_points = min(5, len(bg_pixels))
            neg_indices = np.random.choice(len(bg_pixels), num_neg_points, replace=False)
            for idx in neg_indices:
                py, px = bg_pixels[idx]
                points.append((x_start + px, y_start + py))
                labels.append(0)
        
        return points, labels
    
    def intelligent_fusion(self, core_region, sam_mask, clipseg_mask):
        """智能融合CLIPSeg和SAM结果"""
        core_area = np.sum(core_region)
        sam_area = np.sum(sam_mask)
        
        if sam_area == 0:
            return clipseg_mask
        
        core_preserved = np.sum(sam_mask & core_region) / max(core_area, 1)
        
        if core_preserved > 0.95:
            return sam_mask
        elif core_preserved > 0.7:
            combined_mask = core_region.astype(np.uint8)
            sam_extension = sam_mask & (~core_region)
            combined_mask = np.maximum(combined_mask, sam_extension)
            return combined_mask
        else:
            clipseg_eroded = binary_erosion(clipseg_mask, iterations=3)
            sam_boundary = sam_mask & (~clipseg_eroded)
            refined_mask = np.maximum(clipseg_mask, sam_boundary)
            return refined_mask
    
    def refine_class_hybrid(self, prob_map: np.ndarray, config: dict) -> np.ndarray:
        """混合优化单个类别的掩码"""
        # 分割区域
        core_region, boundary_region, background_region, binary_mask = self.split_regions(prob_map, config)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > config['min_area']]
        
        if not filtered_contours:
            return np.zeros_like(prob_map, dtype=np.uint8)
        
        # 处理每个实例
        final_mask = np.zeros_like(prob_map, dtype=np.uint8)
        
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            instance_mask = np.zeros_like(binary_mask)
            cv2.drawContours(instance_mask, [contour], -1, 1, -1)
            
            instance_core = core_region & instance_mask.astype(bool)
            
            # 生成自适应点
            points, labels = self.get_adaptive_points(
                instance_core, boundary_region, background_region, contour
            )
            
            box = np.array([x, y, x + w, y + h])
            
            if len(points) > 0:
                point_coords = np.array(points)
                point_labels = np.array(labels)
            else:
                point_coords = None
                point_labels = None
            
            try:
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box[None, :],
                    multimask_output=True
                )
                
                best_idx = np.argmax(scores)
                sam_mask = masks[best_idx].astype(np.uint8)
                
                refined_instance = self.intelligent_fusion(
                    instance_core, sam_mask, instance_mask
                )
                
                final_mask = np.maximum(final_mask, refined_instance)
                
            except Exception as e:
                print(f"  SAM failed for an instance: {e}")
                final_mask = np.maximum(final_mask, instance_mask)
        
        # 后处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        return final_mask
    
    def process_model_outputs(self, model_name: str):
        """处理单个模型的所有输出"""
        print(f"\nProcessing {model_name}...")
        
        # 路径设置
        inference_dir = EXPERIMENT_OUTPUT_DIR / "inference" / model_name
        probs_dir = inference_dir / "probabilities"
        output_dir = EXPERIMENT_OUTPUT_DIR / "sam_refined" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not probs_dir.exists():
            print(f"  WARNING: No probability maps found for {model_name}")
            return
        
        # 获取测试图像列表
        test_images = sorted(TEST_IMAGES_DIR.glob("*.tif"))
        
        # 统计信息
        stats = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "num_images": len(test_images),
            "refined_classes": list(REFINEMENT_CONFIG.keys())
        }
        
        # 处理每张图像
        for img_path in tqdm(test_images, desc=f"SAM refinement for {model_name}"):
            img_name = img_path.stem
            
            # 设置SAM图像
            image = np.array(Image.open(img_path).convert("RGB"))
            self.predictor.set_image(image)
            
            # 处理每个要优化的类别
            for class_name, config in REFINEMENT_CONFIG.items():
                safe_class_name = class_name.replace(' ', '_')
                prob_path = probs_dir / f"{img_name}_{safe_class_name}_prob.npy"
                
                if prob_path.exists():
                    # 加载概率图
                    prob_map = np.load(prob_path)
                    
                    # 应用SAM优化
                    refined_mask = self.refine_class_hybrid(prob_map, config)
                    
                    # 保存优化后的掩码
                    output_path = output_dir / f"{img_name}_{safe_class_name}_refined.tif"
                    tifffile.imwrite(str(output_path), refined_mask)
        
        # 保存统计信息
        stats_path = output_dir / "sam_refinement_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  Completed SAM refinement for {model_name}")


def main():
    """主函数：批量处理所有模型"""
    print("SAM-Enhanced Sample Size Experiment - Step 2: SAM Refinement")
    print(f"SAM checkpoint: {SAM_CHECKPOINT}")
    print(f"Processing classes: {list(REFINEMENT_CONFIG.keys())}")
    
    # 初始化SAM
    refiner = SAMRefinementBatch(
        sam_checkpoint=SAM_CHECKPOINT,
        model_type="vit_h"
    )
    
    # 处理预训练模型
    print("\n1. Processing pretrained model")
    refiner.process_model_outputs("pretrained")
    
    # 处理各个样本量的模型
    print("\n2. Processing fine-tuned models")
    for sample_size in SAMPLE_SIZES:
        model_name = f"model_{sample_size}"
        refiner.process_model_outputs(model_name)
    
    # 保存批处理元数据
    metadata = {
        "experiment": "sam_enhanced_sample_size",
        "step": "sam_refinement",
        "timestamp": datetime.now().isoformat(),
        "sam_model": "vit_h",
        "refinement_config": REFINEMENT_CONFIG,
        "models_processed": ["pretrained"] + [f"model_{s}" for s in SAMPLE_SIZES]
    }
    
    metadata_path = EXPERIMENT_OUTPUT_DIR / "sam_refinement_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*50)
    print("✓ SAM refinement completed for all models!")
    print(f"Results saved to: {EXPERIMENT_OUTPUT_DIR / 'sam_refined'}")


if __name__ == "__main__":
    main()