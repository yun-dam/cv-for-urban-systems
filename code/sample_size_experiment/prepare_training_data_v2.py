#!/usr/bin/env python3
"""
为每个训练池准备增强后的训练数据 - 修订版
与原始pipeline格式完全一致

功能：
1. 对每个训练池（5, 10, 20, 40, 80, 160, 320）进行80/20划分
2. 80%的图像进行数据增强（每张10个增强版本）作为训练集
3. 20%的图像保持原始作为验证集（不增强，避免数据泄露）
4. 保存划分信息，确保可重复性

使用与原项目相同的albumentations库进行数据增强
"""

import sys
import shutil
import random
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
import albumentations as A
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

# 实验配置
EXPERIMENT_DATA_DIR = DATA_DIR / "Vaihingen" / "sample_size_experiment_v2"
SAMPLE_SIZES = [5, 10, 20, 40, 80, 160, 320]
TRAIN_SPLIT = 0.8  # 80%用于训练（会被增强）
VAL_SPLIT = 0.2    # 20%用于验证（保持原始）
NUM_AUGMENTATIONS = AUGMENTATION_CONFIG['num_augmentations_per_image']  # 使用config中的配置
CLASSES = [cls.replace(' ', '_').replace('/', '_') for cls in URBAN_CLASSES]

# ==============================================================================
# 数据增强管道（使用与原项目相同的albumentations配置）
# ==============================================================================
# 1) 几何变换
geom_transforms = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.1,      # 最大位移: 10%
        scale_limit=0.2,      # 最大缩放: 20%
        rotate_limit=25,      # 最大旋转: +/- 25度
        border_mode=cv2.BORDER_CONSTANT,
        p=0.7
    ),
    # 随机裁剪（如果图像大小 >= 512）
    A.RandomCrop(height=512, width=512, p=0.5),
]

# 2) 光度变换
photo_transforms = [
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.3),
]

# 组合：先几何变换，后光度变换
transform = A.Compose(
    geom_transforms + photo_transforms,
    additional_targets={f"mask{i}": "mask" for i in range(len(CLASSES))}
)


def process_image_set(file_list, input_img_dir, input_mask_dir,
                      output_img_dir, output_mask_dir,
                      should_augment, desc):
    """
    处理一组图像。根据should_augment标志决定是否进行数据增强或仅复制原始文件。
    修改为读取扁平PNG结构的mask文件
    
    Args:
        file_list: 要处理的文件名列表
        input_img_dir: 输入图像目录
        input_mask_dir: 输入mask目录
        output_img_dir: 输出图像目录
        output_mask_dir: 输出mask目录
        should_augment: 是否进行数据增强
        desc: tqdm进度条的描述文本
    """
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    
    for img_name in tqdm(file_list, desc=desc):
        img_path = input_img_dir / img_name
        img_stem = img_path.stem
        
        # 读取图像
        image_np = np.array(Image.open(img_path).convert("RGB"))
        h, w = image_np.shape[:2]
        
        # 加载所有类别的masks
        masks_np = []
        for cls in CLASSES:
            # 修改：从扁平结构读取PNG文件（与prepare_ft_data.py生成的格式一致）
            mask_filename = f"{img_stem}_{cls}.png"  # 改为PNG
            mask_path = input_mask_dir / mask_filename  # 直接在mask目录下，不是子目录
            
            if mask_path.exists():
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                # mask已经是0/255格式，保持不变
                masks_np.append(mask)
            else:
                # 如果mask不存在，创建空的
                print(f"Warning: Mask not found: {mask_path}")
                masks_np.append(np.zeros((h, w), dtype=np.uint8))
        
        # 保存原始版本（总是执行）
        Image.fromarray(image_np).save(output_img_dir / f"{img_stem}_orig.tif")
        for i, cls in enumerate(CLASSES):
            # 输出也保持扁平结构（与原项目保持一致）
            Image.fromarray(masks_np[i]).save(output_mask_dir / f"{img_stem}_orig_{cls}.png")
        
        # 如果需要，执行并保存增强版本
        if should_augment:
            # NUM_AUGMENTATIONS包括原始图像，所以我们生成N-1个增强版本
            for aug_idx in range(NUM_AUGMENTATIONS - 1):
                # 应用增强
                transformed = transform(
                    image=image_np,
                    **{f"mask{i}": masks_np[i] for i in range(len(CLASSES))}
                )
                
                aug_image = transformed["image"]
                aug_masks = [transformed[f"mask{i}"] for i in range(len(CLASSES))]
                
                # 保存增强的图像和masks
                Image.fromarray(aug_image).save(output_img_dir / f"{img_stem}_aug{aug_idx:02d}.tif")
                for i, cls in enumerate(CLASSES):
                    # 输出保持扁平结构（与原项目保持一致）
                    Image.fromarray(aug_masks[i]).save(output_mask_dir / f"{img_stem}_aug{aug_idx:02d}_{cls}.png")


def split_train_val(image_names, train_ratio=0.8, random_seed=42):
    """
    将图像列表划分为训练集和验证集
    
    Args:
        image_names: 图像名称列表
        train_ratio: 训练集比例
        random_seed: 随机种子
    
    Returns:
        train_names: 训练集图像名称
        val_names: 验证集图像名称
    """
    # 使用sklearn的train_test_split确保随机性
    train_names, val_names = train_test_split(
        image_names, 
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    return train_names, val_names


def process_training_pool(pool_size):
    """
    处理一个特定大小的训练池
    
    Args:
        pool_size: 训练池大小（5, 10, 20, 40, 80, 160, 320）
    """
    print(f"\n{'='*60}")
    print(f"Processing pool_{pool_size}")
    print(f"{'='*60}")
    
    # 输入输出路径
    pool_dir = EXPERIMENT_DATA_DIR / "training_pools" / f"pool_{pool_size}"
    augmented_dir = EXPERIMENT_DATA_DIR / "augmented_training_data" / f"pool_{pool_size}_augmented"
    
    # 检查输入数据是否存在
    if not pool_dir.exists():
        print(f"❌ Error: Pool directory not found: {pool_dir}")
        print("Please run prepare_experiment_data_v2.py first.")
        return False
    
    # 获取所有图像文件
    image_files = sorted(list((pool_dir / "images").glob("*.tif")))
    if not image_files:
        print(f"❌ Error: No images found in {pool_dir / 'images'}")
        return False
    
    print(f"Found {len(image_files)} images in pool_{pool_size}")
    
    # 计算实际的训练/验证分割
    # 对于小数据集，确保至少有1张验证图像
    n_train = max(1, int(len(image_files) * TRAIN_SPLIT))
    n_val = len(image_files) - n_train
    
    if n_val < 1:
        # 如果数据太少，至少保留1张作为验证
        n_val = 1
        n_train = len(image_files) - 1
    
    print(f"Split: {n_train} training (will be augmented), {n_val} validation (original)")
    
    # 执行训练/验证分割
    image_names = [f.name for f in image_files]
    train_names, val_names = split_train_val(image_names, train_ratio=n_train/len(image_names))
    
    # 处理训练集（进行增强）
    train_output_dir = augmented_dir / "train"
    process_image_set(
        train_names,
        pool_dir / "images",
        pool_dir / "masks",
        train_output_dir / "images",
        train_output_dir / "masks",
        should_augment=True,
        desc="Processing training set (with augmentation)"
    )
    
    # 处理验证集（不增强）
    val_output_dir = augmented_dir / "val"
    process_image_set(
        val_names,
        pool_dir / "images",
        pool_dir / "masks",
        val_output_dir / "images",
        val_output_dir / "masks",
        should_augment=False,
        desc="Processing validation set (no augmentation)"
    )
    
    # 保存分割信息
    split_info = {
        'pool_size': pool_size,
        'total_images': len(image_files),
        'train_images': len(train_names),
        'val_images': len(val_names),
        'train_names': sorted(train_names),
        'val_names': sorted(val_names),
        'augmentations_per_image': NUM_AUGMENTATIONS,
        'timestamp': datetime.now().isoformat()
    }
    
    split_info_path = augmented_dir / 'train_val_split.json'
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✓ Saved split information to {split_info_path}")
    
    # 统计最终文件数
    final_train_images = len(list((train_output_dir / "images").glob("*.tif")))
    final_val_images = len(list((val_output_dir / "images").glob("*.tif")))
    
    print(f"\n✓ Pool_{pool_size} processing complete!")
    print(f"  - Training images: {final_train_images} (augmented from {len(train_names)})")
    print(f"  - Validation images: {final_val_images} (original)")
    
    return True


def main():
    """主函数：处理所有训练池"""
    print("=" * 60)
    print("Preparing Augmented Training Data V2")
    print("Reading from flat PNG structure")
    print("=" * 60)
    
    # 检查数据是否已准备好
    if not EXPERIMENT_DATA_DIR.exists():
        print(f"❌ Error: Experiment data directory not found: {EXPERIMENT_DATA_DIR}")
        print("Please run prepare_experiment_data_v2.py first.")
        return
    
    # 处理每个训练池
    success_count = 0
    for size in SAMPLE_SIZES:
        if process_training_pool(size):
            success_count += 1
    
    # 打印总结
    print("\n" + "=" * 60)
    print("Data Augmentation Complete!")
    print("=" * 60)
    print(f"Successfully processed: {success_count}/{len(SAMPLE_SIZES)} pools")
    print(f"\nAugmented data saved to: {EXPERIMENT_DATA_DIR / 'augmented_training_data'}")
    print("\nKey improvements in V2:")
    print("  ✓ Reads from flat PNG mask structure")
    print("  ✓ Maintains 0/255 value range")
    print("  ✓ Compatible with original pipeline format")


if __name__ == "__main__":
    main()