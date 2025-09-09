#!/usr/bin/env python3
"""
准备样本量实验的数据 - 修订版
与原始pipeline完全一致的数据格式

功能：
1. 从459张裁剪后的图像中随机分离出139张作为固定测试集
2. 剩余320张作为训练池
3. 从训练池中为每个样本大小独立随机采样：5, 10, 20, 40, 80, 160, 320张
4. 使用与prepare_ft_data.py完全一致的mask生成逻辑
"""

import sys
import shutil
import random
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import tifffile
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

# 实验配置
EXPERIMENT_DATA_DIR = DATA_DIR / "Vaihingen" / "sample_size_experiment_v2"
FIXED_TEST_SIZE = 139  # 固定测试集大小
SAMPLE_SIZES = [5, 10, 20, 40, 80, 160, 320]  # 训练集大小

# 源数据路径
SOURCE_IMG_DIR = DATA_DIR / "Vaihingen" / "top_cropped_512"
SOURCE_MASK_DIR = DATA_DIR / "Vaihingen" / "ground_truth_cropped_512"


def prepare_directories():
    """创建实验所需的目录结构"""
    # 创建主实验目录
    EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 创建固定测试集目录
    test_dir = EXPERIMENT_DATA_DIR / "fixed_test_set"
    (test_dir / "images").mkdir(parents=True, exist_ok=True)
    (test_dir / "masks").mkdir(parents=True, exist_ok=True)  # 扁平结构，不创建子目录
    
    # 创建训练池目录
    training_pools_dir = EXPERIMENT_DATA_DIR / "training_pools"
    training_pools_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个样本大小创建目录
    for size in SAMPLE_SIZES:
        pool_dir = training_pools_dir / f"pool_{size}"
        (pool_dir / "images").mkdir(parents=True, exist_ok=True)
        (pool_dir / "masks").mkdir(parents=True, exist_ok=True)  # 扁平结构
    
    print("✓ Created directory structure")
    return test_dir, training_pools_dir


def get_all_image_names():
    """获取所有可用的图像名称（不含扩展名）"""
    all_images = list(SOURCE_IMG_DIR.glob("*.tif"))
    image_names = [img.stem for img in all_images]
    
    # 验证每个图像都有对应的ground truth
    valid_names = []
    for name in image_names:
        if (SOURCE_MASK_DIR / f"{name}.tif").exists():
            valid_names.append(name)
        else:
            print(f"⚠ Warning: No ground truth found for {name}")
    
    print(f"✓ Found {len(valid_names)} valid image-mask pairs")
    return valid_names


def split_test_and_training_pool(image_names, random_seed=42):
    """
    随机将图像分为固定测试集和训练池
    
    Args:
        image_names: 所有图像名称列表
        random_seed: 随机种子
    
    Returns:
        test_names: 测试集图像名称
        pool_names: 训练池图像名称
    """
    # 设置随机种子确保可重复性
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 随机打乱
    shuffled_names = image_names.copy()
    random.shuffle(shuffled_names)
    
    # 分割
    test_names = shuffled_names[:FIXED_TEST_SIZE]
    pool_names = shuffled_names[FIXED_TEST_SIZE:]
    
    print(f"✓ Split data: {len(test_names)} test, {len(pool_names)} training pool")
    
    return test_names, pool_names


def create_independent_training_sets(pool_names, random_seed=42):
    """
    为每个样本大小独立随机采样（非嵌套）
    
    Args:
        pool_names: 训练池中的所有图像名称
        random_seed: 随机种子
    
    Returns:
        training_sets: 字典，键为样本大小，值为图像名称列表
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 验证最大样本量不超过可用图像数
    max_size = max(SAMPLE_SIZES)
    if max_size > len(pool_names):
        raise ValueError(f"Maximum sample size {max_size} exceeds available pool size {len(pool_names)}")
    
    # 为每个大小独立随机采样
    training_sets = {}
    for size in SAMPLE_SIZES:
        # 每个集合都是独立从训练池中随机采样
        sampled = random.sample(pool_names, size)
        training_sets[size] = sampled
    
    print("✓ Created independent training sets")
    return training_sets


def copy_data_to_directory(image_names, source_img_dir, source_mask_dir, target_dir):
    """
    将图像和mask复制到目标目录
    使用与prepare_ft_data.py完全一致的逻辑
    
    Args:
        image_names: 要复制的图像名称列表
        source_img_dir: 源图像目录
        source_mask_dir: 源mask目录
        target_dir: 目标目录
    """
    for name in tqdm(image_names, desc=f"Processing {target_dir.name}"):
        # 1. 复制图像文件（与prepare_ft_data.py第68行一致）
        src_img = source_img_dir / f"{name}.tif"
        dst_img = target_dir / "images" / f"{name}.tif"
        shutil.copy(str(src_img), str(dst_img))
        
        # 2. 创建并保存二值masks（与prepare_ft_data.py第70-82行完全一致）
        label_path = source_mask_dir / f"{name}.tif"
        label_img = tifffile.imread(str(label_path))
        
        base_name = name  # 已经是stem了
        for class_name, color in CLASS_COLORS_RGB.items():
            # 创建二值mask
            mask = np.all(label_img == np.array(color), axis=-1)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            
            # 替换类名中的空格和斜杠
            safe_class_name = class_name.replace(" ", "_").replace("/", "_")
            mask_filename = f"{base_name}_{safe_class_name}.png"
            
            # 保存到扁平的masks目录
            mask_path = target_dir / "masks" / mask_filename
            mask_img.save(str(mask_path))


def save_split_info(test_names, training_sets, experiment_dir):
    """
    保存数据划分信息
    
    Args:
        test_names: 测试集图像名称
        training_sets: 训练集字典
        experiment_dir: 实验目录
    """
    split_info = {
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'total_images': len(test_names) + len(training_sets[max(SAMPLE_SIZES)]),
        'test_set': {
            'size': len(test_names),
            'images': sorted(test_names)
        },
        'training_sets': {}
    }
    
    # 添加每个训练集的信息
    for size, names in training_sets.items():
        split_info['training_sets'][f'pool_{size}'] = {
            'size': len(names),
            'images': sorted(names)
        }
    
    # 保存到JSON文件
    info_path = experiment_dir / 'data_split_info.json'
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✓ Saved split information to {info_path}")


def main():
    """主函数：执行数据准备流程"""
    print("=" * 60)
    print("Sample Size Experiment - Data Preparation V2")
    print("Using random split and format identical to original pipeline")
    print("=" * 60)
    
    # 1. 创建目录结构
    print("\n1. Creating directory structure...")
    test_dir, training_pools_dir = prepare_directories()
    
    # 2. 获取所有图像名称
    print("\n2. Loading image names...")
    all_image_names = get_all_image_names()
    
    # 3. 随机分割测试集和训练池
    print("\n3. Randomly splitting test set and training pool...")
    test_names, pool_names = split_test_and_training_pool(all_image_names, RANDOM_SEED)
    
    # 4. 创建独立的训练集（非嵌套）
    print("\n4. Creating independent training sets...")
    training_sets = create_independent_training_sets(pool_names, RANDOM_SEED)
    
    # 5. 复制测试集数据
    print("\n5. Copying test set data...")
    copy_data_to_directory(test_names, SOURCE_IMG_DIR, SOURCE_MASK_DIR, test_dir)
    print("   ✓ Test set ready")
    
    # 6. 复制训练集数据
    print("\n6. Copying training set data...")
    for size in SAMPLE_SIZES:
        print(f"\n   Processing pool_{size} ({len(training_sets[size])} images)...")
        pool_dir = training_pools_dir / f"pool_{size}"
        copy_data_to_directory(training_sets[size], SOURCE_IMG_DIR, SOURCE_MASK_DIR, pool_dir)
    print("   ✓ All training pools ready")
    
    # 7. 保存数据划分信息
    print("\n7. Saving split information...")
    save_split_info(test_names, training_sets, EXPERIMENT_DATA_DIR)
    
    # 8. 打印总结
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"Test set: {len(test_names)} images (randomly selected)")
    print("Training sets (independently sampled):")
    for size in SAMPLE_SIZES:
        print(f"  - pool_{size}: {len(training_sets[size])} images")
    print(f"\nData saved to: {EXPERIMENT_DATA_DIR}")
    print("\nKey improvements in V2:")
    print("  ✓ Random split instead of sequential")
    print("  ✓ Independent sampling for each training size")
    print("  ✓ PNG format with 0/255 values")
    print("  ✓ Flat directory structure for masks")
    print("  ✓ Identical to original pipeline format")


if __name__ == "__main__":
    main()