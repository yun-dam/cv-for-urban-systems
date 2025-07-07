import os
import sys
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from pathlib import Path
from PIL import Image
import albumentations as A

# ==============================================================================
# 脚本核心配置
# ==============================================================================
# 控制是否对交叉验证中的验证集进行数据增强
# 警告：为了保证评估的无偏性，强烈建议保持此值为 False。
# 验证集应该反映模型在真实、未修改数据上的表现。
AUGMENT_VALIDATION_SET = False

# ==============================================================================
# 环境设置与导入
# ==============================================================================
# 将项目根目录添加到Python路径，以便导入config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *

# ==============================================================================
# 从 config.py 加载配置
# ==============================================================================
# 输入目录
FINETUNE_POOL_IMG_DIR = FINETUNE_DATA_DIR / "finetune_pool" / "images"
FINETUNE_POOL_MASK_DIR = FINETUNE_DATA_DIR / "finetune_pool" / "masks"
# POOL_IMG_DIR = ORIGINAL_IMG_DIR
# POOL_MASK_DIR = ORIGINAL_MASK_DIR

# 输出目录
BASE_OUTPUT_DIR = FINETUNE_DATA_DIR / "cv_prepared_data"
CV_FOLDS_DIR = BASE_OUTPUT_DIR / "cv_folds"
ALL_AUGMENTED_DIR = BASE_OUTPUT_DIR / "all_data_for_final_train"
FINAL_TRAIN_DIR = ALL_AUGMENTED_DIR / "train"
FINAL_VAL_DIR = ALL_AUGMENTED_DIR / "val"

# 参数
N_AUG_PER_IMAGE = AUGMENTATION_CONFIG['num_augmentations_per_image']
N_SPLITS = DATASET_CONFIG['n_cv_folds']
RANDOM_SEED = DATASET_CONFIG['random_seed']
FINAL_VAL_SPLIT = DATASET_CONFIG['final_val_split']
CLASSES = [cls.replace(' ', '_').replace('/', '_') for cls in URBAN_CLASSES]

# ==============================================================================
# 数据增强管道 (Albumentations)
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
        value=0,
        mask_value=0,
        p=0.7
    ),
    # 如果图像尺寸 >= 512，应用随机裁剪
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

# ==============================================================================
# 核心处理函数
# ==============================================================================
def process_image_set(file_list: list, input_img_dir: Path, input_mask_dir: Path,
                      output_img_dir: Path, output_mask_dir: Path,
                      should_augment: bool, desc: str):
    """
    处理一个图片集合。根据 should_augment 标志决定是进行数据增强还是仅复制原始文件。

    Args:
        file_list (list): 要处理的文件名列表。
        input_img_dir (Path): 原始图片输入目录。
        input_mask_dir (Path): 原始掩码输入目录。
        output_img_dir (Path): 处理后图片的输出目录。
        output_mask_dir (Path): 处理后掩码的输出目录。
        should_augment (bool): 是否执行数据增强。
        desc (str): tqdm进度条的描述文字。
    """
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    for img_name in tqdm(file_list, desc=desc):
        img_path = input_img_dir / img_name
        img_stem = img_path.stem

        # --- 处理图片 ---
        image_np = np.array(Image.open(img_path).convert("RGB"))
        h, w = image_np.shape[:2]

        # --- 加载所有类别的掩码 ---
        masks_np = []
        for cls in CLASSES:
            mask_path = input_mask_dir / f"{img_stem}_{cls}.png"
            if mask_path.exists():
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                masks_np.append(mask)
            else:
                # 如果掩码不存在，创建一个空的
                masks_np.append(np.zeros((h, w), dtype=np.uint8))

        # --- 保存原始版本（总是执行） ---
        Image.fromarray(image_np).save(output_img_dir / f"{img_stem}_orig.tif")
        for i, cls in enumerate(CLASSES):
            Image.fromarray(masks_np[i]).save(output_mask_dir / f"{img_stem}_orig_{cls}.png")

        # --- 如果需要，执行并保存增强版本 ---
        if should_augment:
            # N_AUG_PER_IMAGE 包含原始图像，所以我们生成 N-1 个增强版本
            for aug_idx in range(N_AUG_PER_IMAGE - 1):
                # 应用增强
                transformed = transform(
                    image=image_np,
                    **{f"mask{i}": masks_np[i] for i in range(len(CLASSES))}
                )
                
                aug_image = transformed["image"]
                aug_masks = [transformed[f"mask{i}"] for i in range(len(CLASSES))]

                # 保存增强后的图片和掩码
                Image.fromarray(aug_image).save(output_img_dir / f"{img_stem}_aug{aug_idx:02d}.tif")
                for i, cls in enumerate(CLASSES):
                    Image.fromarray(aug_masks[i]).save(output_mask_dir / f"{img_stem}_aug{aug_idx:02d}_{cls}.png")

# ==============================================================================
# 主逻辑
# ==============================================================================
def main():
    """
    执行完整的数据准备流程：
    1. 为交叉验证创建数据折。
    2. 为最终训练创建全量增强数据集。
    """
    print("🚀 开始统一数据准备流程...")
    print(f"ℹ️  增强验证集: {'是' if AUGMENT_VALIDATION_SET else '否'}")

    # 设置随机种子以保证可复现性
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 清理旧的输出目录，确保从头开始
    if BASE_OUTPUT_DIR.exists():
        print(f"🧹 清理旧目录: {BASE_OUTPUT_DIR}")
        shutil.rmtree(BASE_OUTPUT_DIR)

    # 加载原始图片文件列表
    image_files = np.array(sorted([f.name for f in FINETUNE_POOL_IMG_DIR.glob('*.tif')]))
    if len(image_files) == 0:
        raise FileNotFoundError(f"在 {FINETUNE_POOL_IMG_DIR} 中未找到任何 .tif 图片文件。")
    print(f"🏞️ 找到 {len(image_files)} 张原始图片用于处理。")

    # --- 任务 1: 创建 K-Fold 交叉验证数据集 ---
    print("\n--- 任务 1: 创建 K-Fold CV 数据集 ---")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(image_files)):
        fold_num = fold_idx + 1
        print(f"\n===== 正在处理 Fold {fold_num}/{N_SPLITS} =====")

        train_files = image_files[train_indices]
        val_files = image_files[val_indices]

        # 定义该折的输出路径
        fold_dir = CV_FOLDS_DIR / f"fold_{fold_num}"
        train_output_img_dir = fold_dir / "train/images"
        train_output_mask_dir = fold_dir / "train/masks"
        val_output_img_dir = fold_dir / "val/images"
        val_output_mask_dir = fold_dir / "val/masks"

        # 处理训练集（总是增强）
        process_image_set(train_files, FINETUNE_POOL_IMG_DIR, FINETUNE_POOL_MASK_DIR,
                          train_output_img_dir, train_output_mask_dir,
                          should_augment=True, desc=f"Fold {fold_num} [训练集]")

        # 处理验证集（根据标志决定是否增强）
        process_image_set(val_files, FINETUNE_POOL_IMG_DIR, FINETUNE_POOL_MASK_DIR,
                          val_output_img_dir, val_output_mask_dir,
                          should_augment=AUGMENT_VALIDATION_SET, desc=f"Fold {fold_num} [验证集]")

    # --- 任务 2: 创建用于最终训练的数据集（训练/验证分离） ---
    print("\n--- 任务 2: 创建用于最终训练的数据集 ---")
    
    # 在原始图像级别分割训练集和验证集
    from sklearn.model_selection import train_test_split
    final_train_files, final_val_files = train_test_split(
        image_files, test_size=FINAL_VAL_SPLIT, random_state=RANDOM_SEED
    )
    
    print(f"最终训练数据分割：")
    print(f"  - 训练集: {len(final_train_files)} 张原始图片 (将增强到 {len(final_train_files) * N_AUG_PER_IMAGE} 张)")
    print(f"  - 验证集: {len(final_val_files)} 张原始图片 (保持原始)")
    
    # 处理最终训练集（增强）
    final_train_img_dir = FINAL_TRAIN_DIR / 'images'
    final_train_mask_dir = FINAL_TRAIN_DIR / 'masks'
    process_image_set(final_train_files, FINETUNE_POOL_IMG_DIR, FINETUNE_POOL_MASK_DIR,
                      final_train_img_dir, final_train_mask_dir,
                      should_augment=True, desc="最终训练集增强")
    
    # 处理最终验证集（不增强，只复制原始）
    final_val_img_dir = FINAL_VAL_DIR / 'images'
    final_val_mask_dir = FINAL_VAL_DIR / 'masks'
    process_image_set(final_val_files, FINETUNE_POOL_IMG_DIR, FINETUNE_POOL_MASK_DIR,
                      final_val_img_dir, final_val_mask_dir,
                      should_augment=False, desc="最终验证集（不增强）")

    print("\n✅ 所有数据准备完成!")
    print(f"📁 K-Fold CV 数据保存在: {CV_FOLDS_DIR}")
    print(f"📁 最终训练数据保存在: {ALL_AUGMENTED_DIR}")
    print(f"    - 训练集: {FINAL_TRAIN_DIR}")
    print(f"    - 验证集: {FINAL_VAL_DIR}")

if __name__ == "__main__":
    main()