"""
配置文件 - Stanford UGVR CV项目
V2版：将超参数搜索和最终训练的配置分离，结构更清晰。
"""

import os
import json
from pathlib import Path
import torch
import random
import numpy as np
import albumentations as A

# ==============================================================================
# 1. 核心路径配置 (Core Path Configuration)
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

ORIGINAL_DIR = DATA_DIR / "Vaihingen"
# CROPPED_IMAGES_DIR = ORIGINAL_DIR / "top_cropped_512"
# CROPPED_LABELS_DIR = ORIGINAL_DIR / "ground_truth_cropped_512"


FINETUNE_DATA_DIR = ORIGINAL_DIR / "finetune_data"
# FINETUNE_POOL_IMG_DIR = FINETUNE_DATA_DIR / "finetune_pool" / "images"
# FINETUNE_POOL_MASK_DIR = FINETUNE_DATA_DIR / "finetune_pool" / "masks"

FINETUNED_MODEL_DIR = MODELS_DIR / "clipseg_finetuned"
HYPERPARAMETER_SEARCH_DIR = MODELS_DIR / "hyperparameter_search"

# ==============================================================================
# 2. 数据集与模型通用配置 (General Data & Model Configuration)
# ==============================================================================
# CLIPSeg预训练模型
PRETRAINED_MODEL = "CIDAS/clipseg-rd64-refined"

# 城市分割类别
URBAN_CLASSES = [
    'impervious surface', 'building', 'low vegetation', 'tree', 'car', 'background'
]


CROP_SIZE = 512  # 图像裁剪尺寸

# 添加类别颜色定义（BGR格式）
CLASS_COLORS_BGR = {
    'impervious surface': [255, 255, 255],
    'building': [0, 0, 255],
    'low vegetation': [0, 255, 255],
    'tree': [0, 255, 0],
    'car': [255, 255, 0],
    'background': [255, 0, 0]
}

# For visualization purposes (RGB format for Matplotlib)
SEGMENTATION_COLORS_RGB = {
    "impervious surface": (255, 255, 255),  # White
    "building":           (0, 0, 255),      # Blue
    "low vegetation":     (0, 255, 255),    # Cyan
    "tree":               (0, 255, 0),      # Green
    "car":                (255, 255, 0),    # Yellow
    "background":         (255, 0, 0)       # Red
}

# 数据集参数
DATASET_CONFIG = {
    'finetune_pool_size': 4,  # LAPTOP: 4 | SERVER: 20 (原始图片数量)
    'n_cv_folds': 2,          # LAPTOP: 2 | SERVER: 5 (交叉验证fold数)
    'random_seed': 42         # 随机种子
}

# 数据增强参数
AUGMENTATION_CONFIG = {
    'num_augmentations_per_image': 2,  # LAPTOP: 3 | SERVER: 5
    # 'horizontal_flip': True,
    # 'vertical_flip': True,
    # 'rotation_range': (-30, 30),
    # 'brightness_limit': 0.1,
    # 'contrast_limit': 0.1,
}

# ==============================================================================
# 3. 超参数搜索配置 (Hyperparameter Search Configuration)
# ==============================================================================
# 用于 run_hyperparameter_search.py
SEARCH_CONFIG = {
    # Optuna 框架设置
    'n_trials': 2,               # LAPTOP: 2 | SERVER: 20 (总共尝试多少组超参数)
    'study_name': 'clipseg_cv_search_laptop',
    'direction': 'minimize',     # 优化方向：最小化验证集损失

    # 每个Trial的训练设置
    'num_epochs': 2,             # LAPTOP: 2 | SERVER: 10 (每次试验训练的轮数)
    'patience': 3,            # LAPTOP: 3 | SERVER: 5 (早停策略)

    # 超参数搜索空间
    'space': {
        'learning_rate': {'type': 'loguniform', 'low': 1e-6, 'high': 1e-4},
        'dice_weight':   {'type': 'uniform', 'low': 0.5, 'high': 0.9},
        'batch_size':    {'type': 'categorical', 'choices': [2, 4]}, # 笔记本建议使用小batch
    }
}

# ==============================================================================
# 4. 最终模型训练配置 (Final Model Training Configuration)
# ==============================================================================
# 用于 train_final_model.py
FINAL_TRAIN_CONFIG = {
    'num_epochs': 5,             # LAPTOP: 5 | SERVER: 50 (最终模型总共训练的轮数)

    # 注意: 下面的 learning_rate, batch_size, dice_weight
    # 将会被超参数搜索找到的最佳值覆盖。
    # 这里的值仅用于调试或在没有搜索结果时作为备用。
    'default_learning_rate': 5e-5,
    'default_batch_size': 4,
    'default_dice_weight': 0.8,
}

# ==============================================================================
# 5. 系统与评估配置 (System & Evaluation Configuration)
# ==============================================================================
RANDOM_SEED = DATASET_CONFIG['random_seed']
DEVICE = 'cpu'  # 'auto', 'cuda', 'cpu', 'mps'
NUM_WORKERS = 0 # 设为0便于调试

EVALUATION_CONFIG = {
    'default_model_path': FINETUNED_MODEL_DIR / "best_model",
    'default_test_data': FINETUNE_DATA_DIR / "test" / "images",
    'default_output_dir': OUTPUT_DIR / "evaluation",
    'batch_size': 2,
    'num_visualization_samples': 5,
    'device': 'cpu',  # 'auto', 'cuda', 'cpu', 'mps'
    'figure_size': (16, 16),  # 4面板可视化的图像尺寸
    'visualization_dpi': 150,  # 可视化图像的DPI
}

# ==============================================================================
# 6. 辅助函数 (Utility Functions)
# ==============================================================================
def get_augmentation_transform():
    """获取数据增强变换，基于配置文件"""
    transforms = []
    if AUGMENTATION_CONFIG.get('horizontal_flip'):
        transforms.append(A.HorizontalFlip(p=0.5))
    if AUGMENTATION_CONFIG.get('vertical_flip'):
        transforms.append(A.VerticalFlip(p=0.5))
    if 'rotation_range' in AUGMENTATION_CONFIG:
        min_angle, max_angle = AUGMENTATION_CONFIG['rotation_range']
        transforms.append(A.Rotate(limit=(min_angle, max_angle), p=0.5))
    if 'brightness_limit' in AUGMENTATION_CONFIG:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=AUGMENTATION_CONFIG['brightness_limit'],
            contrast_limit=AUGMENTATION_CONFIG.get('contrast_limit', 0.2),
            p=0.5
        ))
    return A.Compose(transforms)

def get_hyperparameter_from_trial(trial, param_name):
    """从Optuna trial中获取超参数"""
    param_config = SEARCH_CONFIG['space'][param_name]
    if param_config['type'] == 'loguniform':
        return trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
    elif param_config['type'] == 'uniform':
        return trial.suggest_float(param_name, param_config['low'], param_config['high'])
    elif param_config['type'] == 'categorical':
        return trial.suggest_categorical(param_name, param_config['choices'])
    raise ValueError(f"Unknown parameter type: {param_config['type']}")

def set_seed(seed=RANDOM_SEED):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    """确保所有必要的目录存在"""
    dirs_to_create = [
        DATA_DIR, MODELS_DIR, OUTPUT_DIR,
        FINETUNED_MODEL_DIR, HYPERPARAMETER_SEARCH_DIR
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_device():
    """获取可用的设备"""
    if DEVICE == 'auto':
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(DEVICE)

def save_config(config_dict, filename):
    """保存配置到JSON文件"""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

if __name__ == "__main__":
    print("✅ config.py: 配置文件加载成功。")
    print(f"  - 搜索试验次数: {SEARCH_CONFIG['n_trials']}")
    print(f"  - 最终训练轮数: {FINAL_TRAIN_CONFIG['num_epochs']}")
