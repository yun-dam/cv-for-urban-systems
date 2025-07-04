"""
简单的配置文件 - 为Stanford UGVR CV项目统一管理参数
适度改进原始代码，保持简洁易用
"""

import os
from pathlib import Path

# ==============================================================================
# 项目路径配置
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CODE_DIR = PROJECT_ROOT / "code"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ==============================================================================
# 数据相关配置
# ==============================================================================
# Vaihingen数据集路径
VAIHINGEN_DIR = DATA_DIR / "Vaihingen"
VAIHINGEN_IMAGES_DIR = VAIHINGEN_DIR / "top_cropped_512"
VAIHINGEN_LABELS_DIR = VAIHINGEN_DIR / "ground_truth_cropped_512"
FINETUNE_DATA_DIR = VAIHINGEN_DIR / "finetune_data"

# 原始图片池路径（用于5-fold CV）
ORIGINAL_IMG_DIR = FINETUNE_DATA_DIR / "finetune_pool" / "images"
ORIGINAL_MASK_DIR = FINETUNE_DATA_DIR / "finetune_pool" / "masks"

# ==============================================================================
# 模型相关配置
# ==============================================================================
# CLIPSeg预训练模型
PRETRAINED_MODEL = "CIDAS/clipseg-rd64-refined"

# 城市分割类别
URBAN_CLASSES = [
    'impervious surface', 
    'building', 
    'low vegetation', 
    'tree', 
    'car', 
    'background'
]

# ==============================================================================
# 数据处理参数配置
# ==============================================================================
# 数据集参数
CROP_SIZE = 512
NUM_ORIGINAL_IMAGES = 20  # 原始图片数量（用于5-fold CV）
N_CV_FOLDS = 5           # 交叉验证fold数

# 数据增强参数
AUGMENTATION_FACTOR = 5   # 每张图片的增强倍数
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'rotation_range': (-45, 45),
    'brightness_limit': 0.2,
    'contrast_limit': 0.2,
    'gaussian_noise_var': (10.0, 50.0),
    'blur_limit': 3,
}

# ==============================================================================
# 训练参数配置
# ==============================================================================
# 基础训练参数（用于最终训练）
TRAIN_CONFIG = {
    'num_epochs': 50,
    'batch_size': 4,
    'learning_rate': 5e-5,
    'patience': 5,              # 早停耐心值
    'min_delta': 0.001,         # 早停最小改进阈值
    'dice_weight': 0.8,         # Dice loss权重
    'bce_weight': 0.2,          # BCE loss权重
    'gradient_clip_val': 1.0,   # 梯度裁剪值
    'lr_scheduler': 'cosine',   # 学习率调度器类型
    'warmup_epochs': 2,         # 热身epoch数
}

# ==============================================================================
# 超参数搜索空间配置
# ==============================================================================
HYPERPARAMETER_SEARCH_SPACE = {
    # 学习率搜索范围
    'learning_rate': {
        'type': 'loguniform',
        'low': 1e-6,
        'high': 1e-3,
    },
    
    # Dice loss权重搜索范围
    'dice_weight': {
        'type': 'uniform',
        'low': 0.6,
        'high': 0.9,
    },
    
    # 批次大小选项（针对笔记本电脑优化）
    'batch_size': {
        'type': 'categorical',
        'choices': [2, 4, 8],
    },
    
    # 学习率调度器类型
    'lr_scheduler': {
        'type': 'categorical',
        'choices': ['cosine', 'step', 'exponential'],
    },
    
    # 梯度裁剪值
    'gradient_clip_val': {
        'type': 'categorical',
        'choices': [0.5, 1.0, 2.0],
    },
}

# Optuna超参数搜索配置
OPTUNA_CONFIG = {
    'n_trials': 20,              # 搜索试验次数
    'n_jobs': 1,                 # 并行任务数（笔记本电脑建议1）
    'study_name': 'clipseg_cv_search',
    'direction': 'maximize',     # 优化方向（最大化IoU）
    'sampler': 'TPE',           # 采样器类型
    'pruner': 'median',         # 剪枝策略
}

# ==============================================================================
# 系统参数配置
# ==============================================================================
# 系统参数
NUM_WORKERS = 0  # 数据加载器线程数（设为0便于调试）
RANDOM_SEED = 42
DEVICE = 'auto'  # 'auto', 'cuda', 'cpu', 'mps'

# 日志配置
LOG_CONFIG = {
    'log_level': 'INFO',
    'log_dir': OUTPUT_DIR / 'logs',
    'tensorboard': True,
    'wandb': False,  # 可选的wandb集成
}

# ==============================================================================
# 输出路径配置
# ==============================================================================
# 模型保存路径
FINETUNED_MODEL_DIR = MODELS_DIR / "clipseg_finetuned"
HYPERPARAMETER_SEARCH_DIR = MODELS_DIR / "hyperparameter_search"

# 结果输出路径
TRAINING_OUTPUT_DIR = OUTPUT_DIR / "training"
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluation"
COMPARISON_OUTPUT_DIR = OUTPUT_DIR / "model_comparison"

# ==============================================================================
# 实用函数
# ==============================================================================
def ensure_dirs():
    """确保所有必要的目录存在"""
    dirs_to_create = [
        DATA_DIR, MODELS_DIR, OUTPUT_DIR,
        FINETUNED_MODEL_DIR, HYPERPARAMETER_SEARCH_DIR,
        TRAINING_OUTPUT_DIR, EVALUATION_OUTPUT_DIR, COMPARISON_OUTPUT_DIR
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ 已确保所有目录存在")

def get_device():
    """获取可用的设备"""
    import torch
    
    if DEVICE == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✅ 使用Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("⚠️ 使用CPU进行训练")
    else:
        device = torch.device(DEVICE)
        print(f"✅ 使用指定设备: {DEVICE}")
    
    return device

def set_seed(seed=RANDOM_SEED):
    """设置随机种子以确保可重现性"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"✅ 设置随机种子: {seed}")

def get_augmentation_transform():
    """获取数据增强变换，基于配置文件"""
    import albumentations as A
    
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
    
    if 'gaussian_noise_var' in AUGMENTATION_CONFIG:
        var_limit = AUGMENTATION_CONFIG['gaussian_noise_var']
        transforms.append(A.GaussNoise(var_limit=var_limit, p=0.3))
    
    if 'blur_limit' in AUGMENTATION_CONFIG:
        transforms.append(A.GaussianBlur(blur_limit=AUGMENTATION_CONFIG['blur_limit'], p=0.3))
    
    return A.Compose(transforms)

def get_hyperparameter_from_trial(trial, param_name):
    """从Optuna trial中获取超参数"""
    param_config = HYPERPARAMETER_SEARCH_SPACE[param_name]
    
    if param_config['type'] == 'loguniform':
        return trial.suggest_loguniform(param_name, param_config['low'], param_config['high'])
    elif param_config['type'] == 'uniform':
        return trial.suggest_uniform(param_name, param_config['low'], param_config['high'])
    elif param_config['type'] == 'categorical':
        return trial.suggest_categorical(param_name, param_config['choices'])
    else:
        raise ValueError(f"Unknown parameter type: {param_config['type']}")

def save_config(config_dict, filename):
    """保存配置到JSON文件"""
    import json
    from pathlib import Path
    
    # 确保输出目录存在
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换Path对象为字符串
    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        return obj
    
    config_to_save = convert_paths(config_dict)
    
    with open(output_path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    
    print(f"✅ 配置已保存到: {output_path}")

if __name__ == "__main__":
    # 运行配置测试
    print("🔧 配置文件测试")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型目录: {MODELS_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    print("\n📊 数据处理配置:")
    print(f"原始图片数量: {NUM_ORIGINAL_IMAGES}")
    print(f"交叉验证折数: {N_CV_FOLDS}")
    print(f"数据增强倍数: {AUGMENTATION_FACTOR}")
    
    print("\n🔍 超参数搜索配置:")
    print(f"搜索试验次数: {OPTUNA_CONFIG['n_trials']}")
    print(f"搜索参数: {list(HYPERPARAMETER_SEARCH_SPACE.keys())}")
    
    ensure_dirs()
    print("\n✅ 配置文件测试完成")