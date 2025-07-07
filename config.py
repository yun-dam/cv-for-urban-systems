"""
配置文件 - Stanford UGVR CV项目
V2版：将超参数搜索和最终训练的配置分离，结构更清晰。
"""

from pathlib import Path

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
    'finetune_pool_size': 10,  # LAPTOP: 4 | SERVER: 20 (原始图片数量)
    'n_cv_folds': 3,          # LAPTOP: 2 | SERVER: 5 (交叉验证fold数)
    'random_seed': 42,        # 随机种子
    'final_val_split': 0.2    # 最终训练时验证集比例（原始图像级别）
}

# 数据增强参数
AUGMENTATION_CONFIG = {
    'num_augmentations_per_image': 5,  # LAPTOP: 3 | SERVER: 5
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
    'n_trials': 10,               # LAPTOP: 2 | SERVER: 20 (总共尝试多少组超参数)
    'study_name': 'clipseg_cv_search_laptop',
    'direction': 'minimize',     # 优化方向：最小化验证集损失

    # 每个Trial的训练设置
    'num_epochs': 10,             # LAPTOP: 2 | SERVER: 10 (每次试验训练的轮数)
    'patience': 3,            # LAPTOP: 3 | SERVER: 5 (早停策略)

    # 超参数搜索空间
    'space': {
        'learning_rate': {'type': 'loguniform', 'low': 1e-6, 'high': 5e-4},
        'dice_weight':   {'type': 'uniform', 'low': 0.5, 'high': 0.9},
        'batch_size':    {'type': 'categorical', 'choices': [2, 4]}, # 笔记本建议使用小batch
    }
}

# ==============================================================================
# 4. 最终模型训练配置 (Final Model Training Configuration)
# ==============================================================================
# 用于 train_final_model.py
FINAL_TRAIN_CONFIG = {
    'num_epochs': 50,             # LAPTOP: 5 | SERVER: 50 (最终模型总共训练的轮数)
    'patience': 5,                # 早停patience (如果验证损失不下降的轮数)
    'min_delta': 1e-4,            # 最小改善阈值
    
    # 注意: 验证集分割已在 DATASET_CONFIG['final_val_split'] 中定义
    # 数据在 data_augmentation.py 中预先分割
    
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
DEVICE = 'cpu'  # 全局设备配置: 'auto', 'cuda', 'cpu', 'mps'
NUM_WORKERS = 0 # 设为0便于调试

EVALUATION_CONFIG = {
    'default_model_path': FINETUNED_MODEL_DIR / "best_model",
    'default_test_data': FINETUNE_DATA_DIR / "test" / "images",
    'default_output_dir': OUTPUT_DIR / "evaluation",
    'batch_size': 2,
    'num_visualization_samples': 5,
    # 'device' 已移除，统一使用全局 DEVICE 配置
    'figure_size': (16, 16),  # 4面板可视化的图像尺寸
    'visualization_dpi': 150,  # 可视化图像的DPI
}

# ==============================================================================
# 注意：所有辅助函数已移至 utils.py
# config.py 现在只包含配置参数
# ==============================================================================

if __name__ == "__main__":
    print("✅ config.py: 配置文件加载成功。")
    print(f"  - 搜索试验次数: {SEARCH_CONFIG['n_trials']}")
    print(f"  - 最终训练轮数: {FINAL_TRAIN_CONFIG['num_epochs']}")
