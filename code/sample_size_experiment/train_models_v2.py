#!/usr/bin/env python3
"""
为不同样本大小训练模型 - 修订版
使用v2数据格式（扁平PNG结构）

功能：
1. 加载最佳超参数（从之前的超参数搜索结果）
2. 为每个样本大小（5, 10, 20, 40, 80, 160, 320）训练独立的模型
3. 使用与final_train.py相同的训练方式（早停、checkpoint、日志等）
4. 将每个模型保存到对应的目录
"""

import os
import sys
import json
from glob import glob
from pathlib import Path
from typing import Dict
from transformers import CLIPSegProcessor

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Add code directory to path (where utils.py is located)
code_dir = project_root / "code"
sys.path.insert(0, str(code_dir))

# Import configuration and utility functions
from config import *
from utils import (
    create_data_loader, create_model_and_optimizer, train_one_epoch,
    evaluate_model, save_model, set_seed, ensure_dirs, get_device,
    create_training_logger, update_training_log, save_training_log, get_current_lr,
    CheckpointManager
)

# Disable HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message="The following named arguments are not valid")
warnings.filterwarnings("ignore", message="Using a slow image processor")

# 实验配置 - 修改为v2路径
EXPERIMENT_DATA_DIR = DATA_DIR / "Vaihingen" / "sample_size_experiment_v2"
EXPERIMENT_MODEL_DIR = MODELS_DIR / "sample_size_experiment_v2"
SAMPLE_SIZES = [5, 10, 20, 40, 80, 160, 320]
# SAMPLE_SIZES = [5, 10, 20]  # 测试时可以只运行部分


def load_best_hyperparameters() -> Dict:
    """加载最佳超参数"""
    # 首先尝试从标准位置加载
    params_file = HYPERPARAMETER_SEARCH_DIR / "best_hyperparams.json"
    
    # 如果不存在，尝试从logs中加载
    if not params_file.exists():
        params_file = HYPERPARAMETER_SEARCH_DIR / "best_hyperparams_from_logs.json"
    
    if not params_file.exists():
        # 如果仍然不存在，使用默认参数
        print("⚠️ Best hyperparameters not found. Using default parameters.")
        return {
            'learning_rate': FINAL_TRAIN_CONFIG['default_learning_rate'],
            'batch_size': FINAL_TRAIN_CONFIG['default_batch_size'],
            'dice_weight': FINAL_TRAIN_CONFIG['default_dice_weight']
        }
    
    with open(params_file, 'r') as f:
        best_params = json.load(f)
    
    print(f"✅ Loaded best hyperparameters: {best_params}")
    return best_params


def train_model_for_sample_size(sample_size: int, best_params: Dict):
    """为特定样本大小训练模型"""
    print(f"\n{'='*60}")
    print(f"Training model for {sample_size} samples")
    print(f"{'='*60}")
    
    # 检查模型是否已存在
    model_save_dir = EXPERIMENT_MODEL_DIR / f"model_{sample_size}"
    best_model_dir = model_save_dir / "best_model"
    
    # 检查多种可能的模型文件格式
    model_files_exist = (
        (best_model_dir / "pytorch_model.bin").exists() or 
        (best_model_dir / "model.safetensors").exists()
    )
    
    if best_model_dir.exists() and model_files_exist:
        print(f"✅ Model for {sample_size} samples already exists at {best_model_dir}")
        print(f"   Skipping training to avoid overwriting existing model.")
        print(f"   If you want to retrain, please delete the directory first.")
        return True
    
    device = get_device()
    processor = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL)
    
    # 数据路径 - 使用v2路径
    augmented_data_dir = EXPERIMENT_DATA_DIR / "augmented_training_data" / f"pool_{sample_size}_augmented"
    train_data_dir = augmented_data_dir / "train"
    val_data_dir = augmented_data_dir / "val"
    
    # 检查数据是否存在
    if not train_data_dir.exists() or not val_data_dir.exists():
        print(f"❌ Error: Augmented data not found for pool_{sample_size}")
        print(f"Expected path: {augmented_data_dir}")
        print("Please run prepare_training_data_v2.py first.")
        return False
    
    # 加载训练和验证数据
    train_images = sorted(glob(str(train_data_dir / "images/*.tif")))
    val_images = sorted(glob(str(val_data_dir / "images/*.tif")))
    
    if not train_images or not val_images:
        print(f"❌ Error: No images found in {augmented_data_dir}")
        return False
    
    train_mask_dir = str(train_data_dir / "masks")
    val_mask_dir = str(val_data_dir / "masks")
    
    print(f"  Training set: {len(train_images)} images (augmented)")
    print(f"  Validation set: {len(val_images)} images (original)")
    
    # 创建数据加载器 - 使用标准的create_data_loader，它会自动处理PNG格式
    train_loader = create_data_loader(
        train_images, train_mask_dir, URBAN_CLASSES, processor,
        best_params['batch_size'], shuffle=True
    )
    
    val_loader = create_data_loader(
        val_images, val_mask_dir, URBAN_CLASSES, processor,
        best_params['batch_size'], shuffle=False
    )
    
    # 创建模型和优化器
    model, optimizer = create_model_and_optimizer(best_params['learning_rate'], device)
    
    # 创建模型保存目录
    model_save_dir = EXPERIMENT_MODEL_DIR / f"model_{sample_size}"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建checkpoint管理器
    checkpoint_manager = CheckpointManager(
        model_save_dir / "checkpoints",
        max_checkpoints=FINAL_TRAIN_CONFIG.get('max_checkpoints', 3)
    )
    
    # 创建训练日志
    training_logger = create_training_logger()
    training_logger['metadata']['experiment'] = 'sample_size_experiment_v2'
    training_logger['metadata']['sample_size'] = sample_size
    training_logger['metadata']['hyperparameters'] = best_params
    training_logger['metadata']['train_images'] = len(train_images)
    training_logger['metadata']['val_images'] = len(val_images)
    training_logger['metadata']['data_source'] = str(augmented_data_dir)
    
    # 训练参数
    best_val_loss = float('inf')
    best_epoch_model = None
    num_epochs = FINAL_TRAIN_CONFIG['num_epochs']
    patience = FINAL_TRAIN_CONFIG.get('patience', 10)
    min_delta = FINAL_TRAIN_CONFIG.get('min_delta', 1e-4)
    patience_counter = 0
    
    print(f"\n  Training settings:")
    print(f"    - Epochs: {num_epochs}")
    print(f"    - Early stopping: patience={patience}, min_delta={min_delta}")
    print(f"    - Learning rate: {best_params['learning_rate']}")
    print(f"    - Batch size: {best_params['batch_size']}")
    print(f"    - Dice weight: {best_params['dice_weight']}")
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        train_desc = f"Model_{sample_size} Epoch {epoch}/{num_epochs} [Training]"
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            best_params['dice_weight'], desc_str=train_desc
        )
        
        # 验证阶段
        val_desc = f"Model_{sample_size} Epoch {epoch}/{num_epochs} [Validation]"
        val_loss, val_iou = evaluate_model(
            model, val_loader, device,
            best_params['dice_weight'], desc_str=val_desc
        )
        
        # 更新训练日志
        current_lr = get_current_lr(optimizer)
        update_training_log(training_logger, epoch, train_loss, val_loss, current_lr)
        
        # 打印进度
        print(f"  Epoch {epoch:3d}/{num_epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_iou={val_iou:.4f}, lr={current_lr:.2e}")
        
        # 检查是否是最佳模型
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch_model = epoch
            patience_counter = 0
            
            # 保存最佳模型
            best_model_dir = model_save_dir / "best_model"
            save_model(model, processor, best_model_dir, metadata={
                'epoch': epoch,
                'val_loss': float(val_loss),
                'val_iou': float(val_iou),
                'sample_size': sample_size,
                'hyperparameters': best_params
            })
            print(f"  💾 Saved best model (epoch {epoch})")
        else:
            patience_counter += 1
            
        # 定期保存checkpoint
        if epoch % FINAL_TRAIN_CONFIG.get('checkpoint_interval', 5) == 0:
            checkpoint_manager.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch_model,
                'patience_counter': patience_counter,
                'training_log': training_logger
            }, epoch, is_best=(epoch == best_epoch_model))
        
        # 早停检查
        if patience_counter >= patience:
            print(f"\n  ⏹️  Early stopping triggered at epoch {epoch}")
            training_logger['metadata']['early_stopped'] = True
            training_logger['metadata']['stopped_at_epoch'] = epoch
            break
    
    # 保存训练日志
    log_path = model_save_dir / "training_log.json"
    save_training_log(training_logger, log_path)
    
    # 打印训练总结
    print(f"\n  Training completed for model_{sample_size}!")
    print(f"    - Best epoch: {best_epoch_model}")
    print(f"    - Best validation loss: {best_val_loss:.4f}")
    print(f"    - Model saved to: {model_save_dir}")
    
    return True


def main():
    """主函数：训练所有模型"""
    print("=" * 60)
    print("Sample Size Experiment - Model Training V2")
    print("Using flat PNG mask format")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    # 确保必要的目录存在
    ensure_dirs()
    EXPERIMENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载最佳超参数
    print("\nLoading best hyperparameters...")
    best_params = load_best_hyperparameters()
    
    # 训练每个模型
    success_count = 0
    for sample_size in SAMPLE_SIZES:
        if train_model_for_sample_size(sample_size, best_params):
            success_count += 1
        print("\n" + "-" * 60)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("Model Training Complete!")
    print("=" * 60)
    print(f"Successfully trained: {success_count}/{len(SAMPLE_SIZES)} models")
    print(f"Models saved to: {EXPERIMENT_MODEL_DIR}")
    print("\nNext steps:")
    print("  1. Run inference_all_v2.py to generate predictions")
    print("  2. Run evaluate_each_model_v2.py to evaluate performance")


if __name__ == "__main__":
    main()