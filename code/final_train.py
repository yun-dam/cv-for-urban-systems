import os
from glob import glob
from transformers import CLIPSegProcessor
from pathlib import Path
import json
from typing import Dict
import sys

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 导入配置和更新后的工具函数
from config import *
from utils import (
    create_data_loader, create_model_and_optimizer, train_one_epoch,
    save_model, set_seed, ensure_dirs, get_device,
    create_training_logger, update_training_log, save_training_log, get_current_lr
)

# 禁用HuggingFace警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_best_hyperparameters() -> Dict:
    """从文件加载最佳超参数。"""
    params_file = HYPERPARAMETER_SEARCH_DIR / "best_hyperparams.json"
    if not params_file.exists():
        raise FileNotFoundError(
            f"未找到最佳超参数文件: {params_file}\n"
            "请先运行 'hyperparameter_search.py' 脚本。"
        )
    
    with open(params_file, 'r') as f:
        best_params = json.load(f)
    
    print(f"✅ 成功加载最佳超参数: {best_params}")
    return best_params

def train_final_model(best_params: Dict):
    """使用最佳参数在全量增强数据上训练最终模型。"""
    print("\n🎯 开始训练最终模型...")
    
    device = get_device()
    processor = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL)
    
    # 使用由 data_augmentation.py 创建的全量增强数据
    final_data_dir = FINETUNE_DATA_DIR / "cv_prepared_data" / "all_data_for_final_train"
    all_augmented_images = glob(str(final_data_dir / "images/*.tif"))
    all_mask_dir = str(final_data_dir / "masks")

    if not all_augmented_images:
        raise FileNotFoundError(f"在 {final_data_dir} 中未找到用于最终训练的图片。")

    print(f"  总训练图片数: {len(all_augmented_images)}")
    
    train_loader = create_data_loader(
        all_augmented_images, all_mask_dir, URBAN_CLASSES, processor,
        best_params['batch_size'], shuffle=True
    )
    
    model, optimizer = create_model_and_optimizer(best_params['learning_rate'], device)
    
    # 创建训练日志记录器
    final_logger = create_training_logger()
    final_logger['metadata']['model_type'] = 'final_model'
    final_logger['metadata']['hyperparameters'] = best_params
    final_logger['metadata']['total_images'] = len(all_augmented_images)
    final_logger['metadata']['data_source'] = str(final_data_dir)
    
    best_loss = float('inf')
    best_epoch_model = None
    final_epochs = FINAL_TRAIN_CONFIG['num_epochs']
    
    for epoch in range(1, final_epochs + 1):
        # 构造详细的描述信息
        train_desc = f"最终训练 Epoch {epoch}/{final_epochs}"
        
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            best_params.get('dice_weight', FINAL_TRAIN_CONFIG['default_dice_weight']),
            desc_str=train_desc
        )
        
        # 更新训练日志
        current_lr = get_current_lr(optimizer)
        update_training_log(final_logger, epoch, avg_loss, lr=current_lr)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch_model = epoch
            model_save_path = FINETUNED_MODEL_DIR / "best_model"
            
            # 准备要保存的元数据
            metadata = {
                'best_loss': best_loss,
                'epoch': epoch,
                'hyperparameters': best_params,
                'total_epochs_planned': final_epochs
            }
            save_model(model, processor, model_save_path, metadata)
            print(f"  ✨ 保存最佳模型 (损失: {avg_loss:.4f})")
    
    # 保存训练日志
    final_logger['metadata']['best_model_saved_at_epoch'] = best_epoch_model
    log_path = FINETUNED_MODEL_DIR / "final_training_log.json"
    save_training_log(final_logger, log_path)
    
    print(f"\n✅ 最终模型训练完成! ")
    print(f"   - 模型保存在: {FINETUNED_MODEL_DIR / 'best_model'}")
    print(f"   - 训练日志保存在: {log_path}")
    print(f"   - 最佳模型来自 epoch {best_epoch_model}/{final_epochs}")

def main():
    """主函数：加载参数 -> 训练模型 -> 保存模型"""
    print("🚀 步骤 3: 训练最终模型")
    print("=" * 60)
    
    set_seed()
    ensure_dirs()
    
    try:
        best_params = load_best_hyperparameters()
        train_final_model(best_params)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ 错误: {e}")
        return

    print("\n🎉 最终模型训练流程执行完成!")

if __name__ == "__main__":
    main()
