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
    evaluate_model, save_model, set_seed, ensure_dirs, get_device,
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
    
    # 使用由 data_augmentation.py 创建的预先分割好的数据
    final_data_dir = FINETUNE_DATA_DIR / "cv_prepared_data" / "all_data_for_final_train"
    
    # 分别加载训练集和验证集
    train_data_dir = final_data_dir / "train"
    val_data_dir = final_data_dir / "val"
    
    train_images = sorted(glob(str(train_data_dir / "images/*.tif")))
    val_images = sorted(glob(str(val_data_dir / "images/*.tif")))
    
    train_mask_dir = str(train_data_dir / "masks")
    val_mask_dir = str(val_data_dir / "masks")
    
    if not train_images or not val_images:
        raise FileNotFoundError(
            f"未找到预先分割的训练/验证数据。\n"
            f"请先运行 data_augmentation.py 生成数据。\n"
            f"期望路径: {train_data_dir} 和 {val_data_dir}"
        )
    
    print(f"  训练集: {len(train_images)} 张图片（增强后）")
    print(f"  验证集: {len(val_images)} 张图片（原始）")
    
    # 创建数据加载器
    train_loader = create_data_loader(
        train_images, train_mask_dir, URBAN_CLASSES, processor,
        best_params['batch_size'], shuffle=True
    )
    
    val_loader = create_data_loader(
        val_images, val_mask_dir, URBAN_CLASSES, processor,
        best_params['batch_size'], shuffle=False
    )
    
    model, optimizer = create_model_and_optimizer(best_params['learning_rate'], device)
    
    # 创建训练日志记录器
    final_logger = create_training_logger()
    final_logger['metadata']['model_type'] = 'final_model'
    final_logger['metadata']['hyperparameters'] = best_params
    final_logger['metadata']['train_images'] = len(train_images)
    final_logger['metadata']['val_images'] = len(val_images)
    final_logger['metadata']['data_source'] = str(final_data_dir)
    final_logger['metadata']['val_split_method'] = 'pre-split at original image level'
    
    best_val_loss = float('inf')
    best_epoch_model = None
    final_epochs = FINAL_TRAIN_CONFIG['num_epochs']
    patience = FINAL_TRAIN_CONFIG.get('patience', 10)
    min_delta = FINAL_TRAIN_CONFIG.get('min_delta', 1e-4)
    patience_counter = 0
    
    print(f"\n  早停设置: patience={patience}, min_delta={min_delta}")
    
    for epoch in range(1, final_epochs + 1):
        # 训练阶段
        train_desc = f"最终训练 Epoch {epoch}/{final_epochs} [训练]"
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            best_params.get('dice_weight', FINAL_TRAIN_CONFIG['default_dice_weight']),
            desc_str=train_desc
        )
        
        # 验证阶段
        val_desc = f"最终训练 Epoch {epoch}/{final_epochs} [验证]"
        val_loss, val_iou = evaluate_model(
            model, val_loader, device,
            best_params.get('dice_weight', FINAL_TRAIN_CONFIG['default_dice_weight']),
            desc_str=val_desc
        )
        
        # 更新训练日志
        current_lr = get_current_lr(optimizer)
        update_training_log(final_logger, epoch, train_loss, val_loss, current_lr)
        
        print(f"  Epoch {epoch}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}, 验证IoU={val_iou:.4f}")
        
        # 检查是否为最佳模型
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch_model = epoch
            patience_counter = 0
            
            model_save_path = FINETUNED_MODEL_DIR / "best_model"
            
            # 准备要保存的元数据
            metadata = {
                'best_train_loss': train_loss,
                'best_val_loss': val_loss,
                'best_val_iou': val_iou,
                'epoch': epoch,
                'hyperparameters': best_params,
                'total_epochs_planned': final_epochs
            }
            save_model(model, processor, model_save_path, metadata)
            print(f"  ✨ 保存最佳模型 (验证损失: {val_loss:.4f}, IoU: {val_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  🛑 早停触发: 验证损失已经 {patience} 轮没有改善")
                final_logger['metadata']['early_stopped'] = True
                final_logger['metadata']['stopped_at_epoch'] = epoch
                break
    
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
