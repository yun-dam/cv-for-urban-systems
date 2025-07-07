import os
import torch
import numpy as np
from glob import glob
from transformers import CLIPSegProcessor
from pathlib import Path
import optuna
import json
from datetime import datetime
from typing import Dict
import sys

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 导入配置和更新后的工具函数
from config import *
from utils import (
    create_data_loader, create_model_and_optimizer, train_one_epoch,
    evaluate_model, set_seed, ensure_dirs, get_device,
    get_hyperparameter_from_trial, save_config,
    create_training_logger, update_training_log, save_training_log, get_current_lr
)

# 禁用HuggingFace警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_fold_info() -> Dict:
    """扫描并加载预处理好的交叉验证数据折信息。"""
    print("📂 正在加载预处理的交叉验证折信息...")
    fold_info = {}
    cv_folds_dir = FINETUNE_DATA_DIR / "cv_prepared_data" / "cv_folds"

    if not cv_folds_dir.exists():
        raise FileNotFoundError(
            f"CV folds 目录不存在: {cv_folds_dir}\n"
            "请先运行 'data_augmentation.py' 脚本来生成数据。"
        )

    for fold_dir in sorted(cv_folds_dir.glob('fold_*')):
        fold_idx = int(fold_dir.name.split('_')[-1]) - 1
        train_images = glob(str(fold_dir / "train/images/*.tif"))
        val_images = glob(str(fold_dir / "val/images/*.tif"))
        
        if not train_images or not val_images:
            print(f"⚠️ 警告: Fold {fold_idx + 1} 数据不完整，跳过。")
            continue

        fold_info[fold_idx] = {
            "train_images": train_images, "val_images": val_images,
            "train_mask_dir": str(fold_dir / "train/masks"),
            "val_mask_dir": str(fold_dir / "val/masks")
        }
        print(f"  - Fold {fold_idx + 1}: 找到 {len(train_images)} 训练图片, {len(val_images)} 验证图片。")

    if not fold_info:
        raise ValueError(f"在 {cv_folds_dir} 中未能加载任何有效的fold数据。")
    
    print("✅ CV fold数据加载完成。")
    return fold_info

def train_and_evaluate_fold(fold_idx: int, trial_num: int, fold_data: Dict, hyperparams: Dict, device, processor) -> Dict:
    """在单个fold上训练和评估模型，并提供详细的进度信息。返回包含最佳损失和训练日志的字典。"""
    train_loader = create_data_loader(
        fold_data['train_images'], fold_data['train_mask_dir'], URBAN_CLASSES, processor,
        hyperparams['batch_size'], shuffle=True
    )
    val_loader = create_data_loader(
        fold_data['val_images'], fold_data['val_mask_dir'], URBAN_CLASSES, processor,
        hyperparams['batch_size'], shuffle=False
    )
    
    model, optimizer = create_model_and_optimizer(hyperparams['learning_rate'], device)
    
    # 创建训练日志记录器
    fold_logger = create_training_logger()
    fold_logger['metadata']['trial_num'] = trial_num
    fold_logger['metadata']['fold_idx'] = fold_idx
    fold_logger['metadata']['hyperparameters'] = hyperparams
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = SEARCH_CONFIG.get('patience', 5)
    search_epochs = SEARCH_CONFIG['num_epochs']

    for epoch in range(1, search_epochs + 1):
        # 构造详细的描述信息
        train_desc = f"Trial {trial_num+1} Fold {fold_idx+1} Epoch {epoch}/{search_epochs} [Train]"
        val_desc = f"Trial {trial_num+1} Fold {fold_idx+1} Epoch {epoch}/{search_epochs} [Val]"

        train_loss = train_one_epoch(model, train_loader, optimizer, device, hyperparams['dice_weight'], desc_str=train_desc)
        val_loss, _ = evaluate_model(model, val_loader, device, hyperparams['dice_weight'], desc_str=val_desc)
        
        # 更新训练日志
        current_lr = get_current_lr(optimizer)
        update_training_log(fold_logger, epoch, train_loss, val_loss, current_lr)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -- 早停在 epoch {epoch}")
                fold_logger['metadata']['early_stopped'] = True
                fold_logger['metadata']['stopped_at_epoch'] = epoch
                break
    
    return {
        'best_val_loss': best_val_loss,
        'fold_logger': fold_logger
    }

def objective(trial, fold_info: Dict, device, processor) -> float:
    """Optuna目标函数。"""
    hyperparams = {
        'learning_rate': get_hyperparameter_from_trial(trial, 'learning_rate'),
        'dice_weight': get_hyperparameter_from_trial(trial, 'dice_weight'),
        'batch_size': get_hyperparameter_from_trial(trial, 'batch_size'),
    }
    
    fold_scores = []
    trial_loggers = []
    
    for fold_idx, fold_data in fold_info.items():
        try:
            result = train_and_evaluate_fold(fold_idx, trial.number, fold_data, hyperparams, device, processor)
            fold_scores.append(result['best_val_loss'])
            trial_loggers.append(result['fold_logger'])
        except Exception as e:
            print(f"Fold {fold_idx} 训练失败: {e}")
            return float('inf')
    
    avg_score = np.mean(fold_scores) if fold_scores else float('inf')
    trial.set_user_attr("mean_val_loss", avg_score)
    trial.set_user_attr("fold_scores", fold_scores)
    
    # 保存该trial的所有fold训练日志
    trial_log_dir = HYPERPARAMETER_SEARCH_DIR / "trial_logs" / f"trial_{trial.number}"
    trial_log_dir.mkdir(parents=True, exist_ok=True)
    
    for fold_idx, fold_logger in enumerate(trial_loggers):
        log_path = trial_log_dir / f"fold_{fold_idx + 1}_log.json"
        save_training_log(fold_logger, log_path)
    
    # 保存trial总结
    trial_summary = {
        'trial_number': trial.number,
        'hyperparameters': hyperparams,
        'fold_scores': fold_scores,
        'mean_val_loss': avg_score,
        'timestamp': datetime.now().isoformat()
    }
    summary_path = trial_log_dir / "trial_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(trial_summary, f, indent=2)
    
    return avg_score

def run_hyperparameter_search(fold_info: Dict):
    """运行超参数搜索并保存结果。"""
    n_trials = SEARCH_CONFIG['n_trials']
    print(f"\n🔍 开始超参数搜索 ({n_trials} trials)...")
    
    device = get_device()
    processor = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL)

    study = optuna.create_study(direction='minimize', study_name=SEARCH_CONFIG['study_name'])
    study.optimize(lambda trial: objective(trial, fold_info, device, processor), n_trials=n_trials)
    
    print("\n" + "="*50)
    print("✅ 超参数搜索完成!")
    print(f"  最佳试验: Trial {study.best_trial.number}")
    print(f"  最佳分数 (损失): {study.best_value:.4f}")
    print(f"  最佳参数: {study.best_params}")
    print("="*50)
    
    # 保存最佳超参数
    results_file = HYPERPARAMETER_SEARCH_DIR / "best_hyperparams.json"
    save_config(study.best_params, results_file)
    print(f"\n💾 最佳超参数已保存到: {results_file}")
    
    # 保存完整的搜索结果
    search_summary = {
        'study_name': SEARCH_CONFIG['study_name'],
        'n_trials': n_trials,
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'all_trials': []
    }
    
    for trial in study.trials:
        trial_info = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'user_attrs': trial.user_attrs,
            'state': str(trial.state)
        }
        search_summary['all_trials'].append(trial_info)
    
    # 保存搜索总结
    summary_path = HYPERPARAMETER_SEARCH_DIR / "search_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(search_summary, f, indent=2)
    print(f"📊 搜索总结已保存到: {summary_path}")

def main():
    """主函数：加载数据 -> 运行搜索 -> 保存结果"""
    print("🚀 步骤 2: 运行超参数搜索")
    print("=" * 60)
    
    set_seed()
    ensure_dirs()
    
    try:
        fold_info = load_fold_info()
        run_hyperparameter_search(fold_info)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ 错误: {e}")
        return

    print("\n🎉 超参数搜索流程执行完成!")

if __name__ == "__main__":
    main()
