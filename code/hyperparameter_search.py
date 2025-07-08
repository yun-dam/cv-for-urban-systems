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

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import configuration and updated utility functions
from config import *
from utils import (
    create_data_loader, create_model_and_optimizer, train_one_epoch,
    evaluate_model, set_seed, ensure_dirs, get_device,
    get_hyperparameter_from_trial, save_config,
    create_training_logger, update_training_log, save_training_log, get_current_lr
)

# Disable HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_fold_info() -> Dict:
    """Scan and load preprocessed cross-validation fold information."""
    print("📂 Loading preprocessed cross-validation fold information...")
    fold_info = {}
    cv_folds_dir = FINETUNE_DATA_DIR / "cv_prepared_data" / "cv_folds"

    if not cv_folds_dir.exists():
        raise FileNotFoundError(
            f"CV folds directory does not exist: {cv_folds_dir}\n"
            "Please run 'data_augmentation.py' script first to generate data."
        )

    for fold_dir in sorted(cv_folds_dir.glob('fold_*')):
        fold_idx = int(fold_dir.name.split('_')[-1]) - 1
        train_images = glob(str(fold_dir / "train/images/*.tif"))
        val_images = glob(str(fold_dir / "val/images/*.tif"))
        
        if not train_images or not val_images:
            print(f"⚠️ Warning: Fold {fold_idx + 1} data incomplete, skipping.")
            continue

        fold_info[fold_idx] = {
            "train_images": train_images, "val_images": val_images,
            "train_mask_dir": str(fold_dir / "train/masks"),
            "val_mask_dir": str(fold_dir / "val/masks")
        }
        print(f"  - Fold {fold_idx + 1}: Found {len(train_images)} training images, {len(val_images)} validation images.")

    if not fold_info:
        raise ValueError(f"Unable to load any valid fold data in {cv_folds_dir}.")
    
    print("✅ CV fold data loading completed.")
    return fold_info

def train_and_evaluate_fold(fold_idx: int, trial_num: int, fold_data: Dict, hyperparams: Dict, device, processor) -> Dict:
    """Train and evaluate model on a single fold with detailed progress information. Returns dictionary containing best loss and training log."""
    train_loader = create_data_loader(
        fold_data['train_images'], fold_data['train_mask_dir'], URBAN_CLASSES, processor,
        hyperparams['batch_size'], shuffle=True
    )
    val_loader = create_data_loader(
        fold_data['val_images'], fold_data['val_mask_dir'], URBAN_CLASSES, processor,
        hyperparams['batch_size'], shuffle=False
    )
    
    model, optimizer = create_model_and_optimizer(hyperparams['learning_rate'], device)
    
    # Create training logger
    fold_logger = create_training_logger()
    fold_logger['metadata']['trial_num'] = trial_num
    fold_logger['metadata']['fold_idx'] = fold_idx
    fold_logger['metadata']['hyperparameters'] = hyperparams
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = SEARCH_CONFIG.get('patience', 5)
    search_epochs = SEARCH_CONFIG['num_epochs']

    for epoch in range(1, search_epochs + 1):
        # Construct detailed description information
        train_desc = f"Trial {trial_num+1} Fold {fold_idx+1} Epoch {epoch}/{search_epochs} [Train]"
        val_desc = f"Trial {trial_num+1} Fold {fold_idx+1} Epoch {epoch}/{search_epochs} [Val]"

        train_loss = train_one_epoch(model, train_loader, optimizer, device, hyperparams['dice_weight'], desc_str=train_desc)
        val_loss, _ = evaluate_model(model, val_loader, device, hyperparams['dice_weight'], desc_str=val_desc)
        
        # Update training log
        current_lr = get_current_lr(optimizer)
        update_training_log(fold_logger, epoch, train_loss, val_loss, current_lr)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -- Early stopping at epoch {epoch}")
                fold_logger['metadata']['early_stopped'] = True
                fold_logger['metadata']['stopped_at_epoch'] = epoch
                break
    
    return {
        'best_val_loss': best_val_loss,
        'fold_logger': fold_logger
    }

def objective(trial, fold_info: Dict, device, processor) -> float:
    """Optuna objective function."""
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
            print(f"Fold {fold_idx} training failed: {e}")
            return float('inf')
    
    avg_score = np.mean(fold_scores) if fold_scores else float('inf')
    trial.set_user_attr("mean_val_loss", avg_score)
    trial.set_user_attr("fold_scores", fold_scores)
    
    # Save all fold training logs for this trial
    trial_log_dir = HYPERPARAMETER_SEARCH_DIR / "trial_logs" / f"trial_{trial.number}"
    trial_log_dir.mkdir(parents=True, exist_ok=True)
    
    for fold_idx, fold_logger in enumerate(trial_loggers):
        log_path = trial_log_dir / f"fold_{fold_idx + 1}_log.json"
        save_training_log(fold_logger, log_path)
    
    # Save trial summary
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
    """Run hyperparameter search and save results."""
    n_trials = SEARCH_CONFIG['n_trials']
    print(f"\n🔍 Starting hyperparameter search ({n_trials} trials)...")
    
    device = get_device()
    processor = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL)

    study = optuna.create_study(direction='minimize', study_name=SEARCH_CONFIG['study_name'])
    study.optimize(lambda trial: objective(trial, fold_info, device, processor), n_trials=n_trials)
    
    print("\n" + "="*50)
    print("✅ Hyperparameter search completed!")
    print(f"  Best trial: Trial {study.best_trial.number}")
    print(f"  Best score (loss): {study.best_value:.4f}")
    print(f"  Best parameters: {study.best_params}")
    print("="*50)
    
    # Save best hyperparameters
    results_file = HYPERPARAMETER_SEARCH_DIR / "best_hyperparams.json"
    save_config(study.best_params, results_file)
    print(f"\n💾 Best hyperparameters saved to: {results_file}")
    
    # Save complete search results
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
    
    # Save search summary
    summary_path = HYPERPARAMETER_SEARCH_DIR / "search_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(search_summary, f, indent=2)
    print(f"📊 Search summary saved to: {summary_path}")

def main():
    """Main function: load data -> run search -> save results"""
    print("🚀 Step 2: Run hyperparameter search")
    print("=" * 60)
    
    set_seed()
    ensure_dirs()
    
    try:
        fold_info = load_fold_info()
        run_hyperparameter_search(fold_info)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        return

    print("\n🎉 Hyperparameter search workflow completed!")

if __name__ == "__main__":
    main()
