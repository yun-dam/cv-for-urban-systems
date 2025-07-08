import os
from glob import glob
from transformers import CLIPSegProcessor
from pathlib import Path
import json
from typing import Dict
import sys

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import configuration and updated utility functions
from config import *
from utils import (
    create_data_loader, create_model_and_optimizer, train_one_epoch,
    evaluate_model, save_model, set_seed, ensure_dirs, get_device,
    create_training_logger, update_training_log, save_training_log, get_current_lr,
    CheckpointManager
)

# Disable HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_best_hyperparameters() -> Dict:
    """Load best hyperparameters from file."""
    params_file = HYPERPARAMETER_SEARCH_DIR / "best_hyperparams.json"
    if not params_file.exists():
        raise FileNotFoundError(
            f"Best hyperparameters file not found: {params_file}\n"
            "Please run 'hyperparameter_search.py' script first."
        )
    
    with open(params_file, 'r') as f:
        best_params = json.load(f)
    
    print(f"✅ Successfully loaded best hyperparameters: {best_params}")
    return best_params

def train_final_model(best_params: Dict):
    """Train final model using best parameters on full augmented dataset."""
    print("\n🎯 Starting final model training...")
    
    device = get_device()
    processor = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL)
    
    # Use pre-split data created by data_augmentation.py
    final_data_dir = FINETUNE_DATA_DIR / "cv_prepared_data" / "all_data_for_final_train"
    
    # Load training and validation sets separately
    train_data_dir = final_data_dir / "train"
    val_data_dir = final_data_dir / "val"
    
    train_images = sorted(glob(str(train_data_dir / "images/*.tif")))
    val_images = sorted(glob(str(val_data_dir / "images/*.tif")))
    
    train_mask_dir = str(train_data_dir / "masks")
    val_mask_dir = str(val_data_dir / "masks")
    
    if not train_images or not val_images:
        raise FileNotFoundError(
            f"Pre-split training/validation data not found.\n"
            f"Please run data_augmentation.py first to generate data.\n"
            f"Expected paths: {train_data_dir} and {val_data_dir}"
        )
    
    print(f"  Training set: {len(train_images)} images (augmented)")
    print(f"  Validation set: {len(val_images)} images (original)")
    
    # Create data loaders
    train_loader = create_data_loader(
        train_images, train_mask_dir, URBAN_CLASSES, processor,
        best_params['batch_size'], shuffle=True
    )
    
    val_loader = create_data_loader(
        val_images, val_mask_dir, URBAN_CLASSES, processor,
        best_params['batch_size'], shuffle=False
    )
    
    model, optimizer = create_model_and_optimizer(best_params['learning_rate'], device)
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        FINETUNED_MODEL_DIR / "checkpoints",
        max_checkpoints=FINAL_TRAIN_CONFIG.get('max_checkpoints', 3)
    )
    
    # Create training logger
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
    start_epoch = 1
    
    # Check if we should resume from checkpoint
    if FINAL_TRAIN_CONFIG.get('resume_from_checkpoint', True):
        available_checkpoints = checkpoint_manager.get_available_checkpoints()
        if available_checkpoints:
            print(f"\n📂 Found checkpoints: {available_checkpoints}")
            
            # Ask user if they want to resume
            while True:
                user_input = input("\nDo you want to resume from checkpoint? (y/n): ").lower().strip()
                if user_input in ['y', 'yes', 'n', 'no']:
                    break
                print("Please enter 'y' for yes or 'n' for no.")
            
            if user_input in ['y', 'yes']:
                # Try to load latest checkpoint
                checkpoint = checkpoint_manager.load_checkpoint('latest')
                if checkpoint:
                    # Restore model and optimizer state
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    # Restore training state
                    start_epoch = checkpoint['epoch'] + 1
                    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                    best_epoch_model = checkpoint.get('best_epoch', None)
                    patience_counter = checkpoint.get('patience_counter', 0)
                    
                    # Restore training logger if available
                    if 'training_log' in checkpoint:
                        final_logger = checkpoint['training_log']
                    
                    print(f"✅ Resuming training from epoch {start_epoch}")
                    print(f"   Best validation loss so far: {best_val_loss:.4f}")
                    print(f"   Patience counter: {patience_counter}/{patience}")
            else:
                print("🆕 Starting fresh training (ignoring checkpoints).")
        else:
            print("\n📂 No checkpoints found. Starting fresh training.")
    
    print(f"\n  Early stopping settings: patience={patience}, min_delta={min_delta}")
    
    for epoch in range(start_epoch, final_epochs + 1):
        # Training phase
        train_desc = f"Final Training Epoch {epoch}/{final_epochs} [Training]"
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            best_params.get('dice_weight', FINAL_TRAIN_CONFIG['default_dice_weight']),
            desc_str=train_desc
        )
        
        # Validation phase
        val_desc = f"Final Training Epoch {epoch}/{final_epochs} [Validation]"
        val_loss, val_iou = evaluate_model(
            model, val_loader, device,
            best_params.get('dice_weight', FINAL_TRAIN_CONFIG['default_dice_weight']),
            desc_str=val_desc
        )
        
        # Update training log
        current_lr = get_current_lr(optimizer)
        update_training_log(final_logger, epoch, train_loss, val_loss, current_lr)
        
        print(f"  Epoch {epoch}: Training loss={train_loss:.4f}, Validation loss={val_loss:.4f}, Validation IoU={val_iou:.4f}")
        
        # Check if this is the best model
        is_best = val_loss < best_val_loss - min_delta
        if is_best:
            best_val_loss = val_loss
            best_epoch_model = epoch
            patience_counter = 0
            
            model_save_path = FINETUNED_MODEL_DIR / "best_model"
            
            # Prepare metadata to save
            metadata = {
                'best_train_loss': train_loss,
                'best_val_loss': val_loss,
                'best_val_iou': val_iou,
                'epoch': epoch,
                'hyperparameters': best_params,
                'total_epochs_planned': final_epochs
            }
            save_model(model, processor, model_save_path, metadata)
            print(f"  ✨ Saved best model (Validation loss: {val_loss:.4f}, IoU: {val_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  🛑 Early stopping triggered: Validation loss has not improved for {patience} epochs")
                final_logger['metadata']['early_stopped'] = True
                final_logger['metadata']['stopped_at_epoch'] = epoch
                break
        
        # Save checkpoint
        if epoch % FINAL_TRAIN_CONFIG.get('checkpoint_interval', 5) == 0 or is_best:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch_model,
                'patience_counter': patience_counter,
                'training_log': final_logger,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'hyperparameters': best_params
            }
            checkpoint_manager.save_checkpoint(checkpoint_state, epoch, is_best=is_best)
    
    # Save final checkpoint before exiting
    final_checkpoint_state = {
        'epoch': epoch if 'epoch' in locals() else final_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch_model,
        'patience_counter': patience_counter,
        'training_log': final_logger,
        'hyperparameters': best_params,
        'training_completed': True
    }
    checkpoint_manager.save_checkpoint(final_checkpoint_state, epoch if 'epoch' in locals() else final_epochs, is_best=False)
    
    # Save training log
    final_logger['metadata']['best_model_saved_at_epoch'] = best_epoch_model
    log_path = FINETUNED_MODEL_DIR / "final_training_log.json"
    save_training_log(final_logger, log_path)
    
    print(f"\n✅ Final model training completed! ")
    print(f"   - Model saved at: {FINETUNED_MODEL_DIR / 'best_model'}")
    print(f"   - Checkpoints saved at: {FINETUNED_MODEL_DIR / 'checkpoints'}")
    print(f"   - Training log saved at: {log_path}")
    print(f"   - Best model from epoch {best_epoch_model}/{final_epochs}")

def main():
    """Main function: load parameters -> train model -> save model"""
    print("🚀 Step 3: Train final model")
    print("=" * 60)
    
    set_seed()
    ensure_dirs()
    
    try:
        best_params = load_best_hyperparameters()
        train_final_model(best_params)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        return

    print("\n🎉 Final model training workflow completed!")

if __name__ == "__main__":
    main()
