import os
import random
import torch
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import cv2
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import optuna
import json
from datetime import datetime

# Disable HuggingFace Tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==============================================================================
# Configuration
# ==============================================================================
# Input directories - using augmented data with proper train/val split
USE_AUGMENTED_DATA = True  # Set to True to use augmented data

if USE_AUGMENTED_DATA:
    TRAIN_IMG_DIR = "./data/Vaihingen/finetune_data/train_augmented/images"
    TRAIN_MASK_DIR = "./data/Vaihingen/finetune_data/train_augmented/masks"
    VAL_IMG_DIR = "./data/Vaihingen/finetune_data/val_augmented/images"
    VAL_MASK_DIR = "./data/Vaihingen/finetune_data/val_augmented/masks"
else:
    TRAIN_IMG_DIR = "./data/Vaihingen/finetune_data/train/images"
    TRAIN_MASK_DIR = "./data/Vaihingen/finetune_data/train/masks"
    VAL_IMG_DIR = "./data/Vaihingen/finetune_data/val/images"
    VAL_MASK_DIR = "./data/Vaihingen/finetune_data/val/masks"

# Output directories
OUTPUT_DIR = "./clipseg_finetuned_model_searched"
OUTPUT_SEARCH_DIR = "./clipseg_hyperparameter_search"

# Class definitions
CLASSES = ['impervious surface', 'building', 'low vegetation', 'tree', 'car', 'background']

# Training hyperparameters
PRETRAINED_MODEL = "CIDAS/clipseg-rd64-refined"
NUM_EPOCHS = 15  # Reduced for laptop testing
PATIENCE = 3     # Reduced for faster convergence
NUM_WORKERS = 0  # For debugging on CPU

# Hyperparameter search settings
RUN_HYPERPARAMETER_SEARCH = True
N_TRIALS = 6  # Small number for laptop testing (3×2=6 combinations)

# Note: When USE_AUGMENTED_DATA=True, both train and val come from augmented data
# but from different original images to avoid data leakage

# ==============================================================================
# Dataset Class
# ==============================================================================
class FineTuneDataset(Dataset):
    def __init__(self, image_paths, mask_dir, classes, processor):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.classes = classes
        self.processor = processor
        self.n_classes = len(classes)

    def __len__(self):
        return len(self.image_paths) * self.n_classes

    def __getitem__(self, idx):
        img_idx = idx // self.n_classes
        cls_idx = idx % self.n_classes

        image_path = self.image_paths[img_idx]
        cls_name = self.classes[cls_idx]
        
        image = Image.open(image_path).convert("RGB")

        base_fn = Path(image_path).stem
        safe_class_name = cls_name.replace(" ", "_").replace("/", "_")
        mask_path = os.path.join(self.mask_dir, f"{base_fn}_{safe_class_name}.png")
        
        if not os.path.isfile(mask_path):
            mask = np.zeros((352, 352), dtype=np.uint8)
        else:
            mask = np.array(Image.open(mask_path).convert("L"))

        inputs = self.processor(
            text=[cls_name],
            images=[image],
            return_tensors="pt",
            padding=True 
        )
        
        mask_resized = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32)).unsqueeze(0)

        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": mask_tensor
        }

# ==============================================================================
# Loss function and training utilities
# ==============================================================================
def dice_loss(logits, targets, eps=1e-7):
    preds = torch.sigmoid(logits)
    num = 2 * (preds * targets).sum(dim=[2, 3])
    den = (preds + targets).sum(dim=[2, 3]) + eps
    return (1 - (num / den)).mean()

def train_model(learning_rate, dice_weight, batch_size, trial=None):
    """
    Train a single model with given hyperparameters.
    Returns validation loss for hyperparameter optimization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL)
    model = CLIPSegForImageSegmentation.from_pretrained(PRETRAINED_MODEL).to(device)

    # Custom collate_fn to handle variable-length text inputs
    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # Create datasets and dataloaders
    train_image_paths = sorted(glob(os.path.join(TRAIN_IMG_DIR, "*.tif")))
    val_image_paths = sorted(glob(os.path.join(VAL_IMG_DIR, "*.tif")))
    
    if not train_image_paths:
        raise RuntimeError(f"No training images found in {TRAIN_IMG_DIR}")
    if not val_image_paths:
        raise RuntimeError(f"No validation images found in {VAL_IMG_DIR}")

    train_dataset = FineTuneDataset(train_image_paths, TRAIN_MASK_DIR, CLASSES, processor)
    val_dataset = FineTuneDataset(val_image_paths, VAL_MASK_DIR, CLASSES, processor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss_total = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits.unsqueeze(1)

            bce = F.binary_cross_entropy_with_logits(logits, labels)
            dsc = dice_loss(logits, labels)
            loss = bce + dice_weight * dsc
            
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
        
        avg_train_loss = train_loss_total / len(train_loader)

        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits.unsqueeze(1)
                
                bce = F.binary_cross_entropy_with_logits(logits, labels)
                dsc = dice_loss(logits, labels)
                loss = bce + dice_weight * dsc
                val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Report to Optuna if in hyperparameter search mode
        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Only save model if not in hyperparameter search mode
            if trial is None:
                print(f"  ✨ Validation loss decreased. Saving model...")
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
                processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    return best_val_loss

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    """
    # Sample hyperparameters - laptop-friendly small search space
    learning_rate = trial.suggest_categorical("learning_rate", [1e-6, 5e-6, 1e-5])
    dice_weight = trial.suggest_categorical("dice_weight", [0.5, 0.8])
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    
    print(f"\nTrial {trial.number}: lr={learning_rate}, dice_weight={dice_weight}, batch_size={batch_size}")
    
    # Train model and return validation loss
    try:
        val_loss = train_model(learning_rate, dice_weight, batch_size, trial)
        return val_loss
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')

def run_hyperparameter_search():
    """
    Run hyperparameter search using Optuna.
    """
    print("🔍 Starting hyperparameter search...")
    print(f"Search space: lr={[1e-6, 5e-6, 1e-5]}, dice_weight={[0.5, 0.8]}, batch_size={[2, 4]}")
    print(f"Total trials: {N_TRIALS}")
    
    os.makedirs(OUTPUT_SEARCH_DIR, exist_ok=True)
    
    # Create study with pruning for efficiency
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Save results
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "search_date": datetime.now().isoformat(),
        "data_source": "augmented" if USE_AUGMENTED_DATA else "original",
        "trials": [
            {
                "number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state.name
            }
            for trial in study.trials
        ]
    }
    
    results_path = os.path.join(OUTPUT_SEARCH_DIR, "hyperparameter_search_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Hyperparameter search completed!")
    print(f"Best parameters: {study.best_params}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print(f"Results saved to: {results_path}")
    
    return study.best_params

def main():
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Using {'augmented' if USE_AUGMENTED_DATA else 'original'} training data")
    
    if USE_AUGMENTED_DATA:
        print("📊 Data strategy: Train and validation both from augmented data (no data leakage)")
    else:
        print("📊 Data strategy: Using original train/val split")
    
    if RUN_HYPERPARAMETER_SEARCH:
        # Run hyperparameter search
        best_params = run_hyperparameter_search()
        
        print(f"\n🚀 Training final model with best parameters...")
        # Train final model with best parameters
        final_val_loss = train_model(
            learning_rate=best_params["learning_rate"],
            dice_weight=best_params["dice_weight"],
            batch_size=best_params["batch_size"]
        )
        
        print(f"\n✅ Final model training complete!")
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Best model saved at: {os.path.join(OUTPUT_DIR, 'best_model')}")
        
    else:
        # Train with default parameters
        print("🚀 Training with default parameters...")
        val_loss = train_model(
            learning_rate=5e-6,  # Conservative learning rate
            dice_weight=0.8,     # Default dice weight
            batch_size=4         # Default batch size
        )
        print(f"\n✅ Training complete! Validation loss: {val_loss:.4f}")
        print(f"Model saved at: {os.path.join(OUTPUT_DIR, 'best_model')}")

if __name__ == "__main__":
    main()