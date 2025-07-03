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
# --- CORRECTED PATHS ---
# Base directory for all data prepared by the new data_augmentation.py script
PREPARED_DATA_DIR = "./data/Vaihingen/finetune_data/cv_prepared_data"
CV_FOLDS_DIR = os.path.join(PREPARED_DATA_DIR, "cv_folds")
ALL_AUGMENTED_DIR = os.path.join(PREPARED_DATA_DIR, "all_data_augmented")

# Output directories
OUTPUT_DIR = "./clipseg_finetuned_model_cv"
OUTPUT_SEARCH_DIR = "./clipseg_hyperparameter_search_cv"

# Class definitions from the augmentation script
CLASSES = [
    "impervious_surface", "building", "low_vegetation",
    "tree", "car", "background"
]

# Training hyperparameters
PRETRAINED_MODEL = "CIDAS/clipseg-rd64-refined"
NUM_EPOCHS = 3
PATIENCE = 5
NUM_WORKERS = 0

# Hyperparameter search settings
RUN_HYPERPARAMETER_SEARCH = True
N_TRIALS = 3
N_SPLITS = 5

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

        img_stem = Path(image_path).stem
        safe_class_name = cls_name.replace(" ", "_").replace("/", "_")
        mask_filename = f"{img_stem}_{safe_class_name}.png"
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        if not os.path.isfile(mask_path):
            mask = np.zeros((352, 352), dtype=np.uint8)
        else:
            mask = np.array(Image.open(mask_path).convert("L"))

        inputs = self.processor(text=[cls_name], images=[image], return_tensors="pt", padding=True)
        
        mask_resized = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32)).unsqueeze(0)

        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": mask_tensor
        }

# ==============================================================================
# Core Training and Evaluation Logic
# ==============================================================================
def dice_loss(logits, targets, eps=1e-7):
    preds = torch.sigmoid(logits)
    num = 2 * (preds * targets).sum(dim=[2, 3])
    den = (preds + targets).sum(dim=[2, 3]) + eps
    return (1 - (num / den)).mean()

def train_and_evaluate(model, processor, train_loader, val_loader, optimizer, scheduler, params, device, is_final_training=False):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.unsqueeze(1)

            bce = F.binary_cross_entropy_with_logits(logits, labels)
            dsc = dice_loss(logits, labels)
            loss = bce + params['dice_weight'] * dsc
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.unsqueeze(1)
                
                bce = F.binary_cross_entropy_with_logits(logits, labels)
                dsc = dice_loss(logits, labels)
                loss = bce + params['dice_weight'] * dsc
                val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Epoch {epoch}: Avg Val Loss = {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if is_final_training:
                print(f"  ✨ New best model found! Saving to {OUTPUT_DIR}/best_model")
                model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
                processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  ⚠️ Early stopping triggered after {epoch} epochs.")
                break
    
    return best_val_loss

# ==============================================================================
# Optuna Objective Function
# ==============================================================================
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        'dice_weight': trial.suggest_float("dice_weight", 0.3, 1.0),
        'batch_size': trial.suggest_categorical("batch_size", [2, 4]),
        'weight_decay': trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_losses = []

    for fold_num in range(1, N_SPLITS + 1):
        print(f"\n--- Starting Trial {trial.number}, Fold {fold_num}/{N_SPLITS} ---")
        
        fold_dir = os.path.join(CV_FOLDS_DIR, f"fold_{fold_num}")
        train_img_dir = os.path.join(fold_dir, "train_augmented/images")
        train_mask_dir = os.path.join(fold_dir, "train_augmented/masks")
        val_img_dir = os.path.join(fold_dir, "val_augmented/images")
        val_mask_dir = os.path.join(fold_dir, "val_augmented/masks")

        processor = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL)
        model = CLIPSegForImageSegmentation.from_pretrained(PRETRAINED_MODEL).to(device)
        
        train_image_paths = sorted(glob(os.path.join(train_img_dir, "*.tif")))
        val_image_paths = sorted(glob(os.path.join(val_img_dir, "*.tif")))
        
        train_dataset = FineTuneDataset(train_image_paths, train_mask_dir, CLASSES, processor)
        val_dataset = FineTuneDataset(val_image_paths, val_mask_dir, CLASSES, processor)

        def collate_fn(batch):
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
            attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
            return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

        optimizer = AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

        loss = train_and_evaluate(model, processor, train_loader, val_loader, optimizer, scheduler, params, device)
        fold_losses.append(loss)
        
        trial.report(loss, fold_num)
        if trial.should_prune():
            raise optuna.TrialPruned()

    avg_loss = np.mean(fold_losses)
    return avg_loss

# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SEARCH_DIR, exist_ok=True)

    if RUN_HYPERPARAMETER_SEARCH:
        print("🔍 Starting 5-Fold Cross-Validation Hyperparameter Search...")
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
        study.optimize(objective, n_trials=N_TRIALS)
        
        results = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "search_date": datetime.now().isoformat(),
            "trials": [{"value": t.value, "params": t.params} for t in study.trials]
        }
        results_path = os.path.join(OUTPUT_SEARCH_DIR, "cv_search_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Hyperparameter search completed! Best params: {study.best_params}")
        print(f"Results saved to {results_path}")
        best_params = study.best_params
    else:
        print("⏩ Skipping search, using default parameters.")
        best_params = {'learning_rate': 1e-5, 'dice_weight': 0.8, 'batch_size': 4, 'weight_decay': 0.01}

    print("\n🚀 Training final model on ALL augmented data with best parameters...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPSegProcessor.from_pretrained(PRETRAINED_MODEL)
    model = CLIPSegForImageSegmentation.from_pretrained(PRETRAINED_MODEL).to(device)

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Use the full augmented set for training
    final_train_img_dir = os.path.join(ALL_AUGMENTED_DIR, 'images')
    final_train_mask_dir = os.path.join(ALL_AUGMENTED_DIR, 'masks')
    final_train_paths = sorted(glob(os.path.join(final_train_img_dir, "*.tif")))
    final_train_dataset = FineTuneDataset(final_train_paths, final_train_mask_dir, CLASSES, processor)
    final_train_loader = DataLoader(final_train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Use one of the CV folds as a validation set for monitoring this final run
    print("Using fold 5 validation set for final training monitoring.")
    final_val_img_dir = os.path.join(CV_FOLDS_DIR, "fold_5/val_augmented/images")
    final_val_mask_dir = os.path.join(CV_FOLDS_DIR, "fold_5/val_augmented/masks")
    final_val_paths = sorted(glob(os.path.join(final_val_img_dir, "*.tif")))
    final_val_dataset = FineTuneDataset(final_val_paths, final_val_mask_dir, CLASSES, processor)
    final_val_loader = DataLoader(final_val_dataset, batch_size=best_params['batch_size'], shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    train_and_evaluate(model, processor, final_train_loader, final_val_loader, optimizer, scheduler, best_params, device, is_final_training=True)
    
    print(f"\n✅ Final model training complete! Model and processor saved to: {os.path.join(OUTPUT_DIR, 'best_model')}")

if __name__ == "__main__":
    main()