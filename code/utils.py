import os
import random
import torch
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import cv2
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
import json
import albumentations as A

from config import *

# ==============================================================================
# Dataset Classes
# ==============================================================================
class FineTuneDataset(Dataset):
    """CLIPSeg fine-tuning dataset"""
    def __init__(self, image_paths: List[str], mask_dir: str, classes: List[str], processor):
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
        mask = np.array(Image.open(mask_path).convert("L")) if os.path.isfile(mask_path) else np.zeros((352, 352), dtype=np.uint8)
        inputs = self.processor(text=[cls_name], images=[image], return_tensors="pt", padding=True)
        mask_resized = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32)).unsqueeze(0)
        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": mask_tensor,
            "class_idx": cls_idx
        }

# ==============================================================================
# Data Loading and Model Creation
# ==============================================================================
def collate_fn(batch, processor):
    """Data batch processing function"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    class_indices = torch.tensor([item["class_idx"] for item in batch], dtype=torch.long)
    return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "class_indices": class_indices}

def create_data_loader(image_paths: List[str], mask_dir: str, classes: List[str], processor, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create data loader"""
    dataset = FineTuneDataset(image_paths, mask_dir, classes, processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS, collate_fn=lambda b: collate_fn(b, processor))

def create_model_and_optimizer(learning_rate: float, device):
    """Create model and optimizer"""
    model = CLIPSegForImageSegmentation.from_pretrained(PRETRAINED_MODEL).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    return model, optimizer

# ==============================================================================
# Loss and Evaluation
# ==============================================================================
def dice_loss(logits, targets, eps=1e-7):
    """Dice loss function"""
    preds = torch.sigmoid(logits)
    num = 2 * (preds * targets).sum(dim=[2, 3])
    den = preds.sum(dim=[2, 3]) + targets.sum(dim=[2, 3]) + eps
    return (1 - (num / den)).mean()

def calculate_combined_loss(logits, labels, dice_weight: float = 0.8):
    """Calculate combined loss (BCE + Dice)"""
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    dsc = dice_loss(logits, labels)
    return bce + dice_weight * dsc

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """Calculate IoU metric"""
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = (true_mask > 0).astype(np.uint8)
    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()
    return (intersection / union) if union > 0 else 1.0

# ==============================================================================
# Training and Evaluation Core Loops
# ==============================================================================
def train_one_epoch(model, train_loader, optimizer, device, dice_weight: float, desc_str: str = "Training"):
    """Train one epoch."""
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=desc_str, leave=False):
        optimizer.zero_grad()
        outputs = model(pixel_values=batch["pixel_values"].to(device), input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        logits = outputs.logits.unsqueeze(1)
        labels = batch["labels"].to(device)
        loss = calculate_combined_loss(logits, labels, dice_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, device, dice_weight: float, desc_str: str = "Evaluating"):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_ious = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=desc_str, leave=False):
            outputs = model(pixel_values=batch["pixel_values"].to(device), input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            logits = outputs.logits.unsqueeze(1)
            labels = batch["labels"].to(device)
            loss = calculate_combined_loss(logits, labels, dice_weight)
            total_loss += loss.item()
            predictions = torch.sigmoid(logits)
            iou = calculate_iou(predictions.cpu().numpy(), labels.cpu().numpy())
            all_ious.append(iou)
    avg_loss = total_loss / len(val_loader)
    avg_iou = np.mean(all_ious) if all_ious else 0.0
    return avg_loss, avg_iou

# ==============================================================================
# Other Public Functions
# ==============================================================================
def get_augmentation_transform():
    """Get data augmentation transforms based on configuration file"""
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
    return A.Compose(transforms)

def save_model(model, processor, save_path: Path, metadata: Dict = None):
    """Save model, processor and metadata."""
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    if metadata:
        with open(save_path / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

def set_seed(seed=RANDOM_SEED):
    """Set random seed to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    """
    [Fixed] Ensure all necessary directories exist.
    Now only creates top-level directories explicitly defined in config.py.
    """
    dirs_to_create = [
        DATA_DIR, 
        MODELS_DIR, 
        OUTPUT_DIR,
        FINETUNED_MODEL_DIR, 
        HYPERPARAMETER_SEARCH_DIR
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_device():
    """Get available device"""
    if DEVICE == 'auto':
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(DEVICE)

def get_hyperparameter_from_trial(trial, param_name):
    """
    [Fixed] Get hyperparameters from Optuna trial.
    Now reads search space from SEARCH_CONFIG['space'].
    """
    # Get parameter definition from new configuration structure
    param_config = SEARCH_CONFIG['space'][param_name]
    
    if param_config['type'] == 'loguniform':
        return trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
    elif param_config['type'] == 'uniform':
        return trial.suggest_float(param_name, param_config['low'], param_config['high'])
    elif param_config['type'] == 'categorical':
        return trial.suggest_categorical(param_name, param_config['choices'])
    raise ValueError(f"Unknown parameter type: {param_config['type']}")

def save_config(config_dict, filename):
    """Save configuration to JSON file"""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

# ==============================================================================
# Training Logging Functionality
# ==============================================================================
def create_training_logger():
    """Create training logger"""
    return {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'timestamps': [],
        'best_epoch': None,
        'best_loss': float('inf'),
        'metadata': {}
    }

def update_training_log(logger, epoch, train_loss, val_loss=None, lr=None):
    """Update training log"""
    from datetime import datetime
    
    logger['epochs'].append(epoch)
    logger['train_losses'].append(float(train_loss))
    
    if val_loss is not None:
        logger['val_losses'].append(float(val_loss))
        # Update best record
        if val_loss < logger['best_loss']:
            logger['best_loss'] = float(val_loss)
            logger['best_epoch'] = epoch
    
    if lr is not None:
        logger['learning_rates'].append(float(lr))
    
    logger['timestamps'].append(datetime.now().isoformat())
    
def save_training_log(logger, save_path):
    """Save training log to JSON file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add summary information
    if logger['epochs']:
        logger['metadata']['total_epochs'] = logger['epochs'][-1]
        logger['metadata']['final_train_loss'] = logger['train_losses'][-1]
        if logger['val_losses']:
            logger['metadata']['final_val_loss'] = logger['val_losses'][-1]
        logger['metadata']['best_epoch'] = logger['best_epoch']
        logger['metadata']['best_loss'] = logger['best_loss']
    
    with open(save_path, 'w') as f:
        json.dump(logger, f, indent=2)
    
    print(f"📊 Training log saved to: {save_path}")

def get_current_lr(optimizer):
    """Get current learning rate"""
    return optimizer.param_groups[0]['lr']

def load_model(model_path: Path, device):
    """Load trained model, processor and metadata
    
    Args:
        model_path: Model directory path
        device: Computation device
        
    Returns:
        model: Loaded model
        processor: CLIPSeg processor
        metadata: Training metadata (if exists)
    """
    # Load processor
    processor = CLIPSegProcessor.from_pretrained(model_path)
    
    # Load model
    model = CLIPSegForImageSegmentation.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Try to load metadata
    metadata = None
    metadata_path = model_path / "training_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, processor, metadata

# ==============================================================================
# Checkpoint Management Functionality
# ==============================================================================
class CheckpointManager:
    """Manage training checkpoint saving and loading"""
    
    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 3):
        """
        Args:
            checkpoint_dir: checkpoint save directory
            max_checkpoints: maximum number of checkpoints to keep (excluding best and latest)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
    def save_checkpoint(self, state: Dict, epoch: int, is_best: bool = False):
        """Save checkpoint
        
        Args:
            state: dictionary containing model state
            epoch: current epoch
            is_best: whether this is the best model
        """
        # Save latest checkpoint (using temporary file to avoid corruption)
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        temp_path = latest_path.with_suffix('.tmp')
        torch.save(state, temp_path)
        temp_path.rename(latest_path)
        
        # If it's the best model, save an additional copy
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            temp_path = best_path.with_suffix('.tmp')
            torch.save(state, temp_path)
            temp_path.rename(best_path)
            print(f"💾 Saved best model checkpoint (epoch {epoch})")
        
        # Periodically saved checkpoint
        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        temp_path = epoch_path.with_suffix('.tmp')
        torch.save(state, temp_path)
        temp_path.rename(epoch_path)
        print(f"💾 Saved checkpoint: {epoch_path.name}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self):
        """Delete old checkpoints, keeping only the most recent ones"""
        # Get all epoch checkpoints (excluding best and latest)
        epoch_checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        # If exceeding maximum number, delete the oldest ones
        while len(epoch_checkpoints) > self.max_checkpoints:
            oldest = epoch_checkpoints.pop(0)
            oldest.unlink()
            print(f"🗑️  Deleted old checkpoint: {oldest.name}")
    
    def load_checkpoint(self, checkpoint_name: str = "latest"):
        """Load checkpoint
        
        Args:
            checkpoint_name: name of checkpoint to load ('latest', 'best', or 'epoch_N')
            
        Returns:
            checkpoint dictionary, or None if not found
        """
        if checkpoint_name == "latest":
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        elif checkpoint_name == "best":
            checkpoint_path = self.checkpoint_dir / "checkpoint_best.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_name}.pt"
            
        if checkpoint_path.exists():
            print(f"📂 Loading checkpoint: {checkpoint_path.name}")
            # PyTorch 2.6+ requires weights_only=False to load checkpoints containing non-tensor objects
            return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        else:
            return None
            
    def get_available_checkpoints(self):
        """Get list of all available checkpoints"""
        checkpoints = []
        
        # Check special checkpoints
        for special in ["latest", "best"]:
            path = self.checkpoint_dir / f"checkpoint_{special}.pt"
            if path.exists():
                checkpoints.append(special)
                
        # Check epoch checkpoints
        epoch_checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        for cp in epoch_checkpoints:
            epoch_num = cp.stem.split('_')[-1]
            checkpoints.append(f"epoch_{epoch_num}")
            
        return checkpoints

def save_checkpoint(state: Dict, checkpoint_path: Path):
    """Save checkpoint (simple version for backward compatibility)"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use temporary file to avoid interruption during writing
    temp_path = checkpoint_path.with_suffix('.tmp')
    torch.save(state, temp_path)
    temp_path.rename(checkpoint_path)
    
def load_checkpoint(checkpoint_path: Path, model, optimizer=None, device='cpu'):
    """Load checkpoint (simple version for backward compatibility)
    
    Returns:
        loaded epoch number, returns 0 if no checkpoint found
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        return 0
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state (if provided)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # Return starting epoch
    return checkpoint.get('epoch', 0) + 1
