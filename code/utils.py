"""
工具函数模块 - Stanford UGVR CV项目
V2版：修正了与新config.py的兼容性问题。
"""

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

# 导入配置
# 这里的 '*' 会导入新版config.py中的所有全局变量和配置字典
from config import *

# ==============================================================================
# 数据集类 (无变化)
# ==============================================================================
class FineTuneDataset(Dataset):
    """CLIPSeg微调数据集"""
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
            "labels": mask_tensor
        }

# ==============================================================================
# 数据加载与模型创建 (无变化)
# ==============================================================================
def collate_fn(batch, processor):
    """数据批处理函数"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def create_data_loader(image_paths: List[str], mask_dir: str, classes: List[str], processor, batch_size: int, shuffle: bool = True) -> DataLoader:
    """创建数据加载器"""
    dataset = FineTuneDataset(image_paths, mask_dir, classes, processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS, collate_fn=lambda b: collate_fn(b, processor))

def create_model_and_optimizer(learning_rate: float, device):
    """创建模型和优化器"""
    model = CLIPSegForImageSegmentation.from_pretrained(PRETRAINED_MODEL).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    return model, optimizer

# ==============================================================================
# 损失与评估 (无变化)
# ==============================================================================
def dice_loss(logits, targets, eps=1e-7):
    """Dice损失函数"""
    preds = torch.sigmoid(logits)
    num = 2 * (preds * targets).sum(dim=[2, 3])
    den = preds.sum(dim=[2, 3]) + targets.sum(dim=[2, 3]) + eps
    return (1 - (num / den)).mean()

def calculate_combined_loss(logits, labels, dice_weight: float = 0.8):
    """计算组合损失（BCE + Dice）"""
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    dsc = dice_loss(logits, labels)
    return bce + dice_weight * dsc

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """计算IoU指标"""
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = (true_mask > 0).astype(np.uint8)
    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()
    return (intersection / union) if union > 0 else 1.0

# ==============================================================================
# 训练与评估核心循环 (无变化)
# ==============================================================================
def train_one_epoch(model, train_loader, optimizer, device, dice_weight: float, desc_str: str = "训练中"):
    """训练一个epoch。"""
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

def evaluate_model(model, val_loader, device, dice_weight: float, desc_str: str = "评估中"):
    """评估模型。"""
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
# 其他公共函数 (已修正)
# ==============================================================================
def save_model(model, processor, save_path: Path, metadata: Dict = None):
    """保存模型、处理器和元数据。"""
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    if metadata:
        with open(save_path / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

def set_seed(seed=RANDOM_SEED):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    """
    【已修正】确保所有必要的目录存在。
    现在只创建在config.py中明确定义的顶级目录。
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
    """获取可用的设备"""
    if DEVICE == 'auto':
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(DEVICE)

def get_hyperparameter_from_trial(trial, param_name):
    """
    【已修正】从Optuna trial中获取超参数。
    现在从 SEARCH_CONFIG['space'] 中读取搜索空间。
    """
    # 从新的配置结构中获取参数定义
    param_config = SEARCH_CONFIG['space'][param_name]
    
    if param_config['type'] == 'loguniform':
        return trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
    elif param_config['type'] == 'uniform':
        return trial.suggest_float(param_name, param_config['low'], param_config['high'])
    elif param_config['type'] == 'categorical':
        return trial.suggest_categorical(param_name, param_config['choices'])
    raise ValueError(f"未知的参数类型: {param_config['type']}")

def save_config(config_dict, filename):
    """保存配置到JSON文件"""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

# ==============================================================================
# 训练日志记录功能
# ==============================================================================
def create_training_logger():
    """创建训练日志记录器"""
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
    """更新训练日志"""
    from datetime import datetime
    
    logger['epochs'].append(epoch)
    logger['train_losses'].append(float(train_loss))
    
    if val_loss is not None:
        logger['val_losses'].append(float(val_loss))
        # 更新最佳记录
        if val_loss < logger['best_loss']:
            logger['best_loss'] = float(val_loss)
            logger['best_epoch'] = epoch
    
    if lr is not None:
        logger['learning_rates'].append(float(lr))
    
    logger['timestamps'].append(datetime.now().isoformat())
    
def save_training_log(logger, save_path):
    """保存训练日志到JSON文件"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 添加总结信息
    if logger['epochs']:
        logger['metadata']['total_epochs'] = logger['epochs'][-1]
        logger['metadata']['final_train_loss'] = logger['train_losses'][-1]
        if logger['val_losses']:
            logger['metadata']['final_val_loss'] = logger['val_losses'][-1]
        logger['metadata']['best_epoch'] = logger['best_epoch']
        logger['metadata']['best_loss'] = logger['best_loss']
    
    with open(save_path, 'w') as f:
        json.dump(logger, f, indent=2)
    
    print(f"📊 训练日志已保存到: {save_path}")

def get_current_lr(optimizer):
    """获取当前学习率"""
    return optimizer.param_groups[0]['lr']
