# clipseg_finetune_module.py
# Author: [Your Name]
# Description: Fine-tuning module for CLIPSeg using Optuna for hyperparameter tuning

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
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import albumentations as A
import cv2
import optuna


class ClipSegFineTuner:
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 classes: list,
                 output_dir: str,
                 pretrained_model_name: str = "CIDAS/clipseg-rd64-refined",
                 val_split: float = 0.2,
                 random_seed: int = 42,
                 device: str = None,
                 batch_size: int = 8,
                 num_epochs: int = 30,
                 patience: int = 3,
                 lr_patience: int = 2):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.classes = classes
        self.output_dir = output_dir
        self.pretrained_model_name = pretrained_model_name
        self.val_split = val_split
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.lr_patience = lr_patience

        os.makedirs(output_dir, exist_ok=True)

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = CLIPSegProcessor.from_pretrained(self.pretrained_model_name)
        self.resize_transform = A.Compose([A.Resize(352, 352)])

    @staticmethod
    def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps=1e-7):
        preds = torch.sigmoid(logits)
        num = 2 * (preds * targets).sum(dim=[2, 3])
        den = (preds + targets).sum(dim=[2, 3]) + eps
        return (1 - (num / den)).mean()

    class ClipSegFineTuneDataset(Dataset):
        def __init__(self, image_paths, mask_dir, classes, processor, augment=None):
            self.image_paths = image_paths
            self.mask_dir = mask_dir
            self.classes = classes
            self.processor = processor
            self.augment = augment
            self.n_classes = len(classes)

        def __len__(self):
            return len(self.image_paths) * self.n_classes

        def __getitem__(self, idx):
            img_idx = idx // self.n_classes
            cls_idx = idx % self.n_classes

            image_path = self.image_paths[img_idx]
            image = np.array(Image.open(image_path).convert("RGB"))
            h, w = image.shape[:2]

            base_fn = os.path.splitext(os.path.basename(image_path))[0]
            cls_name = self.classes[cls_idx]
            mask_name = f"{base_fn}_{cls_name}.png"
            mask_path = os.path.join(self.mask_dir, mask_name)
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            mask = np.array(Image.open(mask_path).convert("L"))
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            resized = A.Compose([A.Resize(352, 352)])(image=image)
            image_resized = resized["image"]
            mask_resized = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)
            mask_resized = (mask_resized > 127).astype(np.uint8)

            pil_img = Image.fromarray(image_resized)
            encoding = self.processor(
                images=[pil_img],
                text=[cls_name],
                return_tensors="pt",
                padding=True
            )

            return {
                "pixel_values": encoding.pixel_values.squeeze(0),
                "input_ids": encoding.input_ids.squeeze(0),
                "attention_mask": encoding.attention_mask.squeeze(0),
                "labels": torch.from_numpy(mask_resized.astype(np.float32)).unsqueeze(0)
            }

    def collate_fn(self, batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
        labels = torch.stack([item["labels"] for item in batch], dim=0)
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)

        return {
            "pixel_values": pixel_values.to(self.device),
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device)
        }

    def load_data(self):
        all_image_paths = sorted(glob(os.path.join(self.image_dir, "*.png")))
        if not all_image_paths:
            raise RuntimeError(f"No .png files found in: {self.image_dir}")
        random.seed(self.random_seed)
        random.shuffle(all_image_paths)
        num_val = int(len(all_image_paths) * self.val_split)
        return all_image_paths[num_val:], all_image_paths[:num_val]

    def create_dataloaders(self, train_paths, val_paths):
        train_ds = self.ClipSegFineTuneDataset(train_paths, self.mask_dir, self.classes, self.processor)
        val_ds = self.ClipSegFineTuneDataset(val_paths, self.mask_dir, self.classes, self.processor)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=2)
        return train_loader, val_loader

    def train_one_epoch(self, model, loader, optimizer, dice_weight):
        model.train()
        train_losses = []
        for batch in tqdm(loader, desc="Training"):
            optimizer.zero_grad()
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits.unsqueeze(1)
            bce = F.binary_cross_entropy_with_logits(logits, batch["labels"])
            dsc = self.dice_loss(logits, batch["labels"])
            loss = bce + dice_weight * dsc
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        return np.mean(train_losses)

    def evaluate(self, model, loader, dice_weight):
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                logits = outputs.logits.unsqueeze(1)
                bce = F.binary_cross_entropy_with_logits(logits, batch["labels"])
                dsc = self.dice_loss(logits, batch["labels"])
                val_losses.append(bce + dice_weight * dsc)
        return np.mean(val_losses)

    def save_model(self, model, processor, directory):
        os.makedirs(directory, exist_ok=True)
        model.save_pretrained(directory)
        processor.save_pretrained(directory)

    def fine_tune(self, lr, weight_decay, dice_weight):
        train_paths, val_paths = self.load_data()
        train_loader, val_loader = self.create_dataloaders(train_paths, val_paths)

        model = CLIPSegForImageSegmentation.from_pretrained(self.pretrained_model_name).to(self.device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=self.lr_patience)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_one_epoch(model, train_loader, optimizer, dice_weight)
            val_loss = self.evaluate(model, val_loader, dice_weight)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(model, self.processor, os.path.join(self.output_dir, "best_model"))
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        self.save_model(model, self.processor, os.path.join(self.output_dir, "final_model"))
        print(f"\n✅ Fine-tuning completed. Final model saved to: {self.output_dir}/final_model")


if __name__ == "__main__":
    # Example usage
    IMAGE_DIR = "./augmented_tiles/images"
    MASK_DIR = "./augmented_tiles/masks_augmented"
    CLASSES = ["tree", "road", "built_area", "building"]
    OUTPUT_DIR = "./clipseg_finetune_optuna"

    tuner = ClipSegFineTuner(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        classes=CLASSES,
        output_dir=OUTPUT_DIR
    )

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
        dw = trial.suggest_float("dice_weight", 0.1, 1.0)
        
        val_loss = tuner.fine_tune(lr=lr, weight_decay=wd, dice_weight=dw, return_val_only=True)
        
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("Best Params:", study.best_params)
    # Use best parameters for final fine-tuning
    # Active when you want to fine-tune with the best parameters
    # tuner.fine_tune(
    #     lr=study.best_params["lr"],
    #     weight_decay=study.best_params["weight_decay"],
    #     dice_weight=study.best_params["dice_weight"]
    # )
