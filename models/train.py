"""
Unified training script for CustomCNN and MobileNetV2 emotion detectors.

Supports:
- Class-imbalanced training via WeightedRandomSampler
- Learning rate scheduling (CosineAnnealingLR)
- Progressive unfreezing for MobileNetV2
- Early stopping on validation accuracy
- Checkpoint saving with best model persistence
- Training history logging to JSON
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset import AffectNetDataset, TRAIN_TRANSFORMS, VAL_TRANSFORMS
from utils.constants import LABEL_TO_EMOTION, EMOTION_NAMES
from models.custom_cnn import CustomCNN
from models.mobilenet_model import MobileNetV2Model


class Trainer:
    """
    Unified trainer for emotion detection models.
    
    Handles:
    - Model initialization and device selection
    - Training loop with validation
    - Early stopping and checkpoint saving
    - Learning rate scheduling
    - Progressive unfreezing (MobileNetV2)
    - History logging
    """
    
    def __init__(
        self,
        model_type: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str = None
    ) -> None:
        """
        Initialize Trainer.
        
        Args:
            model_type: "custom_cnn" or "mobilenet"
            epochs: Number of training epochs
            batch_size: Batch size for training/validation
            learning_rate: Initial learning rate
            device: Device string ("cuda", "mps", "cpu"). Auto-detects if None
        """
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        print(f"\n✅ Using device: {self.device}")
        
        # Initialize model
        if model_type == "custom_cnn":
            self.model = CustomCNN()
        elif model_type == "mobilenet":
            self.model = MobileNetV2Model(pretrained=True)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.model = self.model.to(self.device)
        
        # Loss function (sampler handles imbalance)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer
        weight_decay = 1e-4
        if model_type == "custom_cnn":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:  # mobilenet
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Early stopping
        self.patience = 8
        self.patience_counter = 0
        self.best_val_accuracy = 0.0
        
        # Checkpoint path
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / f"{model_type}_best.pth"
        
        # History tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "epochs": []
        }
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load AffectNet from HuggingFace and wrap with Dataset.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        print("\n📊 Loading AffectNet dataset...")
        affectnet = load_dataset("Mauregato/affectnet_short")
        train_hf = affectnet["train"]
        val_hf = affectnet["val"]
        
        # Wrap with AffectNetDataset
        train_dataset = AffectNetDataset(train_hf, transform=TRAIN_TRANSFORMS)
        val_dataset = AffectNetDataset(val_hf, transform=VAL_TRANSFORMS)
        
        # Create dataloaders
        train_sampler = train_dataset.get_sampler()
        # MPS (Apple Silicon) doesn't support pin_memory or multi-workers
        _num_workers = 0 if str(self.device) in ("mps", "cpu") else 4
        _pin_memory  = True if str(self.device) == "cuda" else False

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=_num_workers,
            pin_memory=_pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=_num_workers,
            pin_memory=_pin_memory
        )
        
        print(f"  Train: {len(train_dataset):,} samples")
        print(f"  Val:   {len(val_dataset):,} samples")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Execute one training epoch.
        
        Args:
            train_loader: DataLoader for training split
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        with tqdm(train_loader, desc=f"Train Epoch", leave=False) as pbar:
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  
                self.optimizer.step()
                
                total_loss += loss.item() * images.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict[int, float]]:
        """
        Validate on validation split.
        
        Args:
            val_loader: DataLoader for validation split
            
        Returns:
            Tuple of (val_loss, val_accuracy, per_class_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(8)}
        class_total = {i: 0 for i in range(8)}
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Val Epoch", leave=False) as pbar:
                for images, labels in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    
                    total_loss += loss.item() * images.size(0)
                    
                    # Accuracy
                    _, preds = torch.max(logits, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    # Per-class accuracy
                    for i in range(8):
                        mask = labels == i
                        class_total[i] += mask.sum().item()
                        class_correct[i] += (preds[mask] == labels[mask]).sum().item()
                    
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(val_loader.dataset)
        accuracy = correct / total
        
        per_class_accuracy = {
            i: (class_correct[i] / class_total[i]) if class_total[i] > 0 else 0.0
            for i in range(8)
        }
        
        return avg_loss, accuracy, per_class_accuracy
    
    def save_checkpoint(self, epoch: int, val_accuracy: float) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_accuracy: Validation accuracy achieved
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else 0.0,
            "label_map": LABEL_TO_EMOTION
        }
        
        torch.save(checkpoint, self.checkpoint_path)
        print(f"  ✅ Checkpoint saved: {self.checkpoint_path}")
    
    def train(self) -> Dict:
        """
        Execute full training loop.
        
        Handles:
        - Dataset loading
        - Progressive unfreezing (MobileNetV2)
        - Early stopping
        - Checkpoint management
        
        Returns:
            Training history dictionary
        """
        train_loader, val_loader = self.load_data()
        
        print(f"\n{'='*70}")
        print(f"Training {self.model_type.upper()}")
        print(f"{'='*70}")
        print(f"  Model:   {self.model_type}")
        print(f"  Epochs:  {self.epochs}")
        print(f"  Batch:   {self.batch_size}")
        print(f"  LR:      {self.learning_rate}")
        print(f"{'='*70}\n")
        
        for epoch in range(self.epochs):
            # MobileNetV2: unfreeze backbone at epoch 8
            if self.model_type == "mobilenet":
                if epoch == 0:
                    self.model.freeze_backbone()
                    print(f"Epoch {epoch}: Backbone FROZEN (training classifier only)")
                elif epoch == 12:
                    self.model.unfreeze_backbone()
                    # Reduce learning rate for fine-tuning
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = 3e-5
                    print(f"Epoch {epoch}: Backbone UNFROZEN (fine-tuning all layers, lr=1e-4)")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_accuracy, per_class_acc = self.validate(val_loader)
            
            # Learning rate update
            self.scheduler.step()
            
            # Logging
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)
            self.history["epochs"].append(epoch)
            
            # Find worst emotion class
            worst_emotion_idx = min(per_class_acc, key=per_class_acc.get)
            worst_emotion = LABEL_TO_EMOTION[worst_emotion_idx]
            worst_acc = per_class_acc[worst_emotion_idx]
            
            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.4f} | "
                  f"Worst: {worst_emotion} ({worst_acc:.4f})")
            
            # Early stopping
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_accuracy)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\n⚠️  Early stopping at epoch {epoch+1} (patience exceeded)")
                    break
        
        # Save history to JSON
        history_path = self.checkpoint_dir / f"{self.model_type}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ Training complete!")
        print(f"  Best Val Accuracy: {self.best_val_accuracy:.4f}")
        print(f"  Best Epoch: {np.argmax(self.history['val_accuracy'])}")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  History: {history_path}")
        print(f"{'='*70}\n")
        
        return self.history


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train emotion detection model on AffectNet"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["custom_cnn", "mobilenet"],
        required=True,
        help="Model architecture to train"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs (default 60)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default 64)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (uses model defaults if not specified)"
    )
    
    args = parser.parse_args()
    
    # Use model-specific defaults if lr not provided
    if args.lr is None:
        if args.model == "custom_cnn":
            args.lr = 1e-3
        else:  # mobilenet
            args.lr = 3e-4
    
    # Create trainer and run
    trainer = Trainer(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
