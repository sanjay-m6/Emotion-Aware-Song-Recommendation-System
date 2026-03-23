"""
PyTorch Dataset wrapper for AffectNet with class-weighted sampling.

Provides:
- AffectNetDataset: wraps HuggingFace AffectNet split
- get_sampler(): WeightedRandomSampler for imbalance handling
- Supports separate transforms for train/val splits
"""

import sys
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import CLASS_WEIGHTS


# Standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transforms for training split (with augmentation)
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Transforms for validation split (no augmentation)
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


class AffectNetDataset(Dataset):
    """
    PyTorch Dataset wrapper for AffectNet HuggingFace dataset.
    
    Wraps a HuggingFace dataset split and provides:
    - Image preprocessing and augmentation via torchvision transforms
    - Per-sample class weights for imbalance handling
    - WeightedRandomSampler for balanced mini-batches
    
    Attributes:
        hf_split: HuggingFace Dataset object for one split
        transform: Callable to preprocess PIL images
        sample_weights: Per-sample weights (precomputed from CLASS_WEIGHTS)
    """
    
    def __init__(self, hf_split, transform: Optional[Callable] = None) -> None:
        """
        Initialize AffectNetDataset.
        
        Args:
            hf_split: HuggingFace Dataset object (e.g., affectnet["train"])
            transform: Optional torchvision.transforms.Compose object.
                      If None, uses identity transform.
        """
        self.hf_split = hf_split
        self.transform = transform if transform is not None else transforms.ToTensor()
        
        # Precompute per-sample weights using CLASS_WEIGHTS
        self.sample_weights = []
        for idx in range(len(self.hf_split)):
            label = self.hf_split[idx]["label"]
            weight = CLASS_WEIGHTS.get(label, 1.0)
            self.sample_weights.append(weight)
    
    def get_sampler(self) -> WeightedRandomSampler:
        """
        Create and return a WeightedRandomSampler.
        
        Enables balanced sampling across imbalanced emotion classes.
        Use this sampler in DataLoader for training split:
        
        Example:
            dataset = AffectNetDataset(hf_split, transform=TRAIN_TRANSFORMS)
            sampler = dataset.get_sampler()
            loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=2)
        
        Returns:
            WeightedRandomSampler with replacement, num_samples=len(self)
        """
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self),
            replacement=True
        )
    
    def __len__(self) -> int:
        """Return total number of samples in this split."""
        return len(self.hf_split)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample by index.
        
        Handles corrupted images by retrying with linear interpolation.
        
        Args:
            idx: Sample index (0-indexed)
            
        Returns:
            Tuple of (image_tensor, emotion_label_int) where:
            - image_tensor: RGB tensor of shape (3, 96, 96), normalized
            - emotion_label_int: emotion class label (0-7)
            
        Raises:
            RuntimeError: If image cannot be loaded after retry
        """
        try:
            sample = self.hf_split[idx]
            
            # Get PIL image (already RGB from AffectNet)
            pil_image = sample["image"]
            
            # Ensure RGB (in case of edge cases)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            
            # Apply transforms
            tensor_image = self.transform(pil_image)
            
            # Get label as int
            label = int(sample["label"])
            
            return tensor_image, label
            
        except Exception as e:
            # BUG FIX: Retry with alternative resizing method for corrupted images
            try:
                sample = self.hf_split[idx]
                pil_image = sample["image"]
                
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                
                # Fallback: resize with LANCZOS instead of stored default
                pil_image = pil_image.resize((96, 96), Image.LANCZOS)
                tensor_image = torch.from_numpy(np.array(pil_image)).float()
                tensor_image = tensor_image.permute(2, 0, 1) / 255.0
                
                # Apply normalization
                mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
                std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
                tensor_image = (tensor_image - mean) / std
                
                label = int(sample["label"])
                return tensor_image, label
                
            except Exception as retry_err:
                raise RuntimeError(f"Failed to load image at index {idx}: {retry_err}")
