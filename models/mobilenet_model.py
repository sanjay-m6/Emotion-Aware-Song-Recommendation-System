"""
Transfer learning model using MobileNetV2 for emotion detection.

Leverages pretrained ImageNet weights and fine-tunes on AffectNet 8-class task.
Supports backbone freezing/unfreezing for progressive unfreezing strategy.
Input: RGB 96x96 | Output: 8 emotion logits
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import LABEL_TO_EMOTION


class MobileNetV2Model(nn.Module):
    """
    Transfer learning emotion detector using pretrained MobileNetV2.
    
    Replaces classifier head:
    - Freezes backbone initially (can be unfrozen via unfreeze_backbone())
    - Custom classifier: 1280 → 512 → 8
    
    Supports two training phases:
    1. Frozen backbone (learns classifier only)
    2. Unfrozen backbone (fine-tunes entire network)
    
    Attributes:
        backbone: MobileNetV2 feature extractor (pretrained on ImageNet)
        classifier: Custom FC head (1280 → 512 → 8)
        backbone_frozen: Boolean tracking freeze state
    """
    
    def __init__(self, pretrained: bool = True) -> None:
        """
        Initialize MobileNetV2 model.
        
        Args:
            pretrained: Whether to load ImageNet pretrained weights (default True)
        """
        super(MobileNetV2Model, self).__init__()
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Extract backbone (features) and keep input layer for RGB
        self.backbone = mobilenet.features
        
        # Get output channels from last conv layer
        num_input_features = mobilenet.classifier[0].in_features  # 1280
        
        # Replace classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_input_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 8)
        )
        
        # Average pooling for spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Track freeze state
        self.backbone_frozen = False
    
    def freeze_backbone(self) -> None:
        """
        Freeze all backbone layers (feature extractor).
        
        Only the classifier head is trainable. Useful for initial training
        when only learning the task-specific head.
        
        Example:
            model = MobileNetV2Model()
            model.freeze_backbone()
            # Only classifier parameters will have gradients
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone_frozen = True
    
    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone layers (feature extractor).
        
        All network weights become trainable. Useful for fine-tuning
        after initial training on the classifier head.
        
        Example:
            model = MobileNetV2Model()
            model.freeze_backbone()
            # ... train for a few epochs ...
            model.unfreeze_backbone()
            # ... continue training with lower learning rate ...
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        self.backbone_frozen = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MobileNetV2.
        
        Args:
            x: Input tensor of shape (batch, 3, 96, 96)
            
        Returns:
            Logits tensor of shape (batch, 8)
        """
        # Feature extraction via backbone
        x = self.backbone(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        
        # Flatten
        x = self.flatten(x)
        
        # Classification head
        x = self.classifier(x)
        
        return x
    
    def get_confidence(self, logits: torch.Tensor) -> Tuple[str, float]:
        """
        Get predicted emotion and confidence from logits.
        
        Args:
            logits: Raw output tensor of shape (batch, 8) or (8,) for single sample
            
        Returns:
            Tuple of (emotion_name: str, confidence: float in [0.0, 1.0])
            
        Example:
            logits = model(images)  # (batch, 8)
            emotion, conf = model.get_confidence(logits[0])  # "happy", 0.92
        """
        # Handle batch or single sample
        if logits.dim() > 1:
            logits = logits[0] if logits.shape[0] == 1 else logits
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get highest confidence emotion
        confidence, pred_idx = torch.max(probs, dim=-1)
        confidence_float = float(confidence.item())
        emotion_name = LABEL_TO_EMOTION[int(pred_idx.item())]
        
        return emotion_name, confidence_float
    
    def get_all_scores(self, logits: torch.Tensor) -> Dict[str, float]:
        """
        Get probability scores for all 8 emotions from logits.
        
        Args:
            logits: Raw output tensor of shape (batch, 8) or (8,)
            
        Returns:
            Dictionary mapping emotion names to probabilities {emotion: float}
            
        Example:
            logits = model(images)
            scores = model.get_all_scores(logits[0])
            # {"anger": 0.01, "surprise": 0.02, ..., "disgust": 0.45}
        """
        # Handle batch or single sample
        if logits.dim() > 1:
            logits = logits[0] if logits.shape[0] == 1 else logits
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Build dict of emotion → probability
        scores = {
            LABEL_TO_EMOTION[i]: float(probs[i].item())
            for i in range(8)
        }
        
        return scores
