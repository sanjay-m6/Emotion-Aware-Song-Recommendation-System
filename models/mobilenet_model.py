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
        backbone: Full MobileNetV2 with replaced classifier head
        backbone_frozen: Boolean tracking freeze state
    """
    
    def __init__(self, pretrained: bool = True) -> None:
        """
        Initialize MobileNetV2 model.
        
        Args:
            pretrained: Whether to load ImageNet pretrained weights (default True)
        """
        super(MobileNetV2Model, self).__init__()
        
        # ── FIX 1: Use weights= instead of deprecated pretrained= ──
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        
        # ── FIX 2: classifier[1] is Linear, classifier[0] is Dropout ──
        in_features = backbone.classifier[1].in_features  # 1280
        
        # ── FIX 3: Replace entire classifier inside backbone ──
        # Stronger head: Dropout → Linear(1280→512) → ReLU → Dropout → Linear(512→8)
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 8),
        )
        
        # Store full backbone (features + classifier together)
        self.backbone = backbone
        
        # Track freeze state
        self.backbone_frozen = False
    
    def freeze_backbone(self) -> None:
        """
        Freeze all feature extraction layers — only train classifier head.
        
        Useful for first 12 epochs where only the new classifier
        should be learning.
        
        Example:
            model = MobileNetV2Model()
            model.freeze_backbone()
            # Only classifier parameters will have gradients
        """
        # ── FIX 4: freeze backbone.features, not backbone ──
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        
        self.backbone_frozen = True
        print("🔒 Backbone FROZEN — training classifier head only")
    
    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all feature extraction layers for full fine-tuning.
        
        Call this at epoch 12 with a reduced LR (3e-5) to avoid
        overwriting pretrained ImageNet weights aggressively.
        
        Example:
            model.unfreeze_backbone()
            for pg in optimizer.param_groups:
                pg["lr"] = 3e-5
        """
        # ── FIX 4: unfreeze backbone.features, not backbone ──
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        
        self.backbone_frozen = False
        print("🔓 Backbone UNFROZEN — fine-tuning all layers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MobileNetV2.
        
        Args:
            x: Input tensor of shape (batch, 3, 96, 96)
            
        Returns:
            Logits tensor of shape (batch, 8)
        """
        # ── FIX 5: single call — backbone handles features + 
        #           pooling + classifier internally ──
        return self.backbone(x)
    
    def get_confidence(self, logits: torch.Tensor) -> Tuple[str, float]:
        """
        Get predicted emotion and confidence from logits.
        
        Args:
            logits: Raw output tensor of shape (batch, 8) or (8,)
            
        Returns:
            Tuple of (emotion_name: str, confidence: float 0.0–1.0)
            
        Example:
            logits = model(images)
            emotion, conf = model.get_confidence(logits)
            # "happy", 0.92
        """
        if logits.dim() > 1:
            logits = logits[0] if logits.shape[0] == 1 else logits
        
        probs = F.softmax(logits, dim=-1)
        confidence, pred_idx = torch.max(probs, dim=-1)
        
        return LABEL_TO_EMOTION[int(pred_idx.item())], float(confidence.item())
    
    def get_all_scores(self, logits: torch.Tensor) -> Dict[str, float]:
        """
        Get probability scores for all 8 emotions from logits.
        
        Args:
            logits: Raw output tensor of shape (batch, 8) or (8,)
            
        Returns:
            Dict mapping emotion names to probabilities
            
        Example:
            scores = model.get_all_scores(logits)
            # {"anger": 0.01, "happy": 0.85, ...}
        """
        if logits.dim() > 1:
            logits = logits[0] if logits.shape[0] == 1 else logits
        
        probs = F.softmax(logits, dim=-1)
        
        return {
            LABEL_TO_EMOTION[i]: float(probs[i].item())
            for i in range(8)
        }