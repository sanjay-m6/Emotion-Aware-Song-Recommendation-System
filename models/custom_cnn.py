"""
Custom CNN architecture trained from scratch on AffectNet 8-class emotion detection.

Architecture: 5 convolutional blocks + fully-connected head
Input: RGB 96x96 | Output: 8 emotion logits
Includes confidence score and per-emotion probability methods.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import LABEL_TO_EMOTION


class ResidualBlock(nn.Module):
    """
    Residual convolutional block with batch norm, ReLU, max pool, and dropout.
    Uses a shortcut connection to prevent vanishing gradients.
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.25) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # Residual connection
        out = self.relu(out)
        
        out = self.pool(out)
        out = self.dropout(out)
        return out


class CustomCNN(nn.Module):
    """
    Custom CNN for 8-class emotion detection from 96x96 RGB images.
    
    Architecture:
    - Res Block 1: 3 → 32 channels
    - Res Block 2: 32 → 64 channels
    - Res Block 3: 64 → 128 channels
    - Res Block 4: 128 → 256 channels
    - Res Block 5: 256 → 512 channels
    - Global Avg Pool + FC Head: 512 → 8
    
    Input shape: (batch, 3, 96, 96)
    Output shape: (batch, 8) — logits
    """
    
    def __init__(self) -> None:
        """Initialize CustomCNN with 5 residual blocks and Global Avg Pool head."""
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv_block1 = ResidualBlock(3, 32, dropout_rate=0.25)
        self.conv_block2 = ResidualBlock(32, 64, dropout_rate=0.25)
        self.conv_block3 = ResidualBlock(64, 128, dropout_rate=0.25)
        self.conv_block4 = ResidualBlock(128, 256, dropout_rate=0.25)
        self.conv_block5 = ResidualBlock(256, 512, dropout_rate=0.25)
        
        # Fully connected head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 3, 96, 96)
            
        Returns:
            Logits tensor of shape (batch, 8)
        """
        # Convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        
        # Global Average Pooling and FC
        x = self.global_pool(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
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
