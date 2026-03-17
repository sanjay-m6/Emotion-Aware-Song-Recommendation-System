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


class ConvBlock(nn.Module):
    """
    Standard convolutional block with batch norm, ReLU, max pool, and dropout.
    
    Attributes:
        conv: Conv2d layer
        bn: BatchNorm2d layer
        relu: ReLU activation
        pool: MaxPool2d layer
        dropout: Spatial dropout (Dropout2d)
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.25) -> None:
        """
        Initialize ConvBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout_rate: Dropout rate (default 0.25)
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through conv block.
        
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch, out_channels, height/2, width/2)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CustomCNN(nn.Module):
    """
    Custom CNN for 8-class emotion detection from 96x96 RGB images.
    
    Architecture:
    - Conv Block 1: 3 → 32 channels
    - Conv Block 2: 32 → 64 channels
    - Conv Block 3: 64 → 128 channels
    - Conv Block 4: 128 → 256 channels
    - Conv Block 5: 256 → 512 channels
    - FC Head: 512*3*3 → 1024 → 256 → 8
    
    Input shape: (batch, 3, 96, 96)
    Output shape: (batch, 8) — logits
    """
    
    def __init__(self) -> None:
        """Initialize CustomCNN with 5 conv blocks and FC head."""
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv_block1 = ConvBlock(3, 32, dropout_rate=0.25)
        self.conv_block2 = ConvBlock(32, 64, dropout_rate=0.25)
        self.conv_block3 = ConvBlock(64, 128, dropout_rate=0.25)
        self.conv_block4 = ConvBlock(128, 256, dropout_rate=0.25)
        self.conv_block5 = ConvBlock(256, 512, dropout_rate=0.25)
        
        # Fully connected head
        # After 5 max pools: 96 → 48 → 24 → 12 → 6 → 3
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 8)
    
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
        
        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
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
