"""
Evaluation and comparison tools for emotion detection models.

Provides:
- Model checkpoint loading
- Confusion matrix visualization
- Classification report
- Training curve plots
- Inference benchmarking
- Side-by-side model comparison table
"""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import LABEL_TO_EMOTION, EMOTION_NAMES
from models.custom_cnn import CustomCNN
from models.mobilenet_model import MobileNetV2Model


def load_model(
    model_type: str,
    checkpoint_path: str,
    device: str = "cpu"
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_type: "custom_cnn" or "mobilenet"
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on ("cpu", "cuda", "mps")
        
    Returns:
        Model in eval mode on specified device
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint format is invalid
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Initialize model
    if model_type == "custom_cnn":
        model = CustomCNN()
    elif model_type == "mobilenet":
        model = MobileNetV2Model(pretrained=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def plot_training_curves(
    cnn_history: Dict,
    mobilenet_history: Dict
) -> plt.Figure:
    """
    Plot training curves comparing both models.
    
    Creates 2x2 grid:
    [0,0] Train Loss | [0,1] Val Loss
    [1,0] Val Accuracy | [1,1] Accuracy Gap (MobileNet - CNN)
    
    Args:
        cnn_history: Dict with keys ["train_loss", "val_loss", "val_accuracy", "epochs"]
        mobilenet_history: Same structure as cnn_history
        
    Returns:
        matplotlib.figure.Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison: Training Curves", fontsize=16, fontweight="bold")
    
    epochs_cnn = cnn_history["epochs"]
    epochs_mobilenet = mobilenet_history["epochs"]
    
    # [0, 0] Train Loss
    ax = axes[0, 0]
    ax.plot(epochs_cnn, cnn_history["train_loss"], marker="o", label="CNN", linewidth=2)
    ax.plot(epochs_mobilenet, mobilenet_history["train_loss"], marker="s", label="MobileNetV2", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Train Loss", fontsize=11)
    ax.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # [0, 1] Val Loss
    ax = axes[0, 1]
    ax.plot(epochs_cnn, cnn_history["val_loss"], marker="o", label="CNN", linewidth=2)
    ax.plot(epochs_mobilenet, mobilenet_history["val_loss"], marker="s", label="MobileNetV2", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val Loss", fontsize=11)
    ax.set_title("Validation Loss", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # [1, 0] Val Accuracy
    ax = axes[1, 0]
    ax.plot(epochs_cnn, cnn_history["val_accuracy"], marker="o", label="CNN", linewidth=2)
    ax.plot(epochs_mobilenet, mobilenet_history["val_accuracy"], marker="s", label="MobileNetV2", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Accuracy", fontsize=11)
    ax.set_title("Validation Accuracy", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # [1, 1] Accuracy Gap
    ax = axes[1, 1]
    min_epochs = min(len(epochs_cnn), len(epochs_mobilenet))
    gap = np.array(mobilenet_history["val_accuracy"][:min_epochs]) - np.array(cnn_history["val_accuracy"][:min_epochs])
    ax.plot(epochs_cnn[:min_epochs], gap, marker="D", color="green", linewidth=2, label="Gap (MobileNetV2 - CNN)")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy Difference", fontsize=11)
    ax.set_title("MobileNetV2 vs CNN Accuracy Gap", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    title: str = "",
    save_path: str = None
) -> plt.Figure:
    """
    Compute and plot row-normalized confusion matrix.
    
    Args:
        model: Trained model in eval mode
        dataloader: DataLoader with validation data
        device: Device model is on
        title: Figure title
        save_path: Optional path to save figure
        
    Returns:
        matplotlib.figure.Figure object
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(8)))
    
    # Row normalize (each row sums to 1)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=cm,  # Annotate with raw counts
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTION_NAMES,
        yticklabels=EMOTION_NAMES,
        cbar_kws={"label": "Accuracy"},
        ax=ax
    )
    
    ax.set_xlabel("Predicted Emotion", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Emotion", fontsize=12, fontweight="bold")
    
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def print_classification_report(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu"
) -> str:
    """
    Print sklearn classification report (precision, recall, F1).
    
    Highlights contemplation F1 score separately (typically lowest).
    
    Args:
        model: Trained model in eval mode
        dataloader: DataLoader with validation data
        device: Device model is on
        
    Returns:
        Classification report as formatted string
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Generate report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=EMOTION_NAMES,
        digits=4
    )
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70 + "\n")
    print(report)
    
    # Extract contempt F1 separately
    lines = report.split("\n")
    for line in lines:
        if "contempt" in line:
            parts = line.split()
            f1_idx = -1  # F1 is last numeric value
            try:
                f1_score = float(parts[f1_idx])
                print("="*70)
                print(f"⚠️  Contempt F1 Score: {f1_score:.4f} ← WATCH THIS (typically lowest)")
                print("="*70)
            except (ValueError, IndexError):
                pass
    
    return report


def benchmark_inference(
    model: nn.Module,
    n_runs: int = 200,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Benchmark model inference speed and size.
    
    Args:
        model: Trained model
        n_runs: Number of inference runs
        device: Device model is on
        
    Returns:
        Dict with keys:
        - mean_ms: Mean inference time in milliseconds
        - std_ms: Std dev of inference time
        - model_size_mb: Model size in MB
        - params_total: Total parameters
        - params_trainable: Trainable parameters
    """
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size
    state_dict_size = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)
    
    # Benchmark inference
    dummy_input = torch.randn(1, 3, 96, 96).to(device)
    
    times = []
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        
        # Benchmark
        for _ in range(n_runs):
            start = time.time()
            _ = model(dummy_input)
            times.append((time.time() - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "model_size_mb": float(state_dict_size),
        "params_total": int(total_params),
        "params_trainable": int(trainable_params)
    }


def print_comparison_table(
    cnn_metrics: Dict,
    mobilenet_metrics: Dict
) -> None:
    """
    Print formatted comparison table of model metrics.
    
    Args:
        cnn_metrics: Dict with keys from benchmark_inference + val_accuracy + contempt_f1
        mobilenet_metrics: Same structure
    """
    print("\n" + "="*90)
    print("MODEL COMPARISON TABLE")
    print("="*90)
    print(
        f"{'Metric':<25} | {'Custom CNN':<20} | {'MobileNetV2':<20}"
    )
    print("-" * 90)
    
    # Val Accuracy
    cnn_acc = cnn_metrics.get("val_accuracy", 0.0)
    mobilenet_acc = mobilenet_metrics.get("val_accuracy", 0.0)
    print(f"{'Val Accuracy':<25} | {cnn_acc:>19.4f} | {mobilenet_acc:>19.4f}")
    
    # Contempt F1 Score
    cnn_f1 = cnn_metrics.get("contempt_f1", 0.0)
    mobilenet_f1 = mobilenet_metrics.get("contempt_f1", 0.0)
    print(f"{'Contempt F1 Score':<25} | {cnn_f1:>19.4f} | {mobilenet_f1:>19.4f} ⭐")
    
    # Parameters
    cnn_params = cnn_metrics.get("params_total", 0) / 1e6
    mobilenet_params = mobilenet_metrics.get("params_total", 0) / 1e6
    print(f"{'Parameters (M)':<25} | {cnn_params:>19.2f} | {mobilenet_params:>19.2f}")
    
    # Inference Time
    cnn_time = cnn_metrics.get("mean_ms", 0.0)
    mobilenet_time = mobilenet_metrics.get("mean_ms", 0.0)
    print(f"{'Inference Time (ms)':<25} | {cnn_time:>19.3f} | {mobilenet_time:>19.3f}")
    
    # Model Size
    cnn_size = cnn_metrics.get("model_size_mb", 0.0)
    mobilenet_size = mobilenet_metrics.get("model_size_mb", 0.0)
    print(f"{'Model Size (MB)':<25} | {cnn_size:>19.2f} | {mobilenet_size:>19.2f}")
    
    # Best Emotion (hypothetical, depends on training)
    cnn_best = cnn_metrics.get("best_emotion", "N/A")
    mobilenet_best = mobilenet_metrics.get("best_emotion", "N/A")
    print(f"{'Best Emotion':<25} | {cnn_best:>19} | {mobilenet_best:>19}")
    
    # Worst Emotion
    cnn_worst = cnn_metrics.get("worst_emotion", "N/A")
    mobilenet_worst = mobilenet_metrics.get("worst_emotion", "N/A")
    print(f"{'Worst Emotion':<25} | {cnn_worst:>19} | {mobilenet_worst:>19}")
    
    print("="*90 + "\n")
