"""
Emotion detection service — singleton wrapper using custom CNN.

Uses the CustomCNN PyTorch model trained on AffectNet for real-time
facial emotion detection, replacing the previous DeepFace dependency.
"""

import base64
import io
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.constants import EMOTION_DISPLAY_MAPPING, EMOTION_COLORS, EMOTION_NAMES
from utils.dataset import VAL_TRANSFORMS
from models.custom_cnn import CustomCNN


class EmotionService:
    """
    Singleton service that uses CustomCNN for emotion detection.
    """

    _instance: Optional["EmotionService"] = None
    _ready: bool = False

    def __new__(cls) -> "EmotionService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, **kwargs) -> None:
        """
        Initialize CustomCNN and OpenCV face detector.
        """
        if self._ready:
            return

        try:
            # Set up device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load CustomCNN
            self.model = CustomCNN()
            checkpoint_path = PROJECT_ROOT / "models" / "checkpoints" / "custom_cnn_best.pth"
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
                
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Initialize Haar Cascade for face detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            self._ready = True
            print(f"[OK] EmotionService initialized with CustomCNN on {self.device}")
            
        except Exception as e:
            print(f"[ERROR] EmotionService init failed: {e}")
            raise

    @property
    def is_ready(self) -> bool:
        return self._ready

    def detect_from_base64(self, base64_str: str) -> Dict:
        """
        Detect emotion from a base64-encoded image using CustomCNN.

        Args:
            base64_str: Base64-encoded JPEG/PNG image string.
                        May include the data URI prefix.

        Returns:
            Dict with emotion, navarasa, confidence, all_scores,
            face_found, color, and navarasa metadata.
        """
        if not self.is_ready:
            raise RuntimeError("EmotionService not initialized")

        # Strip data URI prefix if present
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        # Decode base64 -> OpenCV RGB/Gray
        image_bytes = base64.b64decode(base64_str)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame_rgb = np.array(pil_image, dtype=np.uint8)
        
        # Convert to grayscale for face detection
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        
        try:
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )
            
            if len(faces) == 0:
                return self._get_fallback_response(face_found=False)
                
            # Get largest face by area
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Add a 20% margin around the face to capture more context (forehead, chin)
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            img_h, img_w = frame_rgb.shape[:2]
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img_w, x + w + margin_x)
            y2 = min(img_h, y + h + margin_y)
            
            # Crop the face region with margin from the RGB frame
            face_crop = frame_rgb[y1:y2, x1:x2]
            face_pil = Image.fromarray(face_crop)
            
            # Preprocess the face using the validation transforms
            tensor_image = VAL_TRANSFORMS(face_pil)
            tensor_image = tensor_image.unsqueeze(0).to(self.device)  # Add batch dim
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(tensor_image)
            
            # Extract predictions
            emotion, confidence = self.model.get_confidence(logits)
            all_scores = self.model.get_all_scores(logits)
            
            # Round scores for clean output
            for k, v in all_scores.items():
                all_scores[k] = round(v, 4)
                
            # DeepFace vs AffectNet label mapping unification
            # Ensure the output matches what the frontend expects
            if emotion == "contempt":
                emotion = "disgust"  # Map contempt to disgust for simplicity if not handled
                
            display_info = EMOTION_DISPLAY_MAPPING.get(emotion, EMOTION_DISPLAY_MAPPING.get("neutral", {}))

            return {
                "emotion": emotion,
                "confidence": round(confidence, 4),
                "face_found": True,
                "bbox": [int(x), int(y), int(w), int(h)],
                "frame_size": [int(img_w), int(img_h)],
                "all_scores": all_scores,
                "display_name": display_info.get("display_name", "Neutral"),
                "display_meaning": display_info.get("meaning", "Peace"),
                "display_emoji": display_info.get("emoji", ""),
                "color": EMOTION_COLORS.get(emotion, "#808080"),
            }

        except Exception as e:
            print(f"[WARN] CustomCNN detection failed: {e}")
            return self._get_fallback_response(face_found=False)
            
    def _get_fallback_response(self, face_found: bool) -> Dict:
        """Helper to return a neutral fallback response."""
        display_info = EMOTION_DISPLAY_MAPPING.get("neutral", {})
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "face_found": face_found,
            "bbox": None,
            "frame_size": None,
            "all_scores": {e: 0.0 for e in EMOTION_NAMES},
            "display_name": display_info.get("display_name", "Neutral"),
            "display_meaning": display_info.get("meaning", "Peace"),
            "display_emoji": display_info.get("emoji", "😌"),
            "color": EMOTION_COLORS.get("neutral", "#808080"),
        }


# Module-level singleton
emotion_service = EmotionService()
