"""
Real-time emotion detection from webcam frames.

Uses OpenCV Haar Cascade for face detection, preprocesses detected faces
to 96x96 RGB tensors, and runs inference via trained emotion detector model.
Provides frame annotation with bounding boxes, emotion labels, and confidence.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.evaluate import load_model
from utils.constants import LABEL_TO_EMOTION, NAVARASA_MAPPING, EMOTION_CV_COLORS


class EmotionDetector:
    """
    Real-time emotion detector using webcam + CNN.
    
    Combines:
    - OpenCV Haar Cascade face detection
    - Trained CNN for 8-class emotion classification
    - Real-time FPS tracking
    - Frame annotation with emotions and Navarasa labels
    
    Attributes:
        model: Trained emotion detection model
        cascade_classifier: Haar Cascade for face detection
        device: Torch device (cuda/mps/cpu)
        fps_deque: Rolling window for FPS calculation
        cap: OpenCV VideoCapture object
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "mobilenet",
        device: str = None
    ) -> None:
        """
        Initialize EmotionDetector.
        
        Args:
            model_path: Path to .pth checkpoint file
            model_type: "custom_cnn" or "mobilenet"
            device: "cuda", "mps", or "cpu". Auto-detects if None.
            
        Raises:
            FileNotFoundError: If model checkpoint or cascade doesn't exist
            RuntimeError: If model load fails
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        print(f"✅ Emotion detector using device: {self.device}")
        
        # Load model
        try:
            self.model = load_model(model_type, model_path, device=str(self.device))
            print(f"✅ Loaded {model_type} model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade_classifier = cv2.CascadeClassifier(cascade_path)
        if self.cascade_classifier.empty():
            raise FileNotFoundError("Could not load face cascade classifier")
        
        # FPS tracker (last 30 frames)
        self.fps_deque = deque(maxlen=30)
        
        # Video capture
        self.cap = None
    
    def preprocess_face(self, face_roi: np.ndarray) -> torch.Tensor:
        """
        Preprocess detected face region to model input.
        
        Steps:
        1. Convert BGR (OpenCV) → RGB
        2. Convert np.ndarray → PIL Image
        3. Resize to 96x96
        4. Convert to tensor and normalize (ImageNet stats)
        
        Args:
            face_roi: BGR image array from OpenCV (H, W, 3)
            
        Returns:
            Preprocessed tensor of shape (1, 3, 96, 96) on correct device
        """
        # BGR → RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # np.ndarray → PIL Image
        pil_image = Image.fromarray(face_rgb)
        
        # Resize to 96x96
        pil_image = pil_image.resize((96, 96), Image.LANCZOS)
        
        # Convert to tensor
        tensor = torch.from_numpy(np.array(pil_image)).float()  # (96, 96, 3)
        tensor = tensor.permute(2, 0, 1)  # (3, 96, 96)
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor / 255.0 - mean) / std
        
        # Add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect emotion in a video frame.
        
        Steps:
        1. Convert frame to grayscale
        2. Detect faces with Haar Cascade
        3. Select largest face (by bounding box area)
        4. Preprocess face to 96x96 tensor
        5. Run inference
        6. Extract emotion, confidence, and all scores
        7. Annotate frame
        8. Update FPS counter
        
        Args:
            frame: BGR frame from webcam (H, W, 3)
            
        Returns:
            Dict with keys:
            - frame: Annotated frame (np.ndarray)
            - emotion: Predicted emotion name (str)
            - navarasa: Navarasa name (str)
            - navarasa_meaning: Rasa meaning (str)
            - navarasa_emoji: Emoji representation (str)
            - confidence: Confidence score (0.0-1.0)
            - all_scores: {emotion_name: probability} for all 8 emotions
            - face_found: Whether face was detected (bool)
            - fps: Frames per second (float)
        """
        result = {
            "frame": frame.copy(),
            "emotion": "neutral",
            "navarasa": "Shanta",
            "navarasa_meaning": "Peace",
            "navarasa_emoji": "😌",
            "confidence": 0.0,
            "all_scores": {},
            "face_found": False,
            "fps": 0.0
        }
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.cascade_classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Select largest face
                face_areas = [(w * h, i) for i, (x, y, w, h) in enumerate(faces)]
                _, largest_idx = max(face_areas)
                x, y, w, h = faces[largest_idx]
                
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Preprocess
                tensor_face = self.preprocess_face(face_roi)
                
                # Inference
                with torch.no_grad():
                    logits = self.model(tensor_face)
                
                # Extract emotion and confidence
                emotion, confidence = self.model.get_confidence(logits)
                all_scores = self.model.get_all_scores(logits)
                
                # Get Navarasa info
                navarasa_info = NAVARASA_MAPPING.get(emotion, {})
                
                result.update({
                    "frame": result["frame"],
                    "emotion": emotion,
                    "navarasa": navarasa_info.get("navarasa", "Unknown"),
                    "navarasa_meaning": navarasa_info.get("meaning", ""),
                    "navarasa_emoji": navarasa_info.get("emoji", ""),
                    "confidence": confidence,
                    "all_scores": all_scores,
                    "face_found": True
                })
                
                # Annotate frame
                result["frame"] = self.draw_overlay(result["frame"], result, [x, y, w, h])
        
        except Exception as e:
            print(f"⚠️  Detection error: {e}")
        
        # Update FPS
        current_time = cv2.getTickCount()
        end_time = (current_time) / cv2.getTickFrequency()
        self.fps_deque.append(end_time)
        
        if len(self.fps_deque) > 1:
            fps = len(self.fps_deque) / (self.fps_deque[-1] - self.fps_deque[0]) if self.fps_deque[-1] > self.fps_deque[0] else 0
            result["fps"] = fps
        
        return result
    
    def draw_overlay(
        self,
        frame: np.ndarray,
        result: Dict,
        face_bbox: Optional[list] = None
    ) -> np.ndarray:
        """
        Draw emotion detection overlay on frame.
        
        Includes:
        - Colored bounding box around detected face
        - Navarasa label + meaning below box
        - Confidence percentage
        - Mini emotion score bar chart (top-left)
        - FPS counter (top-right)
        
        Args:
            frame: Frame to annotate
            result: Result dict from detect()
            face_bbox: [x, y, w, h] face bounding box coordinates
            
        Returns:
            Annotated frame (np.ndarray)
        """
        annotated = frame.copy()
        
        if face_bbox:
            x, y, w, h = face_bbox
            emotion = result["emotion"]
            
            # Draw colored bounding box
            color = EMOTION_CV_COLORS.get(emotion, (255, 255, 255))
            thickness = 3
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            
            # Label: Navarasa + meaning + confidence
            emoji = result["navarasa_emoji"]
            navarasa = result["navarasa"]
            meaning = result["navarasa_meaning"]
            confidence = result["confidence"]
            
            label_line1 = f"{emoji} {navarasa} ({meaning})"
            label_line2 = f"Confidence: {confidence:.1%}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            
            # Line 1: below box
            cv2.putText(
                annotated,
                label_line1,
                (x, y + h + 25),
                font,
                font_scale,
                color,
                thickness
            )
            
            # Line 2: below line 1
            cv2.putText(
                annotated,
                label_line2,
                (x, y + h + 45),
                font,
                font_scale,
                color,
                thickness
            )
        
        # Top-right: FPS counter
        fps = result["fps"]
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            annotated,
            fps_text,
            (annotated.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return annotated
    
    def start_webcam(self, camera_id: int = 0) -> bool:
        """
        Start webcam capture.
        
        Args:
            camera_id: Camera device ID (default 0)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                return False
            return True
        except Exception as e:
            print(f"⚠️  Webcam error: {e}")
            return False
    
    def release(self) -> None:
        """Release webcam and close all OpenCV windows."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
