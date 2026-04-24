# """
# Real-time emotion detection from webcam frames.

# Uses OpenCV Haar Cascade for face detection, preprocesses detected faces
# to 96x96 RGB tensors, and runs inference via trained emotion detector model.
# Provides frame annotation with bounding boxes, emotion labels, and confidence.
# """

# import sys
# from pathlib import Path
# from typing import Dict, Optional
# from collections import deque

# import cv2
# import numpy as np
# import torch
# from PIL import Image

# # Add parent directory to path
# sys.path.insert(0, str(Path(__file__).parent.parent))

# from utils.evaluate import load_model
# from utils.constants import LABEL_TO_EMOTION, NAVARASA_MAPPING, EMOTION_CV_COLORS


# class EmotionDetector:
#     """
#     Real-time emotion detector using webcam + CNN.
    
#     Combines:
#     - OpenCV Haar Cascade face detection
#     - Trained CNN for 8-class emotion classification
#     - Real-time FPS tracking
#     - Frame annotation with emotions and Navarasa labels
    
#     Attributes:
#         model: Trained emotion detection model
#         cascade_classifier: Haar Cascade for face detection
#         device: Torch device (cuda/mps/cpu)
#         fps_deque: Rolling window for FPS calculation
#         cap: OpenCV VideoCapture object
#     """
    
#     def __init__(
#         self,
#         model_path: str,
#         model_type: str = "mobilenet",
#         device: str = None
#     ) -> None:
#         """
#         Initialize EmotionDetector.
        
#         Args:
#             model_path: Path to .pth checkpoint file
#             model_type: "custom_cnn" or "mobilenet"
#             device: "cuda", "mps", or "cpu". Auto-detects if None.
            
#         Raises:
#             FileNotFoundError: If model checkpoint or cascade doesn't exist
#             RuntimeError: If model load fails
#         """
#         # Auto-detect device
#         if device is None:
#             if torch.cuda.is_available():
#                 device = "cuda"
#             elif torch.backends.mps.is_available():
#                 device = "mps"
#             else:
#                 device = "cpu"
        
#         self.device = torch.device(device)
#         print(f"✅ Emotion detector using device: {self.device}")
        
#         # Load model
#         try:
#             self.model = load_model(model_type, model_path, device=str(self.device))
#             print(f"✅ Loaded {model_type} model from {model_path}")
#         except Exception as e:
#             raise RuntimeError(f"Failed to load model: {e}")
        
#         # Load Haar Cascade for face detection
#         cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#         self.cascade_classifier = cv2.CascadeClassifier(cascade_path)
#         if self.cascade_classifier.empty():
#             raise FileNotFoundError("Could not load face cascade classifier")
        
#         # FPS tracker (last 30 frames)
#         self.fps_deque = deque(maxlen=30)
        
#         # Video capture
#         self.cap = None
    
#     def preprocess_face(self, face_roi: np.ndarray) -> torch.Tensor:
#         """
#         Preprocess detected face region to model input.
        
#         Steps:
#         1. Convert BGR (OpenCV) → RGB
#         2. Convert np.ndarray → PIL Image
#         3. Resize to 96x96
#         4. Convert to tensor and normalize (ImageNet stats)
        
#         Args:
#             face_roi: BGR image array from OpenCV (H, W, 3)
            
#         Returns:
#             Preprocessed tensor of shape (1, 3, 96, 96) on correct device
#         """
#         # BGR → RGB
#         face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
#         # np.ndarray → PIL Image
#         pil_image = Image.fromarray(face_rgb)
        
#         # Resize to 96x96
#         pil_image = pil_image.resize((96, 96), Image.LANCZOS)
        
#         # Convert to tensor
#         tensor = torch.from_numpy(np.array(pil_image)).float()  # (96, 96, 3)
#         tensor = tensor.permute(2, 0, 1)  # (3, 96, 96)
        
#         # Normalize (ImageNet stats) — BUG FIX: move to device BEFORE operations
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
#         tensor = tensor.to(self.device)
#         tensor = (tensor / 255.0 - mean) / std
        
#         # Add batch dimension and move to device
#         tensor = tensor.unsqueeze(0)
        
#         return tensor
    
#     def detect(self, frame: np.ndarray) -> Dict:
#         """
#         Detect emotion in a video frame.
        
#         Steps:
#         1. Convert frame to grayscale
#         2. Detect faces with Haar Cascade
#         3. Select largest face (by bounding box area)
#         4. Preprocess face to 96x96 tensor
#         5. Run inference
#         6. Extract emotion, confidence, and all scores
#         7. Annotate frame
#         8. Update FPS counter
        
#         Args:
#             frame: BGR frame from webcam (H, W, 3)
            
#         Returns:
#             Dict with keys:
#             - frame: Annotated frame (np.ndarray)
#             - emotion: Predicted emotion name (str)
#             - navarasa: Navarasa name (str)
#             - navarasa_meaning: Rasa meaning (str)
#             - navarasa_emoji: Emoji representation (str)
#             - confidence: Confidence score (0.0-1.0)
#             - all_scores: {emotion_name: probability} for all 8 emotions
#             - face_found: Whether face was detected (bool)
#             - fps: Frames per second (float)
#         """
#         result = {
#             "frame": frame.copy(),
#             "emotion": "neutral",
#             "navarasa": "Shanta",
#             "navarasa_meaning": "Peace",
#             "navarasa_emoji": "😌",
#             "confidence": 0.0,
#             "all_scores": {},
#             "face_found": False,
#             "fps": 0.0
#         }
        
#         try:
#             # Convert to grayscale for face detection
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # BUG FIX: Improved face detection parameters for real-world robustness
#             # - scaleFactor: 1.05 (was 1.1) - more sensitive, slower but catches more faces
#             # - minNeighbors: 6 (was 5) - reduce false positives  
#             # - minSize: (40, 40) (was 30, 30) - filter out tiny face artifacts
#             # - maxSize: (500, 500) - filter out overly large detections (screen errors)
#             faces = self.cascade_classifier.detectMultiScale(
#                 gray,
#                 scaleFactor=1.05,
#                 minNeighbors=6,
#                 minSize=(40, 40),
#                 maxSize=(500, 500)
#             )
            
#             if len(faces) > 0:
#                 # Select largest face
#                 face_areas = [(w * h, i) for i, (x, y, w, h) in enumerate(faces)]
#                 _, largest_idx = max(face_areas)
#                 x, y, w, h = faces[largest_idx]
                
#                 # Extract face ROI
#                 face_roi = frame[y:y+h, x:x+w]
                
#                 # Preprocess
#                 tensor_face = self.preprocess_face(face_roi)
                
#                 # Inference
#                 with torch.no_grad():
#                     logits = self.model(tensor_face)
                
#                 # Extract emotion and confidence
#                 emotion, confidence = self.model.get_confidence(logits)
#                 all_scores = self.model.get_all_scores(logits)
                
#                 # BUG FIX: Better confidence validation and logging
#                 # Only trust predictions above 0.4 confidence
#                 min_confidence_threshold = 0.40
#                 if confidence < min_confidence_threshold:
#                     print(f"⚠️ Low confidence detection ({confidence:.2f}). Defaulting to neutral.")
#                     emotion = "neutral"
#                     confidence = 0.5
                
#                 # BUG FIX: Sanity check - ensure emotion is valid
#                 if emotion not in NAVARASA_MAPPING:
#                     print(f"⚠️ Invalid emotion '{emotion}'. Defaulting to neutral.")
#                     emotion = "neutral"
#                     confidence = 0.0
                
#                 # Get Navarasa info
#                 navarasa_info = NAVARASA_MAPPING.get(emotion, {})
                
#                 result.update({
#                     "frame": result["frame"],
#                     "emotion": emotion,
#                     "navarasa": navarasa_info.get("navarasa", "Unknown"),
#                     "navarasa_meaning": navarasa_info.get("meaning", ""),
#                     "navarasa_emoji": navarasa_info.get("emoji", ""),
#                     "confidence": confidence,
#                     "all_scores": all_scores,
#                     "face_found": True
#                 })
                
#                 # Annotate frame
#                 result["frame"] = self.draw_overlay(result["frame"], result, [x, y, w, h])
        
#         except Exception as e:
#             print(f"⚠️  Detection error: {e}")
        
#         # BUG FIX: Improved FPS tracking - use time.time() for better precision
#         import time
#         current_time = time.time()
#         self.fps_deque.append(current_time)
        
#         # Calculate FPS based on rolling window (avoid division by zero)
#         if len(self.fps_deque) >= 2:
#             time_diff = self.fps_deque[-1] - self.fps_deque[0]
#             if time_diff > 0:
#                 fps = len(self.fps_deque) / time_diff
#                 result["fps"] = max(0.0, min(fps, 300.0))  # Clamp between 0-300 FPS (sanity check)
#             else:
#                 result["fps"] = 0.0
        
#         return result
    
#     def draw_overlay(
#         self,
#         frame: np.ndarray,
#         result: Dict,
#         face_bbox: Optional[list] = None
#     ) -> np.ndarray:
#         """
#         Draw emotion detection overlay on frame.
        
#         Includes:
#         - Colored bounding box around detected face
#         - Navarasa label + meaning below box
#         - Confidence percentage
#         - Mini emotion score bar chart (top-left)
#         - FPS counter (top-right)
        
#         Args:
#             frame: Frame to annotate
#             result: Result dict from detect()
#             face_bbox: [x, y, w, h] face bounding box coordinates
            
#         Returns:
#             Annotated frame (np.ndarray)
#         """
#         annotated = frame.copy()
        
#         if face_bbox:
#             x, y, w, h = face_bbox
#             emotion = result["emotion"]
            
#             # Draw colored bounding box
#             color = EMOTION_CV_COLORS.get(emotion, (255, 255, 255))
#             thickness = 3
#             cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            
#             # Label: Navarasa + meaning + confidence
#             emoji = result["navarasa_emoji"]
#             navarasa = result["navarasa"]
#             meaning = result["navarasa_meaning"]
#             confidence = result["confidence"]
            
#             label_line1 = f"{emoji} {navarasa} ({meaning})"
#             label_line2 = f"Confidence: {confidence:.1%}"
            
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 0.6
#             thickness = 1
            
#             # Line 1: below box
#             cv2.putText(
#                 annotated,
#                 label_line1,
#                 (x, y + h + 25),
#                 font,
#                 font_scale,
#                 color,
#                 thickness
#             )
            
#             # Line 2: below line 1
#             cv2.putText(
#                 annotated,
#                 label_line2,
#                 (x, y + h + 45),
#                 font,
#                 font_scale,
#                 color,
#                 thickness
#             )
        
#         # Top-right: FPS counter
#         fps = result["fps"]
#         fps_text = f"FPS: {fps:.1f}"
#         cv2.putText(
#             annotated,
#             fps_text,
#             (annotated.shape[1] - 150, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 0),
#             2
#         )
        
#         return annotated
    
#     def start_webcam(self, camera_id: int = 0) -> bool:
#         """
#         Start webcam capture.
        
#         Args:
#             camera_id: Camera device ID (default 0)
            
#         Returns:
#             True if successful, False otherwise
#         """
#         try:
#             self.cap = cv2.VideoCapture(camera_id)
#             if not self.cap.isOpened():
#                 return False
#             return True
#         except Exception as e:
#             print(f"⚠️  Webcam error: {e}")
#             return False
    
#     def release(self) -> None:
#         """Release webcam and close all OpenCV windows."""
#         if self.cap is not None:
#             self.cap.release()
#         cv2.destroyAllWindows()



"""
Real-time emotion detection from webcam frames.

FIXES APPLIED:
- FIX 1: Color space consistency — accepts RGB input from Streamlit camera
- FIX 2: preprocess_face no longer does BGR→RGB (input is already RGB)
- FIX 3: Grayscale conversion uses COLOR_RGB2GRAY not COLOR_BGR2GRAY
- FIX 4: CLAHE applied correctly on RGB images
- FIX 5: draw_overlay color annotations corrected for RGB frames
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional
from collections import deque

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.evaluate import load_model
from utils.constants import LABEL_TO_EMOTION, NAVARASA_MAPPING, EMOTION_CV_COLORS


# ── Preprocessing transforms (same as training VAL_TRANSFORMS) ──────────────
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class EmotionDetector:
    """
    Real-time emotion detector.

    IMPORTANT: This class expects frames in RGB format (not BGR).
    Streamlit's st.camera_input returns JPEG bytes → PIL opens as RGB → numpy RGB.
    All internal processing is done in RGB.

    Attributes:
        model: Trained emotion detection model (eval mode)
        cascade: Haar Cascade face detector
        device: Torch device
        fps_deque: Rolling window for FPS calculation
        _model_type: String identifier for model type
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "custom_cnn",
        device: str = None
    ) -> None:
        """
        Initialize EmotionDetector.

        Args:
            model_path: Path to .pth checkpoint file
            model_type: "custom_cnn" or "mobilenet"
            device: "cuda", "mps", or "cpu". Auto-detects if None.
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device     = torch.device(device)
        self._model_type = model_type
        print(f"[OK] EmotionDetector using device: {self.device}")

        # Load model
        try:
            self.model = load_model(model_type, model_path, device=str(self.device))
            self.model.eval()
            print(f"[OK] Loaded {model_type} from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Load Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise FileNotFoundError("Haar Cascade not found")

        # FPS tracker
        self.fps_deque = deque(maxlen=30)

    # ─────────────────────────────────────────────────────────────────────────
    # CORE FIX: preprocess_face expects RGB input, NOT BGR
    # ─────────────────────────────────────────────────────────────────────────
    def preprocess_face(self, face_roi_rgb: np.ndarray) -> torch.Tensor:
        """
        Preprocess face ROI to model input tensor.

        Args:
            face_roi_rgb: RGB image array (H, W, 3) — already in RGB format.
                          Do NOT pass BGR here.

        Returns:
            Tensor of shape (1, 3, 96, 96) normalized with ImageNet stats.
        """
        # ── FIX 2: Input is RGB — no color conversion needed ────────────────
        # Old (wrong): face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        # New (correct): input is already RGB from Streamlit camera

        # Apply CLAHE for lighting correction (works on L channel of LAB)
        try:
            # Convert RGB → LAB to apply CLAHE only on luminance
            lab = cv2.cvtColor(face_roi_rgb, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            # Convert back to RGB
            face_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except Exception:
            # Fallback: use original face if CLAHE fails
            face_enhanced = face_roi_rgb

        # Convert to PIL → apply inference transforms
        pil_image = Image.fromarray(face_enhanced.astype(np.uint8))
        tensor = INFERENCE_TRANSFORMS(pil_image)  # (3, 96, 96)
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, 3, 96, 96)

        return tensor

    def detect(self, frame_rgb: np.ndarray) -> Dict:
        """
        Detect emotion from an RGB frame.

        Args:
            frame_rgb: RGB image array (H, W, 3).
                       Streamlit camera_input → PIL.open → np.array gives RGB.

        Returns:
            Dict with emotion, navarasa, confidence, all_scores, face_found, fps, frame.
        """
        result = {
            "frame":            frame_rgb.copy(),
            "emotion":          "neutral",
            "navarasa":         "Shanta",
            "navarasa_meaning": "Peace",
            "navarasa_emoji":   "😌",
            "confidence":       0.0,
            "all_scores":       {e: 0.0 for e in LABEL_TO_EMOTION.values()},
            "face_found":       False,
            "fps":              0.0
        }

        try:
            # ── FIX 3: Use RGB→GRAY not BGR→GRAY ────────────────────────────
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(40, 40),
                maxSize=(500, 500)
            )

            if len(faces) > 0:
                # Select largest face
                largest_idx = max(
                    range(len(faces)),
                    key=lambda i: faces[i][2] * faces[i][3]
                )
                x, y, w, h = faces[largest_idx]

                # ── FIX 1: face_roi is sliced from RGB frame → stays RGB ────
                face_roi_rgb = frame_rgb[y:y+h, x:x+w]

                # Preprocess (RGB → tensor)
                tensor = self.preprocess_face(face_roi_rgb)

                # Inference
                with torch.no_grad():
                    logits = self.model(tensor)

                emotion, confidence   = self.model.get_confidence(logits)
                all_scores            = self.model.get_all_scores(logits)

                # Validate emotion
                if emotion not in NAVARASA_MAPPING:
                    emotion    = "neutral"
                    confidence = 0.0

                navarasa_info = NAVARASA_MAPPING[emotion]

                result.update({
                    "emotion":          emotion,
                    "navarasa":         navarasa_info["navarasa"],
                    "navarasa_meaning": navarasa_info["meaning"],
                    "navarasa_emoji":   navarasa_info["emoji"],
                    "confidence":       confidence,
                    "all_scores":       all_scores,
                    "face_found":       True,
                    "frame":            self.draw_overlay(
                                            frame_rgb.copy(),
                                            emotion, confidence,
                                            [x, y, w, h]
                                        )
                })

        except Exception as e:
            print(f"⚠️ Detection error: {e}")

        # FPS
        self.fps_deque.append(time.time())
        if len(self.fps_deque) >= 2:
            diff = self.fps_deque[-1] - self.fps_deque[0]
            result["fps"] = len(self.fps_deque) / diff if diff > 0 else 0.0

        return result

    def draw_overlay(
        self,
        frame_rgb: np.ndarray,
        emotion: str,
        confidence: float,
        face_bbox: list
    ) -> np.ndarray:
        """
        Draw bounding box and emotion label on RGB frame.

        Args:
            frame_rgb: RGB frame to annotate.
            emotion: Detected emotion string.
            confidence: Confidence score.
            face_bbox: [x, y, w, h] bounding box.

        Returns:
            Annotated RGB frame.
        """
        annotated = frame_rgb.copy()
        x, y, w, h = face_bbox

        # ── FIX 5: EMOTION_CV_COLORS are BGR — convert to RGB for drawing ──
        # OpenCV drawing functions work on the array in-memory.
        # Since our frame is RGB, we need RGB color tuples.
        bgr_color = EMOTION_CV_COLORS.get(emotion, (150, 150, 150))
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])  # BGR → RGB

        # Bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), rgb_color, 3)

        # Labels
        navarasa_info = NAVARASA_MAPPING.get(emotion, {})
        emoji    = navarasa_info.get("emoji", "")
        navarasa = navarasa_info.get("navarasa", emotion)
        meaning  = navarasa_info.get("meaning", "")

        label1 = f"{navarasa} ({meaning})"
        label2 = f"Conf: {confidence:.0%}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated, label1, (x, y + h + 25),
                    font, 0.6, rgb_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, label2, (x, y + h + 45),
                    font, 0.6, rgb_color, 1, cv2.LINE_AA)

        # FPS (top-right)
        fps = 0.0
        if len(self.fps_deque) >= 2:
            diff = self.fps_deque[-1] - self.fps_deque[0]
            fps = len(self.fps_deque) / diff if diff > 0 else 0.0

        cv2.putText(annotated,
                    f"FPS: {fps:.1f}",
                    (annotated.shape[1] - 120, 30),
                    font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        return annotated

    def release(self) -> None:
        """Release all resources."""
        cv2.destroyAllWindows()