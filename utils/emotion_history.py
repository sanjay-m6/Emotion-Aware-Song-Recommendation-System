"""
Emotion detection history tracker with session statistics and visualization.

Tracks detected emotions over time, maintains recently-played track history
for recommendation filtering, and provides methods for session analytics
and timeline visualization.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import NAVARASA_MAPPING, EMOTION_COLORS


class EmotionHistory:
    """
    Session-based emotion history tracker.
    
    Maintains:
    - Full history of detected emotions with timestamps and confidence
    - Recently played track names and genres (for recommendation filtering)
    - Per-emotion counts and session statistics
    
    Attributes:
        max_len: Maximum length of emotion history (default 200)
        emotions: Deque of emotion detection records
        played_tracks: Deque of recently played track names
        played_genres: Deque of recently played track genres
        session_start_time: Timestamp when session began
    """
    
    def __init__(self, max_len: int = 200) -> None:
        """
        Initialize EmotionHistory tracker.
        
        Args:
            max_len: Maximum number of emotion records to store (default 200)
        """
        self.max_len = max_len
        self.emotions = deque(maxlen=max_len)
        self.played_tracks = deque(maxlen=10)
        self.played_genres = deque(maxlen=5)
        self.session_start_time = time.time()
    
    def add(self, emotion: str, confidence: float) -> None:
        """
        Add emotion detection to history.
        
        Args:
            emotion: Emotion name (e.g., "happy", "sad")
            confidence: Confidence score (0.0-1.0)
            
        Example:
            >>> history = EmotionHistory()
            >>> history.add("happy", 0.92)
        """
        record = {
            "emotion": emotion,
            "navarasa": NAVARASA_MAPPING.get(emotion, {}).get("navarasa", "Unknown"),
            "confidence": float(confidence),
            "timestamp": time.time()
        }
        self.emotions.append(record)
    
    def add_played_track(self, track_name: str, genre: str) -> None:
        """
        Record a played song in history.
        
        Used for:
        - Excluding from recommendations (don't repeat recent songs)
        - Inferring Shringara emotion (track recent romantic genres)
        
        Args:
            track_name: Song title
            genre: Genre tag
        """
        self.played_tracks.append(track_name)
        if genre:
            self.played_genres.append(genre)
    
    def get_recent_tracks(self, n: int = 10) -> List[str]:
        """
        Get last n played track names.
        
        Args:
            n: Number of tracks to return (default 10)
            
        Returns:
            List of recently played track names
        """
        return list(self.played_tracks)[-n:]
    
    def get_recent_genres(self, n: int = 5) -> List[str]:
        """
        Get last n played track genres.
        
        Args:
            n: Number of genres to return (default 5)
            
        Returns:
            List of recently played genres
        """
        return list(self.played_genres)[-n:]
    
    def get_dominant(self, window_seconds: int = 30) -> Tuple[str, float]:
        """
        Get most frequent emotion in recent time window.
        
        Args:
            window_seconds: Time window in seconds (default 30)
            
        Returns:
            Tuple of (emotion_name, avg_confidence) for most frequent emotion in window
            Returns ("neutral", 0.0) if no emotions in window
        """
        if not self.emotions:
            return "neutral", 0.0
        
        current_time = time.time()
        recent = [
            e for e in self.emotions
            if (current_time - e["timestamp"]) <= window_seconds
        ]
        
        if not recent:
            return "neutral", 0.0
        
        # Find most frequent emotion in window
        emotion_counts = {}
        emotion_confidences = {}
        
        for record in recent:
            emotion = record["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            emotion_confidences[emotion] = emotion_confidences.get(emotion, [])
            emotion_confidences[emotion].append(record["confidence"])
        
        # Get emotion with highest count
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        avg_confidence = np.mean(emotion_confidences[dominant_emotion])
        
        return dominant_emotion, avg_confidence
    
    def plot_timeline(self) -> plt.Figure:
        """
        Create emotion timeline visualization.
        
        Scatter plot with:
        - X-axis: Elapsed seconds from session start
        - Y-axis: Emotion names
        - Point colors: EMOTION_COLORS
        - Point size: Confidence score (larger = more confident)
        
        Returns:
            matplotlib figure object
        """
        if not self.emotions:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "No emotion detections yet", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Navarasa Emotion Timeline", fontsize=14, fontweight="bold")
            return fig
        
        # Prepare data
        times = []
        emotions = []
        confidences = []
        colors = []
        
        session_start = self.session_start_time
        emotion_to_ypos = {}
        
        for i, record in enumerate(self.emotions):
            elapsed = (record["timestamp"] - session_start) / 60  # Convert to minutes
            emotion = record["emotion"]
            confidence = record["confidence"]
            
            # Assign y position to each emotion (first occurrence)
            if emotion not in emotion_to_ypos:
                emotion_to_ypos[emotion] = len(emotion_to_ypos)
            
            times.append(elapsed)
            emotions.append(emotion)
            confidences.append(confidence)
            colors.append(EMOTION_COLORS.get(emotion, "#808080"))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Map emotions to y positions for unique values
        unique_emotions = sorted(set(emotions), key=lambda x: emotion_to_ypos[x])
        emotion_y = {e: i for i, e in enumerate(unique_emotions)}
        y_positions = [emotion_y[e] for e in emotions]
        
        # Size points by confidence
        sizes = [100 + (conf * 150) for conf in confidences]
        
        # Scatter plot
        scatter = ax.scatter(times, y_positions, s=sizes, c=colors, alpha=0.6, edgecolors="black", linewidth=0.5)
        
        # Formatting
        ax.set_xlabel("Time Elapsed (minutes)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Emotion", fontsize=12, fontweight="bold")
        ax.set_yticks(range(len(unique_emotions)))
        ax.set_yticklabels(unique_emotions)
        ax.set_title("🎭 Navarasa Emotion Timeline", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        # Add confidence legend
        for conf in [0.5, 0.75, 1.0]:
            size = 100 + (conf * 150)
            ax.scatter([], [], s=size, c="gray", alpha=0.6, edgecolors="black", 
                      label=f"Confidence {conf:.0%}")
        ax.legend(loc="upper right", scatterpoints=1, fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def get_session_stats(self) -> Dict:
        """
        Compute session statistics.
        
        Returns:
            Dictionary with keys:
            - emotion_counts: {emotion: count} for each emotion
            - navarasa_counts: {navarasa_name: count} for each rasa
            - dominant_emotion: Most frequent emotion name
            - dominant_navarasa: Most frequent navarasa name
            - total_detections: Total emotion detections this session
            - session_duration_seconds: Time since session started
            
        Example:
            >>> history = EmotionHistory()
            >>> history.add("happy", 0.9)
            >>> history.add("happy", 0.85)
            >>> history.add("sad", 0.7)
            >>> stats = history.get_session_stats()
            >>> print(stats["dominant_emotion"])
            'happy'
        """
        if not self.emotions:
            return {
                "emotion_counts": {},
                "navarasa_counts": {},
                "dominant_emotion": "neutral",
                "dominant_navarasa": "Shanta",
                "total_detections": 0,
                "session_duration_seconds": 0.0
            }
        
        # Count emotions
        emotion_counts = {}
        navarasa_counts = {}
        
        for record in self.emotions:
            emotion = record["emotion"]
            navarasa = record["navarasa"]
            
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            navarasa_counts[navarasa] = navarasa_counts.get(navarasa, 0) + 1
        
        # Find dominant
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        dominant_navarasa = max(navarasa_counts, key=navarasa_counts.get)
        
        # Session duration
        duration = time.time() - self.session_start_time
        
        return {
            "emotion_counts": emotion_counts,
            "navarasa_counts": navarasa_counts,
            "dominant_emotion": dominant_emotion,
            "dominant_navarasa": dominant_navarasa,
            "total_detections": len(self.emotions),
            "session_duration_seconds": duration
        }
    
    def to_json(self, path: str) -> None:
        """
        Save full history to JSON file.
        
        Saves all emotion records and metadata for later analysis.
        
        Args:
            path: File path to save JSON (e.g., "session_log.json")
        """
        # Convert deque to list (JSON serializable)
        emotions_list = list(self.emotions)
        
        data = {
            "session_start_time": self.session_start_time,
            "session_duration_seconds": time.time() - self.session_start_time,
            "emotions": emotions_list,
            "played_tracks": list(self.played_tracks),
            "played_genres": list(self.played_genres),
            "stats": self.get_session_stats()
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Session saved to {path}")
