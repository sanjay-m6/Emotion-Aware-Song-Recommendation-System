"""
Constants and mappings for the Emotion-Aware Music Recommendation System.

This module serves as the single source of truth for all emotion labels,
Navarasa mappings, audio profiles, and styling configurations used
throughout the entire application.
"""

# ============================================================================
# EMOTION LABEL MAPPINGS (AffectNet 8-class)
# ============================================================================

LABEL_TO_EMOTION = {
    0: "anger",
    1: "surprise",
    2: "contempt",
    3: "happy",
    4: "neutral",
    5: "fear",
    6: "sad",
    7: "disgust"
}

EMOTION_TO_LABEL = {v: k for k, v in LABEL_TO_EMOTION.items()}

EMOTION_NAMES = ["anger", "surprise", "contempt", "happy", "neutral", "fear", "sad", "disgust"]

# ============================================================================
# NAVARASA MAPPINGS (Ancient Indian Emotional Framework)
# ============================================================================

NAVARASA_MAPPING = {
    "anger": {
        "navarasa": "Raudra",
        "meaning": "Fury",
        "emoji": "😠"
    },
    "surprise": {
        "navarasa": "Adbhuta",
        "meaning": "Wonder",
        "emoji": "😲"
    },
    "contempt": {
        "navarasa": "Vira",
        "meaning": "Heroism",
        "emoji": "😤"
    },
    "happy": {
        "navarasa": "Hasya",
        "meaning": "Joy",
        "emoji": "😄"
    },
    "neutral": {
        "navarasa": "Shanta",
        "meaning": "Peace",
        "emoji": "😌"
    },
    "fear": {
        "navarasa": "Bhayanaka",
        "meaning": "Terror",
        "emoji": "😨"
    },
    "sad": {
        "navarasa": "Karuna",
        "meaning": "Sorrow",
        "emoji": "😢"
    },
    "disgust": {
        "navarasa": "Bibhatsa",
        "meaning": "Disgust",
        "emoji": "🤢"
    },
    "shringara": {
        "navarasa": "Shringara",
        "meaning": "Love",
        "emoji": "🥰"
    }
}

# ============================================================================
# EMOTION COLOR SCHEMES
# ============================================================================

# Matplotlib/Streamlit UI colors (hex)
EMOTION_COLORS = {
    "anger": "#FF4444",
    "surprise": "#FF8C00",
    "contempt": "#8B4513",
    "happy": "#FFD700",
    "neutral": "#808080",
    "fear": "#9B59B6",
    "sad": "#4169E1",
    "disgust": "#228B22",
    "shringara": "#FF69B4"
}

# OpenCV BGR colors (for video overlay)
EMOTION_CV_COLORS = {
    "anger": (50, 50, 255),          # Red
    "surprise": (0, 140, 255),       # Orange
    "contempt": (30, 80, 160),       # Brown
    "happy": (0, 215, 255),          # Yellow
    "neutral": (150, 150, 150),      # Gray
    "fear": (180, 50, 200),          # Purple
    "sad": (235, 100, 50),           # Blue
    "disgust": (40, 150, 40),        # Green
    "shringara": (255, 105, 180)     # Hot Pink
}

# ============================================================================
# EMOTION-TO-AUDIO MAPPING (Spotify Audio Feature Ranges)
# ============================================================================

EMOTION_AUDIO_PROFILES = {
    "anger": {
        "valence": (0.0, 0.35),
        "energy": (0.75, 1.0),
        "danceability": (0.4, 0.75)
    },
    "surprise": {
        "valence": (0.5, 0.9),
        "energy": (0.65, 1.0),
        "danceability": (0.5, 0.85)
    },
    "contempt": {
        "valence": (0.2, 0.5),
        "energy": (0.5, 0.8),
        "danceability": (0.35, 0.65)
    },
    "happy": {
        "valence": (0.7, 1.0),
        "energy": (0.6, 1.0),
        "danceability": (0.65, 1.0)
    },
    "neutral": {
        "valence": (0.35, 0.65),
        "energy": (0.25, 0.6),
        "danceability": (0.35, 0.65)
    },
    "fear": {
        "valence": (0.05, 0.35),
        "energy": (0.25, 0.6),
        "danceability": (0.15, 0.5)
    },
    "sad": {
        "valence": (0.0, 0.3),
        "energy": (0.0, 0.4),
        "danceability": (0.0, 0.4)
    },
    "disgust": {
        "valence": (0.0, 0.3),
        "energy": (0.4, 0.72),
        "danceability": (0.3, 0.6)
    },
    "shringara": {
        "valence": (0.65, 1.0),
        "energy": (0.3, 0.65),
        "danceability": (0.45, 0.75)
    }
}

# ============================================================================
# MUSIC RECOMMENDATION CONFIGURATION
# ============================================================================

ROMANTIC_GENRES = ["romance", "r&b", "soul", "love", "latin", "indie", "acoustic"]

# ============================================================================
# MODEL TRAINING CONFIGURATION
# ============================================================================

# Class weights to handle AffectNet class imbalance
CLASS_WEIGHTS = {
    0: 2.1,   # anger — underrepresented
    1: 1.4,   # surprise
    2: 3.8,   # contempt — heavily underrepresented
    3: 0.6,   # happy — overrepresented
    4: 0.7,   # neutral — overrepresented
    5: 2.0,   # fear
    6: 1.2,   # sad
    7: 1.9    # disgust
}
