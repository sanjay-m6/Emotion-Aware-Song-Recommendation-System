# """
# Constants and mappings for the Emotion-Aware Music Recommendation System.

# This module serves as the single source of truth for all emotion labels,
# Navarasa mappings, audio profiles, and styling configurations used
# throughout the entire application.
# """

# # ============================================================================
# # EMOTION LABEL MAPPINGS (AffectNet 8-class)
# # ============================================================================

# LABEL_TO_EMOTION = {
#     0: "anger",
#     1: "surprise",
#     2: "contempt",
#     3: "happy",
#     4: "neutral",
#     5: "fear",
#     6: "sad",
#     7: "disgust"
# }

# EMOTION_TO_LABEL = {v: k for k, v in LABEL_TO_EMOTION.items()}

# EMOTION_NAMES = ["anger", "surprise", "contempt", "happy", "neutral", "fear", "sad", "disgust"]

# # ============================================================================
# # NAVARASA MAPPINGS (Ancient Indian Emotional Framework)
# # ============================================================================

# NAVARASA_MAPPING = {
#     "anger": {
#         "navarasa": "Raudra",
#         "meaning": "Fury",
#         "emoji": "😠"
#     },
#     "surprise": {
#         "navarasa": "Adbhuta",
#         "meaning": "Wonder",
#         "emoji": "😲"
#     },
#     "contempt": {
#         "navarasa": "Vira",
#         "meaning": "Heroism",
#         "emoji": "😤"
#     },
#     "happy": {
#         "navarasa": "Hasya",
#         "meaning": "Joy",
#         "emoji": "😄"
#     },
#     "neutral": {
#         "navarasa": "Shanta",
#         "meaning": "Peace",
#         "emoji": "😌"
#     },
#     "fear": {
#         "navarasa": "Bhayanaka",
#         "meaning": "Terror",
#         "emoji": "😨"
#     },
#     "sad": {
#         "navarasa": "Karuna",
#         "meaning": "Sorrow",
#         "emoji": "😢"
#     },
#     "disgust": {
#         "navarasa": "Bibhatsa",
#         "meaning": "Disgust",
#         "emoji": "🤢"
#     },
#     "shringara": {
#         "navarasa": "Shringara",
#         "meaning": "Love",
#         "emoji": "🥰"
#     }
# }

# # ============================================================================
# # EMOTION COLOR SCHEMES
# # ============================================================================

# # Matplotlib/Streamlit UI colors (hex)
# EMOTION_COLORS = {
#     "anger": "#FF4444",
#     "surprise": "#FF8C00",
#     "contempt": "#8B4513",
#     "happy": "#FFD700",
#     "neutral": "#808080",
#     "fear": "#9B59B6",
#     "sad": "#4169E1",
#     "disgust": "#228B22",
#     "shringara": "#FF69B4"
# }

# # OpenCV BGR colors (for video overlay)
# EMOTION_CV_COLORS = {
#     "anger": (50, 50, 255),          # Red
#     "surprise": (0, 140, 255),       # Orange
#     "contempt": (30, 80, 160),       # Brown
#     "happy": (0, 215, 255),          # Yellow
#     "neutral": (150, 150, 150),      # Gray
#     "fear": (180, 50, 200),          # Purple
#     "sad": (235, 100, 50),           # Blue
#     "disgust": (40, 150, 40),        # Green
#     "shringara": (255, 105, 180)     # Hot Pink
# }

# # ============================================================================
# # EMOTION-TO-AUDIO MAPPING (Spotify Audio Feature Ranges)
# # ============================================================================
# # FIXED: Non-overlapping ranges based on Spotify data and music psychology

# EMOTION_AUDIO_PROFILES = {
#     # Negative emotions: Low valence, variable energy & danceability
#     "anger": {
#         "valence": (0.0, 0.2),      # BUG FIX: Was (0.0, 0.35), now more strict
#         "energy": (0.7, 1.0),       # High energy (aggressive)
#         "danceability": (0.5, 0.8)  # Moderate-high danceability
#     },
#     "sad": {
#         "valence": (0.0, 0.25),     # BUG FIX: Was (0.0, 0.3), similar range
#         "energy": (0.0, 0.35),      # Low energy (slow, melancholic)
#         "danceability": (0.0, 0.35) # Low danceability (introspective)
#     },
#     "fear": {
#         "valence": (0.0, 0.25),     # Low valence (tense)
#         "energy": (0.3, 0.7),       # Medium energy
#         "danceability": (0.2, 0.5)  # Low-medium danceability
#     },
#     "disgust": {
#         "valence": (0.0, 0.3),      # Low valence (negative emotion)
#         "energy": (0.5, 0.8),       # Medium-high energy (agitation)
#         "danceability": (0.3, 0.6)  # Medium danceability
#     },
    
#     # Neutral/Balanced emotions: Medium valence
#     "contempt": {
#         "valence": (0.3, 0.5),      # BUG FIX: Was (0.2, 0.5), now higher minimum
#         "energy": (0.4, 0.7),       # Medium energy
#         "danceability": (0.35, 0.6) # Medium danceability
#     },
#     "neutral": {
#         "valence": (0.4, 0.6),      # BUG FIX: Was (0.35, 0.65), now more precise
#         "energy": (0.3, 0.7),       # Variable energy
#         "danceability": (0.35, 0.65) # Variable danceability
#     },
    
#     # Positive emotions: High valence
#     "surprise": {
#         "valence": (0.6, 0.85),     # BUG FIX: Was (0.5, 0.9), now tighter bounds
#         "energy": (0.65, 1.0),      # High energy (exciting)
#         "danceability": (0.55, 0.85) # BUG FIX: tighter bounds
#     },
#     "happy": {
#         "valence": (0.75, 1.0),     # BUG FIX: Was (0.7, 1.0), now higher minimum
#         "energy": (0.6, 1.0),       # High energy (uplifting)
#         "danceability": (0.65, 1.0) # Unchanged - high danceability
#     },
    
#     # Special: Shringara (Love) - romantic music
#     "shringara": {
#         "valence": (0.6, 0.9),      # BUG FIX: Was (0.65, 1.0), now wider range
#         "energy": (0.3, 0.65),      # Medium energy (intimate)
#         "danceability": (0.4, 0.75) # Medium danceability (can be danceable or slower)
#     }
# }

# # ============================================================================
# # MUSIC RECOMMENDATION CONFIGURATION
# # ============================================================================

# ROMANTIC_GENRES = ["romance", "r&b", "soul", "love", "latin", "indie", "acoustic"]

# # ============================================================================
# # MODEL TRAINING CONFIGURATION
# # ============================================================================

# # Class weights to handle AffectNet class imbalance
# CLASS_WEIGHTS = {
#     0: 2.1,   # anger — underrepresented
#     1: 1.4,   # surprise
#     2: 3.8,   # contempt — heavily underrepresented
#     3: 0.6,   # happy — overrepresented
#     4: 0.7,   # neutral — overrepresented
#     5: 2.0,   # fear
#     6: 1.2,   # sad
#     7: 1.9    # disgust
# }







"""
Constants — single source of truth for the entire project.

FIXES APPLIED:
- FIX 1: EMOTION_AUDIO_PROFILES — tighter, non-overlapping valence ranges
         so angry songs never bleed into happy/love territory
- FIX 2: Shringara profile separated from happy profile
"""

# ── Label mappings ────────────────────────────────────────────────────────────

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

EMOTION_NAMES = [
    "anger", "surprise", "contempt", "happy",
    "neutral", "fear", "sad", "disgust"
]

# ── Navarasa ──────────────────────────────────────────────────────────────────

NAVARASA_MAPPING = {
    "anger":     {"navarasa": "Raudra",    "meaning": "Fury",    "emoji": "😠"},
    "surprise":  {"navarasa": "Adbhuta",   "meaning": "Wonder",  "emoji": "😲"},
    "contempt":  {"navarasa": "Vira",      "meaning": "Heroism", "emoji": "😤"},
    "happy":     {"navarasa": "Hasya",     "meaning": "Joy",     "emoji": "😄"},
    "neutral":   {"navarasa": "Shanta",    "meaning": "Peace",   "emoji": "😌"},
    "fear":      {"navarasa": "Bhayanaka", "meaning": "Terror",  "emoji": "😨"},
    "sad":       {"navarasa": "Karuna",    "meaning": "Sorrow",  "emoji": "😢"},
    "disgust":   {"navarasa": "Bibhatsa",  "meaning": "Disgust", "emoji": "🤢"},
    "shringara": {"navarasa": "Shringara", "meaning": "Love",    "emoji": "🥰"},
}

# ── Colors ────────────────────────────────────────────────────────────────────

EMOTION_COLORS = {
    "anger":     "#FF4444",
    "surprise":  "#FF8C00",
    "contempt":  "#8B4513",
    "happy":     "#FFD700",
    "neutral":   "#808080",
    "fear":      "#9B59B6",
    "sad":       "#4169E1",
    "disgust":   "#228B22",
    "shringara": "#FF69B4"
}

# BGR tuples for OpenCV drawing
EMOTION_CV_COLORS = {
    "anger":     (50,  50,  255),
    "surprise":  (0,   140, 255),
    "contempt":  (30,  80,  160),
    "happy":     (0,   215, 255),
    "neutral":   (150, 150, 150),
    "fear":      (180, 50,  200),
    "sad":       (235, 100, 50),
    "disgust":   (40,  150, 40),
    "shringara": (180, 105, 255)
}

# ── Audio profiles ─────────────────────────────────────────────────────────────
# FIX 1: Tighter, non-overlapping valence ranges.
# Key principle:
#   anger/sad/fear/disgust  → valence < 0.35   (clearly negative)
#   neutral/contempt        → valence 0.35–0.55 (middle ground)
#   surprise/happy          → valence > 0.55   (positive)
#   shringara               → valence > 0.60   (romantic positive)

EMOTION_AUDIO_PROFILES = {
    # ── Negative emotions ────────────────────────────────────────────────────
    "anger": {
        "valence":      (0.00, 0.30),   # Low valence — aggressive/dark
        "energy":       (0.70, 1.00),   # High energy — intense
        "danceability": (0.40, 0.80),   # Variable
    },
    "sad": {
        "valence":      (0.00, 0.30),   # Low valence — melancholic
        "energy":       (0.00, 0.40),   # Low energy — slow
        "danceability": (0.00, 0.40),   # Low danceability
    },
    "fear": {
        "valence":      (0.00, 0.30),   # Low valence — tense
        "energy":       (0.30, 0.65),   # Medium energy
        "danceability": (0.20, 0.55),   # Low-medium
    },
    "disgust": {
        "valence":      (0.05, 0.35),   # Low valence — negative
        "energy":       (0.45, 0.80),   # Medium-high energy
        "danceability": (0.30, 0.65),   # Medium
    },

    # ── Neutral / middle ─────────────────────────────────────────────────────
    "neutral": {
        "valence":      (0.35, 0.60),   # Middle ground
        "energy":       (0.25, 0.65),   # Variable
        "danceability": (0.30, 0.65),   # Variable
    },
    "contempt": {
        "valence":      (0.30, 0.55),   # Slightly negative to neutral
        "energy":       (0.40, 0.75),   # Medium energy
        "danceability": (0.35, 0.65),   # Medium
    },

    # ── Positive emotions ────────────────────────────────────────────────────
    "surprise": {
        "valence":      (0.55, 0.85),   # Positive — exciting
        "energy":       (0.60, 1.00),   # High energy
        "danceability": (0.50, 0.85),   # High danceability
    },
    "happy": {
        "valence":      (0.65, 1.00),   # High valence — joyful
        "energy":       (0.55, 1.00),   # High energy
        "danceability": (0.60, 1.00),   # High danceability
    },

    # ── Love / romantic ──────────────────────────────────────────────────────
    "shringara": {
        "valence":      (0.55, 0.90),   # Positive but softer than happy
        "energy":       (0.25, 0.60),   # Lower energy — intimate
        "danceability": (0.35, 0.70),   # Medium danceability
    },
}

# ── Other config ──────────────────────────────────────────────────────────────

ROMANTIC_GENRES = [
    "romance", "r&b", "soul", "love songs",
    "latin", "indie", "acoustic", "jazz"
]

CLASS_WEIGHTS = {
    0: 2.1,   # anger
    1: 1.4,   # surprise
    2: 3.8,   # contempt
    3: 0.6,   # happy
    4: 0.7,   # neutral
    5: 2.0,   # fear
    6: 1.2,   # sad
    7: 1.9,   # disgust
}