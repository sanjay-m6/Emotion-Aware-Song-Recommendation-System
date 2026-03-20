"""
Streamlit web UI for the Emotion-Aware Music Recommendation System.

Features:
- Live webcam emotion detection with Navarasa overlay
- Real-time music recommendations based on detected emotion
- Audio feature visualization (valence, energy, danceability)
- Session statistics and emotion timeline
- Dark theme with Spotify-inspired styling
"""

import sys
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.emotion_history import EmotionHistory
from utils.constants import NAVARASA_MAPPING, EMOTION_COLORS
from music.recommendations import load_songs, get_recommendations
from app.webcam import EmotionDetector


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🎭 Navarasa Music",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING (Dark Theme + Spotify-inspired)
# ============================================================================

st.markdown("""
<style>
.stApp {
    background-color: #121212;
    color: #FFFFFF;
}

.song-card {
    background: #1E1E1E;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    border: 1px solid #282828;
    transition: border 0.2s;
}

.song-card:hover {
    border: 1px solid #1DB954;
}

.song-title {
    font-size: 1.1rem;
    font-weight: bold;
    color: #FFFFFF;
}

.song-artist {
    color: #B3B3B3;
    font-size: 0.9rem;
    margin: 4px 0;
}

.song-genre {
    display: inline-block;
    background: #282828;
    color: #1DB954;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.8rem;
    margin: 4px 0;
}

.song-stats {
    color: #B3B3B3;
    font-size: 0.8rem;
    margin-top: 8px;
}

.play-btn {
    background: #1DB954;
    color: #000000;
    border: none;
    border-radius: 20px;
    padding: 6px 18px;
    font-weight: bold;
    cursor: pointer;
    margin-top: 10px;
    width: 100%;
}

.play-btn:hover {
    background: #1ed760;
}

.navarasa-badge {
    font-size: 1.8rem;
    font-weight: bold;
    padding: 8px 20px;
    border-radius: 30px;
    display: inline-block;
    margin: 8px 0;
    color: #FFFFFF;
}

.emotion-bar {
    height: 8px;
    border-radius: 4px;
    margin: 3px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = EmotionHistory(max_len=200)

if "detector" not in st.session_state:
    st.session_state.detector = None

if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = None

if "songs_df" not in st.session_state:
    try:
        st.session_state.songs_df = load_songs()
    except Exception as e:
        st.error(f"⚠️ Could not load songs database: {e}")
        st.session_state.songs_df = None


# ============================================================================
# HELPER: Convert camera input to numpy array
# ============================================================================

def camera_to_numpy(camera_image) -> np.ndarray:
    """
    Safely convert st.camera_input UploadedFile → RGB numpy array.

    Args:
        camera_image: UploadedFile from st.camera_input

    Returns:
        numpy array of shape (H, W, 3) in RGB format
    """
    # ── FIX 1: camera_image is an UploadedFile, not an array ──
    # Must open via PIL first, then convert to numpy
    pil_image = Image.open(camera_image).convert("RGB")
    return np.array(pil_image)


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Safely convert a frame to RGB numpy array for st.image().

    Handles:
    - BGR (OpenCV) → RGB conversion
    - PIL Image → numpy
    - Validates shape is (H, W, 3)

    Args:
        frame: numpy array or PIL Image

    Returns:
        RGB numpy array of shape (H, W, 3)
    """
    # ── FIX 2: handle PIL Image coming from detector ──
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))

    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    # ── FIX 3: OpenCV returns BGR — convert to RGB for st.image ──
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = frame[:, :, ::-1].copy()

    return frame


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.markdown("## 🎭 Navarasa Music")
st.sidebar.markdown("""
Detect your emotion → Get music matched to your mood using
the ancient Indian Navarasa framework.
""")

st.sidebar.divider()

# Model selector
model_choice = st.sidebar.radio(
    "Select Model:",
    options=["MobileNetV2 (Recommended)", "Custom CNN"],
    index=1   # default to Custom CNN since that's your best model
)
model_type = "mobilenet" if "MobileNetV2" in model_choice else "custom_cnn"

# Confidence threshold slider
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold:",
    min_value=0.3,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Only act on detections above this confidence"
)

st.sidebar.divider()

# Session stats
st.sidebar.markdown("### 📊 Session Stats")
session_stats = st.session_state.emotion_history.get_session_stats()

if session_stats["total_detections"] > 0:
    st.sidebar.metric("Total Detections", session_stats["total_detections"])
    st.sidebar.metric("Dominant Emotion", session_stats["dominant_emotion"].title())

    if session_stats["navarasa_counts"]:
        navarasa_data = pd.DataFrame({
            "Navarasa": list(session_stats["navarasa_counts"].keys()),
            "Count":    list(session_stats["navarasa_counts"].values())
        })

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#121212")
        ax.barh(navarasa_data["Navarasa"], navarasa_data["Count"], color="#1DB954")
        ax.set_xlabel("Count", color="#FFFFFF")
        ax.set_facecolor("#121212")
        ax.tick_params(colors="#FFFFFF")
        for spine in ax.spines.values():
            spine.set_color("#FFFFFF")
        st.sidebar.pyplot(fig, use_container_width=True)
else:
    st.sidebar.info("👀 Detections will appear here")

# Save session button
if st.sidebar.button("💾 Save Session"):
    session_file = Path("session_log.json")
    st.session_state.emotion_history.to_json(str(session_file))
    st.sidebar.success("✅ Session saved!")


# ============================================================================
# MAIN CONTENT: TWO-COLUMN LAYOUT
# ============================================================================

col_left, col_right = st.columns([1.1, 1])

# ============================================================================
# LEFT COLUMN: LIVE EMOTION DETECTION
# ============================================================================

with col_left:
    st.markdown("## 📸 Live Detection")

    camera_image = st.camera_input(
        "Capture your face",
        label_visibility="collapsed"
    )

    if camera_image:

        # ── FIX 1: Correct conversion from UploadedFile → numpy ──
        image_array = camera_to_numpy(camera_image)

        # Initialize detector if model changed or not yet loaded
        current_model = getattr(st.session_state.detector, "model_type", None)
        if st.session_state.detector is None or current_model != model_type:
            checkpoint_dir  = Path(__file__).parent.parent / "models" / "checkpoints"
            checkpoint_path = checkpoint_dir / f"{model_type}_best.pth"

            if checkpoint_path.exists():
                with st.spinner(f"Loading {model_type} model..."):
                    try:
                        st.session_state.detector = EmotionDetector(
                            str(checkpoint_path),
                            model_type=model_type
                        )
                        # Store model_type on detector for change detection
                        st.session_state.detector.model_type = model_type
                    except Exception as e:
                        st.error(f"⚠️ Could not load model: {e}")
                        st.session_state.detector = None
            else:
                st.warning(f"⚠️ Checkpoint not found: {checkpoint_path}")

        # Run detection
        if st.session_state.detector:
            try:
                result = st.session_state.detector.detect(image_array)

                # ── FIX 2 + 3: Safe frame display with BGR→RGB conversion ──
                if result.get("frame") is not None:
                    rgb_frame = frame_to_rgb(result["frame"])
                    st.image(rgb_frame, use_container_width=True)
                else:
                    # Fallback: show original camera image
                    st.image(image_array, use_container_width=True)

                st.session_state.detected_emotion = result

                if result["face_found"] and result["confidence"] >= confidence_threshold:
                    st.session_state.emotion_history.add(
                        result["emotion"],
                        result["confidence"]
                    )

            except Exception as e:
                st.error(f"⚠️ Detection error: {e}")
                # ── FIX 4: fallback always uses use_container_width ──
                st.image(image_array, use_container_width=True)
        else:
            st.image(image_array, use_container_width=True)

    # ── Display emotion result ──────────────────────────────────────────────
    if (st.session_state.detected_emotion
            and st.session_state.detected_emotion.get("face_found")):

        result       = st.session_state.detected_emotion
        emotion      = result["emotion"]
        confidence   = result["confidence"]
        navarasa     = result["navarasa"]
        meaning      = result["navarasa_meaning"]
        emoji        = result["navarasa_emoji"]
        badge_color  = EMOTION_COLORS.get(emotion, "#808080")

        # Navarasa badge
        st.markdown(
            f'<div class="navarasa-badge" style="background-color: {badge_color};">'
            f'{emoji} {navarasa}<br><small>{meaning}</small></div>',
            unsafe_allow_html=True
        )

        # Confidence bar
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")

        # All 8 emotion scores
        with st.expander("🎭 All 8 Emotion Scores"):
            all_scores   = result["all_scores"]
            emotions     = list(all_scores.keys())
            scores       = list(all_scores.values())
            colors_list  = [EMOTION_COLORS.get(e, "#808080") for e in emotions]

            fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1E1E1E")
            ax.barh(emotions, scores, color=colors_list)
            ax.set_xlabel("Probability", color="#FFFFFF")
            ax.set_facecolor("#1E1E1E")
            ax.tick_params(colors="#FFFFFF")
            for spine in ax.spines.values():
                spine.set_color("#FFFFFF")
            ax.set_xlim(0, 1)
            st.pyplot(fig, use_container_width=True)

    elif camera_image:
        st.warning("😐 No face detected — try moving closer to the camera.")
    else:
        st.info("👀 Take a photo above to detect your emotion.")

    # Emotion timeline
    st.markdown("### 📈 Emotion Timeline")
    if st.session_state.emotion_history.get_session_stats()["total_detections"] > 0:
        fig = st.session_state.emotion_history.plot_timeline()
        st.pyplot(fig, use_container_width=True)


# ============================================================================
# RIGHT COLUMN: MUSIC RECOMMENDATIONS
# ============================================================================

with col_right:
    st.markdown("## 🎵 Your Navarasa Playlist")

    if (st.session_state.detected_emotion
            and st.session_state.detected_emotion.get("face_found")):

        result     = st.session_state.detected_emotion
        emotion    = result["emotion"]
        confidence = result["confidence"]

        navarasa_info = NAVARASA_MAPPING.get(emotion, {})
        emoji     = navarasa_info.get("emoji", "🎵")
        navarasa  = navarasa_info.get("navarasa", emotion.title())
        meaning   = navarasa_info.get("meaning", "")

        st.markdown(f"### {emoji} {navarasa} — {meaning}")

        if st.session_state.songs_df is not None:
            st.button("🔄 Refresh Recommendations")

            recommendations = get_recommendations(
                emotion      = emotion,
                confidence   = confidence,
                songs_df     = st.session_state.songs_df,
                history      = st.session_state.emotion_history.get_recent_tracks(10),
                recent_genres= st.session_state.emotion_history.get_recent_genres(5),
                n            = 5
            )

            if recommendations:
                for rec in recommendations:
                    track_name   = rec.get("track_name",   "Unknown")
                    artists      = rec.get("artists",       "Unknown")
                    genre        = rec.get("track_genre",   "")
                    valence      = rec.get("valence",       0)
                    energy       = rec.get("energy",        0)
                    danceability = rec.get("danceability",  0)
                    popularity   = rec.get("popularity",    0)

                    st.markdown(f"""
                    <div class="song-card">
                        <div class="song-title">🎵 {track_name}</div>
                        <div class="song-artist">👤 {artists}</div>
                        <div class="song-genre">{genre}</div>
                        <div class="song-stats">
                            ❤️ Valence: {valence:.2f} &nbsp;&nbsp;
                            ⚡ Energy: {energy:.2f} &nbsp;&nbsp;
                            💃 Dance: {danceability:.2f} &nbsp;&nbsp;
                            ⭐ Popularity: {popularity}
                        </div>
                        <button class="play-btn">▶ Play on Spotify</button>
                    </div>
                    """, unsafe_allow_html=True)

                    st.session_state.emotion_history.add_played_track(
                        track_name, genre
                    )
            else:
                st.info("No songs match this emotion profile. Try a different emotion!")

        else:
            st.warning("⚠️ Songs database not loaded. Run: python music/preprocess_songs.py")

        # About this Navarasa
        with st.expander("📜 About this Navarasa"):
            navarasa_info = NAVARASA_MAPPING.get(emotion, {})
            st.markdown(f"""
### {navarasa_info.get('emoji','')} {navarasa_info.get('navarasa', emotion.title())}

**Meaning:** {navarasa_info.get('meaning', '')}

**Origin:** One of the 9 Rasas from ancient Indian aesthetic theory,
first described in Bharata Muni's *Natyashastra* (200 BCE – 200 CE).

**In Music:** Songs matched to {navarasa_info.get('navarasa', emotion)} are
selected based on their audio features — valence, energy, and danceability —
to match the emotional quality of {navarasa_info.get('meaning', emotion).lower()}.
            """)

    else:
        st.info("👈 Detect your emotion in the left column to get recommendations!")
        st.markdown("""
        ### How it works:
        1. 📸 Click **Take Photo** on the left
        2. 🧠 CNN detects your emotion (8 classes)
        3. 🎭 Mapped to one of the **9 Navarasa**
        4. 🎵 Songs recommended by audio features
        """)