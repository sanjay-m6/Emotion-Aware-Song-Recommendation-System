# """
# Streamlit web UI for the Emotion-Aware Music Recommendation System.

# Features:
# - Live webcam emotion detection with Navarasa overlay
# - Real-time music recommendations based on detected emotion
# - Audio feature visualization (valence, energy, danceability)
# - Session statistics and emotion timeline
# - Dark theme with Spotify-inspired styling
# """

# import sys
# from pathlib import Path
# from typing import Optional

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # Add parent directory to path
# sys.path.insert(0, str(Path(__file__).parent.parent))

# from utils.emotion_history import EmotionHistory
# from utils.constants import NAVARASA_MAPPING, EMOTION_COLORS
# from music.recommendations import load_songs, get_recommendations
# from app.webcam import EmotionDetector


# # ============================================================================
# # PAGE CONFIGURATION
# # ============================================================================

# st.set_page_config(
#     page_title="🎭 Navarasa Music",
#     page_icon="🎭",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ============================================================================
# # CUSTOM CSS STYLING (Dark Theme + Spotify-inspired)
# # ============================================================================

# st.markdown("""
# <style>
# .stApp {
#     background-color: #121212;
#     color: #FFFFFF;
# }

# .song-card {
#     background: #1E1E1E;
#     border-radius: 12px;
#     padding: 16px;
#     margin-bottom: 12px;
#     border: 1px solid #282828;
#     transition: border 0.2s;
# }

# .song-card:hover {
#     border: 1px solid #1DB954;
# }

# .song-title {
#     font-size: 1.1rem;
#     font-weight: bold;
#     color: #FFFFFF;
# }

# .song-artist {
#     color: #B3B3B3;
#     font-size: 0.9rem;
#     margin: 4px 0;
# }

# .song-genre {
#     display: inline-block;
#     background: #282828;
#     color: #1DB954;
#     border-radius: 20px;
#     padding: 2px 10px;
#     font-size: 0.8rem;
#     margin: 4px 0;
# }

# .song-stats {
#     color: #B3B3B3;
#     font-size: 0.8rem;
#     margin-top: 8px;
# }

# .play-btn {
#     background: #1DB954;
#     color: #000000;
#     border: none;
#     border-radius: 20px;
#     padding: 6px 18px;
#     font-weight: bold;
#     cursor: pointer;
#     margin-top: 10px;
#     width: 100%;
# }

# .play-btn:hover {
#     background: #1ed760;
# }

# .navarasa-badge {
#     font-size: 1.8rem;
#     font-weight: bold;
#     padding: 8px 20px;
#     border-radius: 30px;
#     display: inline-block;
#     margin: 8px 0;
#     color: #FFFFFF;
# }

# .emotion-bar {
#     height: 8px;
#     border-radius: 4px;
#     margin: 3px 0;
# }
# </style>
# """, unsafe_allow_html=True)

# # ============================================================================
# # SESSION STATE INITIALIZATION
# # ============================================================================

# if "emotion_history" not in st.session_state:
#     st.session_state.emotion_history = EmotionHistory(max_len=200)

# if "detector" not in st.session_state:
#     st.session_state.detector = None

# if "detected_emotion" not in st.session_state:
#     st.session_state.detected_emotion = None

# if "songs_df" not in st.session_state:
#     try:
#         st.session_state.songs_df = load_songs()
#     except Exception as e:
#         st.error(f"⚠️ Could not load songs database: {e}")
#         st.session_state.songs_df = None


# # ============================================================================
# # HELPER: Convert camera input to numpy array
# # ============================================================================

# def camera_to_numpy(camera_image) -> np.ndarray:
#     """
#     Safely convert st.camera_input UploadedFile → RGB numpy array.

#     Args:
#         camera_image: UploadedFile from st.camera_input

#     Returns:
#         numpy array of shape (H, W, 3) in RGB format
#     """
#     # ── FIX 1: camera_image is an UploadedFile, not an array ──
#     # Must open via PIL first, then convert to numpy
#     pil_image = Image.open(camera_image).convert("RGB")
#     return np.array(pil_image)


# def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
#     """
#     Safely convert a frame to RGB numpy array for st.image().

#     Handles:
#     - BGR (OpenCV) → RGB conversion
#     - PIL Image → numpy
#     - Validates shape is (H, W, 3)

#     Args:
#         frame: numpy array or PIL Image

#     Returns:
#         RGB numpy array of shape (H, W, 3)
#     """
#     # ── FIX 2: handle PIL Image coming from detector ──
#     if isinstance(frame, Image.Image):
#         return np.array(frame.convert("RGB"))

#     if not isinstance(frame, np.ndarray):
#         frame = np.array(frame)

#     # ── FIX 3: OpenCV returns BGR — convert to RGB for st.image ──
#     if len(frame.shape) == 3 and frame.shape[2] == 3:
#         frame = frame[:, :, ::-1].copy()

#     return frame


# # ============================================================================
# # SIDEBAR CONFIGURATION
# # ============================================================================

# st.sidebar.markdown("## 🎭 Navarasa Music")
# st.sidebar.markdown("""
# Detect your emotion → Get music matched to your mood using
# the ancient Indian Navarasa framework.
# """)

# st.sidebar.divider()

# # Model selector
# model_choice = st.sidebar.radio(
#     "Select Model:",
#     options=["MobileNetV2 (Recommended)", "Custom CNN"],
#     index=1   # default to Custom CNN since that's your best model
# )
# model_type = "mobilenet" if "MobileNetV2" in model_choice else "custom_cnn"

# # Confidence threshold slider
# confidence_threshold = st.sidebar.slider(
#     "Confidence Threshold:",
#     min_value=0.3,
#     max_value=0.9,
#     value=0.5,
#     step=0.05,
#     help="Only act on detections above this confidence"
# )

# st.sidebar.divider()

# # BUG FIX: Manual emotion override for testing (if model detection is wrong)
# st.sidebar.markdown("### 🎭 Manual Override (for testing)")
# manual_emotion = st.sidebar.selectbox(
#     "Override detected emotion:",
#     options=["(Use detected emotion)"] + list(NAVARASA_MAPPING.keys()),
#     index=0,
#     help="Select an emotion to override automatic detection for testing recommendations"
# )

# st.sidebar.divider()

# # Session stats
# st.sidebar.markdown("### 📊 Session Stats")
# session_stats = st.session_state.emotion_history.get_session_stats()

# if session_stats["total_detections"] > 0:
#     st.sidebar.metric("Total Detections", session_stats["total_detections"])
#     st.sidebar.metric("Dominant Emotion", session_stats["dominant_emotion"].title())

#     if session_stats["navarasa_counts"]:
#         navarasa_data = pd.DataFrame({
#             "Navarasa": list(session_stats["navarasa_counts"].keys()),
#             "Count":    list(session_stats["navarasa_counts"].values())
#         })

#         fig, ax = plt.subplots(figsize=(8, 4), facecolor="#121212")
#         ax.barh(navarasa_data["Navarasa"], navarasa_data["Count"], color="#1DB954")
#         ax.set_xlabel("Count", color="#FFFFFF")
#         ax.set_facecolor("#121212")
#         ax.tick_params(colors="#FFFFFF")
#         for spine in ax.spines.values():
#             spine.set_color("#FFFFFF")
#         st.sidebar.pyplot(fig, use_container_width=True)
# else:
#     st.sidebar.info("👀 Detections will appear here")

# # Save session button
# if st.sidebar.button("💾 Save Session"):
#     session_file = Path("session_log.json")
#     st.session_state.emotion_history.to_json(str(session_file))
#     st.sidebar.success("✅ Session saved!")


# # ============================================================================
# # MAIN CONTENT: TWO-COLUMN LAYOUT
# # ============================================================================

# col_left, col_right = st.columns([1.1, 1])

# # ============================================================================
# # LEFT COLUMN: LIVE EMOTION DETECTION
# # ============================================================================

# with col_left:
#     st.markdown("## 📸 Live Detection")

#     camera_image = st.camera_input(
#         "Capture your face",
#         label_visibility="collapsed"
#     )

#     if camera_image:

#         # ── FIX 1: Correct conversion from UploadedFile → numpy ──
#         image_array = camera_to_numpy(camera_image)

#         # Initialize detector if model changed or not yet loaded
#         current_model = getattr(st.session_state.detector, "_model_type", None)
#         if st.session_state.detector is None or current_model != model_type:
#             checkpoint_dir  = Path(__file__).parent.parent / "models" / "checkpoints"
#             checkpoint_path = checkpoint_dir / f"{model_type}_best.pth"

#             if checkpoint_path.exists():
#                 with st.spinner(f"Loading {model_type} model..."):
#                     try:
#                         st.session_state.detector = EmotionDetector(
#                             str(checkpoint_path),
#                             model_type=model_type
#                         )
#                     except Exception as e:
#                         st.error(f"⚠️ Could not load model: {e}")
#                         st.session_state.detector = None
#             else:
#                 st.warning(f"⚠️ Checkpoint not found: {checkpoint_path}")

#         # Run detection
#         if st.session_state.detector:
#             try:
#                 result = st.session_state.detector.detect(image_array)

#                 # ── FIX 2 + 3: Safe frame display with BGR→RGB conversion ──
#                 if result.get("frame") is not None:
#                     rgb_frame = frame_to_rgb(result["frame"])
#                     st.image(rgb_frame, use_container_width=True)
#                 else:
#                     # Fallback: show original camera image
#                     st.image(image_array, use_container_width=True)

#                 st.session_state.detected_emotion = result

#                 if result["face_found"] and result["confidence"] >= confidence_threshold:
#                     st.session_state.emotion_history.add(
#                         result["emotion"],
#                         result["confidence"]
#                     )
                    
#                     # BUG FIX: Add debugging output for emotion detection
#                     with st.expander("🔍 Debug Info - Emotion Scores"):
#                         debug_col1, debug_col2 = st.columns(2)
                        
#                         with debug_col1:
#                             st.metric("Detected Emotion", result["emotion"].title())
#                             st.metric("Confidence", f"{result['confidence']:.1%}")
                        
#                         with debug_col2:
#                             st.write("### All Emotion Scores:")
#                             scores_text = ""
#                             for emotion_name, score in sorted(result["all_scores"].items(), 
#                                                            key=lambda x: x[1], reverse=True):
#                                 scores_text += f"{emotion_name:12} : {score:.1%}\n"
#                             st.code(scores_text)

#     # ── Display emotion result ──────────────────────────────────────────────
#             except Exception as e:
#                 st.error(f"❌ Emotion detection error: {e}")

#     # ── Display emotion result ──────────────────────────────────────────────
#     if st.session_state.detected_emotion and st.session_state.detected_emotion.get("face_found"):

#         result       = st.session_state.detected_emotion
#         emotion      = result["emotion"]
#         confidence   = result["confidence"]
#         navarasa     = result["navarasa"]
#         meaning      = result["navarasa_meaning"]
#         emoji        = result["navarasa_emoji"]
#         badge_color  = EMOTION_COLORS.get(emotion, "#808080")

#         # Navarasa badge
#         st.markdown(
#             f'<div class="navarasa-badge" style="background-color: {badge_color};">'
#             f'{emoji} {navarasa}<br><small>{meaning}</small></div>',
#             unsafe_allow_html=True
#         )

#         # Confidence bar
#         st.progress(confidence, text=f"Confidence: {confidence:.1%}")

#         # All 8 emotion scores
#         with st.expander("🎭 All 8 Emotion Scores"):
#             all_scores   = result["all_scores"]
#             emotions     = list(all_scores.keys())
#             scores       = list(all_scores.values())
#             colors_list  = [EMOTION_COLORS.get(e, "#808080") for e in emotions]

#             fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1E1E1E")
#             ax.barh(emotions, scores, color=colors_list)
#             ax.set_xlabel("Probability", color="#FFFFFF")
#             ax.set_facecolor("#1E1E1E")
#             ax.tick_params(colors="#FFFFFF")
#             for spine in ax.spines.values():
#                 spine.set_color("#FFFFFF")
#             ax.set_xlim(0, 1)
#             st.pyplot(fig, use_container_width=True)

#     elif camera_image:
#         st.warning("😐 No face detected — try moving closer to the camera.")
#     else:
#         st.info("👀 Take a photo above to detect your emotion.")

#     # Emotion timeline
#     st.markdown("### 📈 Emotion Timeline")
#     if st.session_state.emotion_history.get_session_stats()["total_detections"] > 0:
#         fig = st.session_state.emotion_history.plot_timeline()
#         st.pyplot(fig, use_container_width=True)


# # ============================================================================
# # RIGHT COLUMN: MUSIC RECOMMENDATIONS
# # ============================================================================

# with col_right:
#     st.markdown("## 🎵 Your Navarasa Playlist")

#     if (st.session_state.detected_emotion
#             and st.session_state.detected_emotion.get("face_found")):

#         result     = st.session_state.detected_emotion
#         emotion    = result["emotion"]
#         confidence = result["confidence"]

#         navarasa_info = NAVARASA_MAPPING.get(emotion, {})
#         emoji     = navarasa_info.get("emoji", "🎵")
#         navarasa  = navarasa_info.get("navarasa", emotion.title())
#         meaning   = navarasa_info.get("meaning", "")

#         st.markdown(f"### {emoji} {navarasa} — {meaning}")

#         if st.session_state.songs_df is not None:
#             st.button("🔄 Refresh Recommendations")

#             # BUG FIX: Use manual override if selected
#             emotion_for_rec = result["emotion"]
#             confidence_for_rec = result["confidence"]
            
#             if manual_emotion != "(Use detected emotion)":
#                 emotion_for_rec = manual_emotion
#                 confidence_for_rec = 1.0  # High confidence for manual selection
#                 st.info(f"ℹ️ Using manual override: **{manual_emotion}** instead of detected **{result['emotion']}**")

#             try:
#                 recommendations = get_recommendations(
#                     emotion      = emotion_for_rec,
#                     confidence   = confidence_for_rec,
#                     songs_df     = st.session_state.songs_df,
#                     history      = st.session_state.emotion_history.get_recent_tracks(10),
#                     recent_genres= st.session_state.emotion_history.get_recent_genres(5),
#                     n            = 5
#                 )

#                 if recommendations:
#                     for rec in recommendations:
#                         track_name   = rec.get("track_name",   "Unknown")
#                         artists      = rec.get("artists",       "Unknown")
#                         genre        = rec.get("track_genre",   "")
#                         valence      = rec.get("valence",       0)
#                         energy       = rec.get("energy",        0)
#                         danceability = rec.get("danceability",  0)
#                         popularity   = rec.get("popularity",    0)

#                         st.markdown(f"""
#                         <div class="song-card">
#                             <div class="song-title">🎵 {track_name}</div>
#                             <div class="song-artist">👤 {artists}</div>
#                             <div class="song-genre">{genre}</div>
#                             <div class="song-stats">
#                                 ❤️ Valence: {valence:.2f} &nbsp;&nbsp;
#                                 ⚡ Energy: {energy:.2f} &nbsp;&nbsp;
#                                 💃 Dance: {danceability:.2f} &nbsp;&nbsp;
#                                 ⭐ Popularity: {popularity}
#                             </div>
#                             <button class="play-btn">▶ Play on Spotify</button>
#                         </div>
#                         """, unsafe_allow_html=True)

#                         # BUG FIX: Only track if user actually interacts (implicit assumption here)
#                         # Note: Streamlit reruns on every interaction, so we use session state to prevent duplicates
#                         if "played_in_session" not in st.session_state:
#                             st.session_state.played_in_session = set()
                        
#                         track_id = f"{track_name}_{artists}"
#                         if track_id not in st.session_state.played_in_session:
#                             st.session_state.emotion_history.add_played_track(track_name, genre)
#                             st.session_state.played_in_session.add(track_id)
#                 else:
#                     st.info("No songs match this emotion profile. Try a different emotion!")
                    
#             except Exception as e:
#                 # BUG FIX: Better error handling for recommendation failures
#                 st.error(f"⚠️ Error generating recommendations: {e}")
#                 print(f"Recommendation error: {e}")

#         else:
#             st.warning("⚠️ Songs database not loaded. Run: python music/preprocess_songs.py")

#         # About this Navarasa
#         with st.expander("📜 About this Navarasa"):
#             navarasa_info = NAVARASA_MAPPING.get(emotion, {})
#             st.markdown(f"""
# ### {navarasa_info.get('emoji','')} {navarasa_info.get('navarasa', emotion.title())}

# **Meaning:** {navarasa_info.get('meaning', '')}

# **Origin:** One of the 9 Rasas from ancient Indian aesthetic theory,
# first described in Bharata Muni's *Natyashastra* (200 BCE – 200 CE).

# **In Music:** Songs matched to {navarasa_info.get('navarasa', emotion)} are
# selected based on their audio features — valence, energy, and danceability —
# to match the emotional quality of {(navarasa_info.get('meaning', emotion) or emotion).lower()}.
#             """)

#     else:
#         st.info("👈 Detect your emotion in the left column to get recommendations!")
#         st.markdown("""
#         ### How it works:
#         1. 📸 Click **Take Photo** on the left
#         2. 🧠 CNN detects your emotion (8 classes)
#         3. 🎭 Mapped to one of the **9 Navarasa**
#         4. 🎵 Songs recommended by audio features
#         """)




"""
Streamlit web UI for the Navarasa Emotion-Aware Music Recommendation System.

FIXES APPLIED:
- FIX 1: camera_to_numpy correctly opens JPEG bytes via PIL → RGB numpy array
- FIX 2: frame_to_rgb removed BGR flip (frame is already RGB from webcam.py)
- FIX 3: All st.image() calls use use_container_width (not deprecated use_column_width)
- FIX 4: Model reload triggered correctly when model_type changes
- FIX 5: Proper error handling throughout
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.emotion_history import EmotionHistory
from utils.constants import NAVARASA_MAPPING, EMOTION_COLORS
from music.recommendations import load_songs, get_recommendations
from app.webcam import EmotionDetector


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎭 Navarasa Music",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #121212; color: #FFFFFF; }
.song-card {
    background: #1E1E1E; border-radius: 12px; padding: 16px;
    margin-bottom: 12px; border: 1px solid #282828; transition: border 0.2s;
}
.song-card:hover { border: 1px solid #1DB954; }
.song-title  { font-size: 1.1rem; font-weight: bold; color: #FFFFFF; }
.song-artist { color: #B3B3B3; font-size: 0.9rem; margin: 4px 0; }
.song-genre  {
    display: inline-block; background: #282828; color: #1DB954;
    border-radius: 20px; padding: 2px 10px; font-size: 0.8rem; margin: 4px 0;
}
.song-stats  { color: #B3B3B3; font-size: 0.8rem; margin-top: 8px; }
.play-btn    {
    background: #1DB954; color: #000; border: none; border-radius: 20px;
    padding: 8px 18px; font-weight: bold; cursor: pointer;
    margin-top: 10px; width: 100%; font-size: 1rem;
}
.play-btn:hover { background: #1ed760; }
.navarasa-badge {
    font-size: 1.8rem; font-weight: bold; padding: 8px 20px;
    border-radius: 30px; display: inline-block; margin: 8px 0; color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = EmotionHistory(max_len=200)
if "detector" not in st.session_state:
    st.session_state.detector = None
if "detector_model_type" not in st.session_state:
    st.session_state.detector_model_type = None
if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = None
if "played_in_session" not in st.session_state:
    st.session_state.played_in_session = set()
if "songs_df" not in st.session_state:
    try:
        st.session_state.songs_df = load_songs()
    except Exception as e:
        st.error(f"⚠️ Could not load songs: {e}")
        st.session_state.songs_df = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def camera_to_numpy(camera_image) -> np.ndarray:
    """
    Convert st.camera_input UploadedFile → RGB numpy array.

    st.camera_input returns an UploadedFile (JPEG bytes).
    PIL.Image.open reads it as RGB automatically.

    Returns:
        np.ndarray of shape (H, W, 3) in RGB.
    """
    pil_image = Image.open(camera_image).convert("RGB")
    return np.array(pil_image)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎭 Navarasa Music")
st.sidebar.markdown(
    "Detect your emotion → Get music matched to your mood "
    "using the ancient Indian Navarasa framework."
)
st.sidebar.divider()

model_choice = st.sidebar.radio(
    "Select Model:",
    options=["Custom CNN (Recommended)", "MobileNetV2"],
    index=0
)
model_type = "custom_cnn" if "Custom CNN" in model_choice else "mobilenet"

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold:",
    min_value=0.3, max_value=0.9, value=0.4, step=0.05,
    help="Only show results above this confidence"
)

st.sidebar.divider()

# Manual override for testing
st.sidebar.markdown("### 🎛️ Manual Override (for testing)")
manual_emotion = st.sidebar.selectbox(
    "Override detected emotion:",
    options=["(Use detected emotion)"] + list(NAVARASA_MAPPING.keys()),
    index=0
)

st.sidebar.divider()

# Session stats
st.sidebar.markdown("### 📊 Session Stats")
session_stats = st.session_state.emotion_history.get_session_stats()

if session_stats["total_detections"] > 0:
    st.sidebar.metric("Total Detections", session_stats["total_detections"])
    st.sidebar.metric(
        "Dominant Emotion",
        session_stats["dominant_emotion"].title()
    )
    if session_stats["navarasa_counts"]:
        df_nav = pd.DataFrame({
            "Navarasa": list(session_stats["navarasa_counts"].keys()),
            "Count":    list(session_stats["navarasa_counts"].values())
        })
        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#121212")
        ax.barh(df_nav["Navarasa"], df_nav["Count"], color="#1DB954")
        ax.set_facecolor("#121212")
        ax.tick_params(colors="#FFFFFF")
        for s in ax.spines.values():
            s.set_color("#FFFFFF")
        st.sidebar.pyplot(fig, use_container_width=True)
        plt.close(fig)
else:
    st.sidebar.info("👀 Detections will appear here")

if st.sidebar.button("💾 Save Session"):
    st.session_state.emotion_history.to_json("session_log.json")
    st.sidebar.success("✅ Saved!")


# ── Main layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1])


# ── LEFT: Detection ──────────────────────────────────────────────────────────
with col_left:
    st.markdown("## 📸 Live Detection")

    camera_image = st.camera_input(
        "Capture your face", label_visibility="collapsed"
    )

    if camera_image:
        # ── FIX 1: Convert JPEG bytes → RGB numpy array ──────────────────────
        image_array = camera_to_numpy(camera_image)

        # Reload detector if model type changed
        if (st.session_state.detector is None or
                st.session_state.detector_model_type != model_type):
            checkpoint_dir  = Path(__file__).parent.parent / "models" / "checkpoints"
            checkpoint_path = checkpoint_dir / f"{model_type}_best.pth"

            if checkpoint_path.exists():
                with st.spinner(f"Loading {model_type} model..."):
                    try:
                        st.session_state.detector = EmotionDetector(
                            str(checkpoint_path),
                            model_type=model_type
                        )
                        st.session_state.detector_model_type = model_type
                    except Exception as e:
                        st.error(f"⚠️ Model load failed: {e}")
                        st.session_state.detector = None
            else:
                st.warning(f"⚠️ Checkpoint not found: {checkpoint_path}")

        if st.session_state.detector:
            try:
                # detect() receives RGB, returns RGB annotated frame
                result = st.session_state.detector.detect(image_array)

                # ── FIX 2: frame is already RGB — display directly ───────────
                frame = result.get("frame")
                if frame is not None and isinstance(frame, np.ndarray):
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        st.image(frame, use_container_width=True)
                    else:
                        st.image(image_array, use_container_width=True)
                else:
                    st.image(image_array, use_container_width=True)

                st.session_state.detected_emotion = result

                if result["face_found"] and result["confidence"] >= confidence_threshold:
                    st.session_state.emotion_history.add(
                        result["emotion"], result["confidence"]
                    )

            except Exception as e:
                st.error(f"❌ Detection error: {e}")
                st.image(image_array, use_container_width=True)
        else:
            st.image(image_array, use_container_width=True)

    # ── Emotion display ───────────────────────────────────────────────────────
    detected = st.session_state.detected_emotion
    if detected and detected.get("face_found"):
        emotion    = detected["emotion"]
        confidence = detected["confidence"]
        navarasa   = detected["navarasa"]
        meaning    = detected["navarasa_meaning"]
        emoji      = detected["navarasa_emoji"]
        color      = EMOTION_COLORS.get(emotion, "#808080")

        st.markdown(
            f'<div class="navarasa-badge" style="background:{color};">'
            f'{emoji} {navarasa}<br><small>{meaning}</small></div>',
            unsafe_allow_html=True
        )
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")

        with st.expander("🎭 All 8 Emotion Scores"):
            scores = detected["all_scores"]
            fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1E1E1E")
            emotions_list = list(scores.keys())
            values        = list(scores.values())
            bar_colors    = [EMOTION_COLORS.get(e, "#808080") for e in emotions_list]
            ax.barh(emotions_list, values, color=bar_colors)
            ax.set_xlim(0, 1)
            ax.set_facecolor("#1E1E1E")
            ax.tick_params(colors="#FFFFFF")
            for s in ax.spines.values():
                s.set_color("#FFFFFF")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    elif camera_image:
        st.warning("😐 No face detected — move closer and try again.")
    else:
        st.info("👀 Take a photo to detect your emotion.")

    # Timeline
    st.markdown("### 📈 Emotion Timeline")
    if session_stats["total_detections"] > 0:
        fig = st.session_state.emotion_history.plot_timeline()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ── RIGHT: Recommendations ────────────────────────────────────────────────────
with col_right:
    st.markdown("## 🎵 Your Navarasa Playlist")

    detected = st.session_state.detected_emotion

    # Determine emotion for recommendations
    emotion_for_rec    = None
    confidence_for_rec = 0.0

    if manual_emotion != "(Use detected emotion)":
        emotion_for_rec    = manual_emotion
        confidence_for_rec = 1.0
        st.info(f"ℹ️ Manual override: **{manual_emotion}**")
    elif detected and detected.get("face_found"):
        emotion_for_rec    = detected["emotion"]
        confidence_for_rec = detected["confidence"]

    if emotion_for_rec:
        nav_info = NAVARASA_MAPPING.get(emotion_for_rec, {})
        emoji    = nav_info.get("emoji", "🎵")
        navarasa = nav_info.get("navarasa", emotion_for_rec.title())
        meaning  = nav_info.get("meaning", "")

        st.markdown(f"### {emoji} {navarasa} — {meaning}")

        if st.session_state.songs_df is not None:
            if st.button("🔄 Refresh Recommendations"):
                # Clear played tracks to get fresh recommendations
                if "played_in_session" in st.session_state:
                    st.session_state.played_in_session = set()

            try:
                recs = get_recommendations(
                    emotion       = emotion_for_rec,
                    confidence    = confidence_for_rec,
                    songs_df      = st.session_state.songs_df,
                    history       = st.session_state.emotion_history.get_recent_tracks(10),
                    recent_genres = st.session_state.emotion_history.get_recent_genres(5),
                    n             = 5
                )

                if recs:
                    for rec in recs:
                        track        = rec.get("track_name",  "Unknown")
                        artist       = rec.get("artists",      "Unknown")
                        genre        = rec.get("track_genre",  "")
                        valence      = rec.get("valence",      0.0)
                        energy       = rec.get("energy",       0.0)
                        dance        = rec.get("danceability", 0.0)
                        popularity   = rec.get("popularity",   0)

                        st.markdown(f"""
                        <div class="song-card">
                            <div class="song-title">🎵 {track}</div>
                            <div class="song-artist">👤 {artist}</div>
                            <div class="song-genre">{genre}</div>
                            <div class="song-stats">
                                ❤️ Valence: {valence:.2f} &nbsp;
                                ⚡ Energy: {energy:.2f} &nbsp;
                                💃 Dance: {dance:.2f} &nbsp;
                                ⭐ {popularity}
                            </div>
                            <button class="play-btn">▶ Play on Spotify</button>
                        </div>
                        """, unsafe_allow_html=True)

                        track_id = f"{track}_{artist}"
                        if track_id not in st.session_state.played_in_session:
                            st.session_state.emotion_history.add_played_track(
                                track, genre
                            )
                            st.session_state.played_in_session.add(track_id)
                else:
                    st.info("No songs found for this emotion profile.")

            except Exception as e:
                st.error(f"⚠️ Recommendation error: {e}")
        else:
            st.warning("⚠️ Run: `python music/preprocess_songs.py` first")

        with st.expander("📜 About this Navarasa"):
            info = NAVARASA_MAPPING.get(emotion_for_rec, {})
            st.markdown(f"""
### {info.get('emoji','')} {info.get('navarasa', emotion_for_rec.title())}
**Meaning:** {info.get('meaning', '')}

**Origin:** One of the 9 Rasas from Bharata Muni's *Natyashastra* (200 BCE–200 CE).

**In music:** Songs for *{info.get('navarasa', emotion_for_rec)}* are filtered
by valence, energy, and danceability ranges that match the
emotional quality of **{info.get('meaning', emotion_for_rec).lower()}**.
            """)
    else:
        st.info("👈 Take a photo on the left to get recommendations!")
        st.markdown("""
        ### How it works:
        1. 📸 Click **Take Photo** on the left
        2. 🧠 CNN detects your emotion (8 classes)
        3. 🎭 Mapped to one of the **9 Navarasa**
        4. 🎵 Songs recommended by Spotify audio features
        """)