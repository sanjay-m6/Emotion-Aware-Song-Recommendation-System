"""
Core API routes for emotion detection and Spotify recommendations.

Endpoints:
    POST /api/detect-emotion     — Detect emotion from base64 webcam frame
    GET  /api/recommendations    — Get Spotify tracks for a detected emotion
    POST /api/playlist/create    — Create a Spotify playlist
    POST /api/track/save         — Save/like a track on Spotify
    GET  /api/mood-history       — Get session mood history
"""

import time
from collections import deque
from typing import Dict, List

from flask import Blueprint, request, jsonify

api_bp = Blueprint("api", __name__, url_prefix="/api")

# In-memory session mood history (per-server process)
_mood_history: deque = deque(maxlen=200)


def _get_access_token() -> str:
    """Extract access token from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1]
    return ""


# ── Emotion detection ─────────────────────────────────────────────────────────

@api_bp.route("/detect-emotion", methods=["POST"])
def detect_emotion():
    """
    Detect emotion from a base64-encoded webcam frame.

    Expects JSON body: { "image": "<base64 string>" }

    Returns JSON with:
        emotion, confidence, face_found, all_scores,
        display_name, display_meaning, display_emoji, color
    """
    from backend.services.emotion_service import emotion_service

    if not emotion_service.is_ready:
        return jsonify({"error": "Emotion model not loaded"}), 503

    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image", "")

    if not image_b64:
        return jsonify({"error": "image field required (base64)"}), 400

    try:
        result = emotion_service.detect_from_base64(image_b64)

        # Record in session history
        _mood_history.append({
            "emotion": result["emotion"],
            "display_name": result["display_name"],
            "confidence": result["confidence"],
            "timestamp": time.time(),
        })

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


# ── Spotify recommendations ──────────────────────────────────────────────────

YOUTUBE_VIDEOS = {
    "sad": [
        {"id": "inpok4MKVLM", "title": "🧘 5-Minute Meditation You Can Do Anywhere"},
        {"id": "z6X5oEIg6Ak", "title": "🎵 Relaxing Music for Stress Relief"},
        {"id": "4q1dgn_C0AU", "title": "💬 How to Deal with Sadness - Therapy Talk"},
        {"id": "F2hc2FLOdhI", "title": "🎶 Peaceful Tamil Instrumental for Calm"},
    ],
    "anger": [
        {"id": "MIc29LN-kME", "title": "🧘 10-Minute Anger Release Meditation"},
        {"id": "O-6f5wQXSu8", "title": "🧘 Let Go of Anger - Guided Meditation"},
        {"id": "BsVq5R_F6RA", "title": "💬 How to Control Your Anger - Motivational"},
        {"id": "z6X5oEIg6Ak", "title": "🎵 Calming Nature Sounds for Inner Peace"},
    ],
    "angry": [
        {"id": "MIc29LN-kME", "title": "🧘 10-Minute Anger Release Meditation"},
        {"id": "O-6f5wQXSu8", "title": "🧘 Let Go of Anger - Guided Meditation"},
        {"id": "BsVq5R_F6RA", "title": "💬 How to Control Your Anger - Motivational"},
        {"id": "z6X5oEIg6Ak", "title": "🎵 Calming Nature Sounds for Inner Peace"},
    ],
    "fear": [
        {"id": "NWv1VdDeoRY", "title": "🧘 10-Minute Meditation for Anxiety"},
        {"id": "O-6f5wQXSu8", "title": "🧘 Let Go of Fear - Guided Meditation"},
        {"id": "ZidGozDhOjg", "title": "💬 Overcoming Fear - Motivational Speech"},
        {"id": "lFcSrYw-ARY", "title": "🎵 Lo-Fi Beats for Anxiety Relief"},
    ],
    "disgust": [
        {"id": "inpok4MKVLM", "title": "🧘 5-Minute Grounding Meditation"},
        {"id": "z6X5oEIg6Ak", "title": "🎵 Soothing Nature Sounds for Relaxation"},
        {"id": "4q1dgn_C0AU", "title": "💬 Emotional Reset - Therapy Talk"},
        {"id": "F2hc2FLOdhI", "title": "🎶 Tamil Soft Melody for Relaxation"},
    ],
    "happy": [
        {"id": "ZbZSe6N_BXs", "title": "🎵 Happy Music - Upbeat Morning Playlist"},
        {"id": "ru0K8uYEZWw", "title": "🎶 Best Tamil Feel-Good Songs Collection"},
        {"id": "y6Sxv-sUYtM", "title": "💬 The Science of Happiness - TED Talk"},
        {"id": "nfs8NYg7yQM", "title": "🎵 Perfect - Ed Sheeran (Feel Good)"},
    ],
    "surprise": [
        {"id": "inpok4MKVLM", "title": "🧘 5-Minute Mindfulness Meditation"},
        {"id": "ZbZSe6N_BXs", "title": "🎵 Uplifting Music to Brighten Your Day"},
        {"id": "ru0K8uYEZWw", "title": "🎶 Energetic Tamil Hits Playlist"},
        {"id": "y6Sxv-sUYtM", "title": "💬 Embrace the Unexpected - Motivational"},
    ],
    "neutral": [
        {"id": "lFcSrYw-ARY", "title": "🎵 Lo-Fi Beats to Relax and Study"},
        {"id": "z6X5oEIg6Ak", "title": "🎵 Calming Background Music for Focus"},
        {"id": "F2hc2FLOdhI", "title": "🎶 Tamil Instrumental - Peaceful Vibes"},
        {"id": "5qap5aO4i9A", "title": "🎵 Chill Vibes - Study & Work Music"},
    ],
    "stressed": [
        {"id": "inpok4MKVLM", "title": "🧘 5-Minute Meditation You Can Do Anywhere"},
        {"id": "NWv1VdDeoRY", "title": "🧘 10-Minute Stress Relief Meditation"},
        {"id": "z6X5oEIg6Ak", "title": "🎵 Nature Sounds for Stress Relief"},
        {"id": "4q1dgn_C0AU", "title": "💬 Managing Stress - Mental Health Tips"},
    ],
    "energetic": [
        {"id": "ZbZSe6N_BXs", "title": "🎵 High Energy Workout Playlist"},
        {"id": "ru0K8uYEZWw", "title": "🎶 Tamil Dance Hits Collection"},
        {"id": "nfs8NYg7yQM", "title": "🎵 Top English Pop Hits to Vibe To"},
        {"id": "y6Sxv-sUYtM", "title": "💬 Power of Positive Energy - Motivation"},
    ],
    "contempt": [
        {"id": "inpok4MKVLM", "title": "🧘 5-Minute Grounding Meditation"},
        {"id": "z6X5oEIg6Ak", "title": "🎵 Peaceful Music for Emotional Reset"},
        {"id": "4q1dgn_C0AU", "title": "💬 Processing Difficult Emotions - Therapy"},
        {"id": "lFcSrYw-ARY", "title": "🎵 Lo-Fi Chill for Mood Reset"},
    ],
}

@api_bp.route("/recommendations")
def recommendations():
    """
    Get Spotify song recommendations based on emotion using Client Credentials.

    Query params:
        emotion  — Detected emotion (default: neutral)
        limit    — Number of tracks (default: 10, max: 50)
    """
    from backend.services.spotify_service import get_recommendations, get_client_credentials_token
    
    access_token = get_client_credentials_token()
    if not access_token:
        return jsonify({"error": "Failed to get Spotify access token from server credentials"}), 500

    emotion = request.args.get("emotion", "neutral").lower().strip()
    limit = min(int(request.args.get("limit", 10)), 50)
    # Default confidence to 0.8 if not provided
    confidence = float(request.args.get("confidence", 0.8))

    # get_recommendations now returns a tuple: (tracks, explanation)
    tracks, explanation = get_recommendations(emotion, access_token, limit)
    
    # Inject YouTube stress-relief videos if the emotion is negative
    youtube_videos = YOUTUBE_VIDEOS.get(emotion, YOUTUBE_VIDEOS.get("neutral", []))
    
    return jsonify({
        "tracks": tracks, 
        "emotion": emotion,
        "explanation": explanation,
        "youtube_videos": youtube_videos
    })


# ── Chat Mode ─────────────────────────────────────────────────────────────────

@api_bp.route("/chat", methods=["POST"])
def chat():
    """
    Chat with the AI and get dynamic song recommendations based on text input.
    """
    from backend.services.ai_service import chat_with_music_ai
    from backend.services.spotify_service import get_recommendations_from_params, get_client_credentials_token
    
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    history = data.get("history", [])
    
    if not message:
        return jsonify({"error": "message is required"}), 400
        
    # Get conversational reply and spotify params from AI
    ai_response = chat_with_music_ai(message, history)
    
    # Get tracks based on those params
    access_token = get_client_credentials_token()
    tracks, emotion = get_recommendations_from_params(ai_response, access_token, limit=10)
    
    # Record chat emotion in mood history
    _mood_history.append({
        "emotion": emotion,
        "display_name": emotion.capitalize(),
        "confidence": 0.85,  # Chat-based detection has high implicit confidence
        "timestamp": time.time(),
        "source": "chat",
    })
    
    # Include YouTube videos based on detected emotion from chat
    youtube_videos = YOUTUBE_VIDEOS.get(emotion, YOUTUBE_VIDEOS.get("neutral", []))
    
    return jsonify({
        "reply": ai_response.get("reply", "Here are some tracks for you!"),
        "emotion": emotion,
        "tracks": tracks,
        "youtube_videos": youtube_videos
    })

# ── Playlist creation ─────────────────────────────────────────────────────────

@api_bp.route("/playlist/create", methods=["POST"])
def create_playlist():
    return jsonify({"error": "Playlist creation requires user login, which is disabled."}), 403

# ── Save / like track ────────────────────────────────────────────────────────

@api_bp.route("/track/save", methods=["POST"])
def save_track():
    return jsonify({"error": "Saving tracks requires user login, which is disabled."}), 403


# ── Mood history ──────────────────────────────────────────────────────────────

@api_bp.route("/mood-history")
def mood_history():
    """
    Return the session mood history.

    Returns JSON: { "history": [...], "stats": {...} }
    """
    history_list = list(_mood_history)

    # Compute stats
    if history_list:
        emotion_counts: Dict[str, int] = {}
        for entry in history_list:
            e = entry["emotion"]
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        dominant = max(emotion_counts, key=emotion_counts.get)
    else:
        emotion_counts = {}
        dominant = "neutral"

    return jsonify({
        "history": history_list,
        "stats": {
            "total_detections": len(history_list),
            "emotion_counts": emotion_counts,
            "dominant_emotion": dominant,
        },
    })
