# """
# Music recommendation engine based on emotion-detected audio profiles.

# Maps detected emotions + confidence levels to Spotify audio feature ranges,
# then filters the cached Spotify tracks to find songs that match the user's
# emotional state. Includes contextual inference of Shringara (Love rasa).
# """

# import sys
# from pathlib import Path
# from typing import List, Dict, Optional

# import pandas as pd

# # Add parent directory to path
# sys.path.insert(0, str(Path(__file__).parent.parent))

# from utils.constants import (
#     EMOTION_AUDIO_PROFILES,
#     ROMANTIC_GENRES,
#     NAVARASA_MAPPING
# )


# def load_songs(cache_path: str = "data/songs_cache.parquet") -> pd.DataFrame:
#     """
#     Load Spotify tracks from cached parquet file with validation.
    
#     Falls back to loading from HuggingFace if cache is missing or corrupt.
#     Validates all required columns exist and have valid ranges.
    
#     Args:
#         cache_path: Path to songs_cache.parquet (relative to project root)
        
#     Returns:
#         DataFrame with track_name, artists, track_genre, audio features, etc.
#         All audio features are validated to be in [0.0, 1.0] range.
        
#     Raises:
#         RuntimeError: If both cache and HuggingFace load fail
#         ValueError: If required columns are missing
#     """
#     cache_path = Path(cache_path)
    
#     # Try to load from cache first
#     if cache_path.exists():
#         try:
#             df = pd.read_parquet(cache_path)
            
#             # BUG FIX: Validate audio features are in valid range
#             audio_features = ["valence", "energy", "danceability"]
#             for feature in audio_features:
#                 if feature in df.columns:
#                     # Clamp out-of-range values (data quality issue)
#                     df[feature] = df[feature].clip(0.0, 1.0)
            
#             return df
#         except Exception as e:
#             print(f"⚠️  Failed to load cache: {e}. Falling back to HuggingFace...")
    
#     # Fallback: load from HuggingFace
#     try:
#         from datasets import load_dataset
#         songs_ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
#         df = songs_ds.to_pandas()
        
#         # Validate required columns
#         required_cols = ["track_name", "artists", "valence", "energy", "danceability"]
#         missing_cols = [c for c in required_cols if c not in df.columns]
#         if missing_cols:
#             raise ValueError(f"Missing required columns: {missing_cols}")
        
#         # Clean data
#         df = df.dropna(subset=["track_name", "artists", "valence", "energy", "danceability"])
#         df = df.drop_duplicates(subset=["track_name", "artists"])
#         if "popularity" in df.columns:
#             df = df[df["popularity"] > 0]
        
#         # BUG FIX: Validate and clamp audio features to [0.0, 1.0]
#         for feature in ["valence", "energy", "danceability"]:
#             if feature in df.columns:
#                 df[feature] = df[feature].clip(0.0, 1.0)
        
#         return df
        
#     except Exception as e:
#         raise RuntimeError(f"Failed to load songs: {e}")


# def infer_shringara(
#     emotion: str,
#     confidence: float,
#     recent_genres: List[str]
# ) -> str:
#     """
#     Infer Shringara (Love) rasa contextually with improved robustness.
    
#     Shringara is triggered when:
#     1. Detected emotion is "happy" 
#     2. Confidence > 0.75 (high certainty)
#     3. User's recent song history includes romantic genres (R&B, soul, indie, etc.)
#     4. BUG FIX: Validate inputs to avoid data corruption
    
#     This contextual inference captures the nuanced emotional state that the
#     user is in a romantic/love-focused mood, even though the face detector
#     only sees happiness.
    
#     Args:
#         emotion: Current detected emotion (e.g., "happy")
#         confidence: Confidence score (0.0-1.0)
#         recent_genres: List of genres from recently played songs
        
#     Returns:
#         "shringara" if conditions met, otherwise original emotion unchanged
        
#     Raises:
#         ValueError: If emotion is not a string or confidence is invalid
#     """
#     # BUG FIX: Input validation
#     try:
#         if not isinstance(emotion, str):
#             emotion = str(emotion).lower()
#         else:
#             emotion = emotion.lower()
        
#         if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
#             return emotion
        
#         if not isinstance(recent_genres, (list, tuple)):
#             recent_genres = []
        
#     except Exception:
#         return emotion
    
#     # Check conditions for Shringara
#     if emotion == "happy" and confidence > 0.75:
#         # Check if any recent song genre matches romantic genres
#         if recent_genres:
#             try:
#                 recent_lower = [g.lower().strip() if isinstance(g, str) else str(g).lower() 
#                               for g in recent_genres]
#                 for genre in recent_lower:
#                     for romantic_genre in ROMANTIC_GENRES:
#                         if romantic_genre in genre:
#                             return "shringara"
#             except Exception as e:
#                 print(f"⚠️ Error in Shringara inference: {e}")
    
#     return emotion


# def get_recommendations(
#     emotion: str,
#     confidence: float,
#     songs_df: pd.DataFrame,
#     history: List[str],
#     recent_genres: List[str],
#     n: int = 5
# ) -> List[Dict]:
#     """
#     Get music recommendations matching the detected emotion.
    
#     IMPROVED ALGORITHM with fallback for poor detections:
#     1. Infer Shringara if contextual conditions met
#     2. Fall back to "neutral" if confidence < 0.5 (uncertain detection)
#     3. Look up EMOTION_AUDIO_PROFILES for valence/energy/danceability ranges
#     4. Filter songs_df by those ranges (STRICT filtering)
#     5. If insufficient results (<3 songs), widen ranges for 2nd attempt (LOOSE filtering)
#     6. Exclude recently played tracks
#     7. Sort by popularity (descending)
#     8. Return top n songs with quality validation
    
#     Args:
#         emotion: Detected emotion name (e.g., "angry", "happy")
#         confidence: Confidence score (0.0-1.0)
#         songs_df: DataFrame of Spotify tracks with audio features
#         history: List of recently played track names (to exclude)
#         recent_genres: List of recently played track genres (for Shringara inference)
#         n: Number of recommendations to return (default 5)
        
#     Returns:
#         List of recommendation dicts with keys:
#         - track_name: Song title
#         - artists: Artist(s)
#         - track_genre: Genre tag
#         - valence: Audio feature (0.0-1.0)
#         - energy: Audio feature (0.0-1.0)
#         - danceability: Audio feature (0.0-1.0)
#         - popularity: Score 0-100
        
#     Raises:
#         ValueError: If songs_df is None or empty
        
#     Example:
#         >>> songs = load_songs()
#         >>> recs = get_recommendations(
#         ...     emotion="happy",
#         ...     confidence=0.87,
#         ...     songs_df=songs,
#         ...     history=["Song A", "Song B"],
#         ...     recent_genres=["pop", "r&b"],
#         ...     n=5
#         ... )
#         >>> print(recs[0]["track_name"])
#     """
#     # BUG FIX: Input validation
#     if songs_df is None or songs_df.empty:
#         raise ValueError("songs_df cannot be None or empty")
    
#     if not isinstance(emotion, str) or emotion.strip() == "":
#         emotion = "neutral"
    
#     if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
#         confidence = 0.5
    
#     if not isinstance(history, (list, tuple)):
#         history = []
    
#     if not isinstance(recent_genres, (list, tuple)):
#         recent_genres = []
    
#     # Step 1: Infer Shringara if applicable
#     emotion = infer_shringara(emotion, confidence, recent_genres)
    
#     # Step 2: Fall back to neutral if confidence too low
#     if confidence < 0.5:
#         emotion = "neutral"
    
#     # Step 3: Get audio profile for this emotion (with fallback)
#     if emotion not in EMOTION_AUDIO_PROFILES:
#         emotion = "neutral"
    
#     profile = EMOTION_AUDIO_PROFILES.get(emotion, EMOTION_AUDIO_PROFILES["neutral"])
#     valence_range = profile["valence"]
#     energy_range = profile["energy"]
#     danceability_range = profile["danceability"]
    
#     # Step 4: STRICT filtering by audio feature ranges
#     try:
#         filtered = songs_df[
#             (songs_df["valence"] >= valence_range[0]) &
#             (songs_df["valence"] <= valence_range[1]) &
#             (songs_df["energy"] >= energy_range[0]) &
#             (songs_df["energy"] <= energy_range[1]) &
#             (songs_df["danceability"] >= danceability_range[0]) &
#             (songs_df["danceability"] <= danceability_range[1])
#         ].copy()
#     except KeyError as e:
#         print(f"⚠️ Audio feature missing: {e}. Falling back to neutral profile.")
#         profile = EMOTION_AUDIO_PROFILES["neutral"]
#         valence_range = profile["valence"]
#         energy_range = profile["energy"]
#         danceability_range = profile["danceability"]
        
#         filtered = songs_df[
#             (songs_df["valence"] >= valence_range[0]) &
#             (songs_df["valence"] <= valence_range[1]) &
#             (songs_df["energy"] >= energy_range[0]) &
#             (songs_df["energy"] <= energy_range[1]) &
#             (songs_df["danceability"] >= danceability_range[0]) &
#             (songs_df["danceability"] <= danceability_range[1])
#         ].copy()
    
#     # BUG FIX: If strict filtering returns too few results, try looser filtering
#     if len(filtered) < 3 * n:  # Less than 3x the requested number
#         print(f"⚠️ Strict filtering returned only {len(filtered)} songs for {emotion}. Trying loose filtering...")
        
#         # LOOSE filtering: only use primary discriminator (valence)
#         filtered_loose = songs_df[
#             (songs_df["valence"] >= valence_range[0] - 0.15) &
#             (songs_df["valence"] <= valence_range[1] + 0.15)
#         ].copy()
        
#         if len(filtered_loose) > len(filtered):
#             filtered = filtered_loose
    
#     # Step 5: Exclude recently played tracks
#     if history:
#         filtered = filtered[~filtered["track_name"].isin(history)]
    
#     # Step 6: Sort by popularity (descending) then by audio features
#     if "popularity" in filtered.columns:
#         filtered = filtered.sort_values("popularity", ascending=False)
#     else:
#         # Fallback: sort by valence + energy
#         filtered["score"] = filtered["valence"] + filtered["energy"]
#         filtered = filtered.sort_values("score", ascending=False)
    
#     # Step 7: Return top n songs as dicts with quality validation
#     result = []
#     columns_to_keep = ["track_name", "artists", "track_genre", "valence", "energy", "danceability", "popularity"]
    
#     for _, row in filtered.head(n * 2).iterrows():  # Get 2x to handle potential filtering
#         rec = {}
#         for col in columns_to_keep:
#             if col in row.index:
#                 rec[col] = row[col]
        
#         # Quality check: validate audio features are in acceptable range
#         # BUG FIX: Ensure recommendations match the emotional profile
#         val = rec.get("valence", 0.5)
#         if valence_range[0] - 0.2 <= val <= valence_range[1] + 0.2:
#             result.append(rec)
#             if len(result) >= n:
#                 break
    
#     # If still no results, fall back to random top popular songs
#     if not result:
#         print(f"⚠️ No songs found for {emotion}. Returning popular songs.")
#         top_popular = songs_df.nlargest(n, "popularity") if "popularity" in songs_df.columns else songs_df.head(n)
#         for _, row in top_popular.iterrows():
#             rec = {}
#             for col in columns_to_keep:
#                 if col in row.index:
#                     rec[col] = row[col]
#             result.append(rec)
    
#     return result


# def get_navarasa_playlist(
#     emotion: str,
#     songs_df: pd.DataFrame,
#     n: int = 20
# ) -> List[Dict]:
#     """
#     Get a longer playlist (20 songs) for a specific emotion/navarasa.
    
#     Similar to get_recommendations but:
#     - Returns more songs (default 20)
#     - No history filtering (can repeat across sessions)
#     - Includes navarasa metadata in result
    
#     Args:
#         emotion: Emotion/navarasa name
#         songs_df: DataFrame of Spotify tracks
#         n: Number of songs in playlist (default 20)
        
#     Returns:
#         List of song dicts with audio features + navarasa metadata
#     """
#     # Get audio profile
#     if emotion not in EMOTION_AUDIO_PROFILES:
#         emotion = "neutral"
    
#     profile = EMOTION_AUDIO_PROFILES[emotion]
#     valence_range = profile["valence"]
#     energy_range = profile["energy"]
#     danceability_range = profile["danceability"]
    
#     # Filter by ranges
#     filtered = songs_df[
#         (songs_df["valence"] >= valence_range[0]) &
#         (songs_df["valence"] <= valence_range[1]) &
#         (songs_df["energy"] >= energy_range[0]) &
#         (songs_df["energy"] <= energy_range[1]) &
#         (songs_df["danceability"] >= danceability_range[0]) &
#         (songs_df["danceability"] <= danceability_range[1])
#     ]
    
#     # Sort by popularity
#     filtered = filtered.sort_values("popularity", ascending=False)
    
#     # Build result with navarasa metadata
#     result = []
#     columns_to_keep = ["track_name", "artists", "track_genre", "valence", "energy", "danceability", "popularity"]
    
#     for _, row in filtered.head(n).iterrows():
#         rec = {}
#         for col in columns_to_keep:
#             if col in row.index:
#                 rec[col] = row[col]
        
#         # Add navarasa metadata if available
#         if emotion in NAVARASA_MAPPING:
#             rec["navarasa"] = NAVARASA_MAPPING[emotion]["navarasa"]
#             rec["meaning"] = NAVARASA_MAPPING[emotion]["meaning"]
#             rec["emoji"] = NAVARASA_MAPPING[emotion]["emoji"]
        
#         result.append(rec)
    
#     return result





"""
Music recommendation engine.

FIXES APPLIED:
- FIX 1: Removed loose fallback that was causing wrong-emotion songs to appear
- FIX 2: Fallback now uses neutral profile (not random popular songs)
- FIX 3: Added debug logging so you can see why songs are selected
"""

import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import (
    EMOTION_AUDIO_PROFILES,
    ROMANTIC_GENRES,
    NAVARASA_MAPPING
)


def load_songs(cache_path: str = "data/songs_cache.parquet") -> pd.DataFrame:
    """Load Spotify songs from parquet cache, fallback to HuggingFace."""
    # Try relative to project root first, then absolute
    paths_to_try = [
        Path(cache_path),
        Path(__file__).parent.parent / cache_path,
    ]

    for p in paths_to_try:
        if p.exists():
            try:
                df = pd.read_parquet(p)
                # Clamp audio features to [0, 1]
                for feat in ["valence", "energy", "danceability"]:
                    if feat in df.columns:
                        df[feat] = df[feat].clip(0.0, 1.0)
                print(f"✅ Loaded {len(df):,} songs from {p}")
                return df
            except Exception as e:
                print(f"⚠️ Cache load failed: {e}")

    # HuggingFace fallback
    print("Cache not found — loading from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
        df = ds.to_pandas()
        df = df.dropna(subset=["track_name", "artists", "valence", "energy", "danceability"])
        df = df.drop_duplicates(subset=["track_name", "artists"])
        if "popularity" in df.columns:
            df = df[df["popularity"] > 0]
        for feat in ["valence", "energy", "danceability"]:
            df[feat] = df[feat].clip(0.0, 1.0)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load songs: {e}")


def infer_shringara(
    emotion: str,
    confidence: float,
    recent_genres: List[str]
) -> str:
    """
    Upgrade 'happy' to 'shringara' if context suggests romantic mood.

    Conditions:
    - emotion == "happy"
    - confidence > 0.75
    - recent genres contain a romantic genre keyword
    """
    if emotion != "happy" or confidence <= 0.75:
        return emotion
    if not recent_genres:
        return emotion

    recent_lower = [g.lower().strip() for g in recent_genres if isinstance(g, str)]
    for g in recent_lower:
        for romantic in ROMANTIC_GENRES:
            if romantic in g:
                print(f"🥰 Shringara inferred (genre match: {g})")
                return "shringara"
    return emotion


def get_recommendations(
    emotion: str,
    confidence: float,
    songs_df: pd.DataFrame,
    history: List[str],
    recent_genres: List[str],
    n: int = 5
) -> List[Dict]:
    """
    Return n songs matching the detected emotion's audio profile.

    Algorithm:
    1. Infer Shringara if applicable
    2. Fall back to neutral if confidence < 0.5
    3. Filter by EMOTION_AUDIO_PROFILES ranges (strict)
    4. If < n results, slightly widen ranges (max 0.1 expansion)
    5. Exclude recently played tracks
    6. Sort by popularity, return top n

    Args:
        emotion: Detected emotion name
        confidence: Detection confidence (0–1)
        songs_df: Spotify DataFrame
        history: Recently played track names to exclude
        recent_genres: Recently played genres for Shringara inference
        n: Number of songs to return

    Returns:
        List of dicts with track_name, artists, track_genre,
        valence, energy, danceability, popularity
    """
    if songs_df is None or songs_df.empty:
        return []

    # Sanitise inputs
    emotion       = str(emotion).lower().strip() if emotion else "neutral"
    confidence    = float(confidence) if isinstance(confidence, (int, float)) else 0.5
    history       = list(history) if history else []
    recent_genres = list(recent_genres) if recent_genres else []

    # Step 1 — Shringara upgrade
    emotion = infer_shringara(emotion, confidence, recent_genres)

    # Step 2 — Low confidence fallback
    if confidence < 0.5:
        print(f"⚠️ Low confidence ({confidence:.2f}) → using neutral profile")
        emotion = "neutral"

    # Step 3 — Validate emotion has a profile
    if emotion not in EMOTION_AUDIO_PROFILES:
        print(f"⚠️ No profile for '{emotion}' → neutral")
        emotion = "neutral"

    profile = EMOTION_AUDIO_PROFILES[emotion]
    v_lo, v_hi = profile["valence"]
    e_lo, e_hi = profile["energy"]
    d_lo, d_hi = profile["danceability"]

    print(f"🎵 Filtering for '{emotion}': "
          f"valence=[{v_lo:.2f},{v_hi:.2f}] "
          f"energy=[{e_lo:.2f},{e_hi:.2f}] "
          f"dance=[{d_lo:.2f},{d_hi:.2f}]")

    # Step 4 — Strict filter
    mask = (
        songs_df["valence"].between(v_lo, v_hi) &
        songs_df["energy"].between(e_lo, e_hi) &
        songs_df["danceability"].between(d_lo, d_hi)
    )
    filtered = songs_df[mask].copy()
    print(f"   Strict filter: {len(filtered)} songs")

    # Slight widen if too few (max +0.1 on each bound)
    if len(filtered) < n:
        expand = 0.10
        mask2 = (
            songs_df["valence"].between(max(0, v_lo - expand),
                                        min(1, v_hi + expand)) &
            songs_df["energy"].between(max(0, e_lo - expand),
                                       min(1, e_hi + expand)) &
            songs_df["danceability"].between(max(0, d_lo - expand),
                                             min(1, d_hi + expand))
        )
        filtered = songs_df[mask2].copy()
        print(f"   Widened filter: {len(filtered)} songs")

    # Step 5 — Exclude history
    if history:
        filtered = filtered[~filtered["track_name"].isin(history)]

    # Step 6 — Sort by popularity
    if "popularity" in filtered.columns:
        filtered = filtered.sort_values("popularity", ascending=False)

    # Build result
    keep_cols = [
        "track_name", "artists", "track_genre",
        "valence", "energy", "danceability", "popularity"
    ]
    keep_cols = [c for c in keep_cols if c in filtered.columns]

    result = []
    for _, row in filtered.head(n * 3).iterrows():
        rec = {c: row[c] for c in keep_cols}
        result.append(rec)
        if len(result) >= n:
            break

    # Final fallback — top popular neutral songs
    if not result:
        print(f"⚠️ No songs for '{emotion}' — falling back to neutral")
        neutral = EMOTION_AUDIO_PROFILES["neutral"]
        mask_n = (
            songs_df["valence"].between(*neutral["valence"]) &
            songs_df["energy"].between(*neutral["energy"]) &
            songs_df["danceability"].between(*neutral["danceability"])
        )
        fallback = songs_df[mask_n].sort_values(
            "popularity", ascending=False
        ).head(n)
        for _, row in fallback.iterrows():
            result.append({c: row[c] for c in keep_cols if c in row.index})

    return result


def get_navarasa_playlist(
    emotion: str,
    songs_df: pd.DataFrame,
    n: int = 20
) -> List[Dict]:
    """Return a longer playlist for one emotion, sorted by popularity."""
    if emotion not in EMOTION_AUDIO_PROFILES:
        emotion = "neutral"

    profile = EMOTION_AUDIO_PROFILES[emotion]
    mask = (
        songs_df["valence"].between(*profile["valence"]) &
        songs_df["energy"].between(*profile["energy"]) &
        songs_df["danceability"].between(*profile["danceability"])
    )
    filtered = songs_df[mask].sort_values("popularity", ascending=False).head(n)

    result = []
    nav = NAVARASA_MAPPING.get(emotion, {})
    keep_cols = [
        "track_name", "artists", "track_genre",
        "valence", "energy", "danceability", "popularity"
    ]
    keep_cols = [c for c in keep_cols if c in filtered.columns]

    for _, row in filtered.iterrows():
        rec = {c: row[c] for c in keep_cols}
        rec.update({
            "navarasa": nav.get("navarasa", ""),
            "meaning":  nav.get("meaning", ""),
            "emoji":    nav.get("emoji", ""),
        })
        result.append(rec)
    return result