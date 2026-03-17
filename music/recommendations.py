"""
Music recommendation engine based on emotion-detected audio profiles.

Maps detected emotions + confidence levels to Spotify audio feature ranges,
then filters the cached Spotify tracks to find songs that match the user's
emotional state. Includes contextual inference of Shringara (Love rasa).
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import (
    EMOTION_AUDIO_PROFILES,
    ROMANTIC_GENRES,
    NAVARASA_MAPPING
)


def load_songs(cache_path: str = "data/songs_cache.parquet") -> pd.DataFrame:
    """
    Load Spotify tracks from cached parquet file.
    
    Falls back to loading from HuggingFace if cache is missing.
    
    Args:
        cache_path: Path to songs_cache.parquet (relative to project root)
        
    Returns:
        DataFrame with track_name, artists, track_genre, audio features, etc.
        
    Raises:
        RuntimeError: If both cache and HuggingFace load fail
    """
    cache_path = Path(cache_path)
    
    # Try to load from cache first
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}. Falling back to HuggingFace...")
    
    # Fallback: load from HuggingFace
    try:
        from datasets import load_dataset
        songs_ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
        df = songs_ds.to_pandas()
        
        # Clean data
        df = df.dropna(subset=["track_name", "artists", "valence", "energy", "danceability"])
        df = df.drop_duplicates(subset=["track_name", "artists"])
        if "popularity" in df.columns:
            df = df[df["popularity"] > 0]
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Failed to load songs: {e}")


def infer_shringara(
    emotion: str,
    confidence: float,
    recent_genres: List[str]
) -> str:
    """
    Infer Shringara (Love) rasa contextually.
    
    Shringara is triggered when:
    1. Detected emotion is "happy"
    2. Confidence > 0.75
    3. User's recent song history includes romantic genres (R&B, soul, indie, etc.)
    
    This contextual inference captures the nuanced emotional state that the
    user is in a romantic/love-focused mood, even though the face detector
    only sees happiness.
    
    Args:
        emotion: Current detected emotion (e.g., "happy")
        confidence: Confidence score (0.0-1.0)
        recent_genres: List of genres from recently played songs
        
    Returns:
        "shringara" if conditions met, otherwise original emotion unchanged
    """
    if emotion == "happy" and confidence > 0.75:
        # Check if any recent song genre matches romantic genres
        if recent_genres:
            recent_lower = [g.lower() for g in recent_genres]
            for genre in recent_lower:
                for romantic_genre in ROMANTIC_GENRES:
                    if romantic_genre in genre:
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
    Get music recommendations matching the detected emotion.
    
    Algorithm:
    1. Infer Shringara if contextual conditions met
    2. Fall back to "neutral" if confidence < 0.5 (uncertain detection)
    3. Look up EMOTION_AUDIO_PROFILES for valence/energy/danceability ranges
    4. Filter songs_df by those ranges
    5. Exclude recently played tracks
    6. Sort by popularity (descending)
    7. Return top n songs
    
    Args:
        emotion: Detected emotion name (e.g., "angry", "happy")
        confidence: Confidence score (0.0-1.0)
        songs_df: DataFrame of Spotify tracks with audio features
        history: List of recently played track names (to exclude)
        recent_genres: List of recently played track genres (for Shringara inference)
        n: Number of recommendations to return (default 5)
        
    Returns:
        List of recommendation dicts with keys:
        - track_name: Song title
        - artists: Artist(s)
        - track_genre: Genre tag
        - valence: Audio feature (0.0-1.0)
        - energy: Audio feature (0.0-1.0)
        - danceability: Audio feature (0.0-1.0)
        - popularity: Score 0-100
        
    Example:
        >>> songs = load_songs()
        >>> recs = get_recommendations(
        ...     emotion="happy",
        ...     confidence=0.87,
        ...     songs_df=songs,
        ...     history=["Song A", "Song B"],
        ...     recent_genres=["pop", "r&b"],
        ...     n=5
        ... )
        >>> print(recs[0]["track_name"])
    """
    # Step 1: Infer Shringara if applicable
    emotion = infer_shringara(emotion, confidence, recent_genres)
    
    # Step 2: Fall back to neutral if confidence too low
    if confidence < 0.5:
        emotion = "neutral"
    
    # Step 3: Get audio profile for this emotion
    if emotion not in EMOTION_AUDIO_PROFILES:
        emotion = "neutral"
    
    profile = EMOTION_AUDIO_PROFILES[emotion]
    valence_range = profile["valence"]
    energy_range = profile["energy"]
    danceability_range = profile["danceability"]
    
    # Step 4: Filter by audio feature ranges
    filtered = songs_df[
        (songs_df["valence"] >= valence_range[0]) &
        (songs_df["valence"] <= valence_range[1]) &
        (songs_df["energy"] >= energy_range[0]) &
        (songs_df["energy"] <= energy_range[1]) &
        (songs_df["danceability"] >= danceability_range[0]) &
        (songs_df["danceability"] <= danceability_range[1])
    ]
    
    # Step 5: Exclude recently played tracks
    if history:
        filtered = filtered[~filtered["track_name"].isin(history)]
    
    # Step 6: Sort by popularity (descending)
    filtered = filtered.sort_values("popularity", ascending=False)
    
    # Step 7: Return top n songs as dicts
    result = []
    columns_to_keep = ["track_name", "artists", "track_genre", "valence", "energy", "danceability", "popularity"]
    for _, row in filtered.head(n).iterrows():
        rec = {}
        for col in columns_to_keep:
            if col in row.index:
                rec[col] = row[col]
        result.append(rec)
    
    return result


def get_navarasa_playlist(
    emotion: str,
    songs_df: pd.DataFrame,
    n: int = 20
) -> List[Dict]:
    """
    Get a longer playlist (20 songs) for a specific emotion/navarasa.
    
    Similar to get_recommendations but:
    - Returns more songs (default 20)
    - No history filtering (can repeat across sessions)
    - Includes navarasa metadata in result
    
    Args:
        emotion: Emotion/navarasa name
        songs_df: DataFrame of Spotify tracks
        n: Number of songs in playlist (default 20)
        
    Returns:
        List of song dicts with audio features + navarasa metadata
    """
    # Get audio profile
    if emotion not in EMOTION_AUDIO_PROFILES:
        emotion = "neutral"
    
    profile = EMOTION_AUDIO_PROFILES[emotion]
    valence_range = profile["valence"]
    energy_range = profile["energy"]
    danceability_range = profile["danceability"]
    
    # Filter by ranges
    filtered = songs_df[
        (songs_df["valence"] >= valence_range[0]) &
        (songs_df["valence"] <= valence_range[1]) &
        (songs_df["energy"] >= energy_range[0]) &
        (songs_df["energy"] <= energy_range[1]) &
        (songs_df["danceability"] >= danceability_range[0]) &
        (songs_df["danceability"] <= danceability_range[1])
    ]
    
    # Sort by popularity
    filtered = filtered.sort_values("popularity", ascending=False)
    
    # Build result with navarasa metadata
    result = []
    columns_to_keep = ["track_name", "artists", "track_genre", "valence", "energy", "danceability", "popularity"]
    
    for _, row in filtered.head(n).iterrows():
        rec = {}
        for col in columns_to_keep:
            if col in row.index:
                rec[col] = row[col]
        
        # Add navarasa metadata if available
        if emotion in NAVARASA_MAPPING:
            rec["navarasa"] = NAVARASA_MAPPING[emotion]["navarasa"]
            rec["meaning"] = NAVARASA_MAPPING[emotion]["meaning"]
            rec["emoji"] = NAVARASA_MAPPING[emotion]["emoji"]
        
        result.append(rec)
    
    return result
