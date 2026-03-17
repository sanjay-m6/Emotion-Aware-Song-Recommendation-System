"""
Spotify tracks dataset preprocessing and caching.

Loads the maharshipandya/spotify-tracks-dataset from HuggingFace,
cleans and filters invalid records, computes normalized audio scores,
and caches to parquet for efficient access during recommendation generation.
"""

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from datasets import load_dataset


def preprocess_spotify() -> int:
    """
    Load, clean, and cache Spotify tracks dataset.
    
    Steps:
    1. Load dataset from HuggingFace
    2. Remove rows with missing critical audio features
    3. Deduplicate by track_name and artists
    4. Keep only tracks with popularity > 0
    5. Compute normalized audio score
    6. Save to data/songs_cache.parquet
    
    Prints:
        - Initial and final row counts
        - Top 10 genres by frequency
        - Value ranges for audio features
        - Cache path confirmation
    
    Returns:
        Number of tracks saved to cache
        
    Raises:
        RuntimeError: If dataset fails to load or save
    """
    print("\n" + "="*70)
    print("🎵 LOADING & PREPROCESSING SPOTIFY DATASET")
    print("="*70)
    
    try:
        # Load from HuggingFace
        print("\nLoading from HuggingFace...")
        songs_ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
        songs_df = songs_ds.to_pandas()
        
        initial_count = len(songs_df)
        print(f"  Initial records: {initial_count:,}")
        
        # Verify required columns exist
        required_cols = {
            "track_name", "artists", "track_genre",
            "valence", "energy", "danceability"
        }
        available_cols = set(songs_df.columns)
        if not required_cols.issubset(available_cols):
            missing = required_cols - available_cols
            raise ValueError(f"Missing columns: {missing}")
        
        # Remove rows with missing critical audio features
        songs_df = songs_df.dropna(subset=["track_name", "artists", "valence", "energy", "danceability"])
        print(f"  After removing NaN:    {len(songs_df):,} (removed {initial_count - len(songs_df):,})")
        
        # Deduplicate by track and artist
        songs_df = songs_df.drop_duplicates(subset=["track_name", "artists"])
        print(f"  After deduplication:  {len(songs_df):,}")
        
        # Filter by popularity (optional column)
        if "popularity" in songs_df.columns:
            songs_df = songs_df[songs_df["popularity"] > 0]
            print(f"  After popularity filter: {len(songs_df):,}")
        
        # Compute normalized audio score (weighted combination of features)
        songs_df["normalized_score"] = (
            0.4 * songs_df["valence"] +
            0.3 * songs_df["energy"] +
            0.3 * songs_df["danceability"]
        )
        
        print(f"\n✅ Dataset 2 (Spotify) cleaned successfully — {len(songs_df):,} tracks")
        
        # Print genre statistics
        print("\n  Top 10 genres by frequency:")
        if "track_genre" in songs_df.columns:
            genre_counts = songs_df["track_genre"].value_counts().head(10)
            for idx, (genre, count) in enumerate(genre_counts.items(), 1):
                percentage = (count / len(songs_df)) * 100
                print(f"    {idx:2}. {genre:25} {count:6,} tracks ({percentage:5.1f}%)")
        
        # Print audio feature ranges
        print("\n  Audio feature ranges:")
        audio_features = ["valence", "energy", "danceability", "tempo"]
        existing_features = [f for f in audio_features if f in songs_df.columns]
        
        for feature in existing_features:
            min_val = songs_df[feature].min()
            max_val = songs_df[feature].max()
            mean_val = songs_df[feature].mean()
            print(f"    {feature:15}: min={min_val:7.3f}, max={max_val:7.3f}, mean={mean_val:7.3f}")
        
        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        cache_path = data_dir / "songs_cache.parquet"
        
        # Save to parquet
        songs_df.to_parquet(cache_path)
        print(f"\n✅ Cached to: {cache_path}")
        print(f"   Cache size: {cache_path.stat().st_size / (1024**2):.2f} MB")
        
        return len(songs_df)
        
    except Exception as e:
        print(f"\n❌ Error preprocessing Spotify dataset: {e}")
        raise RuntimeError(f"Failed to preprocess Spotify dataset: {e}")


def main():
    """
    Main entry point: Load, clean, and cache Spotify dataset.
    """
    print("\n" + "="*70)
    print("🎭 EMOTION-AWARE MUSIC RECOMMENDATION SYSTEM")
    print("Music Dataset Preprocessing")
    print("="*70)
    
    try:
        spotify_tracks = preprocess_spotify()
        
        # Summary
        print("\n" + "="*70)
        print("✅ SPOTIFY DATASET READY FOR RECOMMENDATIONS")
        print("="*70)
        print(f"\n  Total tracks cached: {spotify_tracks:,}")
        print(f"  Location: data/songs_cache.parquet")
        print(f"\n  Ready to generate music recommendations! 🎵")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
