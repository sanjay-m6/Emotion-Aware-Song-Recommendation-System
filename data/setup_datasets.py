"""
HuggingFace dataset loader and validator for AffectNet and Spotify tracks.

This script:
1. Loads AffectNet 8-class emotion dataset from HuggingFace
2. Loads Spotify tracks dataset from HuggingFace
3. Validates data integrity and required columns
4. Prints comprehensive statistics and class distributions
5. Caches cleaned Spotify data to parquet for subsequent use
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import LABEL_TO_EMOTION


def load_affectnet() -> Tuple[int, int]:
    """
    Load AffectNet 8-class emotion dataset from HuggingFace.
    
    Prints:
        - Total samples per split
        - Class distribution for each split
        - Sample per class statistics
        
    Returns:
        Tuple of (train_samples, val_samples)
        
    Raises:
        RuntimeError: If dataset fails to load
    """
    print("\n" + "="*70)
    print("📊 LOADING AFFECTNET EMOTION DATASET")
    print("="*70)
    
    try:
        affectnet = load_dataset("Mauregato/affectnet_short")
        train_data = affectnet["train"]
        val_data = affectnet["val"]
        
        print(f"\n✅ Dataset 1 (AffectNet) loaded successfully — {len(train_data) + len(val_data)} samples")
        
        # Print dataset sizes
        print(f"\n  Train split: {len(train_data):,} images")
        print(f"  Val split:   {len(val_data):,} images")
        
        # Verify required columns
        required_cols = {"image", "label"}
        train_cols = set(train_data.column_names)
        if not required_cols.issubset(train_cols):
            raise ValueError(f"Missing columns in train split. Found: {train_cols}")
        
        # Compute class distribution for train split
        train_labels = train_data["label"]
        class_counts_train = {}
        for label_id in range(8):
            count = sum(1 for l in train_labels if l == label_id)
            class_counts_train[label_id] = count
        
        print("\n  Train set class distribution:")
        for label_id in range(8):
            emotion_name = LABEL_TO_EMOTION[label_id]
            count = class_counts_train[label_id]
            percentage = (count / len(train_data)) * 100
            print(f"    {label_id} ({emotion_name:10}): {count:5,} images ({percentage:5.1f}%)")
        
        # Compute class distribution for val split
        val_labels = val_data["label"]
        class_counts_val = {}
        for label_id in range(8):
            count = sum(1 for l in val_labels if l == label_id)
            class_counts_val[label_id] = count
        
        print("\n  Val set class distribution:")
        for label_id in range(8):
            emotion_name = LABEL_TO_EMOTION[label_id]
            count = class_counts_val[label_id]
            percentage = (count / len(val_data)) * 100
            print(f"    {label_id} ({emotion_name:10}): {count:5,} images ({percentage:5.1f}%)")
        
        return len(train_data), len(val_data)
        
    except Exception as e:
        print(f"\n❌ Error loading AffectNet: {e}")
        raise RuntimeError(f"Failed to load AffectNet dataset: {e}")


def load_spotify() -> int:
    """
    Load Spotify tracks dataset from HuggingFace.
    
    Cleans and caches the dataset by:
    - Removing rows with missing valence/energy/danceability
    - Deduplicating by track_name and artists
    - Filtering out unpopular tracks
    - Computing normalized audio score
    
    Saves cleaned DataFrame to: data/songs_cache.parquet
    
    Prints:
        - Total tracks loaded
        - Tracks after cleaning
        - Top 10 genres by count
        - Value ranges for audio features
        
    Returns:
        Number of tracks saved to cache
        
    Raises:
        RuntimeError: If dataset fails to load or save
    """
    print("\n" + "="*70)
    print("🎵 LOADING SPOTIFY TRACKS DATASET")
    print("="*70)
    
    try:
        songs_ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
        songs_df = songs_ds.to_pandas()
        
        print(f"\n  Initial records: {len(songs_df):,}")
        
        # Clean data
        required_cols = {"track_name", "artists", "track_genre", "valence", "energy", "danceability"}
        available_cols = set(songs_df.columns)
        if not required_cols.issubset(available_cols):
            missing = required_cols - available_cols
            raise ValueError(f"Missing columns: {missing}")
        
        # Remove rows with missing required audio features
        songs_df = songs_df.dropna(subset=["track_name", "artists", "valence", "energy", "danceability"])
        print(f"  After removing NaN:    {len(songs_df):,}")
        
        # Remove duplicates by track and artist
        songs_df = songs_df.drop_duplicates(subset=["track_name", "artists"])
        print(f"  After deduplication:  {len(songs_df):,}")
        
        # Filter by popularity
        songs_df = songs_df[songs_df["popularity"] > 0] if "popularity" in songs_df.columns else songs_df
        print(f"  After popularity filter: {len(songs_df):,}")
        
        # Compute normalized audio score
        songs_df["normalized_score"] = (
            0.4 * songs_df["valence"] +
            0.3 * songs_df["energy"] +
            0.3 * songs_df["danceability"]
        )
        
        print(f"\n✅ Dataset 2 (Spotify) loaded successfully — {len(songs_df):,} tracks cached")
        
        # Print genre statistics
        print("\n  Top 10 genres by frequency:")
        if "track_genre" in songs_df.columns:
            genre_counts = songs_df["track_genre"].value_counts().head(10)
            for idx, (genre, count) in enumerate(genre_counts.items(), 1):
                print(f"    {idx:2}. {genre:20} {count:6,} tracks")
        
        # Print audio feature ranges
        audio_features = ["valence", "energy", "danceability", "tempo"]
        existing_features = [f for f in audio_features if f in songs_df.columns]
        
        print("\n  Audio feature ranges:")
        for feature in existing_features:
            min_val = songs_df[feature].min()
            max_val = songs_df[feature].max()
            mean_val = songs_df[feature].mean()
            print(f"    {feature:15}: min={min_val:7.3f}, max={max_val:7.3f}, mean={mean_val:7.3f}")
        
        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent
        cache_path = data_dir / "songs_cache.parquet"
        
        # Save to parquet
        songs_df.to_parquet(cache_path)
        print(f"\n  ✅ Cached to: {cache_path}")
        
        return len(songs_df)
        
    except Exception as e:
        print(f"\n❌ Error loading Spotify dataset: {e}")
        raise RuntimeError(f"Failed to load Spotify dataset: {e}")


def main():
    """
    Main entry point: Load and validate both datasets.
    """
    print("\n" + "="*70)
    print("🎭 EMOTION-AWARE MUSIC RECOMMENDATION SYSTEM")
    print("Dataset Setup & Validation")
    print("="*70)
    
    try:
        # Load AffectNet
        affectnet_train, affectnet_val = load_affectnet()
        
        # Load Spotify
        spotify_tracks = load_spotify()
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL DATASETS LOADED SUCCESSFULLY")
        print("="*70)
        print(f"\n  AffectNet:  {affectnet_train + affectnet_val:,} emotion images")
        print(f"  Spotify:    {spotify_tracks:,} tracks")
        print(f"\n  Ready to train emotion detector and recommend music! 🎵")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
