#!/usr/bin/env python3
"""
Diagnostic script to identify emotion detection and recommendation issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.constants import LABEL_TO_EMOTION, EMOTION_AUDIO_PROFILES
import pandas as pd

print("\n" + "="*80)
print("DIAGNOSIS: EMOTION DETECTION & RECOMMENDATION SYSTEM")
print("="*80)

# ============================================================================
# ISSUE 1: Check Label-to-Emotion Mapping
# ============================================================================
print("\n1. LABEL-TO-EMOTION MAPPING:")
print("-" * 80)
for label, emotion in LABEL_TO_EMOTION.items():
    print(f"  Label {label}: {emotion}")

# ============================================================================
# ISSUE 2: Check Emotion Audio Profiles (Valence Ranges)
# ============================================================================
print("\n2. EMOTION AUDIO PROFILES (Check for overlapping valence ranges):")
print("-" * 80)
print(f"{'Emotion':<12} | {'Valence':<20} | {'Energy':<20} | {'Danceability':<20}")
print("-" * 80)
for emotion in LABEL_TO_EMOTION.values():
    if emotion == "shringara":
        continue
    profile = EMOTION_AUDIO_PROFILES.get(emotion, {})
    valence = profile.get("valence", (0, 0))
    energy = profile.get("energy", (0, 0))
    danceability = profile.get("danceability", (0, 0))
    
    print(f"{emotion:<12} | ({valence[0]:.2f}-{valence[1]:.2f})        | ({energy[0]:.2f}-{energy[1]:.2f})        | ({danceability[0]:.2f}-{danceability[1]:.2f})")

# ============================================================================
# ISSUE 3: Identify Valence Range Problems
# ============================================================================
print("\n3. CRITICAL ISSUE - VALENCE RANGE ANALYSIS:")
print("-" * 80)

# Group by valence ranges
negative_emotions = ["anger", "sad", "fear", "disgust"]
positive_emotions = ["happy", "surprise"]
neutral_emotions = ["neutral", "contempt"]

print("\nNegative Emotions (should have LOW valence):")
for emotion in negative_emotions:
    profile = EMOTION_AUDIO_PROFILES.get(emotion, {})
    val = profile.get("valence", (0, 0))
    print(f"  {emotion:10} - valence: ({val[0]:.2f}, {val[1]:.2f})")

print("\nPositive Emotions (should have HIGH valence):")
for emotion in positive_emotions:
    profile = EMOTION_AUDIO_PROFILES.get(emotion, {})
    val = profile.get("valence", (0, 0))
    print(f"  {emotion:10} - valence: ({val[0]:.2f}, {val[1]:.2f})")

print("\nNeutral Emotions (should have MEDIUM valence):")
for emotion in neutral_emotions:
    profile = EMOTION_AUDIO_PROFILES.get(emotion, {})
    val = profile.get("valence", (0, 0))
    print(f"  {emotion:10} - valence: ({val[0]:.2f}, {val[1]:.2f})")

# ============================================================================
# ISSUE 4: Check if ranges overlap (danger zone)
# ============================================================================
print("\n4. OVERLAP DETECTION - DO VALENCE RANGES OVERLAP?")
print("-" * 80)

emotions_list = list(LABEL_TO_EMOTION.values())
valence_ranges = {}
for emotion in emotions_list:
    if emotion == "shringara":
        continue
    profile = EMOTION_AUDIO_PROFILES.get(emotion, {})
    valence_ranges[emotion] = profile.get("valence", (0, 0))

overlaps = []
for i, (emotion1, range1) in enumerate(valence_ranges.items()):
    for emotion2, range2 in list(valence_ranges.items())[i+1:]:
        # Check if ranges overlap
        if range1[1] >= range2[0] and range2[1] >= range1[0]:
            overlaps.append((emotion1, range1, emotion2, range2))

if overlaps:
    print("⚠️ WARNING: Overlapping valence ranges detected!")
    for e1, r1, e2, r2 in overlaps:
        print(f"  {e1:10} {r1} OVERLAPS WITH {e2:10} {r2}")
else:
    print("✅ No overlapping ranges")

# ============================================================================
# ISSUE 5: Load and check song data
# ============================================================================
print("\n5. SPOTIFY SONG DATA STATISTICS:")
print("-" * 80)

try:
    from music.recommendations import load_songs
    songs_df = load_songs()
    
    print(f"Total songs: {len(songs_df):,}")
    print(f"\nValence statistics:")
    print(f"  Min: {songs_df['valence'].min():.3f}")
    print(f"  Max: {songs_df['valence'].max():.3f}")
    print(f"  Mean: {songs_df['valence'].mean():.3f}")
    print(f"  Std: {songs_df['valence'].std():.3f}")
    
    print(f"\nEnergy statistics:")
    print(f"  Min: {songs_df['energy'].min():.3f}")
    print(f"  Max: {songs_df['energy'].max():.3f}")
    print(f"  Mean: {songs_df['energy'].mean():.3f}")
    print(f"  Std: {songs_df['energy'].std():.3f}")
    
    # Check how many songs match each emotion profile
    print(f"\n6. SONGS MATCHING EACH EMOTION PROFILE:")
    print("-" * 80)
    
    for emotion in LABEL_TO_EMOTION.values():
        if emotion == "shringara":
            continue
        
        profile = EMOTION_AUDIO_PROFILES.get(emotion, {})
        val_range = profile.get("valence", (0, 1))
        energy_range = profile.get("energy", (0, 1))
        dance_range = profile.get("danceability", (0, 1))
        
        matching = songs_df[
            (songs_df["valence"] >= val_range[0]) &
            (songs_df["valence"] <= val_range[1]) &
            (songs_df["energy"] >= energy_range[0]) &
            (songs_df["energy"] <= energy_range[1]) &
            (songs_df["danceability"] >= dance_range[0]) &
            (songs_df["danceability"] <= dance_range[1])
        ]
        
        percentage = (len(matching) / len(songs_df)) * 100
        print(f"  {emotion:10}: {len(matching):6,} songs ({percentage:5.1f}%)")
    
except Exception as e:
    print(f"❌ Error loading songs: {e}")

print("\n" + "="*80)
print("END DIAGNOSIS")
print("="*80 + "\n")
