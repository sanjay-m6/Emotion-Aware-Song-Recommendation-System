#!/usr/bin/env python3
"""
Test script to validate recommendation improvements.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from music.recommendations import get_recommendations, load_songs
from utils.constants import EMOTION_AUDIO_PROFILES

print("\n" + "="*80)
print("TESTING: Recommendation Logic with Fixed Audio Profiles")
print("="*80)

# Load songs
try:
    songs_df = load_songs()
    print(f"✅ Loaded {len(songs_df):,} songs\n")
except Exception as e:
    print(f"❌ Failed to load songs: {e}")
    sys.exit(1)

# Test each emotion
test_emotions = ["anger", "sad", "fear", "happy", "surprise", "neutral"]

print("Testing recommendations for each emotion:")
print("-" * 80)

for emotion in test_emotions:
    try:
        recs = get_recommendations(
            emotion="sad",  # Force sad first
            confidence=0.9,
            songs_df=songs_df,
            history=[],
            recent_genres=[],
            n=3
        )
        
        print(f"\n{emotion.upper()}:")
        print(f"  Recommendations: {len(recs)} songs")
        
        if recs:
            # Show audio features of first recommendation
            rec = recs[0]
            valence = rec.get("valence", 0)
            energy = rec.get("energy", 0)
            dance = rec.get("danceability", 0)
            
            # Get emotion profile
            profile = EMOTION_AUDIO_PROFILES.get(emotion, {})
            val_range = profile.get("valence", (0, 1))
            eng_range = profile.get("energy", (0, 1))
            dan_range = profile.get("danceability", (0, 1))
            
            # Check if in range
            val_match = val_range[0] <= valence <= val_range[1]
            eng_match = eng_range[0] <= energy <= eng_range[1]
            dan_match = dan_range[0] <= dance <= dan_range[1]
            
            print(f"  Example: {rec.get('track_name', 'Unknown')} by {rec.get('artists', 'Unknown')}")
            print(f"  Audio Features: V={valence:.2f}, E={energy:.2f}, D={dance:.2f}")
            print(f"  Expected Ranges: V={val_range}, E={eng_range}, D={dan_range}")
            print(f"  Match Status: Valence={val_match}✓, Energy={eng_match}✓, Dance={dan_match}✓")
            
            if not (val_match and eng_match and dan_match):
                print(f"  ⚠️ WARNING: Song doesn't match emotion profile!")
        else:
            print(f"  ⚠️ No recommendations found!")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80 + "\n")
