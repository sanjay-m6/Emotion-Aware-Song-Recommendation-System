"""
AI Service for dynamic emotion-based music recommendation parameters.
Uses NVIDIA's LLM API to analyze emotions and output JSON.
"""

import os
import json
import time
from typing import Dict, Any

from openai import OpenAI

def get_ai_client():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        return None
        
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

def get_music_parameters(emotion: str, confidence: float) -> Dict[str, Any]:
    """
    Call the NVIDIA LLM to get dynamic music recommendation parameters.
    
    Args:
        emotion: The detected emotion (e.g. happy, sad)
        confidence: The confidence score of the detection (0.0 to 1.0)
        
    Returns:
        Dict containing JSON parameters for Spotify.
    """
    client = get_ai_client()
    
    # Fallback default if AI fails or isn't configured
    default_fallback = {
        "interpretation": f"A standard {emotion} mood.",
        "genres": ["pop", "chill"] if emotion in ["happy", "neutral", "surprise"] else ["acoustic", "ambient"],
        "explanation": f"We picked some standard tracks for a {emotion} vibe since the AI service is currently unavailable.",
        "spotify_targets": {
            "target_valence": 0.5,
            "target_energy": 0.5,
            "target_danceability": 0.5
        }
    }
    
    if not client:
        print("⚠️ NVIDIA_API_KEY not set. Using fallback parameters.")
        return default_fallback

    # Intensity mapping
    intensity = "moderate"
    if confidence > 0.8:
        intensity = "high"
    elif confidence < 0.4:
        intensity = "low"

    prompt = f"""You are an expert music curator and psychological DJ.
The user's current emotion is "{emotion}" with a "{intensity}" intensity.
Your goal is to choose the absolute best music to match or appropriately improve this mood.
CRITICAL RULE: You must exclusively recommend Tamil and English songs.

You must respond ONLY with a valid JSON object matching this exact structure, nothing else:
{{
  "interpretation": "Short description of their current state (e.g., 'High energy but somewhat aggressive mood')",
  "genres": ["genre1", "genre2"], 
  "tempo": "slow" | "medium" | "fast",
  "intent": "uplift" | "relax" | "focus" | "energize" | "catharsis",
  "explanation": "A friendly 1-sentence message to the user explaining why you picked this music for them.",
  "spotify_targets": {{
    "target_valence": <float 0.0 to 1.0>,
    "target_energy": <float 0.0 to 1.0>,
    "target_danceability": <float 0.0 to 1.0>
  }}
}}

Notes for genres: ONLY use official Spotify genre seeds. To ensure Tamil and English music, YOU MUST ALWAYS include genres like "tamil", "indian", "kollywood", "pop", or "english". Provide exactly 2-3 genres.
Make the explanation conversational, like: "Since you're feeling a bit sad, we've selected some calm, acoustic Tamil and English tracks to help you process and relax."
"""

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=500,
            stream=False,
            # If the model supports JSON mode natively, but we'll try without first to ensure compatibility
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Clean up any potential markdown formatting the model might add
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        response_text = response_text.strip()
        
        data = json.loads(response_text)
        
        # Ensure spotify_targets exists
        if "spotify_targets" not in data:
            data["spotify_targets"] = default_fallback["spotify_targets"]
            
        return data
        
    except Exception as e:
        print(f"❌ AI parameter generation failed: {e}")
        return default_fallback

def chat_with_music_ai(message: str, history: list = None) -> Dict[str, Any]:
    """
    Chat with the AI to get a conversational reply and music parameters.
    """
    client = get_ai_client()
    
    default_fallback = {
        "reply": "I'm having trouble connecting to my brain right now, but here are some nice tunes anyway!",
        "emotion": "neutral",
        "genres": ["tamil", "pop"],
        "spotify_targets": {
            "target_valence": 0.5,
            "target_energy": 0.5,
            "target_danceability": 0.5
        }
    }
    
    if not client:
        return default_fallback

    system_prompt = """You are Emora, an empathetic and cool AI music curator.
Your job is to chat with the user, figure out their mood, and pick the perfect Tamil and English songs for them.
CRITICAL RULE: You must exclusively recommend Tamil and English songs.

You must respond ONLY with a valid JSON object matching this exact structure:
{
  "reply": "Your conversational response to the user's message. Keep it short, friendly, and empathetic.",
  "emotion": "The core emotion you detected (e.g. happy, sad, angry, stressed, energetic)",
  "genres": ["genre1", "genre2"], 
  "spotify_targets": {
    "target_valence": <float 0.0 to 1.0>,
    "target_energy": <float 0.0 to 1.0>,
    "target_danceability": <float 0.0 to 1.0>
  }
}

Notes for genres: ONLY use official Spotify genre seeds. To ensure Tamil and English music, YOU MUST ALWAYS include genres like "tamil", "indian", "kollywood", "pop", or "english". Provide exactly 2-3 genres.
"""

    messages = [{"role": "system", "content": system_prompt}]
    
    if history:
        for msg in history[-5:]:  # Keep context window small
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            
    messages.append({"role": "user", "content": message})

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            max_tokens=500,
            stream=False,
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        return json.loads(response_text.strip())
        
    except Exception as e:
        print(f"❌ AI chat failed: {e}")
        return default_fallback
