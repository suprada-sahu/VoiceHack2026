"""
tts.py
Text-to-Speech module for CareCaller AI.
Uses gTTS (Google Text-to-Speech) — free, no API key needed.
Falls back gracefully if not installed.
"""

import io
import os

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


def is_tts_available() -> bool:
    return TTS_AVAILABLE


def text_to_speech_bytes(text: str, lang: str = "en", slow: bool = False) -> bytes | None:
    """
    Convert text to speech and return as bytes (MP3).
    
    Args:
        text: Text to speak
        lang: Language code ('en' for English, 'hi' for Hindi)
        slow: Speak slowly (comfort mode)
    
    Returns:
        MP3 bytes or None if TTS not available
    """
    if not TTS_AVAILABLE:
        return None

    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        print(f"TTS error: {e}")
        return None


def speak_text(text: str, lang: str = "en", slow: bool = False) -> bytes | None:
    """
    Generate speech audio. Returns bytes for Streamlit audio player.
    Detects Hinglish and uses English engine (works well for Hinglish).
    """
    # Use English engine for Hinglish (Hindi engine doesn't handle Roman script)
    actual_lang = "en" if lang == "hinglish" else lang
    return text_to_speech_bytes(text, lang=actual_lang, slow=slow)
