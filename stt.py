"""
stt.py
Speech-to-Text module for CareCaller AI.
Uses SpeechRecognition library with Google STT (free tier).
Falls back gracefully if microphone not available.
"""

import io

# ── Optional imports (won't crash if not installed) ───────────────────────
try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False

try:
    import sounddevice as sd
    import numpy as np
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


def is_stt_available() -> bool:
    """Check if STT dependencies are installed."""
    return STT_AVAILABLE


def transcribe_from_microphone(timeout: int = 8, phrase_limit: int = 15) -> dict:
    """
    Record audio from microphone and transcribe using Google STT.

    Returns:
        dict with keys: 'success', 'text', 'error'
    """
    if not STT_AVAILABLE:
        return {
            "success": False,
            "text": "",
            "error": "SpeechRecognition not installed. Run: pip install SpeechRecognition pyaudio"
        }

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.pause_threshold  = 1.2

    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)

        # Transcribe
        text = recognizer.recognize_google(audio, language="en-IN")  # en-IN covers Hinglish well
        return {"success": True, "text": text, "error": ""}

    except sr.WaitTimeoutError:
        return {"success": False, "text": "", "error": "No speech detected. Please try again."}
    except sr.UnknownValueError:
        return {"success": False, "text": "", "error": "Could not understand audio. Please speak clearly."}
    except sr.RequestError as e:
        return {"success": False, "text": "", "error": f"STT service error: {str(e)}"}
    except Exception as e:
        return {"success": False, "text": "", "error": f"Microphone error: {str(e)}"}


def transcribe_audio_file(audio_bytes: bytes, language: str = "en-IN") -> dict:
    """
    Transcribe audio from bytes (e.g., uploaded file).
    """
    if not STT_AVAILABLE:
        return {"success": False, "text": "", "error": "SpeechRecognition not installed."}

    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=language)
        return {"success": True, "text": text, "error": ""}
    except Exception as e:
        return {"success": False, "text": "", "error": str(e)}
