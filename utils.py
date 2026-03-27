"""
utils.py
Utility functions for CareCaller AI.
"""

import json
import os
from datetime import datetime


RESPONSES_FILE = "responses.json"


def save_session(session_summary: dict):
    """Append session summary to responses.json."""
    existing = []
    if os.path.exists(RESPONSES_FILE):
        try:
            with open(RESPONSES_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing.append(session_summary)

    with open(RESPONSES_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def load_all_sessions() -> list:
    """Load all past sessions from responses.json."""
    if not os.path.exists(RESPONSES_FILE):
        return []
    try:
        with open(RESPONSES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def format_responses_for_display(responses: dict) -> list[dict]:
    """
    Convert raw responses dict into a display-friendly list.
    Returns list of {question, answer} dicts.
    """
    from conversation import HEALTHCARE_QUESTIONS
    display = []
    for q in HEALTHCARE_QUESTIONS:
        key = q["key"]
        display.append({
            "Question": q["question"],
            "Answer":   responses.get(key, "—"),
            "Status":   "✅" if key in responses else "⬜",
        })
    return display


def emotion_to_emoji(emotion: str) -> str:
    """Map emotion string to emoji."""
    mapping = {
        "calm":     "😌",
        "stressed": "😟",
        "confused": "🤔",
        "pain":     "😣",
        "happy":    "😊",
        "unknown":  "❓",
    }
    return mapping.get(emotion, "❓")


def status_to_emoji(status: str) -> str:
    mapping = {
        "active":      "🟢 Active",
        "completed":   "✅ Completed",
        "ended_early": "🔴 Ended Early",
        "idle":        "⚪ Idle",
    }
    return mapping.get(status, status)


def get_progress_color(pct: float) -> str:
    """Return a hex color based on completion percentage."""
    if pct >= 0.8:
        return "#22c55e"   # green
    elif pct >= 0.5:
        return "#f59e0b"   # amber
    else:
        return "#ef4444"   # red
