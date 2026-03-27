# 🩺 CareCaller AI — Voice Healthcare Assistant

A Streamlit-powered AI healthcare call simulator for medication refill check-ins.
Built for the CareCaller Hackathon 2026.

---

## ✨ What It Does

- 🎙️ **Simulates healthcare phone calls** — AI asks 14 medication refill questions
- 😌 **Emotion detection** — detects if patient is calm, stressed, confused, or in pain
- 🌐 **Multilingual** — handles English and Hinglish naturally
- 🏷️ **Call tagging** — Spotify-style tags (✅ Complete, 😟 Stressed, 🌐 Hinglish...)
- 🤔 **Comfort mode** — slows down automatically if patient seems confused
- 🚨 **Edge cases** — wrong number, not interested, reschedule, escalation
- 📊 **Live JSON output** — structured responses downloadable as JSON
- 🔊 **Text-to-speech** — AI responses read aloud via gTTS

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- A Grok API key from [console.x.ai](https://console.x.ai/) (or Anthropic key)

### 2. Setup

```bash
cd voicehack

# Create virtual environment
python -m venv voicehackenv
voicehackenv\Scripts\activate       # Windows
# source voicehackenv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API key — copy .env from your ai_coding_coach project
# OR create a new .env:
echo GROK_API_KEY=your_key_here > .env
```

### 3. Run

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🎮 Demo Mode (No API Key)

The app works in **demo mode** without an API key — it cycles through realistic simulated responses so you can test the full UI flow.

---

## 📁 Project Structure

```
voicehack/
├── app.py            ← Streamlit UI (main file)
├── conversation.py   ← AI conversation engine + emotion detection
├── stt.py            ← Speech-to-Text (SpeechRecognition)
├── tts.py            ← Text-to-Speech (gTTS)
├── utils.py          ← Helper utilities
├── responses.json    ← Saved session data
├── requirements.txt
└── README.md
```

---

## 🎙️ Enabling Microphone

To use voice input instead of typing:

```bash
pip install SpeechRecognition pyaudio
```

Then uncomment lines in `requirements.txt` and the microphone button will appear.

---

## 🌐 Architecture

```
Patient (Browser)
    ↓ Type / Speak
Streamlit UI (app.py)
    ↓ Patient input
ConversationSession (conversation.py)
    ↓ Builds message history + prompts
Grok API / Anthropic API
    ↓ JSON response
Emotion + Edge Case Detection
    ↓
UI Update (bubbles, tags, JSON, audio)
```

---

## 🔗 FastAPI Backend Integration

This Streamlit app runs **standalone**. If you want to connect to your FastAPI backend (`ai_coding_coach`):

1. Both apps share the same `.env` file
2. The FastAPI backend runs at `http://localhost:8000`
3. The Streamlit app runs at `http://localhost:8501`
4. They can run simultaneously — use the FastAPI `/docs` for API testing and Streamlit for the live call simulation

---

## 🏆 Hackathon Tips

- Demo flow: Start Call → answer a few questions → show emotion timeline → download JSON
- The JSON output is the key deliverable for Problem 1 evaluation
- Stress the system: say "wrong number" or "I'm not interested" to trigger edge cases
- Say "I'm very confused" to trigger comfort mode (slower responses)
