"""
app.py — CareCaller AI: Voice Healthcare Assistant
Run with: streamlit run app.py

A Streamlit-based frontend for AI-powered medication refill check-in calls.
Connects to the FastAPI backend OR runs standalone with direct LLM calls.
"""

import streamlit as st
import json
import time
import os
from datetime import datetime

# ── Local modules ─────────────────────────────────────────────────────────
from conversation import ConversationSession, HEALTHCARE_QUESTIONS
from utils import (
    save_session, format_responses_for_display,
    emotion_to_emoji, status_to_emoji, get_progress_color
)
from tts import speak_text, is_tts_available
from stt import is_stt_available

# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareCaller AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Clean clinical dark theme, medical-grade aesthetic
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root Variables ──────────────────────────────── */
:root {
    --bg-primary:    #0a0f1a;
    --bg-card:       #111827;
    --bg-elevated:   #1a2236;
    --accent-teal:   #14b8a6;
    --accent-blue:   #3b82f6;
    --accent-green:  #22c55e;
    --accent-red:    #ef4444;
    --accent-amber:  #f59e0b;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #475569;
    --border:        #1e2d40;
    --glow-teal:     0 0 20px rgba(20, 184, 166, 0.15);
    --glow-blue:     0 0 20px rgba(59, 130, 246, 0.15);
}

/* ── Global Reset ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(20, 184, 166, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse at 90% 100%, rgba(59, 130, 246, 0.06) 0%, transparent 60%) !important;
}

/* ── Hide Streamlit chrome ────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1400px !important; }

/* ── Header ───────────────────────────────────────── */
.cc-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 28px;
    background: linear-gradient(135deg, rgba(20,184,166,0.12) 0%, rgba(59,130,246,0.08) 100%);
    border: 1px solid rgba(20,184,166,0.25);
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: var(--glow-teal);
}
.cc-header-icon {
    font-size: 2.5rem;
    line-height: 1;
}
.cc-header-title {
    font-size: 1.7rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #14b8a6, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.cc-header-sub {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 2px 0 0 0;
    font-weight: 300;
}
.cc-status-pill {
    margin-left: auto;
    padding: 6px 16px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.status-idle     { background: rgba(148,163,184,0.12); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }
.status-active   { background: rgba(34,197,94,0.12);  color: #22c55e; border: 1px solid rgba(34,197,94,0.3);
                   animation: pulse-green 2s infinite; }
.status-completed { background: rgba(20,184,166,0.12); color: #14b8a6; border: 1px solid rgba(20,184,166,0.3); }
.status-ended    { background: rgba(239,68,68,0.12);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.3); }
    50%       { box-shadow: 0 0 0 6px rgba(34,197,94,0); }
}

/* ── Cards ────────────────────────────────────────── */
.cc-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.cc-card-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Conversation bubbles ─────────────────────────── */
.chat-bubble-wrap { display: flex; align-items: flex-end; gap: 10px; margin: 8px 0; }
.chat-bubble-wrap.assistant { flex-direction: row; }
.chat-bubble-wrap.patient   { flex-direction: row-reverse; }

.chat-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar-ai      { background: linear-gradient(135deg, #14b8a6, #3b82f6); }
.avatar-patient { background: linear-gradient(135deg, #6366f1, #8b5cf6); }

.chat-bubble {
    max-width: 78%;
    padding: 11px 16px;
    border-radius: 16px;
    font-size: 0.9rem;
    line-height: 1.55;
    position: relative;
}
.bubble-ai {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    color: var(--text-primary);
}
.bubble-patient {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
    border: 1px solid rgba(99,102,241,0.25);
    border-bottom-right-radius: 4px;
    color: var(--text-primary);
}
.bubble-meta {
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 4px;
    font-family: 'DM Mono', monospace;
}

/* ── Tags ─────────────────────────────────────────── */
.tag-wrap { display: flex; flex-wrap: wrap; gap: 8px; }
.call-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 5px 12px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 500;
    background: rgba(20,184,166,0.1);
    border: 1px solid rgba(20,184,166,0.25);
    color: #14b8a6;
}

/* ── Progress bar ─────────────────────────────────── */
.prog-bar-bg {
    background: var(--border);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin: 8px 0;
}
.prog-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
    background: linear-gradient(90deg, #14b8a6, #3b82f6);
}

/* ── Question table ───────────────────────────────── */
.q-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.q-row:last-child { border-bottom: none; }
.q-status { font-size: 1rem; flex-shrink: 0; width: 20px; }
.q-text   { color: var(--text-secondary); flex: 1; }
.q-answer { color: var(--text-primary); font-weight: 500; flex: 1; font-family: 'DM Mono', monospace; font-size: 0.8rem; }

/* ── Metric boxes ─────────────────────────────────── */
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.metric-box {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.metric-val  { font-size: 1.6rem; font-weight: 600; line-height: 1; }
.metric-label{ font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }

/* ── Buttons ──────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #14b8a6, #0d9488) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(20,184,166,0.3) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

div[data-testid="stTextInput"] input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.6rem 1rem !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-teal) !important;
    box-shadow: 0 0 0 2px rgba(20,184,166,0.2) !important;
}

/* ── JSON display ─────────────────────────────────── */
.json-block {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #7dd3fc;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 380px;
    overflow-y: auto;
    line-height: 1.7;
}

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: var(--text-secondary) !important; font-size: 0.85rem !important; }

/* ── Selectbox ────────────────────────────────────── */
div[data-testid="stSelectbox"] > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Divider ──────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ── Scrollbar ────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Audio player ─────────────────────────────────── */
audio { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "session":         None,
        "call_active":     False,
        "call_status":     "idle",
        "conversation":    [],          # [{role, text, emotion, timestamp}]
        "responses":       {},
        "call_tags":       [],
        "emotion_history": [],
        "patient_name":    "Patient",
        "audio_enabled":   True,
        "last_audio":      None,
        "input_key":       0,           # To reset text input
        "q_progress":      "0/14",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
def render_header():
    status = st.session_state.call_status
    status_class_map = {
        "idle":        ("status-idle",      "⚪ Idle"),
        "active":      ("status-active",    "🟢 Live"),
        "completed":   ("status-completed", "✅ Done"),
        "ended_early": ("status-ended",     "🔴 Ended"),
    }
    css_cls, label = status_class_map.get(status, ("status-idle", "⚪ Idle"))

    st.markdown(f"""
    <div class="cc-header">
        <div class="cc-header-icon">🩺</div>
        <div>
            <div class="cc-header-title">CareCaller AI</div>
            <div class="cc-header-sub">Voice Healthcare Assistant — Medication Refill Check-in</div>
        </div>
        <div class="cc-status-pill {css_cls}">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_chat_bubble(role: str, text: str, emotion: str, timestamp: str):
    is_ai = (role == "assistant")
    bubble_cls  = "bubble-ai" if is_ai else "bubble-patient"
    wrap_cls    = "assistant" if is_ai else "patient"
    avatar_cls  = "avatar-ai" if is_ai else "avatar-patient"
    avatar_icon = "🩺" if is_ai else "👤"
    emotion_ico = emotion_to_emoji(emotion) if not is_ai else ""
    name_label  = "CareCaller AI" if is_ai else st.session_state.patient_name

    st.markdown(f"""
    <div class="chat-bubble-wrap {wrap_cls}">
        <div class="chat-avatar {avatar_cls}">{avatar_icon}</div>
        <div>
            <div class="chat-bubble {bubble_cls}">{text}</div>
            <div class="bubble-meta">{name_label} · {timestamp} {emotion_ico}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_progress(answered: int, total: int = 14):
    pct = answered / total
    color = get_progress_color(pct)
    st.markdown(f"""
    <div class="prog-bar-bg">
        <div class="prog-bar-fill" style="width:{pct*100:.0f}%; background: linear-gradient(90deg, {color}, {color}aa);"></div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#475569; margin-top:4px;">
        <span>{answered} answered</span>
        <span>{total - answered} remaining</span>
    </div>
    """, unsafe_allow_html=True)


def render_tags(tags: list):
    if not tags:
        st.markdown('<div style="color:#475569; font-size:0.85rem;">No tags yet — start a call.</div>', unsafe_allow_html=True)
        return
    tag_html = "".join(f'<span class="call-tag">{t}</span>' for t in tags)
    st.markdown(f'<div class="tag-wrap">{tag_html}</div>', unsafe_allow_html=True)


def render_q_table(responses: dict):
    from conversation import HEALTHCARE_QUESTIONS
    for q in HEALTHCARE_QUESTIONS:
        key    = q["key"]
        ans    = responses.get(key, "")
        status = "✅" if ans else "⬜"
        ans_display = ans if ans else '<span style="color:#475569">—</span>'
        st.markdown(f"""
        <div class="q-row">
            <div class="q-status">{status}</div>
            <div class="q-text">{q['question']}</div>
            <div class="q-answer">{ans_display}</div>
        </div>
        """, unsafe_allow_html=True)


def play_audio(text: str):
    """Generate and play TTS audio if enabled."""
    if not st.session_state.audio_enabled or not is_tts_available():
        return
    lang = "hinglish" if (
        st.session_state.session and
        st.session_state.session.language == "hinglish"
    ) else "en"
    slow = st.session_state.session.comfort_mode if st.session_state.session else False
    audio_bytes = speak_text(text, lang=lang, slow=slow)
    if audio_bytes:
        st.session_state.last_audio = audio_bytes


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Call Settings")

    patient_name = st.text_input(
        "Patient Name",
        value=st.session_state.patient_name,
        placeholder="e.g. Priya Sharma",
        disabled=st.session_state.call_active,
    )
    if patient_name:
        st.session_state.patient_name = patient_name

    st.markdown("---")
    st.markdown("### 🔊 Audio")
    audio_enabled = st.toggle(
        "Enable Voice (TTS)",
        value=st.session_state.audio_enabled,
        help="Reads AI responses aloud using Google TTS"
    )
    st.session_state.audio_enabled = audio_enabled

    if not is_tts_available():
        st.warning("gTTS not installed.\n`pip install gtts`")

    st.markdown("---")
    st.markdown("### 📊 Session Info")

    if st.session_state.session:
        sess = st.session_state.session
        answered = len(st.session_state.responses)
        st.markdown(f"**Session ID**\n`{sess.session_id}`")
        st.markdown(f"**Language:** `{sess.language}`")
        st.markdown(f"**Comfort Mode:** {'🟢 On' if sess.comfort_mode else '⚫ Off'}")
        st.markdown(f"**Questions:** `{answered}/14`")
        dominant = "calm"
        eh = st.session_state.emotion_history
        if eh:
            dominant = max(set(eh), key=eh.count)
        st.markdown(f"**Dominant Emotion:** {emotion_to_emoji(dominant)} `{dominant}`")
    else:
        st.markdown("*No active session*")

    st.markdown("---")
    st.markdown("### 🔌 Backend")
    backend_url = st.text_input("FastAPI URL", value="http://localhost:8000", help="Optional: Your FastAPI backend URL")
    st.caption("ℹ️ This app runs standalone. The FastAPI backend is optional.")

    st.markdown("---")
    st.markdown("### 📁 History")
    if st.button("📥 Download Session JSON", disabled=not st.session_state.session):
        if st.session_state.session:
            summary = st.session_state.session.get_summary()
            st.download_button(
                label="⬇️ Save JSON",
                data=json.dumps(summary, indent=2),
                file_name=f"carecaller_{st.session_state.session.session_id}.json",
                mime="application/json",
            )


# ─────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────
render_header()

# ── Top control row ────────────────────────────────────────────────────
col_start, col_end, col_reset, col_spacer = st.columns([1.2, 1.2, 1.2, 5])

with col_start:
    if not st.session_state.call_active:
        if st.button("📞 Start Call", use_container_width=True):
            # Create new session
            sess = ConversationSession(patient_name=st.session_state.patient_name)
            st.session_state.session       = sess
            st.session_state.call_active   = True
            st.session_state.call_status   = "active"
            st.session_state.conversation  = []
            st.session_state.responses     = {}
            st.session_state.call_tags     = []
            st.session_state.emotion_history = []
            st.session_state.last_audio    = None

            # Get opening greeting
            with st.spinner("Connecting..."):
                opening = sess.start_call()

            st.session_state.conversation = list(sess.conversation_log)
            play_audio(opening)
            st.rerun()

with col_end:
    if st.session_state.call_active:
        if st.button("📵 End Call", use_container_width=True):
            st.session_state.call_active = False        # ✅ boolean, not {}
            st.session_state.call_status = "ended_early"
            if st.session_state.session:
                st.session_state.session.call_status = "ended_early"
                summary = st.session_state.session.get_summary()
                save_session(summary)
                st.session_state.call_tags = summary.get("call_tags", [])
            st.rerun()

with col_reset:
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session = None
        st.session_state.call_active = False
        st.session_state.call_status = "idle"
        st.session_state.conversation = []
        st.session_state.responses = {}          # ✅ dict, not bool
        st.session_state.call_tags = []
        st.session_state.emotion_history = []
        st.session_state.last_audio = None
        st.session_state.input_key += 1
        st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# MAIN CONTENT — 3 columns
# ─────────────────────────────────────────────────────────────────────────
left_col, mid_col, right_col = st.columns([2.2, 2, 1.8], gap="medium")


# ═══════════════════════════════════════════════
# LEFT — Live Conversation + Input
# ═══════════════════════════════════════════════
with left_col:
    answered_count = len(st.session_state.responses)
    total_q = 14

    st.markdown(f"""
    <div class="cc-card-title">
        💬 Live Conversation
        <span style="margin-left:auto; font-size:0.8rem; color:#475569; font-family:'DM Mono'; text-transform:none; letter-spacing:0;">{answered_count}/{total_q} questions answered</span>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar
    render_progress(answered_count, total_q)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Conversation display
    conversation = st.session_state.conversation or []
    if not conversation:
        st.markdown("""
        <div style="text-align:center; padding: 48px 20px; color:#475569;">
            <div style="font-size:2.5rem; margin-bottom:12px;">🩺</div>
            <div style="font-size:0.95rem;">Press <strong style="color:#14b8a6">Start Call</strong> to begin the healthcare check-in.</div>
            <div style="font-size:0.8rem; margin-top:8px; color:#334155;">CareCaller AI will ask 14 health questions.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show chat bubbles (last 20 to avoid overflow)
        for msg in conversation[-20:]:
            render_chat_bubble(
                role=msg["role"],
                text=msg["text"],
                emotion=msg.get("emotion", "calm"),
                timestamp=msg.get("timestamp", "")
            )

    # Audio player
    if st.session_state.last_audio and st.session_state.audio_enabled:
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        st.audio(st.session_state.last_audio, format="audio/mp3", autoplay=True)
        st.session_state.last_audio = None

    # ── Patient Input ────────────────────────────────────
    if st.session_state.call_active:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="cc-card-title">🎙️ Your Response</div>', unsafe_allow_html=True)

        input_col, btn_col = st.columns([4, 1])
        with input_col:
            user_input = st.text_input(
                label="Type your response",
                placeholder="Type here or speak… (press Enter to send)",
                label_visibility="collapsed",
                key=f"patient_input_{st.session_state.input_key}",
            )
        with btn_col:
            send_clicked = st.button("Send →", use_container_width=True)

        if (send_clicked or (user_input and user_input.endswith("\n"))) and user_input.strip():
            _input = user_input.strip()
            sess   = st.session_state.session

            with st.spinner("CareCaller AI is responding..."):
                result = sess.patient_speaks(_input)

            # Update state
            st.session_state.conversation  = list(sess.conversation_log)
            st.session_state.responses     = dict(sess.responses)
            st.session_state.call_tags     = result.get("call_tags", [])
            st.session_state.emotion_history = list(sess.emotion_history)
            st.session_state.call_status   = result.get("call_status", "active")
            st.session_state.q_progress    = result.get("q_progress", "0/14")

            # End session cleanly when all 14 questions are answered
            if st.session_state.q_progress == "14/14":
                    st.session_state.call_status = "completed"
                    st.session_state.call_active = False
                    summary = sess.get_summary()
                    save_session(summary)

            elif result.get("call_status") in ("completed", "ended_early"):
                    st.session_state.call_active = False
                    summary = sess.get_summary()
                    save_session(summary)


            play_audio(result.get("message", ""))
            st.session_state.input_key += 1
            st.rerun()

        # STT hint
        if not is_stt_available():
            st.caption("💡 Install `SpeechRecognition pyaudio` for microphone support.")


# ═══════════════════════════════════════════════
# MIDDLE — Patient Responses Table
# ═══════════════════════════════════════════════
with mid_col:
    st.markdown('<div class="cc-card-title">📋 Patient Responses</div>', unsafe_allow_html=True)

    responses = st.session_state.responses
    if not responses:
        st.markdown("""
        <div style="text-align:center; padding:32px 16px; color:#475569; font-size:0.85rem;">
            Responses will appear here as questions are answered.
        </div>
        """, unsafe_allow_html=True)
    else:
        render_q_table(responses)

    # Metrics row
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="cc-card-title">📈 Call Metrics</div>', unsafe_allow_html=True)

    total_turns = len(st.session_state.conversation)
    ai_turns    = sum(1 for m in st.session_state.conversation if m["role"] == "assistant")
    pt_turns    = total_turns - ai_turns
    pct_done    = f"{len(responses)/14*100:.0f}%"

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-val" style="color:#14b8a6">{len(responses)}</div>
            <div class="metric-label">Answered</div>
        </div>
        <div class="metric-box">
            <div class="metric-val" style="color:#3b82f6">{pt_turns}</div>
            <div class="metric-label">Patient Turns</div>
        </div>
        <div class="metric-box">
            <div class="metric-val" style="color:#22c55e">{pct_done}</div>
            <div class="metric-label">Complete</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# RIGHT — Call Tags + JSON Output
# ═══════════════════════════════════════════════
with right_col:

    # ── Emotion timeline ──────────────────────
    st.markdown('<div class="cc-card-title">😌 Emotion Timeline</div>', unsafe_allow_html=True)
    eh = st.session_state.emotion_history
    if eh:
        emotion_icons = " → ".join(emotion_to_emoji(e) for e in eh[-8:])
        st.markdown(f'<div style="font-size:1.2rem; letter-spacing:4px; padding:8px 0;">{emotion_icons}</div>', unsafe_allow_html=True)
        dominant = max(set(eh), key=eh.count)
        st.markdown(f'<div style="font-size:0.78rem; color:#475569;">Dominant: <strong style="color:#f1f5f9">{emotion_to_emoji(dominant)} {dominant}</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#475569; font-size:0.85rem; padding:8px 0;">No emotion data yet.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Call Tags ─────────────────────────────
    st.markdown('<div class="cc-card-title">🏷️ Call Tags</div>', unsafe_allow_html=True)
    render_tags(st.session_state.call_tags)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── JSON Output ───────────────────────────
    st.markdown('<div class="cc-card-title">{ } JSON Output</div>', unsafe_allow_html=True)

    if st.session_state.session:
        summary = st.session_state.session.get_summary()
        json_str = json.dumps(summary, indent=2)
        st.markdown(f'<div class="json-block">{json_str}</div>', unsafe_allow_html=True)

        # Download button
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name=f"carecaller_{st.session_state.session.session_id}.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.markdown("""
        <div class="json-block">{
  "status": "waiting",
  "message": "Start a call to see structured output here"
}</div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Call complete banner ───────────────────
    if st.session_state.call_status == "completed":
        st.markdown("""
        <div style="background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3);
             border-radius: 12px; padding: 16px; text-align: center; margin-top: 8px;">
            <div style="font-size:1.5rem;">✅</div>
            <div style="color:#22c55e; font-weight:600; font-size:0.9rem;">Call Complete!</div>
            <div style="color:#475569; font-size:0.78rem; margin-top:4px;">All 14 questions answered</div>
        </div>
        """, unsafe_allow_html=True)

    elif st.session_state.call_status == "ended_early":
        st.markdown("""
        <div style="background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3);
             border-radius: 12px; padding: 16px; text-align: center; margin-top: 8px;">
            <div style="font-size:1.5rem;">📵</div>
            <div style="color:#ef4444; font-weight:600; font-size:0.9rem;">Call Ended Early</div>
            <div style="color:#475569; font-size:0.78rem; margin-top:4px;">Session saved to responses.json</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding: 16px; border-top: 1px solid #1e2d40; color:#334155; font-size:0.75rem;">
    CareCaller AI · Hackathon 2026 · All data is synthetic · Built with Streamlit + Grok API
</div>
""", unsafe_allow_html=True)
#to run use => "streamlit run app.py"
