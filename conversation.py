"""
conversation.py
Core conversation engine for CareCaller AI.
Handles 14 healthcare questions, edge cases, emotion detection, multilingual support.
"""

import json
import re
import time
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────
# CONFIG — uses same .env as backend
# ─────────────────────────────────────
GROK_API_KEY  = os.getenv("GROK_API_KEY", "")
GROK_API_BASE = os.getenv("GROK_API_BASE", "https://api.x.ai/v1")
GROK_MODEL    = os.getenv("GROK_MODEL",    "grok-beta")

# Try Anthropic as fallback if set
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ─────────────────────────────────────
# 14 HEALTHCARE QUESTIONS
# ─────────────────────────────────────
HEALTHCARE_QUESTIONS = [
    {"id": 1,  "key": "weight",               "question": "Can you tell me your current weight?"},
    {"id": 2,  "key": "weight_change",        "question": "Have you noticed any change in your weight recently — gained or lost?"},
    {"id": 3,  "key": "side_effects",         "question": "Have you experienced any side effects from your medication?"},
    {"id": 4,  "key": "side_effect_severity", "question": "If yes, how severe would you say they are — mild, moderate, or severe?"},
    {"id": 5,  "key": "medication_taken",     "question": "Have you been taking your medication as prescribed every day?"},
    {"id": 6,  "key": "missed_doses",         "question": "Have you missed any doses in the past week?"},
    {"id": 7,  "key": "appetite",             "question": "How has your appetite been — normal, increased, or decreased?"},
    {"id": 8,  "key": "energy_level",         "question": "How are your energy levels — do you feel fatigued or energetic?"},
    {"id": 9,  "key": "sleep_quality",        "question": "How has your sleep been — are you sleeping well?"},
    {"id": 10, "key": "mood",                 "question": "How would you describe your mood over the past week?"},
    {"id": 11, "key": "physical_activity",    "question": "Are you able to do light physical activity or exercise?"},
    {"id": 12, "key": "other_medications",    "question": "Are you currently taking any other medications or supplements?"},
    {"id": 13, "key": "allergies",            "question": "Have you noticed any new allergic reactions recently?"},
    {"id": 14, "key": "refill_needed",        "question": "Do you need a medication refill before your next appointment?"},
]

# ─────────────────────────────────────
# EMOTION KEYWORDS
# ─────────────────────────────────────
EMOTION_PATTERNS = {
    "stressed":  ["stressed", "anxiety", "anxious", "worried", "tension", "nervous",
                  "tense", "overwhelmed", "panic", "scared", "frustrated", "upset",
                  "pareshan", "tension hai", "dar", "ghabrahat"],
    "confused":  ["confused", "don't understand", "not sure", "what do you mean",
                  "unclear", "lost", "dizzy", "samajh nahi", "kya matlab", "pata nahi"],
    "calm":      ["fine", "good", "okay", "great", "well", "normal", "stable",
                  "comfortable", "relaxed", "theek", "acha", "sab theek"],
    "pain":      ["pain", "hurt", "ache", "sore", "discomfort", "dard", "takleef"],
    "happy":     ["happy", "better", "improved", "wonderful", "excellent", "khush", "bahut acha"],
}

# ─────────────────────────────────────
# EDGE CASE PATTERNS
# ─────────────────────────────────────
EDGE_CASES = {
    "wrong_number":  ["wrong number", "not the right person", "who is this",
                      "galat number", "mujhe nahi jaante"],
    "not_interested": ["not interested", "don't call again", "remove me",
                       "no thanks", "mat karo call", "nahin chahiye"],
    "reschedule":    ["call later", "busy right now", "not a good time",
                     "can you call", "baad mein", "abhi time nahi"],
    "escalate":      ["emergency", "hospital", "ambulance", "doctor urgently",
                      "can't breathe", "chest pain", "very sick", "bahut bura"],
}

# ─────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────
SYSTEM_PROMPT = """You are CareCaller, a warm and professional AI healthcare assistant making medication refill check-in calls.

Your persona:
- Speak naturally and warmly, like a caring nurse
- Keep responses SHORT (1-2 sentences max per turn)
- Ask ONE question at a time
- Support both English and Hinglish naturally
- Never give medical advice — only collect information
- If patient seems distressed, slow down and show empathy

Edge case handling:
- Wrong number → Apologize politely and end call
- Not interested → Respect their decision, offer to reschedule
- Wants to reschedule → Confirm a callback time and end gracefully  
- Medical emergency → Immediately advise calling emergency services (112 in India / 911 in US)

You are currently doing a medication refill check-in call. You will ask the patient 14 health questions one at a time.
Be conversational — don't make it feel like a form. React to their answers naturally before asking the next question.

Always respond in JSON format:
{
  "message": "your spoken response to patient",
  "question_asked": "the health question you asked (if any)",
  "answer_extracted": "what the patient said about their health (if any)",  
  "question_key": "the key name from the question list (if any)",
  "edge_case": "none | wrong_number | not_interested | reschedule | escalate",
  "emotion": "calm | stressed | confused | pain | happy | unknown",
  "call_status": "active | completed | ended_early"
}"""


# ─────────────────────────────────────────────
# LANGUAGE DETECTOR
# ─────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Simple Hinglish/Hindi detector."""
    hindi_words = ["hai", "hain", "nahi", "acha", "theek", "kya", "mujhe",
                   "abhi", "bahut", "ek", "do", "toh", "lekin", "aur", "se",
                   "mein", "ko", "ka", "ki", "ke", "par", "bhi", "hi", "na"]
    words = text.lower().split()
    hindi_count = sum(1 for w in words if w in hindi_words)
    ratio = hindi_count / max(len(words), 1)
    if ratio > 0.15:
        return "hinglish"
    return "english"


# ─────────────────────────────────────────────
# EMOTION DETECTOR
# ─────────────────────────────────────────────
def detect_emotion(text: str) -> str:
    """Detect patient emotion from their response."""
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in EMOTION_PATTERNS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[emotion] = score
    if not scores:
        return "calm"
    return max(scores, key=scores.get)


# ─────────────────────────────────────────────
# EDGE CASE DETECTOR
# ─────────────────────────────────────────────
def detect_edge_case(text: str) -> str:
    """Detect if patient is expressing an edge case."""
    text_lower = text.lower()
    for case, keywords in EDGE_CASES.items():
        if any(kw in text_lower for kw in keywords):
            return case
    return "none"


# ─────────────────────────────────────────────
# LLM CALLER
# ─────────────────────────────────────────────
def call_llm(messages: list) -> dict:
    """
    Call Grok API (or Anthropic fallback) and parse JSON response.
    Returns a dict with message, emotion, edge_case, etc.
    """
    # ── Try Grok ──────────────────────────
    if GROK_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": GROK_MODEL,
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": 400,
            }
            resp = requests.post(
                f"{GROK_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _parse_llm_response(content)
        except Exception as e:
            print(f"Grok error: {e}")

    # ── Try Anthropic ──────────────────────
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
            user_messages = [m for m in messages if m["role"] != "system"]
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                system=system_msg,
                messages=user_messages
            )
            content = response.content[0].text
            return _parse_llm_response(content)
        except Exception as e:
            print(f"Anthropic error: {e}")

    # ── Demo fallback (no API key) ─────────
    return _demo_response(messages)


def _parse_llm_response(content: str) -> dict:
    """Parse LLM JSON response safely."""
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```json|```", "", content).strip()
        # Find JSON object
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    # Fallback: wrap raw text
    return {
        "message": content,
        "question_asked": "",
        "answer_extracted": "",
        "question_key": "",
        "edge_case": "none",
        "emotion": "calm",
        "call_status": "active"
    }


def _demo_response(messages: list) -> dict:
    """Demo mode when no API key is set."""
    # Count user messages to simulate progression
    user_turns = sum(1 for m in messages if m["role"] == "user")
    q_idx = min(user_turns - 1, len(HEALTHCARE_QUESTIONS) - 1)
    
    if user_turns == 0 or q_idx < 0:
        return {
            "message": "Hello! This is CareCaller AI. I'm calling for your medication refill check-in. Do you have a few minutes?",
            "question_asked": "",
            "answer_extracted": "",
            "question_key": "",
            "edge_case": "none",
            "emotion": "calm",
            "call_status": "active"
        }
    
    q = HEALTHCARE_QUESTIONS[q_idx]
    return {
        "message": f"Got it, thank you! {q['question']}",
        "question_asked": q["question"],
        "answer_extracted": messages[-1]["content"] if messages else "",
        "question_key": q["key"],
        "edge_case": "none",
        "emotion": "calm",
        "call_status": "active" if q_idx < 13 else "completed"
    }


# ─────────────────────────────────────────────
# CONVERSATION SESSION
# ─────────────────────────────────────────────
class ConversationSession:
    """
    Manages a single patient call session.
    Tracks questions asked, responses, emotion history, and call state.
    """

    def __init__(self, patient_name: str = "Patient"):
        self.patient_name     = patient_name
        self.session_id       = f"session_{int(time.time())}"
        self.started_at       = datetime.now().isoformat()
        self.messages         = []          # Full LLM message history
        self.conversation_log = []          # UI display log [{role, text, emotion, timestamp}]
        self.responses        = {}          # {question_key: answer}
        self.questions_asked  = []          # List of question keys asked
        self.emotion_history  = []          # Emotion per turn
        self.call_tags        = set()       # Spotify-style tags
        self.language         = "english"
        self.call_status      = "idle"      # idle | active | completed | ended_early
        self.comfort_mode     = False       # Slow down if confused
        self.current_q_index  = 0
        self.edge_case        = "none"

        # Inject system prompt
        self.messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # ─────────────────────────────────────
    def start_call(self) -> str:
        """Generate the opening greeting."""
        self.call_status = "active"
        opening_prompt = (
            f"Start the medication refill check-in call with {self.patient_name}. "
            f"Greet them warmly, introduce yourself as CareCaller AI, and ask if they have a few minutes. "
            f"Do NOT ask any health questions yet."
        )
        self.messages.append({"role": "user", "content": opening_prompt})
        result = call_llm(self.messages)
        
        ai_msg = result.get("message", "Hello! This is CareCaller AI. How are you today?")
        self.messages.append({"role": "assistant", "content": json.dumps(result)})
        self._log("assistant", ai_msg, result.get("emotion", "calm"))
        self.call_status = result.get("call_status", "active")
        return ai_msg

    # ─────────────────────────────────────
    def patient_speaks(self, patient_input: str) -> dict:
        """
        Process a patient's spoken/typed input.
        Returns a dict with AI response and updated session state.
        """
        if self.call_status not in ("active",):
            return {"message": "The call has ended.", "call_status": self.call_status}

        # Detect language + emotion + edge cases
        detected_lang    = detect_language(patient_input)
        detected_emotion = detect_emotion(patient_input)
        detected_edge    = detect_edge_case(patient_input)

        if detected_lang == "hinglish":
            self.language = "hinglish"
        self.emotion_history.append(detected_emotion)

        # Activate comfort mode if confused
        if detected_emotion == "confused":
            self.comfort_mode = True
            self.call_tags.add("🤔 Needs Clarification")
        if detected_emotion == "stressed":
            self.call_tags.add("😟 Patient Stressed")
        if detected_emotion == "pain":
            self.call_tags.add("⚠️ Pain Reported")
        if self.language == "hinglish":
            self.call_tags.add("🌐 Hinglish")

        # Log patient input
        self._log("patient", patient_input, detected_emotion)

        # Build LLM prompt with context
        next_q = None
        if self.current_q_index < len(HEALTHCARE_QUESTIONS):
            next_q = HEALTHCARE_QUESTIONS[self.current_q_index]

        context = self._build_context_prompt(patient_input, detected_edge, detected_emotion, next_q)
        self.messages.append({"role": "user", "content": context})

        # Comfort mode: add a small delay hint
        if self.comfort_mode:
            time.sleep(0.3)

        result = call_llm(self.messages)

        # Process result
        ai_msg        = result.get("message", "I understand. Please continue.")
        edge_case     = result.get("edge_case", detected_edge)
        answer        = result.get("answer_extracted", "")
        q_key         = result.get("question_key", "")
        call_status   = result.get("call_status", "active")
        result_emotion = result.get("emotion", detected_emotion)

        self.messages.append({"role": "assistant", "content": json.dumps(result)})
        self._log("assistant", ai_msg, result_emotion)

        # Store response if we got an answer
        if q_key and answer and q_key not in self.responses:
            self.responses[q_key] = answer
            if next_q and q_key == next_q["key"]:
                self.current_q_index += 1
                if next_q["key"] not in self.questions_asked:
                    self.questions_asked.append(next_q["key"])

        # Handle edge cases
        if edge_case != "none":
            self.edge_case = edge_case
            self._handle_edge_case(edge_case)

        # Check if all questions answered
        if self.current_q_index >= len(HEALTHCARE_QUESTIONS):
            call_status = "completed"
            self.call_tags.add("✅ All Questions Answered")

        self.call_status = call_status

        # Auto-tag call
        self._auto_tag()

        return {
            "message":       ai_msg,
            "emotion":       result_emotion,
            "edge_case":     edge_case,
            "call_status":   call_status,
            "responses":     self.responses,
            "call_tags":     list(self.call_tags),
            "language":      self.language,
            "comfort_mode":  self.comfort_mode,
            "q_progress":    f"{self.current_q_index}/{len(HEALTHCARE_QUESTIONS)}",
        }

    # ─────────────────────────────────────
    def _build_context_prompt(self, patient_input, edge_case, emotion, next_q) -> str:
        lang_hint = "The patient is speaking in Hinglish — respond in warm Hinglish/English mix." if self.language == "hinglish" else ""
        comfort_hint = "The patient seems confused. Slow down, speak very simply, and reassure them." if self.comfort_mode else ""
        q_hint = f"Next question to ask: '{next_q['question']}' (key: {next_q['key']})" if next_q else "All 14 questions have been asked. Wrap up the call warmly and thank the patient."
        edge_hint = f"EDGE CASE DETECTED: {edge_case}. Handle this appropriately." if edge_case != "none" else ""
        answers_so_far = json.dumps(self.responses, indent=2) if self.responses else "None yet"

        return f"""Patient said: "{patient_input}"
Detected emotion: {emotion}
Questions answered so far ({self.current_q_index}/14): {answers_so_far}
{q_hint}
{lang_hint}
{comfort_hint}
{edge_hint}

Respond naturally to what they said, then ask the next question if appropriate."""

    # ─────────────────────────────────────
    def _handle_edge_case(self, edge_case: str):
        case_tags = {
            "wrong_number":   "❌ Wrong Number",
            "not_interested": "🚫 Opted Out",
            "reschedule":     "📅 Rescheduled",
            "escalate":       "🚨 Escalated",
        }
        if edge_case in case_tags:
            self.call_tags.add(case_tags[edge_case])
        if edge_case in ("wrong_number", "not_interested"):
            self.call_status = "ended_early"

    # ─────────────────────────────────────
    def _auto_tag(self):
        """Automatically add Spotify-style call tags based on session state."""
        answered = len(self.responses)
        total    = len(HEALTHCARE_QUESTIONS)

        if answered == total:
            self.call_tags.add("✅ Complete")
        elif answered >= total * 0.7:
            self.call_tags.add("📊 Mostly Complete")
        elif answered >= total * 0.3:
            self.call_tags.add("⏳ Partial")
        elif answered == 0 and self.current_q_index > 0:
            self.call_tags.add("❓ No Answers Captured")

        emotions = self.emotion_history
        if emotions.count("stressed") >= 2:
            self.call_tags.add("😟 High Stress")
        if emotions.count("confused") >= 2:
            self.call_tags.add("🤔 Multiple Confusions")
        if all(e in ("calm", "happy") for e in emotions) and emotions:
            self.call_tags.add("😊 Smooth Call")

    # ─────────────────────────────────────
    def _log(self, role: str, text: str, emotion: str = "calm"):
        self.conversation_log.append({
            "role":      role,
            "text":      text,
            "emotion":   emotion,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

    # ─────────────────────────────────────
    def get_summary(self) -> dict:
        """Return full session summary as JSON-serializable dict."""
        dominant_emotion = "calm"
        if self.emotion_history:
            dominant_emotion = max(set(self.emotion_history), key=self.emotion_history.count)

        return {
            "session_id":      self.session_id,
            "patient_name":    self.patient_name,
            "started_at":      self.started_at,
            "ended_at":        datetime.now().isoformat(),
            "call_status":     self.call_status,
            "language":        self.language,
            "comfort_mode":    self.comfort_mode,
            "dominant_emotion": dominant_emotion,
            "emotion_history": self.emotion_history,
            "call_tags":       list(self.call_tags),
            "questions_asked": self.questions_asked,
            "responses":       self.responses,
            "q_answered":      len(self.responses),
            "q_total":         len(HEALTHCARE_QUESTIONS),
            "completion_rate": f"{len(self.responses)/len(HEALTHCARE_QUESTIONS)*100:.0f}%",
            "edge_case":       self.edge_case,
        }
