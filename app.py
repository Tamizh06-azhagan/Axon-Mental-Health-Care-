import hashlib
import html as html_lib
import json
import logging
import os
import random
import re
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(
    page_title="Axon",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Config
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path(os.getenv("DATASET_PATH", BASE_DIR / "mentalhealthchatbot.txt"))
CHATS_DIR = BASE_DIR / "user_chats"
CHATS_DIR.mkdir(exist_ok=True)

CHAT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRANSCRIPTION_MODEL_NAME = "tiny"
MAX_AUDIO_BYTES = 5 * 1024 * 1024
SIMILARITY_THRESHOLD = 0.45   # below this, the dataset answer is ignored
HISTORY_TURNS = 6             # previous messages sent to the model
MOODS = ["calm", "okay", "anxious", "stressed", "low", "overwhelmed", "angry", "numb"]

# Immediate danger: thoughts of self-harm, or of harming someone else.
CRISIS_PATTERNS = re.compile(
    r"\b(kill myself|suicide|suicidal|end my life|want to die|don'?t want to live|"
    r"self[- ]?harm|hurt myself|cut myself|no reason to live|"
    r"(kill|hurt|harm|attack|stab|shoot) (him|her|them|someone|somebody|other people|"
    r"my (husband|wife|partner|boss|mom|dad|mother|father|brother|sister|kids?|children|baby)))\b",
    re.IGNORECASE,
)
CRISIS_REPLY = (
    "Thank you for telling me. That sounds incredibly heavy, and your safety matters most right now. "
    "Please contact your local emergency number or a crisis helpline in your country right away. "
    "If you can, reach out to someone near you, a friend, family member, or neighbor, and "
    "try not to be alone with this. I'm not able to provide emergency help, but I'm here to keep "
    "you company while you reach out. Is there someone nearby you can contact right now?"
)

# Ongoing or intense distress: add a gentle nudge toward professional support.
ONGOING_PATTERNS = re.compile(
    r"\b(for (weeks|months|years)|every (day|night)|can'?t (cope|go on|function|take it)|"
    r"panic attacks?|constantly|unbearable|hopeless|nothing helps|can'?t stop crying)\b",
    re.IGNORECASE,
)
SUPPORT_NUDGE = (
    "\n\nWhat you're describing sounds like a lot to carry. When things feel this heavy or "
    "have lasted a while, talking with a licensed mental-health professional or your doctor "
    "can really help. You deserve that support, and I'm happy to keep listening too."
)
FALLBACK_REPLY = "I'm here and listening. What feels most important to share right now?"

UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")

# Optional, self-reported labels shown in the UI (value sent to the model)
MOOD_CHOICES = {
    "😌 Calm": "calm",
    "🙂 Okay": "okay",
    "😟 Anxious": "anxious",
    "😣 Stressed": "stressed",
    "😔 Low": "low",
    "🌊 Overwhelmed": "overwhelmed",
    "😠 Angry": "angry",
    "😶 Numb": "numb",
}
STARTERS = ["I feel anxious", "I can't sleep", "I need to vent", "Help me calm down"]
HELP_CHOICES = ["Talking it through", "A calming idea", "Just venting", "A small next step"]

LOGO_SVG = (
    '<svg width="30" height="30" viewBox="0 0 256 256" fill="currentColor">'
    '<path d="M256 256H128L0 128h128z"/><path d="M256 128H128L0 0h128z"/></svg>'
)
TYPING_HTML = '<div class="row bot"><div class="bub typing"><i></i><i></i><i></i></div></div>'


# -----------------------
# Themes (from the Axon HTML design)
# -----------------------
def _theme(label, ink, bg, card, accent, side, g1, g2, btn, orb1, orb2, dark=False):
    return {
        "label": label, "ink": ink, "bg": bg, "card": card, "accent": accent,
        "side": side, "g1": g1, "g2": g2, "btn": btn, "orb1": orb1, "orb2": orb2,
        "dark": dark,
    }


THEMES = {
    "lavender": _theme("Lavender", "#1B133C", "#EFEBF7", "#FFFFFF", "#6D5BD0", "#E6E0F3", "#F6F3FB", "#EFEBF7", "#1B133C", "#C9BEF0", "#8E7BE0"),
    "ocean": _theme("Calm Ocean", "#0F2A3D", "#E6F1F6", "#FFFFFF", "#2E86AB", "#D6E8F0", "#F1F8FB", "#E6F1F6", "#0F4466", "#BFE3F0", "#4FA3C7"),
    "sage": _theme("Sage", "#1F3326", "#EAF1EA", "#FFFFFF", "#5E9473", "#DCE8DC", "#F4F8F3", "#EAF1EA", "#2F5A40", "#CBE5CF", "#6FA982"),
    "mint": _theme("Mint Breeze", "#123A36", "#E3F5F1", "#FFFFFF", "#2BA58F", "#D1EDE6", "#EFFAF7", "#E3F5F1", "#0F5C52", "#C2F0E4", "#45B8A2"),
    "rose": _theme("Soft Rose", "#3B1F2B", "#F8EAF0", "#FFFFFF", "#D26A93", "#F0DAE4", "#FDF4F7", "#F8EAF0", "#7A2E52", "#F9CFE0", "#E07BA3"),
    "sand": _theme("Warm Sand", "#3A2A1E", "#F5EDE3", "#FFFFFF", "#C08A5B", "#EBDFCE", "#FBF6EE", "#F5EDE3", "#6B4A33", "#F1D9BC", "#D49A62"),
    "sunrise": _theme("Sunrise", "#3D2314", "#FDEEE0", "#FFFFFF", "#E8833A", "#F8DFCB", "#FFF7EE", "#FDEEE0", "#8A3F12", "#FFD9B0", "#F0924A"),
    "midnight": _theme("Midnight", "#E8E6F5", "#14112B", "#1F1B3D", "#8F7DF0", "#1A1636", "#1A1636", "#12102A", "#6D5BD0", "#A99CF5", "#5B49C8", True),
    "ember": _theme("Night Ember", "#F1E9EA", "#14101A", "#1E1823", "#E04A4A", "#181320", "#1B1520", "#120E17", "#B72F36", "#F08A8A", "#B72F36", True),
    "deepsea": _theme("Deep Sea", "#DCEFF5", "#0B1E2B", "#12303F", "#4FB3D9", "#0E2736", "#0F2937", "#0A1B27", "#1F7FA6", "#8ED8F0", "#2A86AD", True),
    "forest": _theme("Night Forest", "#E3F0E4", "#0F1E17", "#172B21", "#6CC48A", "#12241B", "#13261D", "#0D1A13", "#2E8B57", "#A6E3B8", "#3FA870", True),
    "graphite": _theme("Graphite", "#ECECEF", "#17181B", "#222428", "#9AA3B5", "#1C1D21", "#1E2024", "#141518", "#4A5468", "#C9CFDB", "#6B7589", True),
}
THEME_IDS = list(THEMES)

CSS_TEMPLATE = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500&family=Inter:wght@400;500;600&display=swap');

:root {
  --ink: __INK__; --bg: __BG__; --card: __CARD__; --accent: __ACCENT__;
  --side: __SIDE__; --g1: __G1__; --g2: __G2__; --btn: __BTN__;
  --orb1: __ORB1__; --orb2: __ORB2__;
  --ink2: color-mix(in srgb, var(--ink) 62%, transparent);
  --ink3: color-mix(in srgb, var(--ink) 40%, transparent);
  --line: color-mix(in srgb, var(--ink) 10%, transparent);
  color-scheme: __SCHEME__;
}

html, body, [data-testid="stApp"] { background: var(--bg) !important; }
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg, var(--g1), var(--g2)) !important; }
[data-testid="stApp"] { font-family: 'Inter', system-ui, sans-serif; color: var(--ink); }
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 760px; padding-top: 1.2rem; }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background: transparent !important; }

[data-testid="stSidebar"] { background: var(--side) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

h1, h2, h3, .serif { font-family: 'Fraunces', Georgia, serif !important; font-weight: 400 !important; color: var(--ink) !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--ink); }
[data-testid="stCaptionContainer"] { color: var(--ink2) !important; }
[data-testid="stWidgetLabel"] p { color: var(--ink2) !important; font-size: 13px !important; }

button[data-testid="stBaseButton-primary"], button[kind="primary"] {
  background: var(--btn) !important; color: #fff !important; border: 0 !important; border-radius: 14px !important;
}
button[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover { filter: brightness(1.12); }
button[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
  background: var(--card) !important; color: var(--ink) !important; border: 1px solid var(--line) !important; border-radius: 14px !important;
}
[data-testid="stPopover"] button { border-radius: 999px !important; }

[data-testid="stChatInput"] { background: var(--card) !important; border: 1px solid var(--line) !important; border-radius: 22px !important; box-shadow: 0 10px 30px -14px rgba(27,19,60,.25); }
[data-testid="stChatInput"] > div { background: transparent !important; }
[data-testid="stChatInput"] textarea { color: var(--ink) !important; background: transparent !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--ink3) !important; }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent); }

[data-testid="stVerticalBlockBorderWrapper"] { background: var(--card); border-radius: 22px !important; }

/* chat bubbles */
.row { display: flex; width: 100%; margin: 12px 0; }
.row.user { justify-content: flex-end; }
.bub { max-width: 80%; padding: 12px 16px; border-radius: 18px; line-height: 1.55; font-size: 15px; overflow-wrap: anywhere; animation: in .25s ease-out; }
.user .bub { background: var(--btn); color: #fff; border-bottom-right-radius: 5px; }
.bot .bub { background: var(--card); color: var(--ink); border: 1px solid var(--line); border-bottom-left-radius: 5px; max-width: 86%; }
.bot .bub.err { background: color-mix(in srgb, #ea580c 14%, var(--card)); border-color: color-mix(in srgb, #ea580c 35%, var(--card)); }
.cap { display: block; margin-top: 6px; font-size: 13px; opacity: .8; }
@keyframes in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.typing { display: flex; gap: 5px; padding: 16px 18px; }
.typing i { display: block; width: 7px; height: 7px; border-radius: 50%; background: var(--accent); opacity: .3; animation: dot 1.2s infinite; }
.typing i:nth-child(2) { animation-delay: .15s; }
.typing i:nth-child(3) { animation-delay: .3s; }
@keyframes dot { 40% { opacity: 1; transform: translateY(-3px); } }

/* empty state: breathing orb */
.empty { text-align: center; max-width: 500px; margin: 24px auto 8px; }
.orb { width: 120px; height: 120px; margin: 0 auto 28px; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #fff 0, var(--orb1) 38%, var(--orb2) 100%);
  box-shadow: 0 20px 60px -10px color-mix(in srgb, var(--accent) 50%, transparent);
  animation: breathe 8s ease-in-out infinite; }
@keyframes breathe { 0%, 100% { transform: scale(.88); } 50% { transform: scale(1.06); } }
.empty h2 { font-size: 34px !important; line-height: 1.15; margin-bottom: 10px !important; }
.empty p { color: var(--ink2); margin-bottom: 8px; }

/* sidebar */
.brand { display: flex; align-items: center; gap: 10px; font-family: 'Fraunces', serif; font-size: 22px; padding: 4px 4px 14px; color: var(--ink); }
.side-h { font-size: 12px; color: var(--ink3); padding: 6px 4px 6px; }
.side-foot { font-size: 12px; color: var(--ink3); margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--line); line-height: 1.5; }

/* sign in */
.gate-h { text-align: center; font-size: 30px !important; margin: 12px 0 6px !important; }
.gate-p { text-align: center; color: var(--ink2); font-size: 14px; margin-bottom: 18px; }
"""


def theme_css(theme):
    values = {
        "__INK__": theme["ink"], "__BG__": theme["bg"], "__CARD__": theme["card"],
        "__ACCENT__": theme["accent"], "__SIDE__": theme["side"], "__G1__": theme["g1"],
        "__G2__": theme["g2"], "__BTN__": theme["btn"], "__ORB1__": theme["orb1"],
        "__ORB2__": theme["orb2"], "__SCHEME__": "dark" if theme["dark"] else "light",
    }
    css = CSS_TEMPLATE
    for key, value in values.items():
        css = css.replace(key, value)
    return f"<style>{css}</style>"


# -----------------------
# Cached models and data
# -----------------------
@st.cache_resource(show_spinner="Loading the chat model (first run only)...")
def load_chat_model():
    tok = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME)
    mdl = AutoModelForCausalLM.from_pretrained(
        CHAT_MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    mdl.eval()
    return tok, mdl


@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_knowledge_base():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    patterns, pattern_intent = [], []
    for i, intent in enumerate(data["intents"]):
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            pattern_intent.append(i)

    return {
        "data": data,
        "embedder": embedder,
        "pattern_intent": pattern_intent,
        "pattern_embeddings": embedder.encode(patterns),
    }


@st.cache_resource(show_spinner="Loading the transcription model...")
def load_transcriber():
    return WhisperModel(TRANSCRIPTION_MODEL_NAME, device="cpu", compute_type="int8")


@st.cache_resource
def get_compute_lock():
    # One heavy model call at a time on CPU, shared across all sessions.
    return threading.Lock()


# -----------------------
# Chat storage
# -----------------------
def safe_user_dir(user_id):
    safe = re.sub(r"[^A-Za-z0-9_-]", "", str(user_id))
    if not safe:
        raise ValueError("Invalid user id")
    d = CHATS_DIR / safe
    d.mkdir(exist_ok=True)
    return d


def chat_path(user_id, chat_id):
    if not UUID_RE.match(chat_id or ""):
        raise ValueError("Invalid chat id")
    return safe_user_dir(user_id) / f"{chat_id}.json"


def write_chat(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_chat(user_id):
    chat_id = str(uuid.uuid4())
    data = {
        "id": chat_id,
        "created": datetime.now().isoformat(),
        "title": "New Chat",
        "messages": [],
    }
    write_chat(chat_path(user_id, chat_id), data)
    return data


def load_chat(user_id, chat_id):
    path = chat_path(user_id, chat_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_exchange(user_id, chat_id, user_text, bot_text):
    """Append the user message and bot reply together, after a reply succeeded."""
    chat = load_chat(user_id, chat_id)
    if chat is None:
        raise ValueError("Chat not found")

    now = datetime.now().isoformat()
    chat["messages"].append({"sender": "user", "text": user_text, "timestamp": now})
    chat["messages"].append({"sender": "bot", "text": bot_text, "timestamp": now})

    if chat["title"] == "New Chat":
        chat["title"] = user_text[:50].rstrip() + ("..." if len(user_text) > 50 else "")

    write_chat(chat_path(user_id, chat_id), chat)


def recent_messages(user_id, chat_id):
    """Last few messages as chat-template roles, for conversational context."""
    chat = load_chat(user_id, chat_id)
    if not chat:
        return []
    return [
        {"role": "user" if m["sender"] == "user" else "assistant", "content": m["text"]}
        for m in chat["messages"][-HISTORY_TURNS:]
    ]


def list_chats(user_id):
    files = sorted(
        safe_user_dir(user_id).glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    chats = []
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                chats.append(json.load(f))
        except Exception:
            logger.exception("Skipping unreadable chat file %s", p)
    return chats


def ensure_active_chat(user_id):
    """Keep the current chat if it still exists, else reuse an empty one or create one."""
    current = st.session_state.get("chat_id")
    if current and load_chat(user_id, current):
        return current

    empty = next((c for c in list_chats(user_id) if not c["messages"]), None)
    chat_id = empty["id"] if empty else create_chat(user_id)["id"]
    st.session_state.chat_id = chat_id
    return chat_id


# -----------------------
# Core chat logic
# -----------------------
def dataset_answer_for(user_message, kb):
    """Best matching dataset response, or '' if nothing is similar enough."""
    try:
        emb = kb["embedder"].encode([user_message], convert_to_numpy=True)
        sims = cosine_similarity(emb, kb["pattern_embeddings"])[0]
        best = int(np.argmax(sims))
        if sims[best] < SIMILARITY_THRESHOLD:
            return ""
        return random.choice(kb["data"]["intents"][kb["pattern_intent"][best]]["responses"])
    except Exception:
        logger.exception("Dataset similarity lookup failed")
        return ""


def trim_to_sentence(text):
    """Drop a trailing half-finished sentence caused by the token limit."""
    text = text.strip()
    if not text or text[-1] in ".!?\"')":
        return text
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "), text.rfind(".\n"))
    return text[: cut + 1].strip() if cut > 40 else text


def generate_reply(user_message, history, mood, kb):
    user_message = (user_message or "").strip()
    if not user_message:
        return "Please type a message so I can help."

    if CRISIS_PATTERNS.search(user_message):
        return CRISIS_REPLY

    needs_nudge = bool(ONGOING_PATTERNS.search(user_message))
    dataset_answer = dataset_answer_for(user_message, kb)

    try:
        tok, model = load_chat_model()

        system_prompt = (
            "You are Axon, a warm, caring companion for people who want to talk about how "
            "they feel. Speak in plain, gentle, everyday language, never clinical or scripted. "
            "First reflect what the person said in one short sentence so they feel heard. "
            "Then ask exactly ONE gentle, open-ended question that helps them say more. "
            "Do not diagnose, label conditions, give a list of tips, or lecture. "
            "Do not promise confidentiality. Never claim to replace a therapist or doctor. "
            "Keep replies under 90 words, in plain text."
        )
        if mood in MOODS:
            system_prompt += (
                f"\nThe person picked '{mood}' on an optional self-reported mood check-in. "
                "It is only their own quick label, not a diagnosis or a measurement. "
                "Use it to gently invite them to talk. Do not mention scores or percentages."
            )
        if dataset_answer:
            system_prompt += (
                "\nA related reply from the knowledge base, use it only if helpful: "
                + str(dataset_answer)
            )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_message})

        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        with get_compute_lock(), torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=tok.eos_token_id,
            )

        reply = tok.decode(output[0][input_len:], skip_special_tokens=True).strip()
        reply = trim_to_sentence(reply)
        if reply:
            return reply + (SUPPORT_NUDGE if needs_nudge else "")
        logger.warning("Model returned an empty reply; using fallback.")
    except Exception:
        logger.exception("Local model failed; using fallback reply")

    fallback = dataset_answer or FALLBACK_REPLY
    return fallback + (SUPPORT_NUDGE if needs_nudge else "")


def transcribe_audio(audio_bytes):
    model = load_transcriber()
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "recording.wav"
        audio_path.write_bytes(audio_bytes)
        with get_compute_lock():
            segments, _ = model.transcribe(str(audio_path), beam_size=1, vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()


# -----------------------
# UI helpers
# -----------------------
def esc(text):
    """Escape text for raw HTML. Newlines become line breaks, and the result stays on one line."""
    return html_lib.escape(text or "").replace("\n", "<br>")


def bubble(text, who, cls=""):
    st.markdown(
        f'<div class="row {who}"><div class="bub {cls}">{esc(text)}</div></div>',
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        '<div class="empty"><div class="orb"></div>'
        "<h2 class=\"serif\">Hi, I'm Axon. I'm glad you're here.</h2>"
        "<p>This is a space to say what's on your mind, at your own pace. "
        "You can type, or record a voice message below. There's no wrong way to begin.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.pills(
        "Optional: how would you describe today?",
        list(MOOD_CHOICES),
        selection_mode="single",
        key="mood_pick",
        on_change=on_mood_pick,
    )
    st.caption(
        "This is just your own quick take, not a diagnosis or a precise measurement. "
        "Feel free to skip it."
    )
    st.pills(
        "Or start with something like:",
        STARTERS,
        selection_mode="single",
        key="starter_pick",
        on_change=on_starter_pick,
    )


def render_checkin():
    with st.container(border=True):
        st.markdown("**Quick check-in (optional)**")
        st.caption(
            "Your own quick take. It isn't a test, a diagnosis, or an exact measurement. "
            "Skip it any time."
        )
        st.pills(
            "How would you describe right now?",
            list(MOOD_CHOICES),
            selection_mode="single",
            key="ci_mood",
        )
        st.segmented_control(
            "How heavy does it feel? (1 = light, 5 = very heavy)",
            ["1", "2", "3", "4", "5"],
            key="ci_heavy",
        )
        st.pills(
            "What would help most? (pick any)",
            HELP_CHOICES,
            selection_mode="multi",
            key="ci_help",
        )
        a, b, _ = st.columns([2, 2, 3])
        a.button("Share with Axon", type="primary", on_click=share_checkin, use_container_width=True)
        b.button("Not now", on_click=close_checkin, use_container_width=True)


# -----------------------
# Callbacks (run before the script on each rerun)
# -----------------------
def queue_message(text, mood=None):
    st.session_state.pending = (text, mood)


def on_mood_pick():
    label = st.session_state.get("mood_pick")
    if label:
        key = MOOD_CHOICES[label]
        queue_message(f"I'm feeling {key} today.", key)


def on_starter_pick():
    text = st.session_state.get("starter_pick")
    if text:
        queue_message(text)


def share_checkin():
    mood_label = st.session_state.get("ci_mood")
    heavy = st.session_state.get("ci_heavy")
    helps = st.session_state.get("ci_help") or []

    parts = []
    if mood_label:
        parts.append(f"I'm feeling {MOOD_CHOICES[mood_label]}.")
    if heavy:
        parts.append(f"It feels about {heavy} out of 5 in heaviness (my own rating).")
    if helps:
        parts.append(f"What might help: {', '.join(h.lower() for h in helps)}.")
    if not parts:
        return

    st.session_state.checkin_open = False
    st.session_state.checkin_offered_for = st.session_state.get("chat_id")
    mood_key = MOOD_CHOICES[mood_label] if mood_label else None
    queue_message("Quick check-in: " + " ".join(parts), mood_key)


def close_checkin():
    st.session_state.checkin_open = False


def open_checkin():
    st.session_state.checkin_open = True


def start_new_chat(user_id):
    st.session_state.chat_id = create_chat(user_id)["id"]
    st.session_state.checkin_open = False


def open_chat(chat_id):
    st.session_state.chat_id = chat_id
    st.session_state.checkin_open = False


def delete_chat_cb(user_id, chat_id):
    chat_path(user_id, chat_id).unlink(missing_ok=True)
    if st.session_state.get("chat_id") == chat_id:
        st.session_state.pop("chat_id", None)


def set_theme():
    st.session_state.theme = st.session_state.theme_pick


def init_state():
    st.session_state.setdefault("theme", "lavender")
    st.session_state.setdefault("theme_pick", st.session_state.theme)
    st.session_state.setdefault("checkin_open", False)
    st.session_state.setdefault("checkin_offered_for", None)


# -----------------------
# Page
# -----------------------
init_state()
active_theme = THEMES[st.session_state.theme]
st.markdown(theme_css(active_theme), unsafe_allow_html=True)

# Sign-in gate
if not st.user.is_logged_in:
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        with st.container(border=True):
            st.markdown(
                f'<div style="text-align:center;color:var(--ink)">{LOGO_SVG}</div>'
                '<h2 class="serif gate-h">A quiet place to talk</h2>'
                '<p class="gate-p">Sign in to save your conversations and pick up where you left off.</p>',
                unsafe_allow_html=True,
            )
            st.button("Continue with Google", type="primary", use_container_width=True, on_click=st.login)
            st.caption("Axon offers support, not diagnosis or emergency care.")
    st.stop()

# Signed-in app
user_id = st.user.sub
kb = load_knowledge_base()
chat_id = ensure_active_chat(user_id)

# Header: title on the left, account menu on the right
first_name = (st.user.get("name") or "You").split(" ")[0]
header_left, header_right = st.columns([5, 1])
title_slot = header_left.empty()
with header_right:
    with st.popover(first_name):
        if st.user.get("email"):
            st.caption(st.user.get("email"))
        st.markdown("**Theme**")
        st.radio(
            "Theme",
            THEME_IDS,
            format_func=lambda k: ("🌙 " if THEMES[k]["dark"] else "") + THEMES[k]["label"],
            key="theme_pick",
            on_change=set_theme,
            label_visibility="collapsed",
        )
        st.button("Sign out", on_click=st.logout, use_container_width=True)

# Slots that get filled after the new message is processed
history_slot = st.container()
checkin_slot = st.container()

# Composer
pending = st.session_state.pop("pending", None)
notice = None

col_voice, col_checkin = st.columns([1, 1])
with col_voice:
    with st.popover("🎙️ Voice message"):
        audio = st.audio_input("Record, then stop to send", key="voice_in")
with col_checkin:
    st.button("Quick check-in", on_click=open_checkin, use_container_width=True)

if audio is not None:
    audio_bytes = audio.getvalue()
    digest = hashlib.sha256(audio_bytes).hexdigest()
    # Streamlit reruns on every interaction, so transcribe each recording only once.
    if digest != st.session_state.get("last_audio_digest"):
        st.session_state.last_audio_digest = digest
        if not audio_bytes:
            notice = "The recording was empty. Please try again."
        elif len(audio_bytes) > MAX_AUDIO_BYTES:
            notice = "That voice message is too long. Please keep it under a minute."
        else:
            with st.spinner("Listening..."):
                transcript = transcribe_audio(audio_bytes)
            if transcript:
                pending = pending or (transcript, None)
            else:
                notice = "I couldn't make out that recording. Try again, or type your message instead."

typed = st.chat_input("Write what's on your mind")
if typed and not pending:
    pending = (typed.strip(), None)

# Render history, then handle the new message
with history_slot:
    if notice:
        st.warning(notice)

    chat = load_chat(user_id, chat_id)
    if not chat["messages"] and not pending:
        render_empty_state()

    for m in chat["messages"]:
        bubble(m["text"], "user" if m["sender"] == "user" else "bot")

    if pending:
        text, mood = pending
        bubble(text, "user")
        typing_ph = st.empty()
        typing_ph.markdown(TYPING_HTML, unsafe_allow_html=True)
        reply = generate_reply(text, recent_messages(user_id, chat_id), mood, kb)
        typing_ph.empty()
        bubble(reply, "bot")

        try:
            save_exchange(user_id, chat_id, text, reply)
            chat = load_chat(user_id, chat_id)
        except Exception:
            logger.exception("Could not save the chat exchange")
            st.error("I couldn't save this message. Please try again.")

# Title
title = chat["title"] if chat["messages"] else "New chat"
title_slot.markdown(f'<h2 class="serif" style="margin:0">{esc(title)}</h2>', unsafe_allow_html=True)

# Mid-conversation check-in: offered once, after the third message
user_count = sum(1 for m in chat["messages"] if m["sender"] == "user")
if pending and user_count == 3 and st.session_state.checkin_offered_for != chat_id:
    st.session_state.checkin_offered_for = chat_id
    st.session_state.checkin_open = True

with checkin_slot:
    if st.session_state.checkin_open:
        render_checkin()

# Sidebar: rendered last so it shows the newest chat titles
with st.sidebar:
    st.markdown(f'<div class="brand">{LOGO_SVG} Axon</div>', unsafe_allow_html=True)
    st.button(
        "＋  New chat",
        type="primary",
        on_click=start_new_chat,
        args=(user_id,),
        use_container_width=True,
    )
    st.markdown('<div class="side-h">Recent</div>', unsafe_allow_html=True)

    shown = [c for c in list_chats(user_id) if c["messages"] or c["id"] == chat_id]
    if not shown:
        st.caption("Your conversations will appear here.")
    for c in shown:
        col_open, col_del = st.columns([6, 1])
        col_open.button(
            c["title"],
            key=f"open-{c['id']}",
            on_click=open_chat,
            args=(c["id"],),
            type="primary" if c["id"] == chat_id else "secondary",
            use_container_width=True,
        )
        col_del.button(
            "✕",
            key=f"del-{c['id']}",
            on_click=delete_chat_cb,
            args=(user_id, c["id"]),
        )

    st.markdown(
        '<div class="side-foot">If you are in immediate danger, call your local emergency number.</div>',
        unsafe_allow_html=True,
    )