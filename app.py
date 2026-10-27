import hashlib
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

st.set_page_config(page_title="Axon", page_icon="🧠", layout="centered")

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


def open_chat(chat_id):
    st.session_state.chat_id = chat_id


def delete_chat_cb(user_id, chat_id):
    chat_path(user_id, chat_id).unlink(missing_ok=True)
    if st.session_state.get("chat_id") == chat_id:
        st.session_state.pop("chat_id", None)


def start_new_chat(user_id):
    st.session_state.chat_id = create_chat(user_id)["id"]


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


def respond(user_id, chat_id, user_text, mood, kb):
    with st.chat_message("user"):
        st.write(user_text)

    history = recent_messages(user_id, chat_id)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = generate_reply(user_text, history, mood, kb)
        st.write(reply)

    try:
        save_exchange(user_id, chat_id, user_text, reply)
    except Exception:
        logger.exception("Could not save the chat exchange")
        st.error("I couldn't save this message. Please try again.")
        return

    st.rerun()  # refresh the sidebar title and history from disk


# -----------------------
# Auth gate (Google sign-in via Streamlit's built-in OIDC support)
# -----------------------
if not st.user.is_logged_in:
    st.title("Axon")
    st.write(
        "A calm place to talk about how you're feeling. "
        "Sign in with Google to keep your conversations."
    )
    st.button("Sign in with Google", type="primary", on_click=st.login)
    st.caption(
        "Axon is a conversation companion, not a replacement for a therapist, "
        "doctor, or emergency services."
    )
    st.stop()

# -----------------------
# Signed-in app
# -----------------------
user_id = st.user.sub
kb = load_knowledge_base()

with st.sidebar:
    st.markdown(f"**{st.user.get('name') or 'Signed in'}**")
    if st.user.get("email"):
        st.caption(st.user.get("email"))

    st.button("New chat", on_click=start_new_chat, args=(user_id,), use_container_width=True)

    st.divider()
    st.caption("Your chats")
    for c in list_chats(user_id):
        col_open, col_del = st.columns([5, 1])
        col_open.button(
            c["title"],
            key=f"open-{c['id']}",
            on_click=open_chat,
            args=(c["id"],),
            use_container_width=True,
        )
        col_del.button(
            "🗑",
            key=f"del-{c['id']}",
            on_click=delete_chat_cb,
            args=(user_id, c["id"]),
        )

    st.divider()
    st.button("Sign out", on_click=st.logout)

st.title("Axon")

chat_id = ensure_active_chat(user_id)
chat = load_chat(user_id, chat_id)

for m in chat["messages"]:
    with st.chat_message("user" if m["sender"] == "user" else "assistant"):
        st.write(m["text"])

mood_choice = st.selectbox(
    "How are you feeling? (optional, just for you)",
    ["Not sure"] + MOODS,
    index=0,
)
mood = None if mood_choice == "Not sure" else mood_choice

audio = st.audio_input("Or record a voice message")
pending_text = None
if audio is not None:
    audio_bytes = audio.getvalue()
    digest = hashlib.sha256(audio_bytes).hexdigest()
    # Streamlit reruns on every interaction, so only transcribe a recording once.
    if digest != st.session_state.get("last_audio_digest"):
        st.session_state.last_audio_digest = digest
        if not audio_bytes:
            st.warning("The recording was empty. Please try again.")
        elif len(audio_bytes) > MAX_AUDIO_BYTES:
            st.error("That voice message is too long. Please keep it under a minute.")
        else:
            with st.spinner("Listening..."):
                transcript = transcribe_audio(audio_bytes)
            if transcript:
                pending_text = transcript
            else:
                st.warning(
                    "I couldn't make out that recording. Try again, or type your message instead."
                )

typed = st.chat_input("Type how you're feeling...")
user_text = typed or pending_text
if user_text:
    respond(user_id, chat_id, user_text, mood, kb)