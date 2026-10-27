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
from urllib.parse import urlencode

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------
# Config
# -----------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI", "http://localhost:5000/api/auth/google/callback"
)

CHAT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRANSCRIPTION_MODEL_NAME = "tiny"

chat_model = None
chat_tokenizer = None
transcription_model = None
chat_load_lock = threading.Lock()       # guards loading the chat model
chat_gen_lock = threading.Lock()        # one generation at a time on CPU
transcription_model_lock = threading.Lock()

MAX_AUDIO_BYTES = 5 * 1024 * 1024
SIMILARITY_THRESHOLD = 0.45             # below this, the dataset answer is ignored
HISTORY_TURNS = 6                       # previous messages sent to the model

CHATS_DIR = Path("user_chats")
CHATS_DIR.mkdir(exist_ok=True)

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

# Optional, self-reported mood labels. Never scored, never inferred from text.
MOODS = {"calm", "okay", "anxious", "stressed", "low", "overwhelmed", "angry", "numb"}

# -----------------------
# Flask App
# -----------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-your-env-file")
app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_BYTES


# -----------------------
# Dataset & embeddings
# -----------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path(
    os.getenv("DATASET_PATH", str(BASE_DIR / "mentalhealthchatbot.txt"))
)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

patterns = []
pattern_intent = []

for i, intent in enumerate(data["intents"]):
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        pattern_intent.append(i)

pattern_embeddings = embedder.encode(patterns)

print("Mental health chatbot ready. AI responses and transcription run locally.")


# -----------------------
# Chat storage helpers
# -----------------------
UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")


def valid_chat_id(chat_id):
    return bool(chat_id) and bool(UUID_RE.match(chat_id))


def user_dir_for(user_id):
    safe = re.sub(r"[^A-Za-z0-9_-]", "", str(user_id))
    if not safe:
        raise ValueError("Invalid user id")
    d = CHATS_DIR / safe
    d.mkdir(exist_ok=True)
    return d


def chat_path(user_id, chat_id):
    if not valid_chat_id(chat_id):
        raise ValueError("Invalid chat id")
    return user_dir_for(user_id) / f"{chat_id}.json"


def create_chat_file(user_id, chat_id=None):
    chat_id = chat_id or str(uuid.uuid4())
    chat_data = {
        "id": chat_id,
        "created": datetime.now().isoformat(),
        "title": "New Chat",
        "messages": [],
    }
    with open(chat_path(user_id, chat_id), "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2)
    return chat_data


def save_exchange(user_id, chat_id, user_text, bot_text):
    """Append the user message and bot reply together, after a reply succeeded."""
    path = chat_path(user_id, chat_id)
    if not path.exists():
        create_chat_file(user_id, chat_id)
    with open(path, "r", encoding="utf-8") as f:
        chat_data = json.load(f)

    now = datetime.now().isoformat()
    chat_data["messages"].append({"sender": "user", "text": user_text, "timestamp": now})
    chat_data["messages"].append({"sender": "bot", "text": bot_text, "timestamp": now})

    if chat_data["title"] == "New Chat":
        chat_data["title"] = user_text[:50].rstrip() + ("..." if len(user_text) > 50 else "")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2)


def get_chat_history(user_id, chat_id):
    path = chat_path(user_id, chat_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recent_messages(user_id, chat_id):
    """Last few messages as chat-template roles, for conversational context."""
    try:
        chat = get_chat_history(user_id, chat_id)
    except Exception:
        return []
    if not chat:
        return []
    out = []
    for m in chat["messages"][-HISTORY_TURNS:]:
        out.append({"role": "user" if m["sender"] == "user" else "assistant", "content": m["text"]})
    return out


def list_user_chats(user_id):
    chats = []
    for chat_file in sorted(
        user_dir_for(user_id).glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
    ):
        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                c = json.load(f)
            chats.append({
                "id": c["id"],
                "title": c["title"],
                "created": c["created"],
                "message_count": len(c["messages"]),
            })
        except Exception:
            logger.exception("Skipping unreadable chat file %s", chat_file)
    return chats


def delete_chat(user_id, chat_id):
    path = chat_path(user_id, chat_id)
    if path.exists():
        path.unlink()
        return True
    return False

# -----------------------
# Core chat logic
# -----------------------
def load_chat_model():
    global chat_model, chat_tokenizer
    if chat_model is not None and chat_tokenizer is not None:
        return
    with chat_load_lock:
        if chat_model is None or chat_tokenizer is None:
            logger.info("Loading local chat model: %s", CHAT_MODEL_NAME)
            tok = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME)
            mdl = AutoModelForCausalLM.from_pretrained(
                CHAT_MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            mdl.eval()
            chat_tokenizer, chat_model = tok, mdl
            logger.info("Local chat model loaded.")


def dataset_answer_for(user_message):
    """Best matching dataset response, or '' if nothing is similar enough."""
    try:
        emb = embedder.encode([user_message], convert_to_numpy=True)
        sims = cosine_similarity(emb, pattern_embeddings)[0]
        best = int(np.argmax(sims))
        if sims[best] < SIMILARITY_THRESHOLD:
            return ""
        return random.choice(data["intents"][pattern_intent[best]]["responses"])
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


def generate_reply(user_message, history=None, mood=None):
    user_message = (user_message or "").strip()
    if not user_message:
        return "Please type a message so I can help."

    if CRISIS_PATTERNS.search(user_message):
        return CRISIS_REPLY

    needs_nudge = bool(ONGOING_PATTERNS.search(user_message))
    dataset_answer = dataset_answer_for(user_message)

    try:
        load_chat_model()

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

        # Build the prompt as text, then tokenize: works on every transformers version
        # (apply_chat_template(return_tensors="pt") returns a BatchEncoding on new ones).
        prompt = chat_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = chat_tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        with chat_gen_lock, torch.inference_mode():
            output = chat_model.generate(
                **inputs,
                max_new_tokens=140,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=chat_tokenizer.eos_token_id,
            )

        reply = chat_tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
        reply = trim_to_sentence(reply)
        if reply:
            return reply + (SUPPORT_NUDGE if needs_nudge else "")
        logger.warning("Model returned an empty reply; using fallback.")
    except Exception:
        logger.exception("Local model failed; using fallback reply")

    fallback = dataset_answer or FALLBACK_REPLY
    return fallback + (SUPPORT_NUDGE if needs_nudge else "")


def transcribe_audio(audio_bytes, mime_type="audio/webm"):
    global transcription_model
    if transcription_model is None:
        with transcription_model_lock:
            if transcription_model is None:
                transcription_model = WhisperModel(
                    TRANSCRIPTION_MODEL_NAME, device="cpu", compute_type="int8"
                )

    suffix = {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/mp4": ".mp4",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
    }.get(mime_type.split(";", 1)[0].lower(), ".audio")

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / f"recording{suffix}"
        audio_path.write_bytes(audio_bytes)
        segments, _ = transcription_model.transcribe(
            str(audio_path), beam_size=1, vad_filter=True
        )
        return " ".join(s.text.strip() for s in segments).strip()

# -----------------------
# Google OAuth
# -----------------------
@app.route("/api/auth/google")
def google_login():
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }
    return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}")


@app.route("/api/auth/google/callback", methods=["GET"])
def google_callback():
    code = request.args.get("code")
    error = request.args.get("error")

    if error or not code:
        logger.warning("OAuth failed: error=%s code_present=%s", error, bool(code))
        return redirect(url_for("home"))

    try:
        token_response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            timeout=15,
        )
        token_response.raise_for_status()
        token_json = token_response.json()
        if "access_token" not in token_json:
            logger.warning("No access token in Google response")
            return redirect(url_for("home"))

        info_response = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {token_json['access_token']}"},
            timeout=15,
        )
        info_response.raise_for_status()
        info = info_response.json()

        session["user"] = {
            "id": info.get("id"),
            "email": info.get("email"),
            "name": info.get("name"),
            "picture": info.get("picture"),
        }
        logger.info("User signed in: %s", info.get("name"))
    except Exception:
        logger.exception("OAuth callback error")

    return redirect(url_for("home"))


@app.route("/api/auth/status", methods=["GET"])
def auth_status():
    if "user" in session:
        return jsonify({"signed_in": True, "user": session["user"]})
    return jsonify({"signed_in": False})


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})

# -----------------------
# Chat history routes
# -----------------------
def require_user():
    return session.get("user")


@app.route("/api/chats", methods=["GET"])
def get_chats():
    user = require_user()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401
    return jsonify({"chats": list_user_chats(user["id"])})


@app.route("/api/chats/new", methods=["POST"])
def new_chat():
    user = require_user()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401
    return jsonify({"chat": create_chat_file(user["id"])})


@app.route("/api/chats/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    user = require_user()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401
    if not valid_chat_id(chat_id):
        return jsonify({"error": "Invalid chat ID"}), 400
    chat_data = get_chat_history(user["id"], chat_id)
    if not chat_data:
        return jsonify({"error": "Chat not found"}), 404
    return jsonify({"chat": chat_data})


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def delete_chat_route(chat_id):
    user = require_user()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401
    if not valid_chat_id(chat_id):
        return jsonify({"error": "Invalid chat ID"}), 400
    if delete_chat(user["id"], chat_id):
        return jsonify({"success": True})
    return jsonify({"error": "Chat not found"}), 404

# -----------------------
# Chat API
# -----------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "chat_model_loaded": chat_model is not None,
        "transcription_model_loaded": transcription_model is not None,
    })


@app.route("/")
def home():
    return render_template("index.html")


def error_text(exc):
    return str(exc) or exc.__class__.__name__


@app.route("/api/chat", methods=["POST"])
def chat():
    user = require_user()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401

    body = request.get_json(silent=True) or {}
    user_message = (body.get("message") or "").strip()
    chat_id = (body.get("chat_id") or "").strip()
    mood = (body.get("mood") or "").strip().lower()
    mood = mood if mood in MOODS else None

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400
    if not valid_chat_id(chat_id):
        return jsonify({"error": "A valid chat ID is required."}), 400

    try:
        history = recent_messages(user["id"], chat_id)
        reply_text = generate_reply(user_message, history, mood)
        save_exchange(user["id"], chat_id, user_message, reply_text)
        return jsonify({"reply": reply_text})
    except Exception as exc:
        logger.exception("Text chat request failed")
        return jsonify({"error": error_text(exc)}), 500


@app.route("/api/chat/voice", methods=["POST"])
def chat_voice():
    user = require_user()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401

    if "audio" not in request.files:
        return jsonify({"error": "No audio file was received."}), 400

    chat_id = request.form.get("chat_id", "").strip()
    if not valid_chat_id(chat_id):
        return jsonify({"error": "A valid chat ID is required."}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if not audio_bytes:
        return jsonify({"error": "The recording was empty. Please try again."}), 400
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        return jsonify({"error": "That voice message is too long. Please keep it under a minute."}), 400

    mime_type = audio_file.mimetype or "audio/webm"

    try:
        transcript = transcribe_audio(audio_bytes, mime_type)
        if not transcript:
            return jsonify({
                "error": "I couldn't make out that recording. Try again, or type your message instead."
            }), 422

        history = recent_messages(user["id"], chat_id)
        reply_text = generate_reply(transcript, history)
        save_exchange(user["id"], chat_id, transcript, reply_text)
        return jsonify({"transcript": transcript, "reply": reply_text})
    except Exception as exc:
        logger.exception("Voice chat request failed")
        return jsonify({"error": error_text(exc)}), 500

# -----------------------
# Run
# -----------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False,
        threaded=True
    )

    # Preload the chat model so the first message isn't slow.
    threading.Thread(target=load_chat_model, daemon=True).start()
    # use_reloader=False stops debug mode from loading every model twice.
    app.run(debug=True, use_reloader=False, port=5000, threaded=True)