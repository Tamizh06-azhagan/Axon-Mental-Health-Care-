import base64
import html as html_lib
import json
import logging
import os
import random
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(
    page_title="Axon",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" if st.user.is_logged_in else "collapsed",
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
COMPOSER_DIR = BASE_DIR / "composer_component"

# Custom bottom bar (text + voice recorder), built from composer_component/index.html
_composer = components.declare_component("axon_composer", path=str(COMPOSER_DIR))

CHAT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRANSCRIPTION_MODEL_NAME = "tiny"
MAX_AUDIO_BYTES = 5 * 1024 * 1024
SIMILARITY_THRESHOLD = 0.45   # below this, the dataset answer is ignored
HISTORY_TURNS = 6             # previous messages sent to the model
UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")
AUDIO_SUFFIX = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
}

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

# Optional, self-reported moods: label shown in the UI -> value sent to the model
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
MOODS = list(MOOD_CHOICES.values())
HELP_CHOICES = ["Talking it through", "A calming idea", "Just venting", "A small next step"]

LOGO_SVG = (
    '<svg width="26" height="26" viewBox="0 0 256 256" fill="currentColor">'
    '<path d="M256 256H128L0 128h128z"/><path d="M256 128H128L0 0h128z"/></svg>'
)
SKELETON_HTML = (
    '<div class="row bot"><div class="bub skel" aria-label="Axon is replying">'
    '<span class="l1"></span><span class="l2"></span><span class="l3"></span></div></div>'
)


# -----------------------
# Themes (taken from index.html)
# -----------------------
def make_theme(label, ink, bg, card, accent, side, g1, g2, btn, orb1, orb2, dark=False):
    return dict(
        label=label, ink=ink, bg=bg, card=card, accent=accent, side=side,
        g1=g1, g2=g2, btn=btn, orb1=orb1, orb2=orb2, dark=dark,
    )


THEMES = {
    "lavender": make_theme("Lavender", "#1B133C", "#EFEBF7", "#FFFFFF", "#6D5BD0", "#E6E0F3", "#F6F3FB", "#EFEBF7", "#1B133C", "#C9BEF0", "#8E7BE0"),
    "ocean": make_theme("Calm Ocean", "#0F2A3D", "#E6F1F6", "#FFFFFF", "#2E86AB", "#D6E8F0", "#F1F8FB", "#E6F1F6", "#0F4466", "#BFE3F0", "#4FA3C7"),
    "sage": make_theme("Sage", "#1F3326", "#EAF1EA", "#FFFFFF", "#5E9473", "#DCE8DC", "#F4F8F3", "#EAF1EA", "#2F5A40", "#CBE5CF", "#6FA982"),
    "mint": make_theme("Mint Breeze", "#123A36", "#E3F5F1", "#FFFFFF", "#2BA58F", "#D1EDE6", "#EFFAF7", "#E3F5F1", "#0F5C52", "#C2F0E4", "#45B8A2"),
    "rose": make_theme("Soft Rose", "#3B1F2B", "#F8EAF0", "#FFFFFF", "#D26A93", "#F0DAE4", "#FDF4F7", "#F8EAF0", "#7A2E52", "#F9CFE0", "#E07BA3"),
    "sand": make_theme("Warm Sand", "#3A2A1E", "#F5EDE3", "#FFFFFF", "#C08A5B", "#EBDFCE", "#FBF6EE", "#F5EDE3", "#6B4A33", "#F1D9BC", "#D49A62"),
    "sunrise": make_theme("Sunrise", "#3D2314", "#FDEEE0", "#FFFFFF", "#E8833A", "#F8DFCB", "#FFF7EE", "#FDEEE0", "#8A3F12", "#FFD9B0", "#F0924A"),
    "midnight": make_theme("Midnight", "#E8E6F5", "#14112B", "#1F1B3D", "#8F7DF0", "#1A1636", "#1A1636", "#12102A", "#6D5BD0", "#A99CF5", "#5B49C8", True),
    "ember": make_theme("Night Ember", "#F1E9EA", "#14101A", "#1E1823", "#E04A4A", "#181320", "#1B1520", "#120E17", "#B72F36", "#F08A8A", "#B72F36", True),
    "deepsea": make_theme("Deep Sea", "#DCEFF5", "#0B1E2B", "#12303F", "#4FB3D9", "#0E2736", "#0F2937", "#0A1B27", "#1F7FA6", "#8ED8F0", "#2A86AD", True),
    "forest": make_theme("Night Forest", "#E3F0E4", "#0F1E17", "#172B21", "#6CC48A", "#12241B", "#13261D", "#0D1A13", "#2E8B57", "#A6E3B8", "#3FA870", True),
    "graphite": make_theme("Graphite", "#ECECEF", "#17181B", "#222428", "#9AA3B5", "#1C1D21", "#1E2024", "#141518", "#4A5468", "#C9CFDB", "#6B7589", True),
}
THEME_GROUPS = [
    ("Calming", ["lavender", "ocean", "sage", "mint", "rose", "sand", "sunrise"]),
    ("Dark", ["midnight", "ember", "deepsea", "forest", "graphite"]),
]

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500"
    "&family=Inter:wght@400;500;600&display=swap');\n"
)

STATIC_CSS = """
html, body, .stApp { background: var(--bg) !important; }
.stApp { color: var(--ink); font-family: 'Inter', system-ui, sans-serif; }
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg, var(--g1), var(--g2)) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stDecoration"], [data-testid="stToolbar"], [data-testid="stStatusWidget"],
#MainMenu, footer { display: none !important; }
.block-container { max-width: 760px; padding: 1.2rem 1rem 0.2rem; }
.st-key-composer-bar { position: sticky; bottom: 0; z-index: 30; padding: 8px 0 0;
  background: linear-gradient(180deg, transparent 0%, var(--g2) 30%); }
.st-key-composer-bar iframe { display: block; }

/* headings and text */
h1, h2, h3, .serif { font-family: 'Fraunces', Georgia, serif !important; font-weight: 400 !important; color: var(--ink) !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--ink); }
[data-testid="stCaptionContainer"] p { color: var(--ink2) !important; }
[data-testid="stWidgetLabel"] p { color: var(--ink2) !important; font-size: 13px !important; }
[data-testid="stImage"] img { border-radius: 50%; width: 32px; height: 32px; object-fit: cover; }
[data-testid="stDivider"], hr { border-color: var(--line) !important; }

/* buttons: text colour is set on every kind so labels never disappear */
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: var(--btn) !important; color: #FFFFFF !important; border: 0 !important;
  border-radius: 14px !important; font-weight: 500 !important;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
  background: var(--btn2) !important; color: #FFFFFF !important;
}
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
  background: var(--card) !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: 14px !important;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--accent) !important; color: var(--ink) !important;
}
button[kind="tertiary"], button[data-testid="stBaseButton-tertiary"] {
  background: transparent !important; color: var(--ink) !important; border: 0 !important; border-radius: 12px !important;
}
button[kind="tertiary"]:hover, button[data-testid="stBaseButton-tertiary"]:hover {
  background: var(--hover) !important; color: var(--ink) !important;
}
button[kind] p, button[data-testid^="stBaseButton"] p { color: inherit !important; }

/* pills (mood check-in and heaviness scale): light pill, accent when selected */
[data-testid="stPills"] button, button[kind="pills"], button[kind="pillsActive"],
button[data-testid="stBaseButton-pills"], button[data-testid="stBaseButton-pillsActive"] {
  background: var(--card) !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: 999px !important;
}
[data-testid="stPills"] button p, button[kind="pills"] p, button[kind="pillsActive"] p,
button[data-testid="stBaseButton-pills"] p, button[data-testid="stBaseButton-pillsActive"] p {
  color: var(--ink) !important;
}
[data-testid="stPills"] button[aria-checked="true"], button[kind="pillsActive"],
button[data-testid="stBaseButton-pillsActive"] {
  background: color-mix(in srgb, var(--accent) 20%, var(--card)) !important;
  border-color: var(--accent) !important;
}

/* popover (account menu) panel */
[data-testid="stPopover"] button {
  background: var(--card) !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: 999px !important; font-weight: 500 !important;
}
[data-testid="stPopover"] button p { color: var(--ink) !important; }
[data-testid="stPopoverBody"], div[data-baseweb="popover"] > div {
  background: var(--card) !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: 16px !important;
  box-shadow: 0 16px 40px -12px rgba(27,19,60,.25) !important;
}

/* bordered cards (check-in) */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card); border-radius: 22px !important; border-color: var(--line) !important;
}

/* composer: sticky at the bottom, fades into the page background */


/* chat bubbles */
.row { display: flex; width: 100%; margin: 12px 0; }
.row.user { justify-content: flex-end; }
.bub { max-width: 82%; padding: 12px 16px; border-radius: 18px; line-height: 1.55; font-size: 15px;
  overflow-wrap: anywhere; animation: rise .35s cubic-bezier(.2,.8,.2,1); }
.user .bub { background: var(--btn); color: #FFFFFF; border-bottom-right-radius: 6px;
  box-shadow: 0 10px 24px -14px color-mix(in srgb, var(--btn) 70%, transparent); }
.bot .bub { background: var(--card); color: var(--ink); border: 1px solid var(--line);
  border-bottom-left-radius: 6px; max-width: 86%; box-shadow: 0 10px 28px -18px rgba(27,19,60,.35); }
.bot .bub.err { background: color-mix(in srgb, #ea580c 14%, var(--card)); border-color: color-mix(in srgb, #ea580c 35%, var(--card)); }
@keyframes rise { from { opacity: 0; transform: translateY(10px) scale(.98); } to { opacity: 1; transform: none; } }

/* skeleton bubble shown while Axon is replying (replaces the three dots) */
.skel { min-width: 230px; display: flex; flex-direction: column; gap: 9px; padding: 16px 18px; }
.skel span { display: block; height: 10px; border-radius: 999px;
  background: linear-gradient(90deg, color-mix(in srgb, var(--ink) 8%, transparent) 25%,
    color-mix(in srgb, var(--accent) 32%, transparent) 50%, color-mix(in srgb, var(--ink) 8%, transparent) 75%);
  background-size: 300% 100%; animation: shimmer 1.4s linear infinite; }
.skel .l1 { width: 92%; }
.skel .l2 { width: 74%; }
.skel .l3 { width: 46%; }
@keyframes shimmer { from { background-position: 100% 0; } to { background-position: -100% 0; } }

/* voice bubble (sent voice messages) */
.voice { display: flex; align-items: center; gap: 10px; min-width: 190px; }
.mic-dot { width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center;
  background: rgba(255,255,255,.2); color: #FFFFFF; flex: none; }
.wave { display: flex; align-items: center; gap: 2px; height: 22px; flex: 1; }
.wave span { width: 3px; border-radius: 9px; background: rgba(255,255,255,.6); }
.dur { font-size: 12px; opacity: .8; }
.cap { display: block; margin-top: 8px; font-size: 13px; opacity: .85; white-space: pre-wrap; }

/* empty state: breathing orb */
.empty { text-align: center; max-width: 520px; margin: 9vh auto 0; }
.orb { width: 120px; height: 120px; margin: 0 auto 28px; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #fff 0, var(--orb1) 38%, var(--orb2) 100%);
  box-shadow: 0 20px 60px -10px color-mix(in srgb, var(--accent) 50%, transparent);
  animation: breathe 8s ease-in-out infinite; }
@keyframes breathe { 0%, 100% { transform: scale(.88); } 50% { transform: scale(1.06); } }
.empty h2 { font-size: 34px !important; line-height: 1.15; margin-bottom: 12px !important; }
.empty p { color: var(--ink2); font-size: 15px; }

/* header title */
.title { font-family: 'Fraunces', Georgia, serif; font-size: 17px; color: var(--ink);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--side) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebarUserContent"] { padding: 1.2rem 1rem 1rem; }
[data-testid="stSidebar"] button { justify-content: flex-start !important; text-align: left !important; }
[data-testid="stSidebar"] button > div { justify-content: flex-start !important; width: 100%; }
[data-testid="stSidebar"] button p { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
[data-testid="stSidebar"] button[kind="primary"] { justify-content: center !important; }
[data-testid="stSidebar"] button[kind="primary"] > div { justify-content: center !important; }
.brand { display: flex; align-items: center; gap: 10px; font-family: 'Fraunces', serif; font-size: 22px; padding: 4px 4px 16px; color: var(--ink); }
.side-h { font-size: 12px; color: var(--ink3); padding: 14px 4px 6px; }
.side-foot { font-size: 12px; color: var(--ink3); margin-top: 20px; padding-top: 12px; border-top: 1px solid var(--line); line-height: 1.5; }
[class*="st-key-del-"] button { opacity: 0; transition: opacity .15s; }
div[data-testid="stHorizontalBlock"]:hover [class*="st-key-del-"] button { opacity: 1; }
[class*="st-key-del-"] button, [class*="st-key-del-"] button p { color: var(--danger) !important; }
[class*="st-key-del-"] button:hover { background: rgba(194,65,12,.1) !important; }

/* sign-in text */
.gate-h { text-align: center; font-size: 30px !important; margin: 10px 0 6px !important; }
.gate-p { text-align: center; color: var(--ink2); font-size: 14px; margin-bottom: 18px; }

/* phones */
@media (max-width: 640px) {
  .block-container { padding: .8rem .7rem .6rem; }
  .bub { max-width: 90%; }
  .empty { margin-top: 6vh; }
  .empty h2 { font-size: 28px !important; }
  .orb { width: 96px; height: 96px; margin-bottom: 22px; }
  .block-container { padding: .8rem .7rem .2rem; }
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""

# Sign-in page: drifting colour orbs, frosted glass card, Google button
GATE_CSS = """
/* sign-in: soft drifting colour orbs behind a frosted glass card */
.gate-orbs { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
.gate-orbs span { position: absolute; border-radius: 50%; filter: blur(70px);
  animation: drift 16s ease-in-out infinite alternate; }
.gate-orbs .o1 { width: 52vw; height: 52vw; left: -12vw; top: -14vw; background: var(--orb1); opacity: .95; }
.gate-orbs .o2 { width: 46vw; height: 46vw; right: -14vw; bottom: -16vw; background: var(--orb2); opacity: .7; animation-delay: -6s; }
.gate-orbs .o3 { width: 28vw; height: 28vw; left: 46vw; top: 40vh; background: var(--accent); opacity: .35; animation-delay: -11s; }
@keyframes drift {
  from { transform: translate(0, 0) scale(1); }
  to { transform: translate(4vw, 3vh) scale(1.08); }
}

/* frosted glass card */
.st-key-gate-card {
  position: relative; z-index: 1;
  max-width: 440px; margin: 0 auto;
  background: color-mix(in srgb, var(--card) 62%, transparent) !important;
  backdrop-filter: blur(24px) saturate(170%);
  -webkit-backdrop-filter: blur(24px) saturate(170%);
  border: 1px solid color-mix(in srgb, #FFFFFF 60%, transparent) !important;
  border-radius: 26px !important;
  box-shadow: 0 40px 80px -30px rgba(27, 19, 60, .45), inset 0 1px 0 rgba(255, 255, 255, .7) !important;
  padding: 2.6rem 2.4rem 2.2rem !important;
}

/* logo sits in a soft glowing orb */
.gate-logo {
  width: 68px; height: 68px; margin: 0 auto 10px; border-radius: 50%;
  display: grid; place-items: center; color: var(--ink);
  background: radial-gradient(circle at 35% 30%, #FFFFFF 0, var(--orb1) 45%, var(--orb2) 100%);
  box-shadow: 0 14px 34px -12px color-mix(in srgb, var(--accent) 60%, transparent);
}
.gate-logo svg { width: 30px; height: 30px; }
.gate-h { font-size: 28px !important; line-height: 1.2 !important; }

/* sign-in button: pill shape, real Google G in a white badge */
.st-key-gate-login button[kind="primary"] {
  display: flex; align-items: center; justify-content: center; gap: 10px;
  height: 52px; width: 100%; border-radius: 999px !important; font-size: 15px !important;
  box-shadow: 0 12px 28px -12px color-mix(in srgb, var(--btn) 80%, transparent);
  transition: transform .2s, box-shadow .2s;
}
.st-key-gate-login button[kind="primary"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 18px 36px -12px color-mix(in srgb, var(--btn) 85%, transparent);
}
.st-key-gate-login button[kind="primary"]::before {
  content: ""; flex: none; width: 24px; height: 24px; border-radius: 50%;
  background-color: #FFFFFF;
  background-image: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'><path fill='%23EA4335' d='M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z'/><path fill='%234285F4' d='M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z'/><path fill='%23FBBC05' d='M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z'/><path fill='%2334A853' d='M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z'/></svg>");
  background-repeat: no-repeat; background-position: center; background-size: 14px 14px;
}
"""


def theme_css(theme_id):
    t = THEMES[theme_id]
    root = f""":root {{
  --ink: {t['ink']}; --bg: {t['bg']}; --card: {t['card']}; --accent: {t['accent']};
  --side: {t['side']}; --g1: {t['g1']}; --g2: {t['g2']}; --btn: {t['btn']};
  --orb1: {t['orb1']}; --orb2: {t['orb2']}; --danger: #C2410C;
  --btn2: color-mix(in srgb, {t['btn']} 82%, white);
  --ink2: color-mix(in srgb, {t['ink']} 62%, transparent);
  --ink3: color-mix(in srgb, {t['ink']} 40%, transparent);
  --line: color-mix(in srgb, {t['ink']} 10%, transparent);
  --hover: color-mix(in srgb, {t['ink']} 7%, transparent);
  color-scheme: {'dark' if t['dark'] else 'light'};
}}
"""
    # Colour swatch in every theme button, and a highlight on the active theme
    swatches = ""
    for tid, th in THEMES.items():
        swatches += (
            f".st-key-th-{tid} button {{ justify-content: flex-start !important; gap: 8px; }}"
            f".st-key-th-{tid} button::before {{ content: ''; display: inline-block; flex: none; width: 16px; height: 16px;"
            f" border-radius: 50%; border: 1px solid var(--line);"
            f" background: linear-gradient(135deg, {th['bg']} 50%, {th['accent']} 50%); }}"
        )
    active = (
        f".st-key-th-{theme_id} button {{ border-color: var(--accent) !important;"
        f" background: color-mix(in srgb, var(--accent) 14%, var(--card)) !important; }}"
    )
    return f"<style>{FONT_IMPORT}{root}{STATIC_CSS}{GATE_CSS}{swatches}{active}</style>"


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


def delete_chat_file(user_id, chat_id):
    chat_path(user_id, chat_id).unlink(missing_ok=True)


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


def transcribe_audio(audio_bytes, mime="audio/webm"):
    """Turn a recording into text. Returns '' if it can't be understood."""
    try:
        model = load_transcriber()
        suffix = AUDIO_SUFFIX.get((mime or "").split(";")[0].strip().lower(), ".webm")
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / f"recording{suffix}"
            audio_path.write_bytes(audio_bytes)
            with get_compute_lock():
                segments, _ = model.transcribe(str(audio_path), beam_size=1, vad_filter=True)
                return " ".join(s.text.strip() for s in segments).strip()
    except Exception:
        logger.exception("Transcription failed")
        return ""


# -----------------------
# UI helpers
# -----------------------
def esc(text):
    """Escape text for raw HTML. Newlines become line breaks."""
    return html_lib.escape(text or "").replace("\n", "<br>")


def fmt_time(secs):
    return f"{secs // 60}:{secs % 60:02d}"


def bot_bubble_html(text):
    return f'<div class="row bot"><div class="bub">{esc(text)}</div></div>'


def bubble(text, who):
    st.markdown(
        f'<div class="row {who}"><div class="bub">{esc(text)}</div></div>',
        unsafe_allow_html=True,
    )


def voice_bubble(transcript, secs):
    bars = "".join(f'<span style="height:{4 + (i * 7) % 16}px"></span>' for i in range(20))
    cap = f'<span class="cap">{esc(transcript)}</span>' if transcript else ""
    st.markdown(
        f'<div class="row user"><div class="bub"><div class="voice">'
        f'<span class="mic-dot">{MIC_SVG}</span><div class="wave">{bars}</div>'
        f'<span class="dur">{fmt_time(secs)}</span></div>{cap}</div></div>',
        unsafe_allow_html=True,
    )


MIC_SVG = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2.2" stroke-linecap="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>'
    '<path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4"/></svg>'
)


def reveal_reply(reply, slot):
    """Show the reply word by word, like someone typing it out."""
    words = reply.split(" ")
    step = 2
    delay = min(0.07, 2.2 / max(1, len(words) / step))
    for end in range(step, len(words) + step, step):
        slot.markdown(bot_bubble_html(" ".join(words[:end])), unsafe_allow_html=True)
        time.sleep(delay)


def render_empty_state():
    st.markdown(
        '<div class="empty"><div class="orb"></div>'
        '<h2 class="serif">Hi, I\'m Axon. I\'m glad you\'re here.</h2>'
        "<p>This is a space to say what's on your mind, at your own pace.</p></div>",
        unsafe_allow_html=True,
    )


def render_checkin():
    v = st.session_state.ui_v
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
            key=f"ci_mood_{v}",
        )
        st.pills(
            "How heavy does it feel? (1 = light, 5 = very heavy)",
            ["1", "2", "3", "4", "5"],
            selection_mode="single",
            key=f"ci_heavy_{v}",
        )
        st.pills(
            "What would help most? (pick any)",
            HELP_CHOICES,
            selection_mode="multi",
            key=f"ci_help_{v}",
        )
        if st.session_state.ci_hint:
            st.caption("Pick at least one option, or press Not now.")
        a, b, _ = st.columns([2, 2, 3])
        a.button("Share with Axon", type="primary", on_click=share_checkin, use_container_width=True)
        b.button("Not now", on_click=close_checkin, use_container_width=True)


# -----------------------
# Callbacks (run before the script on each rerun)
# -----------------------
def queue_message(text, mood=None, voice=None):
    st.session_state.pending = (text, mood, voice)


def read_incoming(value):
    """The composer returns the same value on every rerun. Act on each message once."""
    if not isinstance(value, dict):
        return None
    message_id = value.get("id")
    if not message_id or message_id == st.session_state.get("last_composer_id"):
        return None
    st.session_state.last_composer_id = message_id
    return value


def show_checkin():
    st.session_state.checkin_open = True
    st.session_state.ci_hint = False
    st.session_state.ui_v += 1


def close_checkin():
    st.session_state.checkin_open = False
    st.session_state.ci_hint = False
    st.session_state.ui_v += 1


def share_checkin():
    v = st.session_state.ui_v
    mood_label = st.session_state.get(f"ci_mood_{v}")
    heavy = st.session_state.get(f"ci_heavy_{v}")
    helps = st.session_state.get(f"ci_help_{v}") or []

    parts = []
    if mood_label:
        parts.append(f"I'm feeling {MOOD_CHOICES[mood_label]}.")
    if heavy:
        parts.append(f"It feels about {heavy} out of 5 in heaviness (my own rating).")
    if helps:
        parts.append(f"What might help: {', '.join(h.lower() for h in helps)}.")
    if not parts:
        st.session_state.ci_hint = True
        return

    mood_key = MOOD_CHOICES[mood_label] if mood_label else None
    st.session_state.checkin_open = False
    st.session_state.ci_hint = False
    st.session_state.ui_v += 1
    queue_message("Quick check-in: " + " ".join(parts), mood_key)


def start_new_chat(user_id):
    st.session_state.chat_id = create_chat(user_id)["id"]
    st.session_state.checkin_open = False
    st.session_state.ui_v += 1


def open_chat(chat_id):
    st.session_state.chat_id = chat_id
    st.session_state.checkin_open = False
    st.session_state.ui_v += 1


def open_delete(chat_id):
    st.session_state.delete_target = chat_id
    st.session_state.delete_open = True


def set_theme(theme_id):
    st.session_state.theme = theme_id
    st.query_params["theme"] = theme_id  # keeps the choice across browser refreshes


def init_state():
    if "theme" not in st.session_state:
        requested = st.query_params.get("theme", "lavender")
        st.session_state.theme = requested if requested in THEMES else "lavender"
    st.session_state.setdefault("ui_v", 0)
    st.session_state.setdefault("checkin_open", False)
    st.session_state.setdefault("ci_hint", False)
    st.session_state.setdefault("checkin_offered_for", None)


@st.dialog("Delete this chat?")
def confirm_delete(user_id, chat_id):
    st.write("This conversation will be removed. This can't be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", use_container_width=True):
        st.rerun()
    if c2.button("Delete", type="primary", use_container_width=True):
        delete_chat_file(user_id, chat_id)
        if st.session_state.get("chat_id") == chat_id:
            st.session_state.pop("chat_id", None)
        st.rerun()


# -----------------------
# Page
# -----------------------
init_state()
st.markdown(theme_css(st.session_state.theme), unsafe_allow_html=True)

# Sign-in card
if not st.user.is_logged_in:
    st.markdown(
        '<div class="gate-orbs" aria-hidden="true">'
        '<span class="o1"></span><span class="o2"></span><span class="o3"></span></div>'
        '<div style="height:8vh"></div>',
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        with st.container(border=True, key="gate-card"):
            st.markdown(
                f'<div class="gate-logo">{LOGO_SVG}</div>'
                '<h2 class="serif gate-h" style="text-align:center;margin:0 0 6px">A quiet place to talk</h2>'
                '<p class="gate-p">Sign in to save your conversations and pick up where you left off.</p>',
                unsafe_allow_html=True,
            )
            st.button(
                "Continue with Google",
                type="primary",
                key="gate-login",
                use_container_width=True,
                on_click=st.login,
            )
            st.caption("Axon offers support, not diagnosis or emergency care.")
    st.stop()

# Signed-in app
user_id = st.user.sub
kb = load_knowledge_base()
chat_id = ensure_active_chat(user_id)

# Header: title, quick check-in, avatar, account menu
first_name = (st.user.get("name") or "You").split(" ")[0]
head_title, head_checkin, head_avatar, head_account = st.columns(
    [6, 2, 1, 2], vertical_alignment="center"
)
title_slot = head_title.empty()
with head_checkin:
    st.button("Quick check-in", on_click=show_checkin, use_container_width=True)
with head_avatar:
    if st.user.get("picture"):
        st.image(st.user.get("picture"), width=32)
with head_account:
    with st.popover(first_name):
        st.markdown(f"**{st.user.get('name') or 'Signed in'}**")
        if st.user.get("email"):
            st.caption(st.user.get("email"))
        st.caption("Theme")
        for group, ids in THEME_GROUPS:
            st.caption(group)
            cols = st.columns(2)
            for i, tid in enumerate(ids):
                cols[i % 2].button(
                    THEMES[tid]["label"],
                    key=f"th-{tid}",
                    on_click=set_theme,
                    args=(tid,),
                    use_container_width=True,
                )
        st.divider()
        st.button("Sign out", on_click=st.logout, use_container_width=True)

st.divider()

# Slots, created in display order. The composer sits at the bottom and is read first.
history_slot = st.container()
checkin_slot = st.container()
composer_slot = st.container(key="composer-bar")

pending = st.session_state.pop("pending", None)
notice = None

with composer_slot:
    composer_value = _composer(
        theme=THEMES[st.session_state.theme],
        placeholder="Write what's on your mind",
        key="composer-input",
        default=None,
    )

incoming = read_incoming(composer_value)
if incoming:
    if incoming.get("kind") == "text":
        typed_text = (incoming.get("text") or "").strip()
        if typed_text:
            pending = (typed_text, None, None)
    elif incoming.get("kind") == "audio":
        try:
            raw_audio = base64.b64decode(incoming.get("audio") or "")
        except Exception:
            raw_audio = b""
        if not raw_audio:
            notice = "The recording was empty. Please try again."
        elif len(raw_audio) > MAX_AUDIO_BYTES:
            notice = "That voice message is too long. Please keep it under a minute."
        else:
            pending = (None, None, (
                raw_audio,
                int(incoming.get("secs") or 0),
                incoming.get("mime") or "audio/webm",
            ))

chat = load_chat(user_id, chat_id)

with history_slot:
    if notice:
        st.warning(notice)

    if not chat["messages"] and not pending:
        render_empty_state()

    for m in chat["messages"]:
        bubble(m["text"], "user" if m["sender"] == "user" else "bot")

    if pending:
        text, mood, voice = pending
        if voice:
            audio_bytes, secs, mime = voice
            with st.spinner("Listening..."):
                text = transcribe_audio(audio_bytes, mime)
            if text:
                voice_bubble(text, secs)
                _, audio_col = st.columns([2, 3])
                audio_col.audio(audio_bytes, format=mime)
            else:
                st.warning("I couldn't make out that recording. Try again, or type your message instead.")
                pending = None
        else:
            bubble(text, "user")

    if pending:
        reply_slot = st.empty()
        reply_slot.markdown(SKELETON_HTML, unsafe_allow_html=True)
        reply = generate_reply(text, recent_messages(user_id, chat_id), mood, kb)
        reveal_reply(reply, reply_slot)

        try:
            save_exchange(user_id, chat_id, text, reply)
            chat = load_chat(user_id, chat_id)
        except Exception:
            logger.exception("Could not save the chat exchange")
            st.error("I couldn't save this message. Please try again.")

# Title
title = chat["title"] if chat["messages"] else "New chat"
title_slot.markdown(f'<div class="title">{esc(title)}</div>', unsafe_allow_html=True)

# Check-in card: offered once, after the third message
user_count = sum(1 for m in chat["messages"] if m["sender"] == "user")
if pending and user_count == 3 and st.session_state.checkin_offered_for != chat_id:
    st.session_state.checkin_offered_for = chat_id
    show_checkin()

with checkin_slot:
    if st.session_state.checkin_open:
        render_checkin()

# Sidebar, rendered last so it shows the newest titles
with st.sidebar:
    st.markdown(f'<div class="brand">{LOGO_SVG}<span>Axon</span></div>', unsafe_allow_html=True)
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
        open_col, del_col = st.columns([6, 1], vertical_alignment="center", gap="small")
        open_col.button(
            c["title"],
            key=f"open-{c['id']}",
            on_click=open_chat,
            args=(c["id"],),
            type="secondary" if c["id"] == chat_id else "tertiary",
            use_container_width=True,
        )
        del_col.button("✕", key=f"del-{c['id']}", on_click=open_delete, args=(c["id"],), type="tertiary")

    st.markdown(
        '<div class="side-foot">If you are in immediate danger, call your local emergency number.</div>',
        unsafe_allow_html=True,
    )

# Delete confirmation (replaces the browser confirm() dialog)
if st.session_state.pop("delete_open", False):
    confirm_delete(user_id, st.session_state.get("delete_target"))