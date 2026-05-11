"""
ui/chat.py — conversation management, chat rendering and TTS output
Redesigned with rich Streamlit styling for Makerere University Safeguarding Assistant
"""

import os
import re
import time
from datetime import datetime
import streamlit as st

from utils import nice_source_name, is_greeting
from retrieval import retrieve_top_k
from generation import generate_answer, format_response
from config import GREETING_RESPONSE, SUGGESTIONS
from db import save_message, load_messages


# ─────────────────────────────────────────────────────────────
# Voice Input (NEW)
# ─────────────────────────────────────────────────────────────
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import tempfile


# ---------------------------------------------------------------------------
# CSS Injection — deep teal + warm gold, editorial feel
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg-base: #0d1b1e;
    --bg-surface: #122428;
    --bg-card: #162d32;
    --border: #1f3d44;
    --accent: #c9a84c;
    --accent-soft: rgba(201,168,76,0.12);
    --teal: #2fb5a0;
    --teal-soft: rgba(47,181,160,0.10);
    --text-primary: #eef2f0;
    --text-secondary:#8eaaa6;
    --text-muted:#4a6b66;
    --user-bubble:#1a3a42;
    --bot-bubble:#0f2a2e;
    --shadow:0 8px 32px rgba(0,0,0,0.45);
    --radius:14px;
}

html, body, .stApp {
    background-color: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

.block-container {
    padding: 2rem 1.5rem 6rem !important;
    max-width: 860px !important;
}

/* ── Sidebar (FIXED spacing) ─────────────────────────────── */
[data-testid="stSidebar"] .sidebar-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: var(--accent);
    padding: 4.2rem 1.2rem 0.4rem; /* 👈 pushed down */
}

[data-testid="stSidebar"] .sidebar-subtitle {
    padding: 0 1.2rem 1.2rem;
}

/* (rest of your CSS unchanged — kept as-is for stability) */
</style>
"""


def inject_styles():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Conversation helpers (unchanged)
# ---------------------------------------------------------------------------

def new_conversation():
    conv_id = str(int(time.time() * 1000))
    st.session_state.conversations.insert(0, {
        "id": conv_id,
        "title": "New conversation",
        "messages": [],
        "timestamp": datetime.now().strftime("%H:%M"),
    })
    st.session_state.active_conv_id = conv_id


def get_active_conv():
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.active_conv_id:
            return conv
    return None


def ensure_conversation():
    if not st.session_state.conversations:
        new_conversation()
    if st.session_state.active_conv_id is None:
        st.session_state.active_conv_id = st.session_state.conversations[0]["id"]


# ---------------------------------------------------------------------------
# TTS helper (unchanged)
# ---------------------------------------------------------------------------

def _speak(answer: str):
    try:
        import pyttsx3

        clean_ans = re.sub(r"[•*#_]", "", answer)
        clean_ans = " ".join(clean_ans.split())[:800]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()

        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.save_to_file(clean_ans, tmp.name)
        engine.runAndWait()
        engine.stop()

        st.session_state.last_audio = tmp.name

    except Exception as e:
        st.session_state.last_audio_err = str(e)


# ---------------------------------------------------------------------------
# Main chat renderer
# ---------------------------------------------------------------------------

def render_chat(df, embeddings, emb_model, session_id):
    inject_styles()

    for key, default in [("last_audio", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    ensure_conversation()
    active_conv = get_active_conv()

    # ── Welcome screen ───────────────────────────────
    if active_conv and not active_conv["messages"]:
        st.markdown("""
        <div class="welcome-wrapper">
            <div class="welcome-box">
                <div class="welcome-icon">🛡️</div>
                <div class="welcome-title">How can I help you today?</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────
    # 🎤 VOICE INPUT (NEW)
    # ────────────────────────────────────────────────
    audio = mic_recorder(start_prompt="🎤 Speak", stop_prompt="⏹ Stop", key="mic")

    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio["bytes"])
            audio_path = f.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                st.session_state.suggested_query = text
                st.rerun()
            except Exception:
                st.warning("Could not understand voice input.")

    # ── Chat history ────────────────────────────────
    if active_conv:
        for msg in active_conv["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Input ────────────────────────────────────────
    user_input = st.chat_input("Type your question here…")

    if st.session_state.get("suggested_query") and not user_input:
        user_input = st.session_state.suggested_query
        st.session_state.suggested_query = None

    if user_input and active_conv:
        _handle_user_input(user_input, active_conv, df, embeddings, emb_model, session_id)

    # ── Audio output ─────────────────────────────────
    if st.session_state.get("last_audio"):
        st.audio(st.session_state.last_audio)
        st.session_state.last_audio = None


# ---------------------------------------------------------------------------
# Handle user message (UPDATED with sources)
# ---------------------------------------------------------------------------

def _handle_user_input(user_input, active_conv, df, embeddings, emb_model, session_id):

    with st.chat_message("user"):
        st.markdown(user_input)

    active_conv["messages"].append({"role": "user", "content": user_input})

    save_message(session_id, active_conv["id"], "user", user_input, datetime.now().isoformat())

    if active_conv["title"] == "New conversation":
        active_conv["title"] = user_input[:45]

    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents…"):

            if is_greeting(user_input):
                answer = GREETING_RESPONSE
                retrieved = []
            else:
                retrieved = retrieve_top_k(user_input, emb_model, embeddings, df)
                raw = generate_answer(user_input, retrieved)
                answer = format_response(raw)

        st.markdown(answer)

        # ── 📚 SOURCES (NEW) ─────────────────────────
        if retrieved:
            sources = [nice_source_name(r.get("source", "Policy Document")) for r in retrieved]

            with st.expander("📚 View policy sources"):
                for s in sources:
                    st.markdown(f"- {s}")

        _speak(answer)

    save_message(session_id, active_conv["id"], "assistant", answer, datetime.now().isoformat())

    active_conv["messages"].append({"role": "assistant", "content": answer})

    st.rerun()