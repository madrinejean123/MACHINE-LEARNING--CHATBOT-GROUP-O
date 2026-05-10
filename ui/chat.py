"""
ui/chat.py — conversation management, chat rendering and TTS output
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

# ✅ ADD DB
from db import save_message, load_messages


# ---------------------------------------------------------------------------
# Conversation helpers
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
# TTS helper (UNCHANGED)
# ---------------------------------------------------------------------------

def _speak(answer: str):
    try:
        import pyttsx3
        import tempfile

        clean_ans = re.sub(r"[•*#_]", "", answer)
        clean_ans = " ".join(clean_ans.split())[:800]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()

        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        engine.save_to_file(clean_ans, tmp.name)
        engine.runAndWait()
        engine.stop()

        size = os.path.getsize(tmp.name)

        if size > 1000:
            st.session_state.last_audio = tmp.name
            st.session_state.last_audio_size = size
            st.session_state.last_audio_err = None
        else:
            st.session_state.last_audio_err = f"wav too small ({size} bytes)"

    except Exception as e:
        st.session_state.last_audio_err = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# MAIN CHAT RENDERER (UPDATED WITH DB + SESSION)
# ---------------------------------------------------------------------------

def render_chat(df, embeddings, emb_model, session_id):

    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None
    if "last_audio_size" not in st.session_state:
        st.session_state.last_audio_size = 0
    if "last_audio_err" not in st.session_state:
        st.session_state.last_audio_err = None

    ensure_conversation()
    active_conv = get_active_conv()

    # ==========================================================
    # LOAD CHAT HISTORY FROM DB (NEW)
    # ==========================================================
    if active_conv:
        db_messages = load_messages(session_id, active_conv["id"])

        if db_messages and not active_conv["messages"]:
            for role, msg, _ in db_messages:
                active_conv["messages"].append({
                    "role": role,
                    "content": msg
                })

    # ==========================================================
    # WELCOME SCREEN
    # ==========================================================
    if active_conv and not active_conv["messages"]:
        st.markdown("""
        <div class="welcome-wrapper">
            <div class="welcome-box">
                <div class="welcome-icon">🛡️</div>
                <div class="welcome-title">How can I help you today?</div>
                <div class="welcome-text">
                    Ask me anything about Makerere University's safeguarding policies,
                    disability rights, sexual harassment procedures, and student protections.
                </div>
                <div class="welcome-hint">
                    🎙️ Speak your question instead of typing
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(len(SUGGESTIONS))
        for col, suggestion in zip(cols, SUGGESTIONS):
            with col:
                if st.button(suggestion, key=f"sug_{suggestion[:20]}"):
                    st.session_state.suggested_query = suggestion
                    st.rerun()

    # ==========================================================
    # CHAT HISTORY DISPLAY
    # ==========================================================
    if active_conv:
        for msg in active_conv["messages"]:
            with st.chat_message(msg["role"], avatar="🛡️" if msg["role"] == "assistant" else "🧑"):
                st.markdown(msg["content"], unsafe_allow_html=True)

    # ==========================================================
    # INPUT
    # ==========================================================
    user_input = st.chat_input("Type your question here…")

    if st.session_state.suggested_query and not user_input:
        user_input = st.session_state.suggested_query
        st.session_state.suggested_query = None

    if user_input and active_conv:
        _handle_user_input(user_input, active_conv, df, embeddings, emb_model, session_id)

    # ==========================================================
    # AUDIO OUTPUT
    # ==========================================================
    if st.session_state.get("last_audio"):
        st.audio(st.session_state.last_audio, format="audio/wav")
        st.session_state.last_audio = None


# ---------------------------------------------------------------------------
# HANDLE USER MESSAGE (UPDATED WITH DB SAVE)
# ---------------------------------------------------------------------------

def _handle_user_input(user_input, active_conv, df, embeddings, emb_model, session_id):

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    active_conv["messages"].append({"role": "user", "content": user_input})

    # SAVE USER MESSAGE
    save_message(
        session_id,
        active_conv["id"],
        "user",
        user_input,
        datetime.now().isoformat()
    )

    if active_conv["title"] == "New conversation":
        active_conv["title"] = user_input[:45]

    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("Searching policy documents..."):

            if is_greeting(user_input):
                answer = GREETING_RESPONSE
                sources = []
            else:
                retrieved = retrieve_top_k(user_input, emb_model, embeddings, df)
                raw = generate_answer(user_input, retrieved)
                answer = format_response(raw)
                sources = []

        st.markdown(answer)

        _speak(answer)

    # SAVE ASSISTANT MESSAGE
    save_message(
        session_id,
        active_conv["id"],
        "assistant",
        answer,
        datetime.now().isoformat()
    )

    active_conv["messages"].append({"role": "assistant", "content": answer})

    st.rerun()