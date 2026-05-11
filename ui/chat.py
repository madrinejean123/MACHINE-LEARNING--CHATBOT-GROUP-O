"""
ui/chat.py — ChatGPT-style chat + REAL voice input (Streamlit + Whisper)
Hugging Face safe version
"""

import os
import time
import re
from datetime import datetime
import streamlit as st

from utils import is_greeting
from retrieval import retrieve_top_k
from generation import generate_answer, format_response
from config import GREETING_RESPONSE, SUGGESTIONS
from db import save_message, load_messages


# ==========================================================
# OPTIONAL MIC (NEW)
# ==========================================================
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except:
    MIC_AVAILABLE = False


# ==========================================================
# CONVERSATION MANAGEMENT
# ==========================================================

def new_conversation():
    cid = str(int(time.time() * 1000))

    st.session_state.conversations.insert(0, {
        "id": cid,
        "title": "New chat",
        "messages": [],
        "timestamp": datetime.now().strftime("%H:%M"),
    })

    st.session_state.active_conv_id = cid


def get_active():
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.active_conv_id:
            return c
    return None


def ensure():
    if not st.session_state.conversations:
        new_conversation()
    if not st.session_state.active_conv_id:
        st.session_state.active_conv_id = st.session_state.conversations[0]["id"]


# ==========================================================
# WHISPER TRANSCRIPTION (REAL VOICE INPUT)
# ==========================================================

def transcribe_audio(audio_bytes):
    try:
        import whisper
        import tempfile

        model = whisper.load_model("base")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.close()

        result = model.transcribe(tmp.name)
        return result["text"].strip()

    except Exception as e:
        st.error(f"Voice error: {e}")
        return ""


# ==========================================================
# CHAT UI (ChatGPT STYLE)
# ==========================================================

def render_welcome():
    st.markdown("""
    <div style="text-align:center;padding:40px;">
        <div style="font-size:48px;">🛡️</div>
        <h2>Safeguarding Companion</h2>
        <p style="color:#aaa;">
            Ask about university policies, rights, and safety procedures.
        </p>
        <p style="color:#777;">Type or use 🎤 voice below</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(SUGGESTIONS))
    for i, s in enumerate(SUGGESTIONS):
        with cols[i]:
            if st.button(s):
                st.session_state.suggested = s
                st.rerun()


# ==========================================================
# MAIN RENDER
# ==========================================================

def render_chat(df, embeddings, emb_model, session_id):

    ensure()
    conv = get_active()

    # LOAD DB HISTORY
    if conv:
        msgs = load_messages(session_id, conv["id"])
        if msgs and not conv["messages"]:
            for r, m, _ in msgs:
                conv["messages"].append({"role": r, "content": m})

    # WELCOME
    if conv and len(conv["messages"]) == 0:
        render_welcome()

    # CHAT HISTORY
    if conv:
        for m in conv["messages"]:
            with st.chat_message(m["role"], avatar="🛡️" if m["role"]=="assistant" else "🧑"):
                st.markdown(m["content"])

    # ======================================================
    # 🔥 VOICE INPUT SECTION (NEW)
    # ======================================================

    st.markdown("### 🎤 Voice Input")

    voice_text = ""

    if MIC_AVAILABLE:
        audio = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="⏹ Stop Recording",
            key="mic"
        )

        if audio and audio.get("bytes"):
            with st.spinner("Transcribing voice..."):
                voice_text = transcribe_audio(audio["bytes"])
    else:
        st.warning("Mic not installed. Run: pip install streamlit-mic-recorder")

    # ======================================================
    # TEXT INPUT
    # ======================================================

    user_input = st.chat_input("Ask something...")

    if st.session_state.get("suggested"):
        user_input = st.session_state.suggested
        st.session_state.suggested = None

    # voice overrides text
    if voice_text:
        user_input = voice_text

    if user_input:
        handle(user_input, conv, df, embeddings, emb_model, session_id)


# ==========================================================
# MESSAGE HANDLER
# ==========================================================

def handle(user_input, conv, df, embeddings, emb_model, session_id):

    with st.chat_message("user"):
        st.markdown(user_input)
🛡️
How can I help you today?
Ask me anything about Makerere University's safeguarding policies, disability rights, sexual harassment procedures, and student protections.
🎙️ Speak your question instead of typing


    conv["messages"].append({"role": "user", "content": user_input})

    save_message(session_id, conv["id"], "user", user_input, datetime.now().isoformat())

    if conv["title"] == "New chat":
        conv["title"] = user_input[:40]

    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("Thinking..."):

            if is_greeting(user_input):
                answer = GREETING_RESPONSE
            else:
                retrieved = retrieve_top_k(user_input, emb_model, embeddings, df)
                raw = generate_answer(user_input, retrieved)
                answer = format_response(raw)

        st.markdown(answer)

    save_message(session_id, conv["id"], "assistant", answer, datetime.now().isoformat())

    conv["messages"].append({"role": "assistant", "content": answer})

    st.rerun()