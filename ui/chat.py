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


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def new_conversation():
    conv_id = str(int(time.time() * 1000))
    st.session_state.conversations.insert(0, {
        "id":        conv_id,
        "title":     "New conversation",
        "messages":  [],
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
    if st.session_state.active_conv_id is None and st.session_state.conversations:
        st.session_state.active_conv_id = st.session_state.conversations[0]["id"]


# ---------------------------------------------------------------------------
# TTS helper
# ---------------------------------------------------------------------------

def _speak(answer: str):
    """Generate a WAV file from answer text using pyttsx3 and store path in session state."""
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
            st.session_state.last_audio      = tmp.name
            st.session_state.last_audio_size = size
            st.session_state.last_audio_err  = None
        else:
            st.session_state.last_audio_err = f"wav too small ({size} bytes)"
    except Exception as e:
        st.session_state.last_audio_err = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Main chat renderer
# ---------------------------------------------------------------------------

def render_chat(df, embeddings, emb_model):
    # make sure session state has needed keys
    if "last_audio"      not in st.session_state: st.session_state.last_audio      = None
    if "last_audio_size" not in st.session_state: st.session_state.last_audio_size = 0
    if "last_audio_err"  not in st.session_state: st.session_state.last_audio_err  = None

    ensure_conversation()
    active_conv = get_active_conv()

    # Welcome screen
    if active_conv and not active_conv["messages"]:
        st.markdown("""
        <div class="welcome-card">
          <div style="font-size:3.2rem;margin-bottom:14px">🛡️</div>
          <h2>How can I help you today?</h2>
          <p>Ask me anything about Makerere University's safeguarding policies,
          disability rights, sexual harassment procedures, and student protections.
          All answers come from official policy documents.</p>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(len(SUGGESTIONS))
        for col, suggestion in zip(cols, SUGGESTIONS):
            with col:
                if st.button(suggestion, use_container_width=True, key=f"sug_{suggestion[:20]}"):
                    st.session_state.suggested_query = suggestion
                    st.rerun()

    # Chat history
    if active_conv:
        for msg in active_conv["messages"]:
            with st.chat_message(msg["role"], avatar="🛡️" if msg["role"] == "assistant" else "🧑"):
                st.markdown(msg["content"], unsafe_allow_html=True)

    # Voice input expander
    with st.expander("🎙️ Speak your question instead of typing", expanded=False):
        st.caption("💡 Allow microphone access in your browser if prompted.")
        audio_input = st.audio_input("🎙️ Click to record", key="mic_input")
        if audio_input is not None:
            with st.spinner("Transcribing…"):
                from voice import transcribe_audio
                transcribed, debug_msgs = transcribe_audio(audio_input)
            for m in debug_msgs:
                st.caption(f"🔍 {m}")
            if transcribed:
                st.success(f"🎙️ Heard: *{transcribed}*")
                if st.button("✅ Submit this voice question", key="submit_voice"):
                    st.session_state.suggested_query = transcribed
                    st.rerun()
            else:
                st.warning("⚠️ Could not transcribe. Type what you said below:")
                manual = st.text_input("Type your question here", key="manual_voice")
                if manual and st.button("✅ Submit", key="submit_manual"):
                    st.session_state.suggested_query = manual
                    st.rerun()

    # Text input
    user_input = st.chat_input("Type your question here…")

    if st.session_state.suggested_query and not user_input:
        user_input = st.session_state.suggested_query
        st.session_state.suggested_query = None

    if user_input and active_conv:
        _handle_user_input(user_input, active_conv, df, embeddings, emb_model)

    # ── TTS audio player (shown after answer is generated) ──────────────────
    if st.session_state.get("last_audio"):
        st.caption(f"🔊 Listen to the answer ({st.session_state.last_audio_size} bytes)")
        st.audio(st.session_state.last_audio, format="audio/wav")
        st.session_state.last_audio = None
    elif st.session_state.get("last_audio_err"):
        st.caption(f"🔇 Voice output unavailable: {st.session_state.last_audio_err}")
        st.session_state.last_audio_err = None


# ---------------------------------------------------------------------------
# Handle a single user message
# ---------------------------------------------------------------------------

def _handle_user_input(user_input, active_conv, df, embeddings, emb_model):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    active_conv["messages"].append({"role": "user", "content": user_input})

    if active_conv["title"] == "New conversation":
        active_conv["title"] = user_input[:45] + ("…" if len(user_input) > 45 else "")

    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("Searching policy documents and generating answer…"):
            if is_greeting(user_input):
                answer    = GREETING_RESPONSE
                sources   = []
                retrieved = None
            else:
                retrieved = retrieve_top_k(user_input, emb_model, embeddings, df)
                raw       = generate_answer(user_input, retrieved)
                answer    = format_response(raw)
                sources   = (
                    list(retrieved["source_document"].unique())
                    if retrieved is not None and not retrieved.empty else []
                )

        st.markdown(answer)

        # generate TTS audio
        _speak(answer)

        if sources:
            tags = "".join(
                f'<span class="src-tag">📄 {nice_source_name(s)}</span>'
                for s in sources
            )
            st.markdown(tags, unsafe_allow_html=True)

        full_content = answer
        if sources:
            full_content += "<br>" + "".join(
                f'<span class="src-tag">📄 {nice_source_name(s)}</span>'
                for s in sources
            )

    active_conv["messages"].append({"role": "assistant", "content": full_content})
    st.rerun()