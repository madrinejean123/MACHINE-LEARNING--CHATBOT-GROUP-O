"""
voice.py — audio transcription using OpenAI Whisper
"""

import os
import re
import streamlit as st
from config import STOP_WORDS


@st.cache_resource(show_spinner=False)
def load_whisper():
    import whisper
    return whisper.load_model("base")


def transcribe_audio(audio_bytes) -> tuple[str, list]:
    import tempfile
    debug_msgs = []
    try:
        raw = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        raw.write(audio_bytes.read())
        raw.flush()
        raw.close()
        debug_msgs.append(f"Audio saved: {os.path.getsize(raw.name)} bytes")
        debug_msgs.append("Loading Whisper model...")
        model = load_whisper()
        debug_msgs.append("Transcribing...")
        result = model.transcribe(raw.name, language="en", task="transcribe")
        os.unlink(raw.name)
        text    = result.get("text", "").strip()
        stripped = re.sub(r"[^a-zA-Z]", "", text)
        if not stripped or len(text.split()) < 2:
            debug_msgs.append(f"Rejected: {repr(text[:60])}")
            return "", debug_msgs
        debug_msgs.append(f"Accepted: {repr(text)}")
        return text, debug_msgs
    except Exception as e:
        debug_msgs.append(f"ERROR: {type(e).__name__}: {e}")
        return "", debug_msgs