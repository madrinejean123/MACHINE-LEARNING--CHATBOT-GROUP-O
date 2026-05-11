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


# ---------------------------------------------------------------------------
# CSS Injection — deep teal + warm gold, editorial feel
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root tokens ─────────────────────────────────────────── */
:root {
    --bg-base:       #0d1b1e;
    --bg-surface:    #122428;
    --bg-card:       #162d32;
    --border:        #1f3d44;
    --accent:        #c9a84c;
    --accent-soft:   rgba(201,168,76,0.12);
    --teal:          #2fb5a0;
    --teal-soft:     rgba(47,181,160,0.10);
    --text-primary:  #eef2f0;
    --text-secondary:#8eaaa6;
    --text-muted:    #4a6b66;
    --user-bubble:   #1a3a42;
    --bot-bubble:    #0f2a2e;
    --shadow:        0 8px 32px rgba(0,0,0,0.45);
    --radius:        14px;
}

/* ── Global reset ────────────────────────────────────────── */
html, body, .stApp {
    background-color: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* Restore normal Streamlit container — don't over-widen */
.block-container {
    padding: 2rem 1.5rem 6rem !important;
    max-width: 860px !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] .sidebar-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    color: var(--accent);
    letter-spacing: 0.02em;
    padding: 1.5rem 1.2rem 0.5rem;
}

[data-testid="stSidebar"] .sidebar-subtitle {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 0 1.2rem 1rem;
}

/* ── New conversation button ─────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--teal) 0%, #1e8a78 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.65rem 1rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    letter-spacing: 0.03em;
}

[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Conversation list items ─────────────────────────────── */
.conv-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.7rem 1.2rem;
    cursor: pointer;
    border-left: 3px solid transparent;
    transition: all 0.18s;
    font-size: 0.84rem;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.conv-item:hover {
    background: var(--teal-soft);
    color: var(--text-primary);
    border-left-color: var(--teal);
}

.conv-item.active {
    background: var(--accent-soft);
    color: var(--accent);
    border-left-color: var(--accent);
    font-weight: 500;
}

.conv-item .conv-icon { font-size: 0.9rem; flex-shrink: 0; }
.conv-item .conv-time {
    margin-left: auto;
    font-size: 0.7rem;
    color: var(--text-muted);
    flex-shrink: 0;
}

/* ── Scrollable chat message area ───────────────────────── */
[data-testid="stChatMessageContainer"],
section.main > div:first-child {
    overflow-y: auto !important;
    max-height: calc(100vh - 160px) !important;
    padding-bottom: 2rem !important;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}

/* ── Welcome screen ──────────────────────────────────────── */
.welcome-wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 55vh;
    padding: 2rem 0;
}

.welcome-box {
    text-align: center;
    max-width: 560px;
    animation: fadeUp 0.7s ease both;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

.welcome-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    filter: drop-shadow(0 0 18px rgba(201,168,76,0.4));
}

.welcome-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
    line-height: 1.25;
}

.welcome-text {
    font-size: 0.95rem;
    color: var(--text-secondary);
    line-height: 1.7;
    margin-bottom: 1.5rem;
}

.welcome-hint {
    display: inline-block;
    background: var(--teal-soft);
    border: 1px solid rgba(47,181,160,0.25);
    color: var(--teal);
    font-size: 0.78rem;
    padding: 0.45rem 1rem;
    border-radius: 999px;
    letter-spacing: 0.04em;
}

/* ── Suggestion chips ────────────────────────────────────── */
.stButton > button[key^="sug_"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    border-radius: 10px !important;
    font-size: 0.78rem !important;
    padding: 0.55rem 0.8rem !important;
    text-align: left !important;
    line-height: 1.4 !important;
    transition: all 0.18s !important;
    height: auto !important;
    white-space: normal !important;
}

.stButton > button[key^="sug_"]:hover {
    border-color: var(--teal) !important;
    color: var(--teal) !important;
    background: var(--teal-soft) !important;
    transform: translateY(-2px) !important;
}

/* ── Chat messages ───────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
    animation: msgIn 0.3s ease both;
}

@keyframes msgIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* User message bubble */
[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
[data-testid="stChatMessage"]:has([alt="🧑"]) .stMarkdown {
    background: var(--user-bubble) !important;
    border: 1px solid rgba(47,181,160,0.2) !important;
    border-radius: var(--radius) var(--radius) 4px var(--radius) !important;
    padding: 0.9rem 1.15rem !important;
    font-size: 0.92rem !important;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow) !important;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:has([alt="🛡️"]) .stMarkdown {
    background: var(--bot-bubble) !important;
    border: 1px solid rgba(201,168,76,0.15) !important;
    border-radius: var(--radius) var(--radius) var(--radius) 4px !important;
    padding: 0.9rem 1.15rem !important;
    font-size: 0.92rem !important;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow) !important;
}

/* Avatar circles */
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
    width: 36px !important;
    height: 36px !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1rem !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, var(--teal) 0%, #1e8a78 100%) !important;
    box-shadow: 0 0 12px rgba(47,181,160,0.35) !important;
}

[data-testid="chatAvatarIcon-user"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
}

/* ── Markdown typography inside bubbles ──────────────────── */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Playfair Display', serif;
    color: var(--accent) !important;
    margin-top: 0.8rem;
}

.stMarkdown ul, .stMarkdown ol {
    padding-left: 1.2rem;
    color: var(--text-secondary);
}

.stMarkdown li { margin-bottom: 0.3rem; }

.stMarkdown strong { color: var(--teal); }

.stMarkdown code {
    background: rgba(47,181,160,0.12);
    color: var(--teal);
    padding: 0.1em 0.4em;
    border-radius: 4px;
    font-size: 0.85em;
}

/* ── Chat input — stays inside the main column only ─────── */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    right: 0 !important;
    /* left respects the sidebar (~21rem default Streamlit sidebar) */
    left: 21rem !important;
    z-index: 100 !important;
    background: linear-gradient(to top, var(--bg-base) 70%, transparent) !important;
    padding: 0.8rem 2rem 1rem !important;
}

[data-testid="stChatInput"] textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.2s !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(47,181,160,0.12) !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

/* Send button */
[data-testid="stChatInput"] button {
    background: var(--teal) !important;
    border: none !important;
    border-radius: 10px !important;
    transition: opacity 0.2s !important;
}

[data-testid="stChatInput"] button:hover { opacity: 0.82 !important; }

/* ── Spinner ─────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: var(--teal) !important;
}

/* ── Audio player ────────────────────────────────────────── */
audio {
    width: 100% !important;
    border-radius: 8px !important;
    margin-top: 0.5rem !important;
    filter: invert(0.85) hue-rotate(140deg);
}

/* ── Divider ─────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1rem 1.2rem !important;
}

/* ── Voice input bar ─────────────────────────────────────── */
.voice-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.55rem 1rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}

.voice-bar.listening {
    border-color: var(--teal);
    box-shadow: 0 0 0 3px rgba(47,181,160,0.12);
}

.mic-btn {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 1.2rem;
    line-height: 1;
    padding: 0;
    transition: transform 0.15s;
    flex-shrink: 0;
}

.mic-btn:hover { transform: scale(1.15); }
.mic-btn.active { animation: pulse 1s infinite; }

@keyframes pulse {
    0%, 100% { filter: drop-shadow(0 0 4px var(--teal)); }
    50%       { filter: drop-shadow(0 0 12px var(--teal)); }
}

.voice-transcript {
    flex: 1;
    font-size: 0.88rem;
    color: var(--text-secondary);
    font-style: italic;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
}

.voice-transcript.has-text { color: var(--text-primary); font-style: normal; }

.voice-send-btn {
    background: var(--teal);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.3rem 0.75rem;
    font-size: 0.78rem;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
    flex-shrink: 0;
}

.voice-send-btn.visible { opacity: 1; pointer-events: auto; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--teal); }
</style>
"""


def inject_styles():
    """Inject global CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
# Sidebar renderer
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render branded sidebar with conversation list."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">🛡️ SafeGuard AI</div>'
            '<div class="sidebar-subtitle">Makerere University</div>',
            unsafe_allow_html=True,
        )

        if st.button("＋  New Conversation", key="new_conv_btn"):
            new_conversation()
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # Conversation list
        for conv in st.session_state.get("conversations", []):
            is_active = conv["id"] == st.session_state.get("active_conv_id")
            css_class = "conv-item active" if is_active else "conv-item"
            title = conv.get("title", "Untitled")[:38]
            ts = conv.get("timestamp", "")

            st.markdown(
                f'<div class="{css_class}">'
                f'  <span class="conv-icon">💬</span>'
                f'  <span style="overflow:hidden;text-overflow:ellipsis;">{title}</span>'
                f'  <span class="conv-time">{ts}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Invisible button overlay to make the item clickable
            if st.button("_", key=f"sel_conv_{conv['id']}", label_visibility="collapsed"):
                st.session_state.active_conv_id = conv["id"]
                st.rerun()

        # Footer
        st.markdown(
            '<div style="position:fixed;bottom:1rem;left:0;right:0;'
            'text-align:center;font-size:0.68rem;color:#4a6b66;">'
            'Powered by Makerere University · SafeGuard v2</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# TTS helper
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
# Main chat renderer
# ---------------------------------------------------------------------------

def render_chat(df, embeddings, emb_model, session_id):
    # Inject styles on every render
    inject_styles()

    # Initialise audio state
    for key, default in [("last_audio", None), ("last_audio_size", 0), ("last_audio_err", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    ensure_conversation()
    active_conv = get_active_conv()

    # ── Load history from DB ──────────────────────────────────
    if active_conv:
        db_messages = load_messages(session_id, active_conv["id"])
        if db_messages and not active_conv["messages"]:
            for role, msg, _ in db_messages:
                active_conv["messages"].append({"role": role, "content": msg})

    # ── Welcome screen ────────────────────────────────────────
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
                <div class="welcome-hint">🔊 Responses are read aloud automatically</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Suggestion chips
        cols = st.columns(len(SUGGESTIONS))
        for col, suggestion in zip(cols, SUGGESTIONS):
            with col:
                if st.button(suggestion, key=f"sug_{suggestion[:20]}"):
                    st.session_state.suggested_query = suggestion
                    st.rerun()

    # ── Chat history ──────────────────────────────────────────
    if active_conv:
        for msg in active_conv["messages"]:
            avatar = "🛡️" if msg["role"] == "assistant" else "🧑"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"], unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────
    user_input = st.chat_input("Type your question here…")

    if st.session_state.get("suggested_query") and not user_input:
        user_input = st.session_state.suggested_query
        st.session_state.suggested_query = None

    if user_input and active_conv:
        _handle_user_input(user_input, active_conv, df, embeddings, emb_model, session_id)

    # ── Audio playback ────────────────────────────────────────
    if st.session_state.get("last_audio"):
        st.audio(st.session_state.last_audio, format="audio/wav")
        st.session_state.last_audio = None


# ---------------------------------------------------------------------------
# Handle user message
# ---------------------------------------------------------------------------

def _handle_user_input(user_input, active_conv, df, embeddings, emb_model, session_id):

    # Show user bubble immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    active_conv["messages"].append({"role": "user", "content": user_input})

    save_message(
        session_id,
        active_conv["id"],
        "user",
        user_input,
        datetime.now().isoformat(),
    )

    # Update conversation title from first message
    if active_conv["title"] == "New conversation":
        active_conv["title"] = user_input[:45]

    # Generate and show assistant response
    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("Searching policy documents…"):
            if is_greeting(user_input):
                answer = GREETING_RESPONSE
            else:
                retrieved = retrieve_top_k(user_input, emb_model, embeddings, df)
                raw = generate_answer(user_input, retrieved)
                answer = format_response(raw)

        st.markdown(answer, unsafe_allow_html=True)
        _speak(answer)

    save_message(
        session_id,
        active_conv["id"],
        "assistant",
        answer,
        datetime.now().isoformat(),
    )

    active_conv["messages"].append({"role": "assistant", "content": answer})
    st.rerun()