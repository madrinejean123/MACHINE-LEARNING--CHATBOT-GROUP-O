"""
app.py — entry point for the eSafeRide Safeguarding Companion
All logic lives in separate modules — this file just wires them together.
"""

import streamlit as st

# ===========================================================================
# 1. PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="Safeguarding Companion",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# 2. SESSION STATE
# ===========================================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "contrast" not in st.session_state:
    st.session_state.contrast = 100

if "font_size" not in st.session_state:
    st.session_state.font_size = 16

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "active_conv_id" not in st.session_state:
    st.session_state.active_conv_id = None

if "suggested_query" not in st.session_state:
    st.session_state.suggested_query = None

# ===========================================================================
# 3. THEME
# ===========================================================================
from ui.styles import get_theme_vars

theme = get_theme_vars(
    st.session_state.dark_mode,
    st.session_state.contrast,
    st.session_state.font_size,
)

# ===========================================================================
# 4. HEADER + CHAT INTRO STYLING ONLY
# ===========================================================================
st.markdown("""
<style>

/* ================= HEADER ================= */
.main-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 18px;
    border-bottom: 1px solid rgba(208,215,222,0.8);
    margin-bottom: 24px;
}

.main-logo {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg,#238636,#1f6feb);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 600;
}

.main-sub {
    font-size: 0.75rem;
    opacity: 0.7;
}

.online-badge {
    margin-left: auto;
    font-size: 0.68rem;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(35,134,54,0.15);
    color: #238636;
    border: 1px solid rgba(35,134,54,0.3);
}

/* ================= CHAT WELCOME ================= */
.chat-welcome {
    text-align: center;
    margin-top: 40px;
    padding: 20px;
}

.chat-welcome h3 {
    font-size: 1.2rem;
    margin-bottom: 10px;
}

.chat-welcome p {
    font-size: 0.95rem;
    opacity: 0.8;
    line-height: 1.6;
    max-width: 650px;
    margin: 0 auto;
}

.chat-hint {
    margin-top: 18px;
    font-size: 0.85rem;
    opacity: 0.7;
}

</style>
""", unsafe_allow_html=True)

# ===========================================================================
# 5. SIDEBAR (UNCHANGED)
# ===========================================================================
from ui.sidebar import render_sidebar
render_sidebar(theme)

# ===========================================================================
# 6. HEADER UI
# ===========================================================================
st.markdown("""
<div class="main-header">
  <div class="main-logo">🛡️</div>
  <div>
    <div class="main-title">Safeguarding Companion</div>
    <div class="main-sub">Makerere University · Policy Q&amp;A</div>
  </div>
  <div class="online-badge">● Online</div>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# 7. CHAT WELCOME TEXT (YOUR PART)
# ===========================================================================
st.markdown("""
<div class="chat-welcome">
    <h3>🛡️ How can I help you today?</h3>

    <p>
        Ask me anything about Makerere University's safeguarding policies,
        disability rights, sexual harassment procedures, and student protections.
        All answers come from official policy documents.
    </p>

    <div class="chat-hint">
        🎙️ Speak your question instead of typing
    </div>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# 8. LOAD MODELS
# ===========================================================================
from models import load_everything

with st.spinner("Loading policy documents — first run takes a moment…"):
    df, embeddings, emb_model = load_everything()

# ===========================================================================
# 9. CHAT SYSTEM
# ===========================================================================
from ui.chat import render_chat
render_chat(df, embeddings, emb_model)