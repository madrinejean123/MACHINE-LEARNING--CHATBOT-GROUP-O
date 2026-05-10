"""
app.py — entry point for the eSafeRide Safeguarding Companion
All logic lives in separate modules — this file just wires them together.
"""

import streamlit as st

# ===========================================================================
# 1. PAGE CONFIG — must be the very first Streamlit call
# ===========================================================================
st.set_page_config(
    page_title="Safeguarding Companion",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# 2. SESSION STATE — initialise before anything reads it
# ===========================================================================
if "dark_mode"       not in st.session_state: st.session_state.dark_mode       = True
if "contrast"        not in st.session_state: st.session_state.contrast        = 100
if "font_size"       not in st.session_state: st.session_state.font_size       = 16
if "conversations"   not in st.session_state: st.session_state.conversations   = []
if "active_conv_id"  not in st.session_state: st.session_state.active_conv_id  = None
if "suggested_query" not in st.session_state: st.session_state.suggested_query = None

# ===========================================================================
# 3. THEME + CSS
# ===========================================================================
from ui.styles import get_theme_vars, inject_css

theme = get_theme_vars(
    st.session_state.dark_mode,
    st.session_state.contrast,
    st.session_state.font_size,
)
inject_css(**theme)

# ===========================================================================
# 4. SIDEBAR
# ===========================================================================
from ui.sidebar import render_sidebar
render_sidebar(theme)

# ===========================================================================
# 5. MAIN HEADER
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
# 6. LOAD MODELS + CHUNKS
# ===========================================================================
from models import load_everything

with st.spinner("Loading policy documents — first run takes a moment…"):
    df, embeddings, emb_model = load_everything()

# ===========================================================================
# 7. CHAT AREA
# ===========================================================================
from ui.chat import render_chat
render_chat(df, embeddings, emb_model)