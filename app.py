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
# 3. THEME (NO CSS FILES)
# ===========================================================================
from ui.styles import get_theme_vars

theme = get_theme_vars(
    st.session_state.dark_mode,
    st.session_state.contrast,
    st.session_state.font_size,
)


# ===========================================================================
# 4. GLOBAL STYLE (DIRECT — NO inject_css)
# ===========================================================================
st.markdown(f"""
<style>

/* =========================
   GLOBAL VARIABLES
========================= */
:root {{
    --font-size: {theme["font"]}px;
    --text: {theme["text"]};
    --subtext: {theme["subtext"]};
    --bg: {theme["bg"]};
    --sidebar: {theme["sidebar"]};
    --border: {theme["border"]};
}}

/* =========================
   BASE APP
========================= */
html, body, .stApp {{
    background-color: var(--bg) !important;
    font-size: var(--font-size) !important;
    color: var(--text) !important;
}}

/* Apply to all readable elements */
p, span, div, label, button, textarea {{
    font-size: var(--font-size) !important;
    color: var(--text) !important;
}}

/* =========================
   SIDEBAR (SAFE ONLY COLOR)
========================= */
[data-testid="stSidebar"] {{
    background-color: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
}}

/* =========================
   CHAT INPUT
========================= */
[data-testid="stChatInput"] textarea {{
    font-size: var(--font-size) !important;
    color: var(--text) !important;
}}

/* =========================
   CHAT MESSAGES
========================= */
[data-testid="stChatMessage"] {{
    font-size: var(--font-size) !important;
    color: var(--text) !important;
    line-height: 1.6;
}}

/* =========================
   HEADER
========================= */
.main-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}}

.main-logo {{
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg,#238636,#1f6feb);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}}

.main-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
}}

.main-sub {{
    font-size: 0.75rem;
    color: var(--subtext);
}}

.online-badge {{
    margin-left: auto;
    font-size: 0.7rem;
    padding: 4px 10px;
    border-radius: 20px;
    background: rgba(35,134,54,0.15);
    color: #3fb950;
    border: 1px solid rgba(35,134,54,0.3);
}}

</style>
""", unsafe_allow_html=True)


# ===========================================================================
# 5. SIDEBAR
# ===========================================================================
from ui.sidebar import render_sidebar
render_sidebar(theme)


# ===========================================================================
# 6. HEADER
# ===========================================================================
st.markdown("""
<div class="main-header">
  <div class="main-logo">🛡️</div>
  <div>
    <div class="main-title">Safeguarding Companion</div>
    <div class="main-sub">Makerere University · Policy Q&A</div>
  </div>
  <div class="online-badge">● Online</div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# 7. LOAD MODELS
# ===========================================================================
from models import load_everything

with st.spinner("Loading policy documents — first run takes a moment…"):
    df, embeddings, emb_model = load_everything()


# ===========================================================================
# 8. CHAT
# ===========================================================================
from ui.chat import render_chat
render_chat(df, embeddings, emb_model)