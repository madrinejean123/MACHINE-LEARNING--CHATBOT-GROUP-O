"""
app.py — entry point for the eSafeRide Safeguarding Companion
All logic lives in separate modules — this file just wires them together.
"""

import streamlit as st
from db import init_db

# Initialize database
init_db()

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
# 2. DB AUTH IMPORT
# ===========================================================================
from db import verify_user, create_user

# ===========================================================================
# 3. SESSION STATE
# ===========================================================================
if "user_email" not in st.session_state:
    st.session_state.user_email = None

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
# 4. LOGIN SYSTEM
# ===========================================================================
if not st.session_state.user_email:

    st.title("🛡️ Safeguarding Companion Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    # ---------------- LOGIN ----------------
    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if verify_user(email, password):
                st.session_state.user_email = email.lower().strip()
                st.rerun()
            else:
                st.error("Invalid email or password")

    # ---------------- REGISTER ----------------
    with tab2:
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            try:
                create_user(new_email, new_password)
                st.success("Account created! Go to Login tab.")
            except:
                st.error("User already exists")

    st.stop()


# ===========================================================================
# 5. THEME
# ===========================================================================
from ui.styles import get_theme_vars

theme = get_theme_vars(
    st.session_state.dark_mode,
    st.session_state.contrast,
    st.session_state.font_size,
)


# ===========================================================================
# 6. GLOBAL CSS VARIABLES
# ===========================================================================
st.markdown(f"""
<style>

:root {{
    --font-size: {theme["_fsize"]}px;
    --text: {theme["TEXT"]};
    --subtext: {theme["SUBTEXT"]};
    --bg: {theme["BG"]};
}}

html, body, .stApp {{
    background-color: var(--bg) !important;
    font-size: var(--font-size) !important;
}}

.main-header {{
    margin-top: 60px;
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 18px;
    border-bottom: 1px solid rgba(208,215,222,0.8);
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
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
}}

.main-sub {{
    font-size: 0.75rem;
    color: var(--subtext);
}}

.online-badge {{
    margin-left: auto;
    font-size: 0.68rem;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(35,134,54,0.15);
    color: #238636;
    border: 1px solid rgba(35,134,54,0.3);
}}

</style>
""", unsafe_allow_html=True)


# ===========================================================================
# 7. SIDEBAR
# ===========================================================================
from ui.sidebar import render_sidebar
render_sidebar(theme)


# ===========================================================================
# 8. HEADER UI
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
# 9. LOAD MODELS
# ===========================================================================
from models import load_everything

with st.spinner("Loading policy documents .... first run takes a moment ..."):
    df, embeddings, emb_model = load_everything()


# ===========================================================================
# 10. CHAT (SESSION HANDLING)
# ===========================================================================
from ui.chat import render_chat

render_chat(
    df,
    embeddings,
    emb_model,
    st.session_state.user_email
)