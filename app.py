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
# 4. GLOBAL THEME CSS — injected BEFORE login so it covers everything
# ===========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root tokens (matches chat.py exactly) ───────────────── */
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
    --shadow:        0 8px 32px rgba(0,0,0,0.45);
    --radius:        14px;
}

/* ── Global reset ────────────────────────────────────────── */
html, body, .stApp {
    background-color: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Streamlit native bar — keep sidebar toggle visible ─── */
header[data-testid="stHeader"] { background: var(--bg-base) !important; }
[data-testid="collapsedControl"] { display: flex !important; }

/* ── Tabs (Login / Register) ─────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}

[data-testid="stTabs"] button[role="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1.5rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.18s !important;
}

[data-testid="stTabs"] button[role="tab"]:hover {
    color: var(--teal) !important;
    background: var(--teal-soft) !important;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    font-weight: 500 !important;
}

/* ── Input fields ────────────────────────────────────────── */
input[type="text"], input[type="email"], input[type="password"],
[data-testid="stTextInput"] input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.2s !important;
}

input[type="text"]:focus, input[type="email"]:focus,
input[type="password"]:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(47,181,160,0.12) !important;
    outline: none !important;
}

[data-testid="stTextInput"] label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
    margin-bottom: 0.3rem !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--teal) 0%, #1e8a78 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    letter-spacing: 0.03em !important;
}

[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Alerts (error / success) ────────────────────────────── */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Main content block ──────────────────────────────────── */
.block-container {
    padding-top: 1.5rem !important;
    max-width: 860px !important;
}

/* ── Main header bar ─────────────────────────────────────── */
.main-header {
    margin-top: 60px;
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}

.main-logo {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--teal) 0%, #1e8a78 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 16px rgba(47,181,160,0.3);
}

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
}

.main-sub {
    font-size: 0.75rem;
    color: var(--text-secondary);
}

.online-badge {
    margin-left: auto;
    font-size: 0.68rem;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(47,181,160,0.12);
    color: var(--teal);
    border: 1px solid rgba(47,181,160,0.3);
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--teal); }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# 5. LOGIN SYSTEM — fully themed
# ===========================================================================
if not st.session_state.user_email:

    # Centre the card using columns
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
        <div style="text-align:center;margin-top:3rem;margin-bottom:1.5rem;
                    animation:fadeUp 0.6s ease both;">
            <div style="font-size:3.2rem;margin-bottom:0.6rem;
                        filter:drop-shadow(0 0 18px rgba(201,168,76,0.4));">🛡️</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.75rem;
                        font-weight:700;color:#eef2f0;margin-bottom:0.3rem;">
                Safeguarding Companion
            </div>
            <div style="font-size:0.8rem;color:#8eaaa6;margin-bottom:2rem;">
                Makerere University · Sign in to continue
            </div>
        </div>
        <style>
        @keyframes fadeUp {
            from { opacity:0; transform:translateY(20px); }
            to   { opacity:1; transform:translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔑  Login", "📝  Register"])

        # ---------------- LOGIN ----------------
        with tab1:
            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            email = st.text_input("Email", placeholder="you@mak.ac.ug", key="login_email")
            password = st.text_input("Password", type="password",
                                     placeholder="••••••••", key="login_pass")
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            if st.button("Sign In →", use_container_width=True, key="login_btn"):
                if verify_user(email, password):
                    st.session_state.user_email = email.lower().strip()
                    st.rerun()
                else:
                    st.error("Invalid email or password")

        # ---------------- REGISTER ----------------
        with tab2:
            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            new_email = st.text_input("Email", placeholder="you@mak.ac.ug", key="reg_email")
            new_password = st.text_input("Password", type="password",
                                         placeholder="Choose a strong password", key="reg_pass")
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            if st.button("Create Account →", use_container_width=True, key="reg_btn"):
                try:
                    create_user(new_email, new_password)
                    st.success("Account created! Switch to the Login tab.")
                except:
                    st.error("An account with that email already exists.")

        st.markdown("""
        <div style='text-align:center;margin-top:1.8rem;
                    font-size:0.7rem;color:#4a6b66;letter-spacing:0.05em;'>
            PROTECTED · MAKERERE UNIVERSITY SAFEGUARD V2
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ===========================================================================
# 6. POST-LOGIN THEME VARS (for sidebar dynamic settings)
# ===========================================================================
from ui.styles import get_theme_vars

theme = get_theme_vars(
    st.session_state.dark_mode,
    st.session_state.contrast,
    st.session_state.font_size,
)


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