"""
app.py — Safeguarding Companion (Streamlit ChatGPT-style UI)
"""

import streamlit as st
from db import init_db, verify_user, create_user

init_db()

# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="Safeguarding Companion",
    page_icon="🛡️",
    layout="wide"
)

# ===========================================================================
# SESSION STATE
# ===========================================================================
if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===========================================================================
# LOGIN UI (CENTERED CHATGPT-STYLE CARD)
# ===========================================================================
if not st.session_state.user:

    st.markdown("""
    <style>
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 90vh;
    }

    .login-box {
        width: 360px;
        padding: 28px;
        border-radius: 16px;
        background: #0f172a;
        border: 1px solid #1e293b;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    }

    .login-title {
        font-size: 22px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 18px;
        color: white;
    }

    .small {
        font-size: 12px;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 18px;
    }

    input {
        width: 100%;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 10px;
        border: 1px solid #334155;
        background: #0b1220;
        color: white;
    }

    .btn {
        width: 100%;
        padding: 10px;
        border-radius: 10px;
        background: #2563eb;
        color: white;
        border: none;
        cursor: pointer;
        margin-top: 8px;
    }

    .btn:hover {
        background: #1d4ed8;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-container"><div class="login-box">', unsafe_allow_html=True)

    st.markdown('<div class="login-title">🛡 Safeguarding Companion</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Makerere University Support System</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if verify_user(email, password):
                st.session_state.user = email
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            create_user(new_email, new_password)
            st.success("Account created")

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


# ===========================================================================
# CHAT GPT STYLE HEADER
# ===========================================================================
st.markdown("""
<style>
.header {
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:12px 18px;
    border-bottom:1px solid #1e293b;
    position:sticky;
    top:0;
    background:#0b1220;
    z-index:100;
}

.title {
    font-weight:600;
    color:white;
}

.status {
    font-size:12px;
    color:#22c55e;
}
</style>

<div class="header">
    <div class="title">🛡 Safeguarding Companion</div>
    <div class="status">● Online</div>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# LOAD MODELS (lazy import recommended)
# ===========================================================================
from models import load_everything
from retrieval import retrieve_top_k
from generation import generate_answer, format_response
from config import GREETING_RESPONSE

df, embeddings, emb_model = load_everything()

# ===========================================================================
# CHAT DISPLAY (ChatGPT style)
# ===========================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ===========================================================================
# INPUT BAR (ChatGPT STYLE)
# ===========================================================================
user_input = st.chat_input("Ask anything about safeguarding policies... 🎤")

# ===========================================================================
# VOICE INPUT (simple upgrade placeholder)
# ===========================================================================
audio = st.audio_input("🎙 Voice input (optional)")

if audio:
    st.info("Voice detected (connect Whisper module here)")

# ===========================================================================
# PROCESS INPUT
# ===========================================================================
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # greeting check
    if user_input.lower().strip() in ["hi", "hello", "hey"]:
        answer = GREETING_RESPONSE
    else:
        retrieved = retrieve_top_k(user_input, emb_model, embeddings, df)
        raw = generate_answer(user_input, retrieved)
        answer = format_response(raw)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)