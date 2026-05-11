"""
ui/chat.py — Clean ChatGPT-style Streamlit chat system (HF-safe)
"""

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
# CONVERSATION SETUP
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
    if "conversations" not in st.session_state:
        st.session_state.conversations = []

    if not st.session_state.conversations:
        new_conversation()

    if not st.session_state.active_conv_id:
        st.session_state.active_conv_id = st.session_state.conversations[0]["id"]


# ==========================================================
# WELCOME SCREEN (FIXED SAFE HTML STRING)
# ==========================================================

def render_welcome():
    st.markdown("""
    <div style="
        text-align:center;
        padding:40px 20px;
        border-radius:16px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(255,255,255,0.08);
        margin-bottom:20px;
    ">
        <div style="font-size:45px;">🛡️</div>

        <h2 style="margin:10px 0; color:white;">
            Safeguarding Companion
        </h2>

        <p style="color:#aaa; max-width:650px; margin:auto;">
            Ask me anything about Makerere University safeguarding policies,
            disability rights, sexual harassment procedures, and student protections.
        </p>

        <p style="margin-top:12px; color:#777;">
            Type your question below to begin
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# MAIN CHAT UI
# ==========================================================

def render_chat(df, embeddings, emb_model, session_id):

    ensure()
    conv = get_active()

    # ---------------- LOAD DB HISTORY ----------------
    if conv:
        db_msgs = load_messages(session_id, conv["id"])
        if db_msgs and not conv["messages"]:
            for role, msg, _ in db_msgs:
                conv["messages"].append({
                    "role": role,
                    "content": msg
                })

    # ---------------- WELCOME ----------------
    if conv and len(conv["messages"]) == 0:
        render_welcome()

    # ---------------- CHAT HISTORY ----------------
    if conv:
        for msg in conv["messages"]:
            with st.chat_message(
                msg["role"],
                avatar="🛡️" if msg["role"] == "assistant" else "🧑"
            ):
                st.markdown(msg["content"])

    # ---------------- INPUT ----------------
    user_input = st.chat_input("Ask your question...")

    if user_input:
        handle(user_input, conv, df, embeddings, emb_model, session_id)

    # ---------------- SUGGESTIONS ----------------
    if conv and len(conv["messages"]) == 0:
        cols = st.columns(len(SUGGESTIONS))
        for i, s in enumerate(SUGGESTIONS):
            with cols[i]:
                if st.button(s):
                    handle(s, conv, df, embeddings, emb_model, session_id)


# ==========================================================
# MESSAGE HANDLER
# ==========================================================

def handle(user_input, conv, df, embeddings, emb_model, session_id):

    # USER MESSAGE
    with st.chat_message("user"):
        st.markdown(user_input)

    conv["messages"].append({"role": "user", "content": user_input})

    save_message(session_id, conv["id"], "user", user_input, datetime.now().isoformat())

    if conv["title"] == "New chat":
        conv["title"] = user_input[:40]

    # ASSISTANT RESPONSE
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