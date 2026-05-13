"""
ui/sidebar.py — FIXED working sidebar (safe + visible)
"""

import time
from datetime import datetime
import streamlit as st


def render_sidebar(theme: dict):
    BORDER = theme["BORDER"]
    TEXT = theme["TEXT"]
    SUBTEXT = theme["SUBTEXT"]

    # ==============================
    # SIDEBAR CONTAINER
    # ==============================
    with st.sidebar:

        # ==============================
        # LOGO
        # ==============================
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;
                    padding:10px 0;border-bottom:1px solid {BORDER};margin-bottom:12px;">
          <div style="width:36px;height:36px;border-radius:10px;
                      background:linear-gradient(135deg,#238636,#1f6feb);
                      display:flex;align-items:center;justify-content:center;">
            🛡️
          </div>
          <div>
            <div style="font-size:14px;font-weight:700;color:{TEXT};">
                Safeguarding Companion
            </div>
            <div style="font-size:11px;color:{SUBTEXT};">
                Makerere University
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ==============================
        # NEW CHAT BUTTON
        # ==============================
        if st.button("➕ New conversation", use_container_width=True):
            conv_id = str(int(time.time() * 1000))

            st.session_state.conversations.insert(0, {
                "id": conv_id,
                "title": "New conversation",
                "messages": [],
                "timestamp": datetime.now().strftime("%H:%M"),
            })

            st.session_state.active_conv_id = conv_id
            st.rerun()

        # ==============================
        # RECENT CHATS
        # ==============================
        st.markdown("### Recent chats")

        if not st.session_state.conversations:
            st.write("No conversations yet.")
        else:
            for conv in st.session_state.conversations:
                if st.button(f"💬 {conv['title']}", key=conv["id"]):
                    st.session_state.active_conv_id = conv["id"]
                    st.rerun()

        st.divider()

        # ==============================
        # SETTINGS
        # ==============================
        st.markdown("### Settings")

        # Dark mode toggle
        dark = st.toggle("Dark mode", value=st.session_state.dark_mode)
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        # ==============================
        # CONTRAST (FIXED)
        # ==============================
        new_contrast = st.slider(
            "Contrast",
            50, 150,
            st.session_state.contrast,
            key="contrast_slider"
        )

        if new_contrast != st.session_state.contrast:
            st.session_state.contrast = new_contrast
            st.rerun()

        # ==============================
        # FONT SIZE (FIXED)
        # ==============================
        new_font = st.slider(
            "Font size",
            12, 22,
            st.session_state.font_size,
            key="font_slider"
        )

        if new_font != st.session_state.font_size:
            st.session_state.font_size = new_font
            st.rerun()

        st.divider()

        # ==============================
        # EMERGENCY BOX
        # ==============================
        st.markdown("""
        <div style="background:#2d0f0f;padding:12px;border-radius:10px;">
            <b style="color:#ff6b6b;">Emergency Contacts</b><br><br>
            📞 Gender Office: +256 414 532 631<br>
            📞 Dean of Students: +256 414 531 543<br>
            📞 Security: +256 414 530 903
        </div>
        """, unsafe_allow_html=True)