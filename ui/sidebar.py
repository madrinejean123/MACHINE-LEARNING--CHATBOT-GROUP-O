import time
from datetime import datetime
import streamlit as st


def render_sidebar(theme: dict):
    BORDER = theme["BORDER"]
    TEXT = theme["TEXT"]
    SUBTEXT = theme["SUBTEXT"]

    with st.sidebar:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;
                    padding:4px 0 18px;border-bottom:1px solid {BORDER};margin-bottom:14px;">
          <div style="width:36px;height:36px;border-radius:10px;
                      background:linear-gradient(135deg,#238636,#1f6feb);
                      display:flex;align-items:center;justify-content:center;
                      font-size:18px;flex-shrink:0;">🛡️</div>
          <div>
            <div style="font-size:0.9rem;font-weight:600;color:{TEXT};">Safeguarding</div>
            <div style="font-size:0.65rem;color:{SUBTEXT};">Makerere University</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✦  New conversation", use_container_width=True, key="new_chat_btn"):
            conv_id = str(int(time.time() * 1000))
            st.session_state.conversations.insert(0, {
                "id": conv_id,
                "title": "New conversation",
                "messages": [],
                "timestamp": datetime.now().strftime("%H:%M"),
            })
            st.session_state.active_conv_id = conv_id
            st.rerun()

        st.markdown(f"""
        <div style="font-size:0.65rem;font-weight:600;letter-spacing:.08em;
                    text-transform:uppercase;color:{SUBTEXT};padding:12px 0 6px;">
          Recent chats
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.conversations:
            st.markdown(
                f'<div style="font-size:0.78rem;padding:8px 0;color:{SUBTEXT};">No conversations yet.</div>',
                unsafe_allow_html=True,
            )
        else:
            for conv in st.session_state.conversations:
                if st.button(
                    f"💬  {conv['title']}",
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                ):
                    st.session_state.active_conv_id = conv["id"]
                    st.rerun()

        st.markdown(f'<hr style="border-color:{BORDER};margin:14px 0;"/>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:0.65rem;font-weight:600;letter-spacing:.08em;
                    text-transform:uppercase;color:{SUBTEXT};padding:4px 0 8px;">
          ⚙️ Settings
        </div>
        """, unsafe_allow_html=True)

        toggled = st.toggle(
            "🌙 Dark mode" if st.session_state.dark_mode else "☀️ Light mode",
            value=st.session_state.dark_mode,
            key="theme_toggle",
        )
        if toggled != st.session_state.dark_mode:
            st.session_state.dark_mode = toggled

        st.markdown(
            f'<div style="font-size:0.72rem;margin-top:10px;margin-bottom:4px;color:{TEXT};">🔆 Contrast</div>',
            unsafe_allow_html=True,
        )
        nc = st.slider(
            "contrast", 50, 150, st.session_state.contrast, 5,
            label_visibility="collapsed", key="contrast_slider"
        )
        if nc != st.session_state.contrast:
            st.session_state.contrast = nc

        st.markdown(
            f'<div style="font-size:0.72rem;margin-top:10px;margin-bottom:4px;color:{TEXT};">🔡 Font size</div>',
            unsafe_allow_html=True,
        )
        nf = st.slider(
            "font", 12, 22, st.session_state.font_size, 1,
            label_visibility="collapsed", key="font_slider"
        )
        if nf != st.session_state.font_size:
            st.session_state.font_size = nf

        st.markdown(f'<hr style="border-color:{BORDER};margin:14px 0;"/>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(255,77,77,0.08);border:1px solid rgba(255,77,77,0.3);
                    border-radius:10px;padding:12px 14px;margin-bottom:12px;">
          <div style="font-size:0.72rem;font-weight:700;color:#ff6b6b;margin-bottom:8px;">
            🚨 NEED IMMEDIATE HELP?
          </div>
          <div style="font-size:0.72rem;line-height:1.9;">
            <b>Gender Mainstreaming Directorate</b><br>
            📞 +256 (0)414 532 631<br>
            📧 gendermainstreaming@mak.ac.ug<br><br>
            <b>Dean of Students Office</b><br>
            📞 +256 (0)414 531 543<br>
            📧 deanofstudents@mak.ac.ug<br><br>
            <b>Security / Emergency</b><br>
            📞 +256 (0)414 530 903<br><br>
            <span style="color:#8b949e;">Mon–Fri · 8 AM – 5 PM<br>Frank Kalimuzo Building</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:0.65rem;line-height:1.6;color:{SUBTEXT};">'
            f'Answers drawn from official Makerere University policy documents.</div>',
            unsafe_allow_html=True,
        )