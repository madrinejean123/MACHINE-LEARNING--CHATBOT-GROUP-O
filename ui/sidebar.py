import streamlit as st

def render_sidebar(theme: dict):
    with st.sidebar:
        st.markdown(
            """
            <div style="
                background:#ffeb3b;
                color:#000;
                padding:16px;
                border-radius:10px;
                font-size:20px;
                font-weight:700;
                margin-bottom:12px;
            ">
            SIDEBAR DEBUG: function is running
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("theme =", theme)
        st.write("If you can read this, the sidebar is working.")