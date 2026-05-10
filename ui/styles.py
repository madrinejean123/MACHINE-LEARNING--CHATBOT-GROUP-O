"""
ui/styles.py — safe Streamlit CSS (fixed sidebar toggle + stable layout)
"""

import streamlit as st


def inject_css(
    BG,
    SIDEBAR_BG,
    BORDER,
    TEXT,
    SUBTEXT,
    INPUT_BG,
    INPUT_BOR,
    _cont,
    _fsize,
):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');

/* =========================================================
   BASE APP
========================================================= */
html, body, .stApp {{
    font-family: 'Sora', sans-serif !important;
    background-color: {BG} !important;
    font-size: {_fsize}px !important;
}}

/* keep contrast working */
[data-testid="stAppViewContainer"] {{
    filter: contrast({_cont});
}}

#MainMenu, footer, header {{
    visibility: hidden;
}}

/* =========================================================
   SIDEBAR (SAFE — DOES NOT BREAK TOGGLE)
========================================================= */

/* ONLY background + border (NO width / position changes) */
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG} !important;
    border-right: 1px solid {BORDER} !important;
}}

/* SAFE spacing only */
[data-testid="stSidebar"] > div {{
    padding: 1rem !important;
}}

/* TEXT ONLY (NO *) */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {{
    color: {TEXT} !important;
}}

/* SIDEBAR BUTTONS */
[data-testid="stSidebar"] .stButton > button {{
    width: 100% !important;
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    padding: 7px 10px !important;
    margin-bottom: 2px !important;
}}

[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(88,166,255,0.07) !important;
    border-color: {BORDER} !important;
}}

/* =========================================================
   CHAT INPUT
========================================================= */
[data-testid="stChatInput"] textarea {{
    background: {INPUT_BG} !important;
    border: 1.5px solid {INPUT_BOR} !important;
    border-radius: 28px !important;
    color: {TEXT} !important;
    font-size: 0.95rem !important;
    padding: 14px 20px !important;
}}

[data-testid="stChatInput"] textarea:focus {{
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,.1) !important;
}}

[data-testid="stChatInput"] button {{
    background: linear-gradient(135deg,#238636,#1f6feb) !important;
    border-radius: 50% !important;
    border: none !important;
}}

/* =========================================================
   CHAT MESSAGES
========================================================= */
[data-testid="stChatMessage"],
[data-testid="stMarkdownContainer"] {{
    font-size: {_fsize}px !important;
    color: {TEXT} !important;
    line-height: 1.6 !important;
}}

/* =========================================================
   HEADER
========================================================= */
.main-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 18px;
    border-bottom: 1px solid {BORDER};
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
    flex-shrink: 0;
}}

.main-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: {TEXT};
}}

.main-sub {{
    font-size: 0.7rem;
    color: {SUBTEXT};
}}

.online-badge {{
    margin-left: auto;
    font-size: 0.68rem;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(35,134,54,0.15);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.3);
}}

</style>
""", unsafe_allow_html=True)


def get_theme_vars(dark_mode: bool, contrast: int, font_size: int) -> dict:
    _cont = contrast / 100
    _fsize = font_size

    if dark_mode:
        return dict(
            BG="#0d1117",
            SIDEBAR_BG="#161b22",
            BORDER="#21262d",
            TEXT=f"rgba(230,237,243,{min(_cont,1)})",
            SUBTEXT=f"rgba(139,148,158,{min(_cont,1)})",
            INPUT_BG="#1c2128",
            INPUT_BOR="#30363d",
            _cont=_cont,
            _fsize=_fsize,
        )

    return dict(
        BG="#ffffff",
        SIDEBAR_BG="#f6f8fa",
        BORDER="#d0d7de",
        TEXT=f"rgba(31,35,40,{min(_cont,1)})",
        SUBTEXT=f"rgba(87,96,106,{min(_cont,1)})",
        INPUT_BG="#ffffff",
        INPUT_BOR="#d0d7de",
        _cont=_cont,
        _fsize=_fsize,
    )