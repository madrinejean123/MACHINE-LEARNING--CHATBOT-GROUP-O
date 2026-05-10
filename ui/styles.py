"""
ui/styles.py — injects all CSS based on current theme settings
"""

import streamlit as st


def inject_css(BG, SIDEBAR_BG, BORDER, TEXT, SUBTEXT, INPUT_BG, INPUT_BOR, _cont, _fsize):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Sora', sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-size: {_fsize}px !important;
}}
[data-testid="stAppViewContainer"] {{ filter: contrast({_cont}); }}
#MainMenu, footer, header {{ visibility: hidden; }}

[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] > div:first-child {{ padding: 1.2rem 1rem !important; }}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    padding: 7px 10px !important;
    margin-bottom: 2px !important;
    transition: background 0.15s;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(88,166,255,0.07) !important;
    border-color: {BORDER} !important;
}}

.main-header {{
    display:flex; align-items:center; gap:14px;
    padding:0 0 18px; border-bottom:1px solid {BORDER}; margin-bottom:24px;
}}
.main-logo {{
    width:42px; height:42px; border-radius:12px;
    background:linear-gradient(135deg,#238636,#1f6feb);
    display:flex; align-items:center; justify-content:center;
    font-size:22px; flex-shrink:0;
}}
.main-title {{ font-family:'Playfair Display',serif; font-size:1.15rem; color:{TEXT}; }}
.main-sub   {{ font-size:0.7rem; color:{SUBTEXT}; margin-top:2px; }}
.online-badge {{
    margin-left:auto; font-size:0.68rem; padding:4px 12px; border-radius:20px;
    background:rgba(35,134,54,0.15); color:#3fb950;
    border:1px solid rgba(63,185,80,0.3);
}}
.welcome-card {{ text-align:center; padding:40px 16px 28px; }}
.welcome-card h2 {{
    font-family:'Playfair Display',serif; font-size:1.7rem;
    margin-bottom:12px; color:{TEXT};
}}
.welcome-card p {{
    color:{SUBTEXT}; font-size:0.88rem; line-height:1.8;
    max-width:500px; margin:0 auto;
}}
.src-tag {{
    display:inline-block; margin:4px 4px 0 0; font-size:0.68rem;
    padding:3px 10px; border-radius:20px;
    background:rgba(31,111,235,.12); color:#58a6ff;
    border:1px solid rgba(88,166,255,.2);
}}
[data-testid="stChatInput"] textarea {{
    background:{INPUT_BG} !important; border:1.5px solid {INPUT_BOR} !important;
    border-radius:28px !important; color:{TEXT} !important;
    font-family:'Sora',sans-serif !important;
    font-size:0.95rem !important; padding:14px 20px !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color:#58a6ff !important;
    box-shadow:0 0 0 3px rgba(88,166,255,.1) !important;
}}
[data-testid="stChatInput"] button {{
    background:linear-gradient(135deg,#238636,#1f6feb) !important;
    border-radius:50% !important; border:none !important;
}}
[data-testid="stChatMessage"] {{
    background:transparent !important; border:none !important; padding:4px 0 !important;
}}
hr {{ border-color:{BORDER} !important; }}
</style>
""", unsafe_allow_html=True)


def get_theme_vars(dark_mode: bool, contrast: int, font_size: int) -> dict:
    """Return a dict of all theme colour variables."""
    _cont  = contrast / 100
    _fsize = font_size

    if dark_mode:
        return dict(
            BG="#0d1117", SIDEBAR_BG="#161b22", BORDER="#21262d",
            TEXT=f"rgba(230,237,243,{min(_cont,1)})",
            SUBTEXT=f"rgba(139,148,158,{min(_cont,1)})",
            INPUT_BG="#1c2128", INPUT_BOR="#30363d",
            _cont=_cont, _fsize=_fsize,
        )
    else:
        return dict(
            BG="#ffffff", SIDEBAR_BG="#f6f8fa", BORDER="#d0d7de",
            TEXT=f"rgba(31,35,40,{min(_cont,1)})",
            SUBTEXT=f"rgba(87,96,106,{min(_cont,1)})",
            INPUT_BG="#ffffff", INPUT_BOR="#d0d7de",
            _cont=_cont, _fsize=_fsize,
        )