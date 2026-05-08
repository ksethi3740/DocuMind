"""
DocuMind — Adaptive Theme Styles (Light + Dark)
"""

import streamlit as st


def inject_styles():
    # Read theme preference
    theme = st.session_state.get("theme", "dark")

    if theme == "dark":
        bg_main      = "#0a0f1e"
        bg_card      = "rgba(15,23,42,0.9)"
        bg_sidebar   = "#080d1a"
        text_primary = "#f1f5f9"
        text_muted   = "#94a3b8"
        border_col   = "rgba(99,102,241,0.3)"
        input_bg     = "rgba(20,30,60,0.9)"
        hero_bg      = "linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(139,92,246,0.12) 100%)"
        upload_bg    = "rgba(99,102,241,0.08)"
        upload_text  = "#c7d2fe"
        btn_text     = "#ffffff"
        tab_list_bg  = "rgba(15,23,42,0.8)"
        divider      = "rgba(99,102,241,0.25)"
        shadow       = "0 8px 32px rgba(0,0,0,0.5)"
    else:
        bg_main      = "#f0f4ff"
        bg_card      = "rgba(255,255,255,0.95)"
        bg_sidebar   = "#e8eeff"
        text_primary = "#1e1b4b"
        text_muted   = "#4b5563"
        border_col   = "rgba(99,102,241,0.35)"
        input_bg     = "#ffffff"
        hero_bg      = "linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%)"
        upload_bg    = "rgba(99,102,241,0.06)"
        upload_text  = "#3730a3"
        btn_text     = "#ffffff"
        tab_list_bg  = "rgba(224,231,255,0.8)"
        divider      = "rgba(99,102,241,0.2)"
        shadow       = "0 8px 32px rgba(99,102,241,0.12)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

    :root {{
        --indigo:      #6366f1;
        --violet:      #8b5cf6;
        --cyan:        #06b6d4;
        --text:        {text_primary};
        --muted:       {text_muted};
        --border:      {border_col};
        --bg-card:     {bg_card};
        --input-bg:    {input_bg};
        --shadow:      {shadow};
    }}
    .dm-mode-groq {{
    color: #f97316; background: rgba(249,115,22,0.1);
    border: 1px solid rgba(249,115,22,0.25);
    border-radius: 6px; padding: 2px 10px; font-size: 0.75rem;
    }}
    .dm-mode-gemini {{
    color: #3b82f6; background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 6px; padding: 2px 10px; font-size: 0.75rem;
    }}
    /* ── App background ── */
    .stApp {{
        background: {bg_main} !important;
        font-family: 'Inter', sans-serif;
        color: {text_primary} !important;
    }}

    /* ── Hide streamlit chrome ── */
    header[data-testid="stHeader"] {{ background: transparent !important; }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {bg_sidebar} !important;
        border-right: 1px solid {border_col} !important;
    }}
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {{
        color: {text_primary} !important;
    }}

    /* ── Upload zone — fix invisible text ── */
/* ── Upload zone — fully visible text ── */
[data-testid="stFileUploader"] {{
    background: {upload_bg} !important;
    border-radius: 14px !important;
    padding: 4px !important;
}}
[data-testid="stFileUploaderDropzone"] {{
    background: {upload_bg} !important;
    border: 2px dashed rgba(99,102,241,0.5) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}}
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span {{
    color: {upload_text} !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    opacity: 1 !important;
    visibility: visible !important;
}}
/* Browse files button */
[data-testid="stFileUploaderDropzone"] button {{
    background: rgba(99,102,241,0.15) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 8px !important;
    color: #a5b4fc !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
}}

    /* ── Hero ── */
    .dm-hero {{
        text-align: center;
        padding: 2.5rem 1rem 1.8rem;
        background: {hero_bg};
        border-radius: 24px;
        border: 1px solid {border_col};
        margin-bottom: 2rem;
        box-shadow: {shadow};
    }}
    .dm-hero h1 {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 40%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    .dm-hero p {{
        color: {text_muted};
        font-size: 1rem;
        margin-top: 0.4rem;
    }}

    /* ── Cards ── */
    .dm-card {{
        background: {bg_card};
        border: 1px solid {border_col};
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: {shadow};
        color: {text_primary};
    }}

    /* ── Send button — fix visibility ── */
    .stButton > button {{
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.55rem 1.5rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
        transition: all 0.2s !important;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,0.55) !important;
        transform: translateY(-1px) !important;
    }}
    .stButton > button p {{
        color: #ffffff !important;
        font-weight: 700 !important;
    }}

    /* ── Text inputs ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background: {input_bg} !important;
        border: 1.5px solid {border_col} !important;
        border-radius: 10px !important;
        color: {text_primary} !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
    }}
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {{
        color: {text_muted} !important;
        opacity: 0.8 !important;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    }}

    /* ── Labels ── */
    label, .stTextInput label, p {{
        color: {text_primary} !important;
    }}

    /* ── Selectbox ── */
    .stSelectbox > div > div {{
        background: {input_bg} !important;
        border: 1.5px solid {border_col} !important;
        border-radius: 10px !important;
        color: {text_primary} !important;
    }}
    .stSelectbox svg {{ fill: {text_muted} !important; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {tab_list_bg} !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid {border_col} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 9px !important;
        color: {text_muted} !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
    }}
    .stTabs [aria-selected="true"] p {{
        color: #ffffff !important;
        font-weight: 600 !important;
    }}

    /* ── Toggle ── */
    .stToggle label p {{ color: {text_primary} !important; }}

    /* ── Suggested prompt buttons ── */
    div[data-testid="column"] .stButton > button {{
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 0.6rem 0.8rem !important;
        white-space: normal !important;
        min-height: 60px !important;
        line-height: 1.3 !important;
    }}

    /* ── Chat bubbles ── */
    .dm-msg-user {{
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15));
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 16px 16px 4px 16px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0 0.5rem 3rem;
        color: {text_primary};
    }}
    .dm-msg-ai {{
        background: {bg_card};
        border: 1px solid {border_col};
        border-radius: 16px 16px 16px 4px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 3rem 0.5rem 0;
        color: {text_primary};
        position: relative;
        box-shadow: {shadow};
    }}
    .dm-msg-ai::before {{
        content: '🧠';
        position: absolute;
        top: -12px; left: -12px;
        font-size: 1.3rem;
    }}

    /* ── Source pill ── */
    .dm-source {{
        background: rgba(6,182,212,0.08);
        border: 1px solid rgba(6,182,212,0.25);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 4px 0;
        font-size: 0.82rem;
        color: {'#0e7490' if theme == 'light' else '#67e8f9'};
    }}

    /* ── Metric cards ── */
    .dm-card div[style*="font-size:1.8rem"] {{
        color: #818cf8 !important;
    }}

    /* ── Divider ── */
    hr {{ border-color: {divider} !important; }}

    /* ── Expander ── */
    details {{
        background: {bg_card} !important;
        border-radius: 12px !important;
        border: 1px solid {border_col} !important;
    }}
    summary {{ color: {text_primary} !important; padding: 0.75rem 1rem !important; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(99,102,241,0.4); border-radius: 3px; }}

    /* ── Mode badges ── */
    .dm-mode-local {{
        color: #16a34a; background: rgba(22,163,74,0.1);
        border: 1px solid rgba(22,163,74,0.25);
        border-radius: 6px; padding: 2px 10px; font-size: 0.75rem;
    }}
    .dm-mode-api {{
        color: #d97706; background: rgba(217,119,6,0.1);
        border: 1px solid rgba(217,119,6,0.25);
        border-radius: 6px; padding: 2px 10px; font-size: 0.75rem;
    }}

    /* ── Progress bar ── */
    .stProgress > div > div {{
        background: linear-gradient(90deg, #6366f1, #06b6d4) !important;
    }}

    /* ── Success/Info/Warning boxes ── */
    .stAlert {{ border-radius: 12px !important; }}

    </style>
    """, unsafe_allow_html=True)