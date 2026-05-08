"""
DocuMind - RAG-Based Intelligent Document Analysis System
Main Application Entry Point
"""

import streamlit as st
import sys
import os

# Add project root to path so all src imports work
sys.path.insert(0, os.path.dirname(__file__))

from src.ui.main_ui import render_main_ui
from src.ui.styles import inject_styles

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "DocuMind – Intelligent RAG-based Document Intelligence System"
    }
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
inject_styles()

# ── Render main UI ────────────────────────────────────────────────────────────
render_main_ui()