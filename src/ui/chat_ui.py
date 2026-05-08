"""
Chat interface — Q&A with source attribution.
Supports mode badges: Groq, Gemini, Claude API, Local LLM
"""

import streamlit as st
from src.core.rag_engine import answer_question
from src.core.vector_store import DocuMindVectorStore


# ── Mode badge config ─────────────────────────────────────────────────────────
MODE_CONFIG = {
    "groq":   {"label": "⚡ Groq / Llama3",  "cls": "dm-mode-groq"},
    "gemini": {"label": "✨ Gemini Flash",    "cls": "dm-mode-gemini"},
    "api":    {"label": "🤖 Claude API",      "cls": "dm-mode-api"},
    "local":  {"label": "🔒 Local LLM",       "cls": "dm-mode-local"},
    "none":   {"label": "⚠️ No Source",       "cls": "dm-mode-local"},
}


def render_chat(vs: DocuMindVectorStore, simple_mode: bool, use_api: bool):
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    chat_history = st.session_state["chat_history"]

    # ── Toolbar (clear chat) ──────────────────────────────────────────────────
    if chat_history:
        col_title, col_clear = st.columns([6, 1])
        with col_title:
            st.markdown(
                f'<div style="color:#94a3b8; font-size:0.82rem; padding-top:6px;">'
                f'💬 {len(chat_history) // 2} message(s)</div>',
                unsafe_allow_html=True
            )
        with col_clear:
            if st.button("🗑️ Clear", key="clear_chat", help="Clear chat history"):
                st.session_state["chat_history"] = []
                st.rerun()

    # ── Display chat history ──────────────────────────────────────────────────
    for msg in chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="dm-msg-user">👤&nbsp;&nbsp;{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            # ── AI response bubble ────────────────────────────────────────────
            st.markdown(
                f'<div class="dm-msg-ai">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

            # ── Source attribution ────────────────────────────────────────────
            if msg.get("sources"):
                with st.expander("🔍 View Source Excerpts", expanded=False):
                    for i, src in enumerate(msg["sources"]):
                        score_pct  = min(int(src["score"] * 100), 100)
                        score_color = (
                            "#86efac" if score_pct >= 70 else
                            "#fbbf24" if score_pct >= 40 else
                            "#f87171"
                        )
                        doc_name = src["metadata"].get("source", "Document")
                        doc_type = src["metadata"].get("type", "").upper()
                        snippet  = src["text"][:300].replace("\n", " ")

                        st.markdown(f"""
                        <div class="dm-source">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <span>📎 <b>{doc_name}</b>
                                    {f'<span style="font-size:0.72rem; color:#64748b; margin-left:6px;">[{doc_type}]</span>' if doc_type else ''}
                                </span>
                                <span style="background:rgba(0,0,0,0.2); border-radius:100px;
                                             padding:2px 10px; font-size:0.75rem; color:{score_color};">
                                    {score_pct}% match
                                </span>
                            </div>
                            <span style="color:#94a3b8; font-size:0.83rem; line-height:1.5;">
                                {snippet}…
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

            # ── Mode badge ────────────────────────────────────────────────────
            mode       = msg.get("mode", "local")
            mode_cfg   = MODE_CONFIG.get(mode, MODE_CONFIG["local"])
            st.markdown(
                f'<span class="{mode_cfg["cls"]}">{mode_cfg["label"]}</span>',
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Input row ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        question = st.text_input(
            "question_field",
            placeholder="Ask anything about your documents — summarise, explain, find details…",
            key="question_input",
            label_visibility="collapsed"
        )
    with col_btn:
        send = st.button("Send 🚀", use_container_width=True, key="send_btn")

    # ── Suggested prompts (shown only when chat is empty) ─────────────────────
    if not chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#94a3b8; font-size:0.9rem; font-weight:600;">💡 Try asking:</p>',
            unsafe_allow_html=True
        )
        prompts = [
            "Summarise the key points",
            "What is the main methodology?",
            "Explain this to me simply",
            "What are the conclusions?",
            "List the important findings"
        ]
        cols = st.columns(len(prompts))
        for i, (col, p) in enumerate(zip(cols, prompts)):
            with col:
                if st.button(p, key=f"sugg_{i}", use_container_width=True):
                    question = p
                    send     = True

    # ── Active mode indicator ─────────────────────────────────────────────────
    if use_api:
        active_modes = []
        import os
        if os.getenv("GROQ_API_KEY"):
            active_modes.append("⚡ Groq")
        if os.getenv("GEMINI_API_KEY"):
            active_modes.append("✨ Gemini")
        from config.settings import ANTHROPIC_API_KEY
        if ANTHROPIC_API_KEY:
            active_modes.append("🤖 Claude")
        if not active_modes:
            active_modes.append("🔒 Local LLM")
        mode_str = " · ".join(active_modes)
        st.markdown(
            f'<div style="text-align:right; color:#475569; font-size:0.75rem; margin-top:4px;">'
            f'Active: {mode_str}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="text-align:right; color:#475569; font-size:0.75rem; margin-top:4px;">'
            'Active: 🔒 Local LLM</div>',
            unsafe_allow_html=True
        )

    # ── Process question ──────────────────────────────────────────────────────
# ── Process question ──────────────────────────────────────────────────────
    if send and question and question.strip():
        if vs.total_chunks == 0:
            st.warning("⚠️ Please upload at least one document first.")
        else:
            # Show which API will be used
            import os
            gkey = os.environ.get("GEMINI_API_KEY","").strip()
            grok = os.environ.get("GROQ_API_KEY","").strip()
            if use_api and gkey:
                st.info("✨ Using Gemini Flash for this answer…")
            elif use_api and grok:
                st.info("⚡ Using Groq / Llama3 for this answer…")
            else:
                st.warning("⚠️ No API key detected — using local extraction. Add GEMINI_API_KEY to .env for better answers.")

            with st.spinner("🧠 Analysing your documents…"):
                result = answer_question(
                    question,
                    vs,
                    simple_mode=simple_mode,
                    use_api=use_api
                )

            chat_history.append({"role": "user", "content": question})
            chat_history.append({
                "role":    "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "mode":    result["mode"]
            })
            st.session_state["chat_history"] = chat_history
            st.rerun()