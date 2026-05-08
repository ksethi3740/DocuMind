"""
Main UI layout — assembles all tabs.
"""

import streamlit as st
from src.core.vector_store import DocuMindVectorStore
from src.ui.sidebar import render_sidebar
from src.ui.chat_ui import render_chat
from src.ui.diagram_ui import render_diagram_tab


def render_main_ui():
    # ── Session state init ────────────────────────────────────────────────────
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = DocuMindVectorStore()
    if "processed_docs" not in st.session_state:
        st.session_state["processed_docs"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    vs = st.session_state["vector_store"]

    # ── Sidebar ───────────────────────────────────────────────────────────────
    processed_docs, simple_mode, use_api = render_sidebar(vs)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="dm-hero">
        <h1>🧠 DocuMind</h1>
        <p>Intelligent Document Analysis · RAG-Powered · Works Offline</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats bar ─────────────────────────────────────────────────────────────
    if processed_docs:
        total_chunks = vs.total_chunks
        doc_types    = list({d["metadata"]["type"] for d in processed_docs})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="dm-card" style="text-align:center;"><div style="font-size:1.8rem;font-weight:700;color:#818cf8;">{len(processed_docs)}</div><div style="color:#64748b;font-size:0.8rem;">Documents</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="dm-card" style="text-align:center;"><div style="font-size:1.8rem;font-weight:700;color:#06b6d4;">{total_chunks:,}</div><div style="color:#64748b;font-size:0.8rem;">Indexed Chunks</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="dm-card" style="text-align:center;"><div style="font-size:1.8rem;font-weight:700;color:#8b5cf6;">{len(st.session_state["chat_history"]) // 2}</div><div style="color:#64748b;font-size:0.8rem;">Questions Asked</div></div>', unsafe_allow_html=True)
        with col4:
            mode_txt  = "⚡ API" if use_api else "🔒 Local"
            mode_col  = "#fbbf24" if use_api else "#86efac"
            st.markdown(f'<div class="dm-card" style="text-align:center;"><div style="font-size:1.5rem;font-weight:700;color:{mode_col};">{mode_txt}</div><div style="color:#64748b;font-size:0.8rem;">AI Mode</div></div>', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_qa, tab_diagram, tab_summary, tab_about = st.tabs([
        "💬 Q&A Chat",
        "🗺️ Diagrams",
        "📋 Summary",
        "ℹ️ About"
    ])

    with tab_qa:
        render_chat(vs, simple_mode=simple_mode, use_api=use_api)

    with tab_diagram:
        render_diagram_tab(processed_docs)

    with tab_summary:
        _render_summary_tab(vs, processed_docs, simple_mode, use_api)

    with tab_about:
        _render_about()


# ── Summary tab ───────────────────────────────────────────────────────────────

def _render_summary_tab(vs, processed_docs, simple_mode, use_api):
    from src.core.rag_engine import answer_question

    st.markdown('<div class="dm-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Document Summary")
    st.markdown("Generate intelligent summaries of your documents.")
    st.markdown("</div>", unsafe_allow_html=True)

    if not processed_docs:
        st.info("📂 Upload documents first.")
        return

    doc_names = [d["name"] for d in processed_docs]
    selected  = st.selectbox("Choose document:", doc_names, key="sum_select")
    mode      = st.radio("Summary style:", ["📚 Detailed", "🗣️ Simple / Layman"], horizontal=True)

    if st.button("✨ Generate Summary"):
        # Get the full text of the selected document directly
        doc_data  = next((d for d in processed_docs if d["name"] == selected), None)
        is_simple = "Simple" in mode

        if not doc_data:
            st.error("Document not found.")
            return

        with st.spinner("Summarising…"):
            # Build context directly from document text — not via vector search
            # This avoids the "Document Source N:" raw dump problem
            full_text = doc_data.get("text", "")

            # Use Gemini/Groq if available, else smart local extraction
            import os
            gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
            groq_key   = os.environ.get("GROQ_API_KEY", "").strip()

            if use_api and (gemini_key or groq_key):
                # Use RAG for API mode — it works well
                result = answer_question(
                    "Provide a comprehensive, well-structured summary of this entire document. "
                    "Cover: main topic, objectives, methodology, key findings, and conclusions.",
                    vs,
                    simple_mode=is_simple,
                    use_api=True
                )
                summary = result["answer"]
            else:
                # Local mode — use the full text directly, not RAG chunks
                from src.core.local_llm import generate_answer_local

                # Take first 4000 chars of full document for local processing
                context = full_text[:4000]
                summary = generate_answer_local(
                    context,
                    "Summarise this document. Cover the main topic, objectives, "
                    "methodology, key findings and conclusions.",
                    is_simple
                )

        st.markdown('<div class="dm-card">', unsafe_allow_html=True)
        st.markdown(summary)
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download Summary",
            data=summary,
            file_name=f"summary_{selected}.txt",
            mime="text/plain"
        )

# ── About tab ─────────────────────────────────────────────────────────────────

def _render_about():
    st.markdown("""
    <div class="dm-card">
        <h3>🧠 About DocuMind</h3>
        <p style="color:#94a3b8;">
            DocuMind is a <b>RAG-based intelligent document analysis system</b> that combines
            semantic search, multimodal document processing, and AI-powered Q&amp;A — entirely offline-capable.
        </p>
    </div>
    """, unsafe_allow_html=True)

    features = [
        ("📄 Multimodal Input",     "Supports PDF, DOCX, PNG, JPG, BMP, TIFF files with full OCR support for scanned images and embedded diagrams."),
        ("🔍 Semantic Search",       "Uses sentence-transformers embeddings + FAISS vector store for lightning-fast semantic retrieval across large documents."),
        ("🤖 Dual AI Modes",         "Works offline with local flan-t5 model OR with Claude API for production-grade quality."),
        ("🗺️ Auto Flowcharts",       "Automatically generates Mermaid flowcharts, research paper structure maps, and concept mind maps."),
        ("🗣️ Layman Mode",          "Toggle simple-language mode to get jargon-free, easy-to-understand explanations."),
        ("📎 Source Attribution",    "Every answer shows ranked source excerpts with relevance scores for full transparency."),
        ("🔒 Privacy-First",         "All processing happens locally. Your documents never leave your machine in offline mode."),
        ("⚡ Scalable Architecture", "Modular design with separate layers for ingestion, embedding, retrieval, and generation."),
    ]

    cols = st.columns(2)
    for i, (title, desc) in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="dm-card">
                <b style="color:#a5b4fc;">{title}</b>
                <p style="color:#94a3b8; font-size:0.88rem; margin-top:0.3rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)