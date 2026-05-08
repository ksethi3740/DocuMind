"""
Sidebar: file upload, settings, document stats.
"""

import streamlit as st
from src.core.document_processor import process_uploaded_file
from src.core.vector_store import DocuMindVectorStore

def render_sidebar(vs):
    # ── Theme toggle ──────────────────────────────────────────────────────────
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

    col_a, col_b = st.sidebar.columns([3, 1])
    with col_b:
        current = st.session_state["theme"]
        icon    = "☀️" if current == "dark" else "🌙"
        if st.button(icon, key="theme_btn", help="Toggle Light/Dark theme"):
            st.session_state["theme"] = "light" if current == "dark" else "dark"
            st.rerun()
def render_sidebar(vs: DocuMindVectorStore) -> list[dict]:
    """Render sidebar and return list of processed documents."""
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.2rem;">🧠</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:700;
                    background:linear-gradient(135deg,#a5b4fc,#06b6d4);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            DocuMind
        </div>
        <div style="font-size:0.72rem; color:#64748b; letter-spacing:0.08em; text-transform:uppercase;">
            Document Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 📂 Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Drop files here",
        type=["pdf", "docx", "png", "jpg", "jpeg", "bmp", "tiff", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    processed_docs = st.session_state.get("processed_docs", [])

    if uploaded_files:
        new_names = {f.name for f in uploaded_files}
        existing  = {d["name"] for d in processed_docs}
        to_add    = [f for f in uploaded_files if f.name not in existing]

        if to_add:
            with st.sidebar:
                prog = st.progress(0, text="Processing documents…")
                for i, f in enumerate(to_add):
                    prog.progress((i + 1) / len(to_add), text=f"Processing {f.name}…")
                    doc = process_uploaded_file(f)
                    vs.add_documents(doc["chunks"], metadata={"source": doc["name"], "type": doc["metadata"]["type"]})
                    processed_docs.append(doc)
                prog.empty()
                st.success(f"✅ {len(to_add)} document(s) indexed!")
            st.session_state["processed_docs"] = processed_docs

    # ── Doc list ──────────────────────────────────────────────────────────────
    if processed_docs:
        st.sidebar.divider()
        st.sidebar.markdown("### 📄 Indexed Documents")
        for doc in processed_docs:
            icon = {"pdf": "📕", "docx": "📘", "image": "🖼️", "text": "📄"}.get(doc["metadata"]["type"], "📄")
            st.sidebar.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; padding:6px 8px;
                        background:rgba(99,102,241,0.07); border-radius:8px; margin:4px 0;
                        border:1px solid rgba(99,102,241,0.15);">
                <span>{icon}</span>
                <span style="font-size:0.82rem; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;">
                    {doc['name']}</span>
                <span style="margin-left:auto; font-size:0.7rem; color:#6366f1; white-space:nowrap;">
                    {doc['chunk_count']} chunks</span>
            </div>
            """, unsafe_allow_html=True)

        if st.sidebar.button("🗑️ Clear All", use_container_width=True):
            vs.clear()
            st.session_state["processed_docs"] = []
            st.session_state["chat_history"]   = []
            st.rerun()

    # ── Settings ──────────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Settings")
    simple_mode = st.sidebar.toggle("🗣️ Layman Mode", value=False,
                                    help="Explains answers in very simple, everyday language")
    use_api     = st.sidebar.toggle("⚡ Use Claude API", value=bool(st.session_state.get("api_key_set")),
                                    help="Use Claude API for best quality (requires API key in .env)")

    st.sidebar.divider()
    st.sidebar.markdown("""
    <div style="text-align:center; font-size:0.7rem; color:#334155; padding:0.5rem;">
        DocuMind v1.0 · RAG + Local LLM<br>
        <span style="color:#6366f1;">Offline-capable ✓</span>
    </div>
    """, unsafe_allow_html=True)

    return processed_docs, simple_mode, use_api