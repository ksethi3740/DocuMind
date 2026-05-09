"""
DocuMind — Semantic Search Tab
Search across ALL uploaded documents simultaneously.
"""

import re
import streamlit as st
from src.core.vector_store import DocuMindVectorStore


def render_search_tab(vs: DocuMindVectorStore, processed_docs: list[dict]):
    st.markdown('<div class="dm-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Semantic Document Search")
    st.markdown("Search across all uploaded documents simultaneously with AI-ranked results.")
    st.markdown("</div>", unsafe_allow_html=True)

    if not processed_docs:
        st.info("📂 Upload documents first.")
        return

    col_s, col_btn = st.columns([5, 1])
    with col_s:
        query = st.text_input(
            "Search:",
            placeholder="Search across all documents… e.g. 'drowsiness detection accuracy'",
            key="search_query",
            label_visibility="collapsed"
        )
    with col_btn:
        search_btn = st.button("🔍 Search", use_container_width=True, key="do_search")

    # Filter by doc type
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        doc_filter = st.multiselect(
            "Filter by document:",
            [d["name"] for d in processed_docs],
            default=[d["name"] for d in processed_docs],
            key="search_doc_filter"
        )
    with col_f2:
        top_k = st.selectbox("Results:", [5, 10, 15, 20], index=1, key="search_topk")

    if search_btn and query.strip():
        with st.spinner("🔍 Searching…"):
            results = vs.search(query.strip(), top_k=top_k * 2)

        # Filter by selected documents
        results = [
            r for r in results
            if r["metadata"].get("source", "") in doc_filter
        ][:top_k]

        if not results:
            st.warning("No relevant results found. Try different search terms.")
            return

        st.markdown(f"**Found {len(results)} relevant sections:**")
        st.markdown("---")

        for i, r in enumerate(results):
            score   = r["score"]
            source  = r["metadata"].get("source", "Document")
            text    = r["text"].strip()
            doc_type= r["metadata"].get("type", "").upper()
            pct     = min(int(score * 100), 100)
            color   = "#86efac" if pct >= 60 else "#fbbf24" if pct >= 35 else "#94a3b8"

            # Highlight query terms
            highlighted = _highlight(text[:350], query)

            st.markdown(
                f"""<div class="dm-card" style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:8px;">
                    <div>
                        <span style="font-weight:600;color:#a5b4fc;">📎 {source}</span>
                        {f'<span style="font-size:0.72rem;color:#64748b;margin-left:8px;">[{doc_type}]</span>' if doc_type else ''}
                    </div>
                    <span style="color:{color};font-size:0.78rem;font-weight:600;
                                 background:rgba(99,102,241,0.1);padding:2px 10px;
                                 border-radius:100px;">{pct}% match</span>
                </div>
                <div style="font-size:0.86rem;color:#cbd5e1;line-height:1.6;">
                    {highlighted}…
                </div>
                </div>""",
                unsafe_allow_html=True
            )

    elif search_btn:
        st.warning("Please enter a search query.")

    # Recent searches
    if "search_history" not in st.session_state:
        st.session_state["search_history"] = []

    if search_btn and query.strip():
        history = st.session_state["search_history"]
        if query not in history:
            history.insert(0, query)
            st.session_state["search_history"] = history[:10]

    if st.session_state.get("search_history"):
        st.markdown("---")
        st.markdown("**🕐 Recent searches:**")
        cols = st.columns(5)
        for i, past in enumerate(st.session_state["search_history"][:5]):
            with cols[i % 5]:
                if st.button(past[:25], key=f"hist_{i}"):
                    st.session_state["search_query"] = past
                    st.rerun()


def _highlight(text: str, query: str) -> str:
    """Highlight query terms in result text."""
    terms = [t for t in query.split() if len(t) > 2]
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text    = pattern.sub(
            f'<mark style="background:rgba(99,102,241,0.3);'
            f'color:#c7d2fe;border-radius:3px;padding:0 2px;">\\g<0></mark>',
            text
        )
    return text