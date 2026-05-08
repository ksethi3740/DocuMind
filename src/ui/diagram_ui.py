"""
Diagram & Flowchart tab UI.
"""

import streamlit as st
from src.utils.flowchart_generator import (
    render_mermaid,
    generate_flowchart_from_text,
    generate_research_flowchart,
    generate_concept_map
)


def render_diagram_tab(processed_docs: list[dict]):
    st.markdown('<div class="dm-card">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Visual Diagrams & Flowcharts")
    st.markdown("Automatically generate visual representations of your documents.")
    st.markdown("</div>", unsafe_allow_html=True)

    if not processed_docs:
        st.info("📂 Upload documents in the sidebar to generate diagrams.")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        doc_names = [d["name"] for d in processed_docs]
        selected_doc = st.selectbox("📄 Select Document", doc_names)
    with col2:
        diagram_type = st.selectbox("📊 Diagram Type", [
            "🔄 Process Flowchart",
            "📖 Research Paper Structure",
            "🧠 Concept Mind Map",
            "✍️ Custom Topic Flowchart"
        ])
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("⚡ Generate", use_container_width=True)

    custom_topic = ""
    if "Custom" in diagram_type:
        custom_topic = st.text_input("Enter topic / section to diagram:", placeholder="e.g. data preprocessing steps")

    # ── Generate ──────────────────────────────────────────────────────────────
    if generate_btn:
        doc_data = next((d for d in processed_docs if d["name"] == selected_doc), None)
        if not doc_data:
            st.error("Document not found.")
            return

        text = doc_data["text"]
        if custom_topic:
            # Find relevant section
            lower = text.lower()
            idx = lower.find(custom_topic.lower())
            text = text[max(0, idx):idx + 2000] if idx != -1 else text[:2000]

        with st.spinner("🎨 Generating diagram…"):
            if "Process" in diagram_type:
                mermaid_code = generate_flowchart_from_text(text, title=selected_doc)
            elif "Research" in diagram_type:
                mermaid_code = generate_research_flowchart(text)
            elif "Mind Map" in diagram_type:
                mermaid_code = generate_concept_map(text)
            else:
                mermaid_code = generate_flowchart_from_text(
                    text, title=custom_topic or "Custom Flow"
                )

        st.markdown('<div class="dm-card">', unsafe_allow_html=True)
        st.markdown(f"#### {diagram_type} — `{selected_doc}`")
        render_mermaid(mermaid_code)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📋 View Mermaid Code"):
            st.code(mermaid_code, language="text")
            st.download_button(
                "⬇️ Download Mermaid Code",
                data=mermaid_code,
                file_name="diagram.mmd",
                mime="text/plain"
            )