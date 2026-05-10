"""
DocuMind — Evaluation Tab UI
Shows accuracy metrics for the last answer or any custom Q&A pair.
"""

import os
import streamlit as st
from src.core.evaluator import evaluate_answer, rouge_l, semantic_similarity


def render_eval_tab(vs, processed_docs: list[dict], use_api: bool):
    st.markdown('<div class="dm-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Answer Quality Evaluation")
    st.markdown(
        "Measure how accurate and relevant DocuMind's answers are using "
        "automated classification metrics."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    has_groq  = bool(use_api and groq_key)

    if not has_groq:
        st.info(
            "💡 Add a **Groq API key** to your `.env` to enable LLM-judge metrics "
            "(Faithfulness, Answer Relevance). Retrieval metrics work without it."
        )

    if not processed_docs:
        st.warning("📂 Upload a document first.")
        return

    # ── Mode selector ─────────────────────────────────────────────────────────
    mode = st.radio(
        "Evaluation mode:",
        ["📋 Evaluate last answer", "✍️ Evaluate custom Q&A"],
        horizontal=True,
        key="eval_mode"
    )

    # ── Get the Q&A to evaluate ───────────────────────────────────────────────
    question, answer, context, chunks = "", "", "", []

    if mode == "📋 Evaluate last answer":
        history = st.session_state.get("chat_history", [])
        ai_msgs = [m for m in history if m["role"] == "assistant"]
        user_msgs = [m for m in history if m["role"] == "user"]

        if not ai_msgs:
            st.info("Ask a question in the Q&A Chat tab first, then come back here.")
            return

        # Show last N exchanges for selection
        st.markdown("**Select which answer to evaluate:**")
        pairs = list(zip(user_msgs[-5:], ai_msgs[-5:]))[::-1]
        labels = [f"Q: {p[0]['content'][:60]}…" for p in pairs]
        selected_idx = st.selectbox("", labels, key="eval_sel",
                                    label_visibility="collapsed")
        idx      = labels.index(selected_idx)
        question = pairs[idx][0]["content"]
        answer   = pairs[idx][1]["content"]
        context  = pairs[idx][1].get("context", "")
        chunks   = pairs[idx][1].get("sources", [])

        st.markdown("**Question:**")
        st.info(question)
        with st.expander("View answer"):
            st.markdown(answer)

    else:
        # Custom mode
        col1, col2 = st.columns(2)
        with col1:
            question = st.text_area(
                "Question:", height=80,
                placeholder="What is the main methodology?",
                key="eval_q"
            )
        with col2:
            answer = st.text_area(
                "AI Answer (paste here):", height=80,
                placeholder="Paste the answer you want to evaluate…",
                key="eval_a"
            )

        # Run retrieval to get context
        if question.strip() and vs.total_chunks > 0:
            chunks  = vs.search(question.strip(), top_k=8)
            context = "\n\n".join(c["text"] for c in chunks)

    # ── Optional reference answer ─────────────────────────────────────────────
    with st.expander("📎 Add reference answer (optional — enables ROUGE-L & semantic similarity)"):
        reference = st.text_area(
            "Your ideal / expected answer:",
            height=100,
            key="eval_ref",
            placeholder="Type what you consider the correct answer…"
        )

    # ── Run evaluation ────────────────────────────────────────────────────────
    reference = st.session_state.get("eval_ref", "")

    if st.button("🔬 Run Evaluation", key="run_eval_btn", use_container_width=True):
        if not question.strip() or not answer.strip():
            st.warning("Please provide both a question and an answer.")
            return

        with st.spinner("🔬 Computing metrics…"):
            result = evaluate_answer(
                question         = question,
                answer           = answer,
                context          = context,
                retrieved_chunks = chunks,
                reference_answer = reference
            )
        st.session_state["eval_result"]   = result
        st.session_state["eval_question"] = question
        st.session_state["eval_answer"]   = answer

    # ── Display results ───────────────────────────────────────────────────────
    if "eval_result" not in st.session_state:
        return

    result = st.session_state["eval_result"]
    _render_results(result)


def _render_results(result: dict):
    st.markdown("---")
    st.markdown("### 📊 Evaluation Results")

    # ── Overall grade ─────────────────────────────────────────────────────────
    overall = result.get("overall_score", 0.0)
    grade   = result.get("grade", "?")
    pct     = int(overall * 100)
    color   = (
        "#86efac" if pct >= 85 else
        "#fbbf24" if pct >= 70 else
        "#fb923c" if pct >= 55 else
        "#f87171"
    )
    label   = (
        "Excellent ✨" if pct >= 85 else
        "Good 👍"      if pct >= 70 else
        "Fair ⚠️"      if pct >= 55 else
        "Poor ❌"
    )

    st.markdown(
        f'<div class="dm-card" style="text-align:center;padding:1.5rem;">'
        f'<div style="font-size:3rem;font-weight:700;color:{color};">'
        f'Grade: {grade}</div>'
        f'<div style="font-size:1.2rem;color:{color};margin-top:4px;">'
        f'{pct}% — {label}</div>'
        f'<div style="color:#64748b;font-size:0.82rem;margin-top:6px;">'
        f'Weighted average across all available metrics</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Individual metric cards ───────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🤖 LLM-Judge Metrics")

        faith  = result.get("faithfulness", {})
        rel    = result.get("answer_relevance", {})

        _metric_card(
            "Faithfulness",
            faith.get("score"),
            faith.get("reason", ""),
            "Every claim in the answer is supported by the document context.",
            threshold=0.70
        )
        _metric_card(
            "Answer Relevance",
            rel.get("score"),
            rel.get("reason", ""),
            "The answer directly addresses what was asked.",
            threshold=0.70
        )

    with col2:
        st.markdown("#### 📐 Retrieval Metrics")

        prec  = result.get("context_precision", {})
        rec   = result.get("context_recall", {})

        _metric_card(
            "Context Precision",
            prec.get("score"),
            f"{prec.get('relevant_count', '?')}/{prec.get('total', '?')} chunks were relevant",
            "Fraction of retrieved chunks that were actually useful.",
            threshold=0.60
        )
        _metric_card(
            "Context Recall",
            rec.get("score"),
            "Semantic overlap between answer and retrieved context",
            "Did retrieval find all information needed to answer?",
            threshold=0.50
        )

    # ── Reference-based metrics ───────────────────────────────────────────────
    sem_sim  = result.get("semantic_similarity")
    rouge    = result.get("rouge_l")

    if sem_sim is not None or rouge is not None:
        st.markdown("#### 📏 Reference-Based Metrics")
        col3, col4 = st.columns(2)

        with col3:
            if sem_sim is not None:
                _metric_card(
                    "Semantic Similarity",
                    sem_sim,
                    "Cosine similarity between answer and reference embeddings",
                    "How close is the answer to your expected answer?",
                    threshold=0.60
                )

        with col4:
            if rouge is not None:
                _metric_card(
                    "ROUGE-L F1",
                    rouge.get("f1"),
                    f"Precision: {rouge.get('precision',0):.2f} | Recall: {rouge.get('recall',0):.2f}",
                    "Longest common subsequence overlap with reference answer.",
                    threshold=0.30
                )

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("#### 💡 Improvement Recommendations")
    _show_recommendations(result)

    # ── Download report ───────────────────────────────────────────────────────
    report = _build_report(result, st.session_state.get("eval_question",""),
                           st.session_state.get("eval_answer",""))
    st.download_button(
        "⬇️ Download Evaluation Report",
        data=report,
        file_name="documind_eval_report.txt",
        mime="text/plain",
        use_container_width=True
    )


def _metric_card(title: str, score, detail: str, description: str,
                 threshold: float = 0.60):
    if score is None:
        badge_html = '<span style="background:rgba(100,116,139,0.15);color:#64748b;font-size:0.75rem;padding:2px 8px;border-radius:6px;">Needs API key</span>'
        bar_html   = ''
        score_html = '<span style="color:#64748b;">N/A</span>'
    else:
        pct        = int(score * 100)
        bar_color  = "#86efac" if score >= threshold else "#fbbf24" if score >= threshold * 0.7 else "#f87171"
        badge_txt  = "Good" if score >= threshold else "Needs work"
        badge_col  = ("rgba(134,239,172,0.15)", "#16a34a") if score >= threshold else ("rgba(248,113,113,0.15)", "#dc2626")
        badge_html = f'<span style="background:{badge_col[0]};color:{badge_col[1]};font-size:0.75rem;padding:2px 8px;border-radius:6px;">{badge_txt}</span>'
        bar_html   = (
            f'<div style="height:6px;background:rgba(99,102,241,0.15);'
            f'border-radius:3px;margin:8px 0 4px;">'
            f'<div style="height:100%;width:{pct}%;background:{bar_color};'
            f'border-radius:3px;"></div></div>'
        )
        score_html = f'<span style="font-size:1.4rem;font-weight:700;color:{bar_color};">{pct}%</span>'

    st.markdown(
        f'<div class="dm-card" style="margin-bottom:10px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<b style="font-size:0.9rem;">{title}</b>{badge_html}</div>'
        f'{bar_html}'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'{score_html}'
        f'<span style="font-size:0.78rem;color:#64748b;text-align:right;max-width:65%;">{detail}</span>'
        f'</div>'
        f'<div style="font-size:0.78rem;color:#475569;margin-top:6px;">{description}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def _show_recommendations(result: dict):
    recs = []
    faith = (result.get("faithfulness") or {}).get("score")
    rel   = (result.get("answer_relevance") or {}).get("score")
    prec  = result.get("context_precision", {}).get("score", 1.0)
    rec   = result.get("context_recall",    {}).get("score", 1.0)

    if faith is not None and faith < 0.70:
        recs.append(("🔴 Low faithfulness",
                     "The answer may contain facts not in the document. "
                     "Increase `TOP_K_RETRIEVAL` in settings.py and lower LLM temperature to 0."))
    if rel is not None and rel < 0.70:
        recs.append(("🔴 Low answer relevance",
                     "The answer doesn't address the question well. "
                     "Improve the query expansion in `rag_engine.py` or rephrase the question."))
    if prec < 0.50:
        recs.append(("🟡 Low context precision",
                     "Too many irrelevant chunks are being retrieved. "
                     "Raise the score threshold in `vector_store.py` from 0.15 to 0.25."))
    if rec < 0.40:
        recs.append(("🟡 Low context recall",
                     "Retrieval is missing relevant content. "
                     "Increase `CHUNK_SIZE` to 1000 and `TOP_K_RETRIEVAL` to 12 in settings.py."))
    if not recs:
        recs.append(("✅ Everything looks good",
                     "All metrics are above their thresholds. "
                     "The system is performing well for this question."))

    for title, desc in recs:
        st.markdown(
            f'<div class="dm-card" style="margin-bottom:8px;">'
            f'<b style="font-size:0.88rem;">{title}</b>'
            f'<p style="color:#94a3b8;font-size:0.82rem;margin-top:4px;">{desc}</p>'
            f'</div>',
            unsafe_allow_html=True
        )


def _build_report(result: dict, question: str, answer: str) -> str:
    lines = [
        "DocuMind — Answer Evaluation Report",
        "=" * 50,
        f"Question:  {question[:200]}",
        f"Answer:    {answer[:300]}",
        "",
        f"OVERALL SCORE: {int(result.get('overall_score', 0)*100)}%  (Grade: {result.get('grade','?')})",
        "",
        "--- Metrics ---",
    ]

    def fmt(label, d, key="score"):
        if isinstance(d, dict):
            sc = d.get(key)
            reason = d.get("reason", "")
        else:
            sc = d
            reason = ""
        val = f"{int(sc*100)}%" if sc is not None else "N/A"
        return f"{label:<28} {val}  {reason}"

    lines += [
        fmt("Faithfulness",      result.get("faithfulness", {})),
        fmt("Answer Relevance",  result.get("answer_relevance", {})),
        fmt("Context Precision", result.get("context_precision", {})),
        fmt("Context Recall",    result.get("context_recall", {})),
    ]

    sem = result.get("semantic_similarity")
    if sem is not None:
        lines.append(f"{'Semantic Similarity':<28} {int(sem*100)}%")

    rouge = result.get("rouge_l")
    if rouge:
        lines.append(f"{'ROUGE-L F1':<28} {int(rouge['f1']*100)}%")
        lines.append(f"{'ROUGE-L Precision':<28} {int(rouge['precision']*100)}%")
        lines.append(f"{'ROUGE-L Recall':<28} {int(rouge['recall']*100)}%")

    return "\n".join(lines)