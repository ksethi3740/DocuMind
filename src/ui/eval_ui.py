"""
DocuMind — Complete Evaluation Tab UI
All metrics: confidence, BLEU, METEOR, MRR, NDCG, hallucination, coherence, completeness, BERTScore.
"""

import os
import re
import streamlit as st
from src.core.evaluator import evaluate_answer_full


def render_eval_tab(vs, processed_docs: list[dict], use_api: bool):
    st.markdown('<div class="dm-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Answer Quality Evaluation")
    st.markdown(
        "Complete accuracy assessment with 12 metrics — "
        "confidence score, hallucination rate, retrieval ranking, lexical and semantic overlap."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    has_groq  = bool(use_api and groq_key)

    if not has_groq:
        st.info("💡 Add `GROQ_API_KEY` to `.env` to enable LLM-judge metrics (Faithfulness, Answer Relevance).")

    if not processed_docs:
        st.warning("📂 Upload a document first.")
        return

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode = st.radio(
        "Evaluation mode:",
        ["📋 Evaluate last answer", "✍️ Evaluate custom Q&A"],
        horizontal=True,
        key="eval_mode"
    )

    question, answer, context, chunks = "", "", "", []

    if mode == "📋 Evaluate last answer":
        history   = st.session_state.get("chat_history", [])
        ai_msgs   = [m for m in history if m["role"] == "assistant"]
        user_msgs = [m for m in history if m["role"] == "user"]

        if not ai_msgs:
            st.info("Ask at least one question in the Q&A Chat tab first.")
            return

        pairs  = list(zip(user_msgs[-5:], ai_msgs[-5:]))[::-1]
        labels = [f"Q: {p[0]['content'][:70]}…" for p in pairs]
        sel    = st.selectbox("Select answer to evaluate:", labels,
                              key="eval_sel", label_visibility="visible")
        idx      = labels.index(sel)
        question = pairs[idx][0]["content"]
        answer   = pairs[idx][1]["content"]
        context  = pairs[idx][1].get("context", "")
        chunks   = pairs[idx][1].get("sources", [])

        st.markdown(f"**Question:** {question}")
        with st.expander("View answer"):
            st.markdown(answer)

    else:
        c1, c2 = st.columns(2)
        with c1:
            question = st.text_area("Question:", height=90,
                                    placeholder="What is the main methodology?",
                                    key="eval_q")
        with c2:
            answer = st.text_area("AI Answer:", height=90,
                                  placeholder="Paste the answer to evaluate…",
                                  key="eval_a")
        if question.strip() and vs.total_chunks > 0:
            chunks  = vs.search(question.strip(), top_k=8)
            context = "\n\n".join(c["text"] for c in chunks)

    # ── Reference answer ──────────────────────────────────────────────────────
    with st.expander("📎 Add reference answer (enables BLEU, METEOR, BERTScore, Exact Match, Semantic Similarity)"):
        reference = st.text_area(
            "Expected / ideal answer:",
            height=100,
            key="eval_ref",
            placeholder="Type or paste the correct answer you expect…"
        )

    reference = st.session_state.get("eval_ref", "")

    # ── BERTScore toggle ──────────────────────────────────────────────────────
    use_bertscore = st.checkbox(
        "🔬 Enable BERTScore (slower, requires `pip install bert-score`)",
        key="use_bs"
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    if st.button("🔬 Run Full Evaluation", key="run_eval_btn", use_container_width=True):
        if not question.strip() or not answer.strip():
            st.warning("Please provide both a question and an answer.")
            return

        with st.spinner("🔬 Computing all 12 metrics…"):
            result = evaluate_answer_full(
                question         = question,
                answer           = answer,
                context          = context,
                retrieved_chunks = chunks,
                reference_answer = reference
            )
            if not use_bertscore:
                result["bertscore"] = None

        st.session_state["eval_result"]   = result
        st.session_state["eval_question"] = question
        st.session_state["eval_answer"]   = answer

    if "eval_result" not in st.session_state:
        return

    _render_full_results(
        st.session_state["eval_result"],
        st.session_state.get("eval_question", ""),
        st.session_state.get("eval_answer", "")
    )


# ══ Results renderer ══════════════════════════════════════════════════════════

def _render_full_results(result: dict, question: str, answer: str):
    st.markdown("---")

    # ── Overall score + grade ─────────────────────────────────────────────────
    overall = result.get("overall_score", 0.0)
    grade   = result.get("grade", "?")
    pct     = int(overall * 100)
    g_color = (
        "#86efac" if pct >= 85 else
        "#fbbf24" if pct >= 70 else
        "#fb923c" if pct >= 55 else
        "#f87171"
    )
    g_label = (
        "Excellent ✨" if pct >= 85 else
        "Good 👍"      if pct >= 70 else
        "Fair ⚠️"      if pct >= 55 else
        "Poor ❌"
    )

    conf     = result.get("confidence", {})
    conf_pct = int(conf.get("score", 0.0) * 100)
    hall     = result.get("hallucination", {})
    hall_pct = int(hall.get("rate", 0.0) * 100)

    # Top summary row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="dm-card" style="text-align:center;">'
            f'<div style="font-size:2.2rem;font-weight:700;color:{g_color};">{grade}</div>'
            f'<div style="font-size:0.85rem;color:{g_color};">{pct}% — {g_label}</div>'
            f'<div style="color:#64748b;font-size:0.75rem;">Overall score</div>'
            f'</div>', unsafe_allow_html=True)
    with c2:
        conf_col = "#86efac" if conf_pct >= 65 else "#fbbf24" if conf_pct >= 45 else "#f87171"
        st.markdown(
            f'<div class="dm-card" style="text-align:center;">'
            f'<div style="font-size:2.2rem;font-weight:700;color:{conf_col};">{conf_pct}%</div>'
            f'<div style="font-size:0.85rem;color:{conf_col};">{conf.get("label","")}</div>'
            f'<div style="color:#64748b;font-size:0.75rem;">Confidence score</div>'
            f'</div>', unsafe_allow_html=True)
    with c3:
        hall_col = "#86efac" if hall_pct <= 10 else "#fbbf24" if hall_pct <= 30 else "#f87171"
        st.markdown(
            f'<div class="dm-card" style="text-align:center;">'
            f'<div style="font-size:2.2rem;font-weight:700;color:{hall_col};">{hall_pct}%</div>'
            f'<div style="font-size:0.85rem;color:{hall_col};">{hall.get("label","")}</div>'
            f'<div style="color:#64748b;font-size:0.75rem;">Hallucination rate</div>'
            f'</div>', unsafe_allow_html=True)
    with c4:
        mrr      = result.get("mrr", {})
        mrr_pct  = int(mrr.get("mrr", 0.0) * 100)
        mrr_col  = "#86efac" if mrr_pct >= 80 else "#fbbf24" if mrr_pct >= 50 else "#f87171"
        st.markdown(
            f'<div class="dm-card" style="text-align:center;">'
            f'<div style="font-size:2.2rem;font-weight:700;color:{mrr_col};">{mrr_pct}%</div>'
            f'<div style="font-size:0.85rem;color:{mrr_col};">{mrr.get("label","")}</div>'
            f'<div style="color:#64748b;font-size:0.75rem;">MRR (retrieval rank)</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 1: LLM Judge + Retrieval ──────────────────────────────────────
    st.markdown("#### 🤖 LLM-Judge Metrics")
    col_a, col_b = st.columns(2)
    with col_a:
        _mcard("Faithfulness",
               result.get("faithfulness",{}).get("score"),
               result.get("faithfulness",{}).get("reason",""),
               "Every claim is supported by the document.",
               threshold=0.70)
    with col_b:
        _mcard("Answer Relevance",
               result.get("answer_relevance",{}).get("score"),
               result.get("answer_relevance",{}).get("reason",""),
               "Answer directly addresses the question.",
               threshold=0.70)

    st.markdown("#### 📐 Retrieval Metrics")
    col_c, col_d = st.columns(2)
    prec = result.get("context_precision", {})
    rec  = result.get("context_recall", {})
    ndcg = result.get("ndcg", {})
    with col_c:
        _mcard("Context Precision",
               prec.get("score"),
               f"{prec.get('relevant_count','?')}/{prec.get('total','?')} chunks relevant",
               "Fraction of retrieved chunks that were useful.")
        _mcard("NDCG@K",
               ndcg.get("ndcg"),
               f"K={ndcg.get('k','?')} | {ndcg.get('label','')}",
               "Ranking quality of retrieved chunks (1.0 = perfect order).",
               threshold=0.75)
    with col_d:
        _mcard("Context Recall",
               rec.get("score"),
               "Semantic overlap: answer ↔ context",
               "Did retrieval find all needed information?",
               threshold=0.50)
        frr = mrr.get("first_relevant_rank")
        _mcard("MRR",
               mrr.get("mrr"),
               f"First relevant result at rank {frr}" if frr else "No relevant chunk found",
               "Mean Reciprocal Rank of first relevant retrieval result.",
               threshold=0.70)

    # ── Section 2: Answer quality ──────────────────────────────────────────────
    st.markdown("#### 📝 Answer Quality Metrics")
    col_e, col_f = st.columns(2)
    comp = result.get("completeness", {})
    coh  = result.get("coherence", {})
    with col_e:
        _mcard("Answer Completeness",
               comp.get("score"),
               f"Covered: {', '.join(comp.get('covered',[])[:4])} | Missing: {', '.join(comp.get('missing',[])[:3])}",
               "Does the answer cover all key topics from the question?",
               threshold=0.70)
    with col_f:
        _mcard("Coherence",
               coh.get("score"),
               f"Sentence flow: {coh.get('label','')}",
               "Sentence-to-sentence semantic consistency.",
               threshold=0.60)

    # ── Confidence breakdown ───────────────────────────────────────────────────
    st.markdown("#### 🔬 Confidence Score Breakdown")
    comp_data = conf.get("components", {})
    if comp_data:
        labels_map = {
            "retrieval_strength": ("Retrieval strength",  "How high the cosine scores of retrieved chunks are"),
            "answer_length":      ("Answer length",        "Is the answer long enough to be complete?"),
            "source_overlap":     ("Source-answer overlap","How many answer words appear in the source context"),
            "query_alignment":    ("Query-answer alignment","Semantic similarity between question and answer"),
        }
        col_g, col_h = st.columns(2)
        items = list(comp_data.items())
        for i, (k, v) in enumerate(items):
            label, desc = labels_map.get(k, (k, ""))
            pct_v = int(v * 100)
            col   = col_g if i % 2 == 0 else col_h
            bar_c = "#86efac" if v >= 0.65 else "#fbbf24" if v >= 0.45 else "#f87171"
            col.markdown(
                f'<div class="dm-card" style="margin-bottom:8px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<b style="font-size:0.85rem;">{label}</b>'
                f'<span style="font-size:1.1rem;font-weight:700;color:{bar_c};">{pct_v}%</span>'
                f'</div>'
                f'<div style="height:5px;background:rgba(99,102,241,0.15);border-radius:3px;margin:6px 0 4px;">'
                f'<div style="height:100%;width:{pct_v}%;background:{bar_c};border-radius:3px;"></div></div>'
                f'<div style="font-size:0.78rem;color:#64748b;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Section 3: Reference-based ─────────────────────────────────────────────
    rouge  = result.get("rouge_l")
    bleu   = result.get("bleu")
    meteor = result.get("meteor")
    em     = result.get("exact_match")
    sem    = result.get("semantic_similarity")
    bscore = result.get("bertscore")

    has_ref = any(x is not None for x in [rouge, bleu, meteor, em, sem, bscore])
    if has_ref:
        st.markdown("#### 📏 Reference-Based Metrics")
        col_i, col_j = st.columns(2)

        with col_i:
            if sem is not None:
                _mcard("Semantic Similarity",
                       sem,
                       "MiniLM cosine distance to reference answer",
                       "How close is the answer to your expected answer?",
                       threshold=0.60)
            if rouge:
                _mcard("ROUGE-L F1",
                       rouge.get("f1"),
                       f"Precision {int(rouge.get('precision',0)*100)}% | Recall {int(rouge.get('recall',0)*100)}%",
                       "Longest common subsequence token overlap.",
                       threshold=0.30)
            if bleu:
                _mcard("BLEU-4",
                       bleu.get("bleu"),
                       f"BLEU-1: {int(bleu.get('bleu_1',0)*100)}% | BLEU-2: {int(bleu.get('bleu_2',0)*100)}%",
                       "4-gram precision against reference answer.",
                       threshold=0.20)

        with col_j:
            if meteor is not None:
                _mcard("METEOR",
                       meteor,
                       "Stemming + synonym-aware precision/recall",
                       "More robust than BLEU for short or paraphrased answers.",
                       threshold=0.30)
            if em:
                ef1_col = "#86efac" if em.get("token_f1",0) >= 0.60 else "#fbbf24"
                st.markdown(
                    f'<div class="dm-card" style="margin-bottom:10px;">'
                    f'<b style="font-size:0.9rem;">Exact Match</b><br>'
                    f'<span style="color:{"#86efac" if em.get("exact") else "#f87171"};">'
                    f'{"✅ Exact match" if em.get("exact") else "❌ Not exact"}</span> &nbsp;'
                    f'<span style="color:{"#fbbf24" if em.get("contains") else "#94a3b8"};">'
                    f'{"✅ Contains reference" if em.get("contains") else "○ Does not contain"}</span><br>'
                    f'<span style="font-size:0.82rem;color:#64748b;">Token F1: '
                    f'<b style="color:{ef1_col};">{int(em.get("token_f1",0)*100)}%</b></span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            if bscore is not None:
                _mcard("BERTScore F1",
                       bscore,
                       "BERT token-level semantic similarity",
                       "Best available semantic overlap metric.",
                       threshold=0.60)

    # ── Recommendations ────────────────────────────────────────────────────────
    st.markdown("#### 💡 Recommendations")
    _recommendations(result)

    # ── Download ───────────────────────────────────────────────────────────────
    report = _build_full_report(result, question, answer)
    st.download_button(
        "⬇️ Download Full Evaluation Report",
        data=report,
        file_name="documind_eval_report.txt",
        mime="text/plain",
        use_container_width=True
    )


def _mcard(title, score, detail, description, threshold=0.60):
    if score is None:
        st.markdown(
            f'<div class="dm-card" style="margin-bottom:10px;opacity:0.6;">'
            f'<b style="font-size:0.9rem;">{title}</b> '
            f'<span style="font-size:0.75rem;color:#64748b;">— needs API key or reference answer</span>'
            f'<p style="font-size:0.78rem;color:#64748b;margin-top:4px;">{description}</p>'
            f'</div>', unsafe_allow_html=True)
        return

    pct      = int(score * 100)
    bar_c    = "#86efac" if score >= threshold else "#fbbf24" if score >= threshold * 0.75 else "#f87171"
    badge_bg = "rgba(134,239,172,0.15)" if score >= threshold else "rgba(248,113,113,0.15)"
    badge_fc = "#16a34a" if score >= threshold else "#dc2626"
    badge_tx = "Good" if score >= threshold else "Needs work"

    st.markdown(
        f'<div class="dm-card" style="margin-bottom:10px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<b style="font-size:0.9rem;">{title}</b>'
        f'<span style="background:{badge_bg};color:{badge_fc};font-size:0.73rem;'
        f'padding:2px 8px;border-radius:6px;">{badge_tx}</span>'
        f'</div>'
        f'<div style="height:5px;background:rgba(99,102,241,0.15);border-radius:3px;margin:7px 0 4px;">'
        f'<div style="height:100%;width:{pct}%;background:{bar_c};border-radius:3px;"></div></div>'
        f'<div style="display:flex;justify-content:space-between;">'
        f'<span style="font-size:1.3rem;font-weight:700;color:{bar_c};">{pct}%</span>'
        f'<span style="font-size:0.78rem;color:#64748b;text-align:right;max-width:68%;">{detail}</span>'
        f'</div>'
        f'<div style="font-size:0.78rem;color:#475569;margin-top:5px;">{description}</div>'
        f'</div>', unsafe_allow_html=True)


def _recommendations(result):
    recs = []

    conf_sc  = result.get("confidence",  {}).get("score",  1.0)
    hall_rt  = result.get("hallucination",{}).get("rate",  0.0)
    prec_sc  = result.get("context_precision",{}).get("score", 1.0)
    rec_sc   = result.get("context_recall",   {}).get("score", 1.0)
    comp_sc  = result.get("completeness",{}).get("score",  1.0)
    coh_sc   = result.get("coherence",   {}).get("score",  1.0)
    faith_sc = (result.get("faithfulness") or {}).get("score")
    rel_sc   = (result.get("answer_relevance") or {}).get("score")
    ndcg_sc  = result.get("ndcg",  {}).get("ndcg",  1.0)
    mrr_sc   = result.get("mrr",   {}).get("mrr",   1.0)

    if hall_rt > 0.30:
        recs.append(("🔴 High hallucination rate",
                     f"{int(hall_rt*100)}% of answer sentences lack context support. "
                     "Lower LLM temperature to 0 and increase TOP_K_RETRIEVAL to 12."))
    if faith_sc is not None and faith_sc < 0.65:
        recs.append(("🔴 Low faithfulness",
                     "Answer contains claims not in the document. "
                     "Use a stricter prompt: add 'Answer ONLY from the context provided.'"))
    if rel_sc is not None and rel_sc < 0.65:
        recs.append(("🔴 Low answer relevance",
                     "Answer doesn't address the question well. "
                     "Improve query expansion or rephrase the question more specifically."))
    if conf_sc < 0.45:
        recs.append(("🟡 Low confidence score",
                     "Retrieval scores are weak and/or answer is too short. "
                     "Try uploading a better quality document or asking a more specific question."))
    if prec_sc < 0.50:
        recs.append(("🟡 Low context precision",
                     "Too many irrelevant chunks retrieved. "
                     "Increase score threshold in vector_store.py from 0.15 → 0.25."))
    if rec_sc < 0.40:
        recs.append(("🟡 Low context recall",
                     "Retrieval missing relevant content. "
                     "Increase CHUNK_SIZE to 1000 and TOP_K_RETRIEVAL to 12 in settings.py."))
    if ndcg_sc < 0.65:
        recs.append(("🟡 Poor retrieval ranking (NDCG)",
                     "The most relevant chunks are not ranked first. "
                     "The RRF re-ranking in rag_engine.py should improve this — ensure it is active."))
    if comp_sc < 0.60:
        missing = result.get("completeness",{}).get("missing",[])
        recs.append(("🟡 Incomplete answer",
                     f"Missing topics: {', '.join(missing[:4])}. "
                     "Ask a follow-up question covering those specific topics."))
    if coh_sc < 0.45:
        recs.append(("⚪ Low coherence",
                     "Answer sentences are not well-connected. "
                     "This is usually improved by using Groq or Gemini instead of local LLM."))

    if not recs:
        recs.append(("✅ All metrics look good",
                     "Every metric is above its threshold. The system is performing well."))

    for title, desc in recs:
        color = "#f87171" if "🔴" in title else "#fbbf24" if "🟡" in title else "#86efac"
        st.markdown(
            f'<div class="dm-card" style="border-left:3px solid {color};margin-bottom:8px;">'
            f'<b style="font-size:0.88rem;">{title}</b>'
            f'<p style="color:#94a3b8;font-size:0.82rem;margin-top:4px;">{desc}</p>'
            f'</div>', unsafe_allow_html=True)


def _build_full_report(result, question, answer) -> str:
    def pct(v):
        return f"{int(v*100)}%" if v is not None else "N/A"

    lines = [
        "DocuMind — Full Evaluation Report",
        "=" * 60,
        f"Question:      {question[:200]}",
        f"Answer:        {answer[:300]}",
        "",
        f"OVERALL SCORE: {pct(result.get('overall_score'))}  Grade: {result.get('grade','?')}",
        "",
        "--- Core Metrics ---",
        f"{'Faithfulness':<30} {pct(result.get('faithfulness',{}).get('score'))}",
        f"{'Answer Relevance':<30} {pct(result.get('answer_relevance',{}).get('score'))}",
        f"{'Context Precision':<30} {pct(result.get('context_precision',{}).get('score'))}",
        f"{'Context Recall':<30} {pct(result.get('context_recall',{}).get('score'))}",
        "",
        "--- Confidence & Quality ---",
        f"{'Confidence Score':<30} {pct(result.get('confidence',{}).get('score'))}  [{result.get('confidence',{}).get('label','')}]",
        f"{'Hallucination Rate':<30} {pct(result.get('hallucination',{}).get('rate'))}  [{result.get('hallucination',{}).get('label','')}]",
        f"{'Answer Completeness':<30} {pct(result.get('completeness',{}).get('score'))}",
        f"{'Coherence':<30} {pct(result.get('coherence',{}).get('score'))}",
        "",
        "--- Retrieval Ranking ---",
        f"{'MRR':<30} {pct(result.get('mrr',{}).get('mrr'))}  Rank {result.get('mrr',{}).get('first_relevant_rank','?')}",
        f"{'NDCG@K':<30} {pct(result.get('ndcg',{}).get('ndcg'))}",
    ]

    rouge  = result.get("rouge_l")
    bleu   = result.get("bleu")
    meteor = result.get("meteor")
    sem    = result.get("semantic_similarity")
    em     = result.get("exact_match")
    bs     = result.get("bertscore")

    if any(x is not None for x in [rouge, bleu, meteor, sem, em, bs]):
        lines += ["", "--- Reference-Based ---"]
        if sem    is not None: lines.append(f"{'Semantic Similarity':<30} {pct(sem)}")
        if rouge:              lines.append(f"{'ROUGE-L F1':<30} {pct(rouge.get('f1'))}")
        if bleu:               lines.append(f"{'BLEU-4':<30} {pct(bleu.get('bleu'))}")
        if meteor is not None: lines.append(f"{'METEOR':<30} {pct(meteor)}")
        if em:                 lines.append(f"{'Token F1 (EM)':<30} {pct(em.get('token_f1'))}")
        if bs     is not None: lines.append(f"{'BERTScore F1':<30} {pct(bs)}")

    return "\n".join(lines)