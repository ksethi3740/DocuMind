# """
# DocuMind — Main UI Layout
# """
"""
DocuMind — Main UI Layout (Fixed)
"""

import os
import re
import streamlit as st

from src.core.vector_store import DocuMindVectorStore

from src.ui.sidebar import render_sidebar
from src.ui.chat_ui import render_chat
from src.ui.diagram_ui import render_diagram_tab
from src.ui.quiz_ui import render_quiz_tab
from src.ui.analytics_ui import render_analytics_tab
from src.ui.search_ui import render_search_tab
from src.ui.eval_ui import render_eval_tab

def render_main_ui():
    # ── Session state init ────────────────────────────────────────────────────
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = DocuMindVectorStore()
    if "processed_docs" not in st.session_state:
        st.session_state["processed_docs"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "search_history" not in st.session_state:
        st.session_state["search_history"] = []

    # ── Get vector store FIRST ────────────────────────────────────────────────
    vs = st.session_state["vector_store"]

    # ── Sidebar (returns docs + settings) ────────────────────────────────────
    processed_docs, simple_mode, use_api = render_sidebar(vs)

    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="dm-hero">
        <h1>🧠 DocuMind</h1>
        <p>Intelligent Document Analysis · RAG-Powered · Works Offline</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats bar ─────────────────────────────────────────────────────────────
    if processed_docs:
        has_api = bool(
            os.environ.get("GROQ_API_KEY",      "").strip() or
            os.environ.get("GEMINI_API_KEY",    "").strip() or
            os.environ.get("ANTHROPIC_API_KEY", "").strip()
        )
        mode_txt = "⚡ API"   if (use_api and has_api) else "🔒 Local"
        mode_col = "#fbbf24"  if (use_api and has_api) else "#86efac"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="dm-card" style="text-align:center;">'
                f'<div style="font-size:1.8rem;font-weight:700;color:#818cf8;">'
                f'{len(processed_docs)}</div>'
                f'<div style="color:#64748b;font-size:0.8rem;">Documents</div>'
                f'</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="dm-card" style="text-align:center;">'
                f'<div style="font-size:1.8rem;font-weight:700;color:#06b6d4;">'
                f'{vs.total_chunks:,}</div>'
                f'<div style="color:#64748b;font-size:0.8rem;">Indexed Chunks</div>'
                f'</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(
                f'<div class="dm-card" style="text-align:center;">'
                f'<div style="font-size:1.8rem;font-weight:700;color:#8b5cf6;">'
                f'{len(st.session_state["chat_history"]) // 2}</div>'
                f'<div style="color:#64748b;font-size:0.8rem;">Questions Asked</div>'
                f'</div>', unsafe_allow_html=True)
        with c4:
            st.markdown(
                f'<div class="dm-card" style="text-align:center;">'
                f'<div style="font-size:1.5rem;font-weight:700;color:{mode_col};">'
                f'{mode_txt}</div>'
                f'<div style="color:#64748b;font-size:0.8rem;">AI Mode</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_qa, tab_diagram, tab_summary, tab_search, tab_quiz, tab_analytics, tab_eval, tab_about = st.tabs([
    "💬 Q&A Chat",
    "🗺️ Diagrams",
    "📋 Summary",
    "🔍 Search",
    "📝 Quiz",
    "📊 Analytics",
    "🎯 Evaluate",
    "ℹ️ About"
])

    with tab_qa:
        render_chat(vs, simple_mode=simple_mode, use_api=use_api)

    with tab_diagram:
        render_diagram_tab(processed_docs)

    with tab_summary:
        _render_summary_tab(vs, processed_docs, simple_mode, use_api)

    with tab_search:
        render_search_tab(vs, processed_docs)

    with tab_quiz:
        render_quiz_tab(processed_docs, use_api)

    with tab_analytics:
        from src.ui.analytics_ui import render_analytics_tab
        render_analytics_tab(processed_docs)
        
    with tab_about:
        _render_about()
    with tab_eval:
        render_eval_tab(vs, processed_docs, use_api)



# ══ Summary tab ════════════════════════════════════════════════════════════════

def _render_summary_tab(vs, processed_docs, simple_mode, use_api):
    st.markdown('<div class="dm-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Document Summary")
    st.markdown("Generate a complete intelligent summary of your document.")
    st.markdown("</div>", unsafe_allow_html=True)

    if not processed_docs:
        st.info("📂 Upload documents first.")
        return

    doc_names = [d["name"] for d in processed_docs]
    selected  = st.selectbox("Choose document:", doc_names, key="sum_select")
    mode      = st.radio("Summary style:",
                         ["📚 Detailed", "🗣️ Simple / Layman"], horizontal=True)

    if st.button("✨ Generate Summary", key="gen_summary_btn"):
        doc_data  = next((d for d in processed_docs if d["name"] == selected), None)
        is_simple = "Simple" in mode

        if not doc_data:
            st.error("Document not found.")
            return

        full_text = doc_data.get("text", "")

        with st.spinner("✨ Generating comprehensive summary…"):
            groq_key   = os.environ.get("GROQ_API_KEY",   "").strip()
            gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

            if use_api and (groq_key or gemini_key):
                summary = _generate_api_summary(full_text, is_simple,
                                                groq_key, gemini_key)
            else:
                summary = _generate_local_summary(full_text, is_simple)

        st.markdown('<div class="dm-card">', unsafe_allow_html=True)
        st.markdown(summary)
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download TXT", data=summary,
                               file_name=f"summary_{selected}.txt",
                               mime="text/plain")
        with c2:
            st.download_button("⬇️ Download MD", data=summary,
                               file_name=f"summary_{selected}.md",
                               mime="text/markdown")


def _generate_api_summary(full_text: str, simple_mode: bool,
                           groq_key: str, gemini_key: str) -> str:
    tone = (
        "Use very simple everyday language. Short sentences. No jargon."
        if simple_mode else
        "Be thorough. Use ## headings and bullet points for each section."
    )
    context = full_text[:6000]
    prompt  = f"""You are DocuMind, an expert document analyst.

Generate a COMPREHENSIVE SUMMARY of the document below.
Cover ALL of these sections:
1. **What is this document about** — main topic, field, purpose
2. **Objectives** — what the authors/writers aim to achieve
3. **Methodology / Approach** — how they went about it
4. **Key Results or Content** — important findings, numbers, key points
5. **Conclusions** — what was concluded, recommendations, future directions

{tone}

DOCUMENT TEXT:
{context}

COMPREHENSIVE SUMMARY:"""

    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp   = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            result = (resp.choices[0].message.content or "").strip()
            if len(result) > 100:
                return result
        except Exception as e:
            print(f"[Summary] Groq error: {e}")

    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            resp   = client.models.generate_content(
                model="gemini-1.5-flash", contents=prompt)
            result = (resp.text or "").strip()
            if len(result) > 100:
                return result
        except Exception as e:
            print(f"[Summary] Gemini error: {e}")

    return _generate_local_summary(full_text, simple_mode)


def _generate_local_summary(full_text: str, simple_mode: bool) -> str:
    text      = re.sub(r'\s+', ' ', full_text).strip()
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                 if len(s.strip()) > 30]

    if not sentences:
        return "Could not extract content from document."

    stopwords = {
        'the','and','for','that','this','with','from','are','was','were',
        'is','in','on','at','to','a','an','of','it','its','be','been',
        'have','has','had','will','would','could','should','may','might',
        'also','which','their','they','we','our','using','by','as','or'
    }

    sections = {
        "🎯 Overview & Objectives": [
            'propose','present','aim','objective','goal','purpose','address',
            'problem','research','study','paper','work','introduce','develop'
        ],
        "🔬 Methodology": [
            'method','approach','algorithm','technique','model','architecture',
            'network','train','implement','pipeline','process','extract',
            'classify','detect','layer','module','framework'
        ],
        "📊 Results & Findings": [
            'accuracy','result','achieve','performance','score','percent',
            'precision','recall','f1','show','demonstrate','obtain','improve',
            'outperform','evaluation','metric','finding'
        ],
        "🏁 Conclusions": [
            'conclude','conclusion','future','limitation','contribute',
            'significant','effective','novel','real-time','potential',
            'recommend','suggest','direction'
        ]
    }

    def score_sents(kws, n=3):
        kw_set = set(kws)
        scored = []
        for sent in sentences:
            words   = set(re.findall(r'\b\w+\b', sent.lower())) - stopwords
            overlap = len(words & kw_set)
            scored.append((overlap, sent))
        scored.sort(key=lambda x: -x[0])
        return [s for sc, s in scored if sc > 0][:n]

    if simple_mode:
        intro  = sentences[:3]
        answer = "## 📄 Simple Summary\n\n" + " ".join(intro) + "\n\n"
        for sec_name, kws in list(sections.items())[:2]:
            top = score_sents(kws, 1)
            if top:
                answer += f"**{sec_name.split()[-1]}:** {top[0]}\n\n"
        return answer + "\n💡 *Add API key for richer summaries*"

    answer = "## 📄 Document Summary\n\n"
    for sec_name, kws in sections.items():
        top = score_sents(kws, 3)
        if top:
            answer += f"### {sec_name}\n"
            for s in top:
                answer += f"- {s}\n"
            answer += "\n"

    return answer + "\n💡 *Add API key for richer summaries*"


# # ══ Search tab ═════════════════════════════════════════════════════════════════

# def _render_search_tab(vs, processed_docs):
#     st.markdown('<div class="dm-card">', unsafe_allow_html=True)
#     st.markdown("### 🔍 Semantic Document Search")
#     st.markdown("Search across all uploaded documents simultaneously with AI-ranked results.")
#     st.markdown("</div>", unsafe_allow_html=True)

#     if not processed_docs:
#         st.info("📂 Upload documents first.")
#         return

#     col_s, col_btn = st.columns([5, 1])
#     with col_s:
#         query = st.text_input(
#             "Search:", placeholder="Search across all documents…",
#             key="search_query", label_visibility="collapsed"
#         )
#     with col_btn:
#         search_btn = st.button("🔍", use_container_width=True, key="do_search")

#     col_f1, col_f2 = st.columns(2)
#     with col_f1:
#         doc_filter = st.multiselect(
#             "Filter by document:",
#             [d["name"] for d in processed_docs],
#             default=[d["name"] for d in processed_docs],
#             key="search_doc_filter"
#         )
#     with col_f2:
#         top_k = st.selectbox("Results:", [5, 10, 15, 20], index=1, key="search_topk")

#     if search_btn and query.strip():
#         with st.spinner("🔍 Searching…"):
#             results = vs.search(query.strip(), top_k=top_k * 2)

#         results = [
#             r for r in results
#             if r["metadata"].get("source", "") in doc_filter
#         ][:top_k]

#         # Save to history
#         history = st.session_state.get("search_history", [])
#         if query not in history:
#             history.insert(0, query)
#             st.session_state["search_history"] = history[:10]

#         if not results:
#             st.warning("No relevant results found. Try different search terms.")
#             return

#         st.markdown(f"**Found {len(results)} relevant sections:**")
#         st.markdown("---")

#         for r in results:
#             score    = r["score"]
#             source   = r["metadata"].get("source", "Document")
#             text     = r["text"].strip()
#             doc_type = r["metadata"].get("type", "").upper()
#             pct      = min(int(score * 100), 100)
#             color    = "#86efac" if pct >= 60 else "#fbbf24" if pct >= 35 else "#94a3b8"

#             # Highlight terms
#             highlighted = text[:350]
#             for term in query.split():
#                 if len(term) > 2:
#                     highlighted = re.sub(
#                         re.escape(term),
#                         f'<mark style="background:rgba(99,102,241,0.3);'
#                         f'color:#c7d2fe;border-radius:3px;padding:0 2px;">'
#                         f'{term}</mark>',
#                         highlighted, flags=re.IGNORECASE
#                     )
#                 doc_badge = ""

#                 if doc_type:
#                     doc_badge = (
#                         f'<span style="font-size:0.72rem;'
#                         f'color:#64748b;margin-left:8px;">'
#                         f'[{doc_type}]</span>'
#                     )


#                 st.markdown(
#                     f'<div class="dm-card" style="margin-bottom:10px;">'
                    
#                     f'<div style="display:flex;justify-content:space-between;'
#                     f'align-items:center;margin-bottom:8px;">'
                    
#                     f'<span style="font-weight:600;color:#a5b4fc;">'
#                     f'🗂 {source}{doc_badge}'
#                     f'</span>'
                    
#                     f'<span style="color:{color};font-size:0.78rem;'
#                     f'font-weight:600;background:rgba(99,102,241,0.1);'
#                     f'padding:2px 10px;border-radius:100px;">'
#                     f'{pct}% match'
#                     f'</span>'
                    
#                     f'</div>'
                    
#                     f'<div style="font-size:0.86rem;color:#cbd5e1;'
#                     f'line-height:1.6;">'
#                     f'{highlighted}...'
#                     f'</div>'
                    
#                     f'</div>',
                    
#                     unsafe_allow_html=True
#                 )

#     # Recent searches
#     if st.session_state.get("search_history"):
#         st.markdown("---")
#         st.markdown("**🕐 Recent searches:**")
#         cols = st.columns(5)
#         for i, past in enumerate(st.session_state["search_history"][:5]):
#             with cols[i % 5]:
#                 if st.button(past[:20], key=f"hist_{i}"):
#                     st.rerun()


# # ══ Quiz tab ═══════════════════════════════════════════════════════════════════

# def _render_quiz_tab(processed_docs, use_api):
#     st.markdown('<div class="dm-card">', unsafe_allow_html=True)
#     st.markdown("### 📝 Auto Quiz Generator")
#     st.markdown("Generate MCQ questions automatically from your document using Groq AI.")
#     st.markdown("</div>", unsafe_allow_html=True)

#     if not processed_docs:
#         st.info("📂 Upload a document first.")
#         return

#     groq_key = os.environ.get("GROQ_API_KEY", "").strip()
#     if not use_api or not groq_key:
#         st.warning("⚠️ Quiz generation needs a **Groq API key** in your `.env` file.")
#         st.code("GROQ_API_KEY=your_key_here")
#         return

#     doc_names = [d["name"] for d in processed_docs]
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         selected   = st.selectbox("📄 Document:", doc_names, key="quiz_doc")
#     with c2:
#         num_q      = st.selectbox("❓ Questions:", [3, 5, 8, 10], index=1)
#     with c3:
#         difficulty = st.selectbox("🎯 Difficulty:", ["Easy","Medium","Hard"], index=1)

#     topic = st.text_input(
#         "🔎 Focus topic (optional):",
#         placeholder="e.g. methodology, CNN architecture, results…",
#         key="quiz_topic"
#     )

#     if st.button("🎯 Generate Quiz", key="gen_quiz_btn", use_container_width=True):
#         doc_data = next((d for d in processed_docs if d["name"] == selected), None)
#         if not doc_data:
#             return

#         text = doc_data.get("text", "")
#         if topic.strip():
#             idx  = text.lower().find(topic.lower())
#             text = text[max(0, idx-100): idx+3000] if idx != -1 else text[:3000]
#         else:
#             text = text[:4000]

#         with st.spinner("🎯 Generating quiz questions…"):
#             quiz = _generate_quiz_questions(text, num_q, difficulty, topic, groq_key)

#         if quiz:
#             st.session_state["current_quiz"]   = quiz
#             st.session_state["quiz_answers"]   = {}
#             st.session_state["quiz_submitted"] = False
#         else:
#             st.error("Failed to generate quiz. Try again.")
#             return

#     if "current_quiz" not in st.session_state:
#         return

#     quiz      = st.session_state["current_quiz"]
#     answers   = st.session_state.get("quiz_answers", {})
#     submitted = st.session_state.get("quiz_submitted", False)

#     st.markdown("---")
#     st.markdown(f"**📋 Quiz — {len(quiz)} Questions**")

#     score = 0
#     for i, q in enumerate(quiz):
#         question = q.get("question", "")
#         options  = q.get("options", [])
#         correct  = q.get("correct", "")
#         explain  = q.get("explanation", "")

#         st.markdown(
#             f'<div class="dm-card"><b>Q{i+1}. {question}</b></div>',
#             unsafe_allow_html=True
#         )

#         if submitted:
#             user_ans = answers.get(i, "")
#             if user_ans == correct:
#                 st.success(f"✅ {user_ans}")
#                 score += 1
#             else:
#                 st.error(f"❌ Your answer: {user_ans}")
#                 st.success(f"✅ Correct: {correct}")
#             if explain:
#                 with st.expander("💡 Explanation"):
#                     st.write(explain)
#         else:
#             if options:
#                 choice       = st.radio(
#                     f"q{i}", options,
#                     key=f"quiz_q_{i}",
#                     label_visibility="collapsed"
#                 )
#                 answers[i]   = choice

#         st.markdown("---")

#     st.session_state["quiz_answers"] = answers

#     if not submitted:
#         if st.button("📤 Submit Answers", key="submit_quiz", use_container_width=True):
#             st.session_state["quiz_submitted"] = True
#             st.rerun()
#     else:
#         total = len(quiz)
#         pct   = int((score / total) * 100) if total else 0
#         color = "#86efac" if pct >= 70 else "#fbbf24" if pct >= 40 else "#f87171"
#         grade = "Excellent! 🏆" if pct >= 80 else "Good! 👍" if pct >= 60 else "Keep practising 📚"

#         st.markdown(
#             f'<div class="dm-card" style="text-align:center;">'
#             f'<div style="font-size:2.5rem;font-weight:700;color:{color};">'
#             f'{score}/{total}</div>'
#             f'<div style="font-size:1.1rem;color:#94a3b8;">{pct}% — {grade}</div>'
#             f'</div>',
#             unsafe_allow_html=True
#         )

#         lines  = [f"DocuMind Quiz — {score}/{total} ({pct}%)\n", "="*50, ""]
#         for i, q in enumerate(quiz):
#             lines += [
#                 f"Q{i+1}. {q['question']}",
#                 *[f"  {'✓' if o==q['correct'] else '○'} {o}" for o in q.get("options",[])],
#                 f"  Your answer: {answers.get(i,'—')}",
#                 f"  Explanation: {q.get('explanation','')}",
#                 ""
#             ]

#         c1, c2 = st.columns(2)
#         with c1:
#             st.download_button("⬇️ Download Results", data="\n".join(lines),
#                                file_name="quiz_results.txt", mime="text/plain")
#         with c2:
#             if st.button("🔄 New Quiz", use_container_width=True):
#                 for k in ["current_quiz","quiz_answers","quiz_submitted"]:
#                     st.session_state.pop(k, None)
#                 st.rerun()


def _generate_quiz_questions(text, num_q, difficulty, topic, groq_key):
    topic_note = f"Focus on: {topic}." if topic.strip() else ""
    prompt = f"""Generate exactly {num_q} MCQ questions. Difficulty: {difficulty}. {topic_note}

DOCUMENT:
{text}

Return ONLY a JSON array. No markdown. No explanation.
Format:
[{{"question":"...","options":["A","B","C","D"],"correct":"A","explanation":"..."}}]

Generate {num_q} questions (JSON only):"""

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        resp   = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=2500, temperature=0.2
        )
        raw   = (resp.choices[0].message.content or "").strip()
        raw   = re.sub(r'^```(?:json)?\s*','',raw)
        raw   = re.sub(r'```\s*$','',raw).strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        st.error(f"Quiz error: {e}")
    return []


# ══ Analytics tab ══════════════════════════════════════════════════════════════

# def _render_analytics_tab(processed_docs):
#     st.markdown('<div class="dm-card">', unsafe_allow_html=True)
#     st.markdown("### 📊 Document Analytics")
#     st.markdown("Visual analysis of your document content and structure.")
#     st.markdown("</div>", unsafe_allow_html=True)

#     if not processed_docs:
#         st.info("📂 Upload a document first.")
#         return

#     doc_names = [d["name"] for d in processed_docs]
#     selected  = st.selectbox("Choose document:", doc_names, key="analytics_doc")
#     doc_data  = next((d for d in processed_docs if d["name"] == selected), None)

#     if not doc_data:
#         return

#     text = doc_data.get("text", "")
#     if not text.strip():
#         st.warning("No text content found.")
#         return

#     stopwords = {
#         'the','and','for','that','this','with','from','are','was','were','is',
#         'in','on','at','to','a','an','of','it','its','be','been','have','has',
#         'had','will','would','could','should','may','might','also','which',
#         'their','they','we','our','using','by','as','or','but','not','all',
#         'can','into','than','then','these','those','about','after','such','each'
#     }

#     words     = re.findall(r'\b[a-zA-Z]\w+\b', text.lower())
#     sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+',text) if len(s.strip())>15]
#     paragraphs= [p.strip() for p in text.split('\n\n') if len(p.strip())>20]
#     unique_w  = set(words)

#     c1,c2,c3,c4,c5 = st.columns(5)
#     c1.metric("📝 Words",        f"{len(words):,}")
#     c2.metric("📄 Sentences",    f"{len(sentences):,}")
#     c3.metric("📑 Paragraphs",   f"{len(paragraphs):,}")
#     c4.metric("🔤 Unique Words", f"{len(unique_w):,}")
#     c5.metric("✂️ Chunks",       f"{doc_data.get('chunk_count',0)}")

#     st.markdown("---")

#     filtered = [w for w in words if w not in stopwords and len(w) > 3]
#     freq     = Counter(filtered)
#     top15    = freq.most_common(15)

#     if top15:
#         col_left, col_right = st.columns([2, 1])

#         with col_left:
#             st.markdown("**🔑 Top Keywords**")
#             try:
#                 import plotly.graph_objects as go
#                 labels = [w for w,_ in top15]
#                 counts = [c for _,c in top15]
#                 fig = go.Figure(go.Bar(
#                     x=counts[::-1], y=labels[::-1],
#                     orientation='h',
#                     marker=dict(
#                         color=counts[::-1],
#                         colorscale=[[0,'#4f46e5'],[0.5,'#7c3aed'],[1,'#06b6d4']],
#                         showscale=False
#                     ),
#                     text=[str(c) for c in counts[::-1]],
#                     textposition='outside',
#                     textfont=dict(size=11, color='#94a3b8')
#                 ))
#                 fig.update_layout(
#                     height=320,
#                     margin=dict(l=0,r=40,t=5,b=5),
#                     paper_bgcolor='rgba(0,0,0,0)',
#                     plot_bgcolor='rgba(0,0,0,0)',
#                     font=dict(color='#94a3b8',size=12),
#                     xaxis=dict(gridcolor='rgba(99,102,241,0.1)',zeroline=False),
#                     yaxis=dict(gridcolor='rgba(0,0,0,0)')
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
#             except ImportError:
#                 for word, count in top15[:8]:
#                     pct = int(count / top15[0][1] * 100)
#                     st.markdown(f"`{word}` {count}", unsafe_allow_html=True)

#         with col_right:
#             st.markdown("**📈 Document Stats**")
#             avg_sl    = round(len(words) / max(len(sentences),1), 1)
#             vocab_r   = round(len(unique_w) / max(len(words),1) * 100, 1)
#             doc_type  = doc_data.get("metadata",{}).get("type","?").upper()
#             stats = [
#                 ("Avg sentence",    f"{avg_sl} words"),
#                 ("Vocab richness",  f"{vocab_r}%"),
#                 ("File type",       doc_type),
#                 ("Total chars",     f"{len(text):,}"),
#             ]
#             for label, val in stats:
#                 st.markdown(
#                     f'<div style="display:flex;justify-content:space-between;'
#                     f'padding:5px 0;border-bottom:0.5px solid rgba(99,102,241,0.15);">'
#                     f'<span style="color:#64748b;font-size:0.82rem;">{label}</span>'
#                     f'<span style="color:#a5b4fc;font-size:0.82rem;font-weight:500;">{val}</span>'
#                     f'</div>',
#                     unsafe_allow_html=True
#                 )

#     st.markdown("---")

#     col_a, col_b = st.columns(2)
#     sent_lens = [len(s.split()) for s in sentences if 2 < len(s.split()) < 80]

#     with col_a:
#         st.markdown("**📏 Sentence Length Distribution**")
#         if sent_lens:
#             short  = sum(1 for l in sent_lens if l <= 10)
#             medium = sum(1 for l in sent_lens if 10 < l <= 25)
#             long_  = sum(1 for l in sent_lens if l > 25)
#             try:
#                 import plotly.graph_objects as go
#                 fig2 = go.Figure(go.Pie(
#                     labels=["Short ≤10","Medium 11-25","Long >25"],
#                     values=[short, medium, long_],
#                     hole=0.55,
#                     marker_colors=['#06b6d4','#6366f1','#8b5cf6'],
#                     textfont=dict(size=12)
#                 ))
#                 fig2.update_layout(
#                     height=220,
#                     margin=dict(l=0,r=0,t=5,b=5),
#                     paper_bgcolor='rgba(0,0,0,0)',
#                     font=dict(color='#94a3b8',size=11),
#                     showlegend=True,
#                     legend=dict(font=dict(color='#94a3b8',size=11))
#                 )
#                 st.plotly_chart(fig2, use_container_width=True)
#             except ImportError:
#                 c1,c2,c3 = st.columns(3)
#                 c1.metric("Short",  short)
#                 c2.metric("Medium", medium)
#                 c3.metric("Long",   long_)

#     with col_b:
#         st.markdown("**🧩 Detected Topics**")
#         t = text.lower()
#         topic_signals = {
#             "Machine Learning":   ['neural','deep learning','cnn','model','train'],
#             "Computer Vision":    ['image','frame','detection','camera','video'],
#             "Data Analysis":      ['dataset','data','analysis','statistic','sample'],
#             "Evaluation/Results": ['accuracy','precision','recall','f1','performance'],
#             "Healthcare":         ['patient','clinical','diagnosis','treatment'],
#             "Security/Safety":    ['security','safety','alert','risk','detect'],
#             "Business":           ['revenue','profit','market','strategy','growth'],
#             "Education":          ['student','learning','course','teaching'],
#         }
#         topic_scores = {}
#         for topic_name, signals in topic_signals.items():
#             cnt = sum(t.count(s) for s in signals)
#             if cnt > 0:
#                 topic_scores[topic_name] = cnt

#         if topic_scores:
#             max_sc = max(topic_scores.values())
#             sorted_topics = sorted(topic_scores.items(), key=lambda x: -x[1])
#             for topic_name, sc in sorted_topics[:6]:
#                 bar_w = int(sc / max_sc * 100)
#                 st.markdown(
#                     f'<div style="margin-bottom:8px;">'
#                     f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
#                     f'<span style="font-size:0.82rem;color:#e2e8f0;">{topic_name}</span>'
#                     f'<span style="font-size:0.75rem;color:#6366f1;">{bar_w}%</span></div>'
#                     f'<div style="height:5px;background:rgba(99,102,241,0.15);border-radius:3px;">'
#                     f'<div style="height:100%;width:{bar_w}%;'
#                     f'background:linear-gradient(90deg,#6366f1,#06b6d4);border-radius:3px;">'
#                     f'</div></div></div>',
#                     unsafe_allow_html=True
#                 )


# ══ About tab ══════════════════════════════════════════════════════════════════

def _render_about():
    st.markdown("""
    <div class="dm-card">
        <h3>🧠 About DocuMind</h3>
        <p style="color:#94a3b8;">
            DocuMind is a <b>RAG-based intelligent document analysis system</b> combining
            semantic search, multimodal processing, AI-powered Q&amp;A, quiz generation,
            analytics, and visual diagrams.
        </p>
    </div>
    """, unsafe_allow_html=True)

    features = [
        ("📄 Multimodal Input",    "PDF, DOCX, PNG, JPG, BMP, TIFF with full OCR."),
        ("🔍 Semantic Search",     "FAISS + sentence-transformers with RRF re-ranking."),
        ("🤖 Multi-API Support",   "Groq, Gemini, Claude — falls back to offline mode."),
        ("🗺️ Auto Flowcharts",     "Process flows, research maps, concept diagrams."),
        ("📝 Quiz Generator",      "Auto MCQs from any document using Groq Llama3."),
        ("🔍 Document Search",     "Search all docs simultaneously with highlights."),
        ("📊 Analytics",           "Keywords, topics, sentence distribution charts."),
        ("📎 Source Attribution",  "Ranked excerpts with relevance scores."),
    ]
    cols = st.columns(2)
    for i, (title, desc) in enumerate(features):
        with cols[i % 2]:
            st.markdown(
                f'<div class="dm-card">'
                f'<b style="color:#a5b4fc;">{title}</b>'
                f'<p style="color:#94a3b8;font-size:0.88rem;margin-top:0.3rem;">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
# import streamlit as st
# from src.core.vector_store import DocuMindVectorStore
# from src.ui.sidebar import render_sidebar
# from src.ui.chat_ui import render_chat
# from src.ui.diagram_ui import render_diagram_tab
# from src.ui.quiz_ui     import render_quiz_tab
# from src.ui.analytics_ui import render_analytics_tab
# from src.ui.search_ui   import render_search_tab


# def render_main_ui():


#     tab_qa, tab_diagram, tab_summary, tab_search, tab_quiz, tab_analytics, tab_about = st.tabs([
#         "💬 Q&A Chat",
#         "🗺️ Diagrams",
#         "📋 Summary",
#         "🔍 Search",
#         "📝 Quiz",
#         "📊 Analytics",
#         "ℹ️ About"
#     ])

#     with tab_qa:
#         render_chat(vs, simple_mode=simple_mode, use_api=use_api)
#     with tab_diagram:
#         render_diagram_tab(processed_docs)
#     with tab_summary:
#         _render_summary_tab(vs, processed_docs, simple_mode, use_api)
#     with tab_search:
#         render_search_tab(vs, processed_docs)
#     with tab_quiz:
#         render_quiz_tab(processed_docs, use_api)
#     with tab_analytics:
#         render_analytics_tab(processed_docs)
#     with tab_about:
#         _render_about()


# # ══ Summary tab — FIXED ════════════════════════════════════════════════════════

# def _render_summary_tab(vs, processed_docs, simple_mode, use_api):
#     st.markdown('<div class="dm-card">', unsafe_allow_html=True)
#     st.markdown("### 📋 Document Summary")
#     st.markdown("Generate a complete, intelligent summary of your document.")
#     st.markdown("</div>", unsafe_allow_html=True)

#     if not processed_docs:
#         st.info("📂 Upload documents first.")
#         return

#     doc_names = [d["name"] for d in processed_docs]
#     selected  = st.selectbox("Choose document:", doc_names, key="sum_select")
#     mode      = st.radio(
#         "Summary style:",
#         ["📚 Detailed", "🗣️ Simple / Layman"],
#         horizontal=True
#     )

#     if st.button("✨ Generate Summary", key="gen_summary_btn"):
#         doc_data  = next((d for d in processed_docs if d["name"] == selected), None)
#         is_simple = "Simple" in mode

#         if not doc_data:
#             st.error("Document not found.")
#             return

#         full_text = doc_data.get("text", "")

#         with st.spinner("✨ Generating comprehensive summary…"):
#             import os
#             groq_key   = os.environ.get("GROQ_API_KEY",   "").strip()
#             gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

#             if use_api and (groq_key or gemini_key):
#                 summary = _generate_api_summary(full_text, is_simple, groq_key, gemini_key)
#             else:
#                 summary = _generate_local_summary(full_text, is_simple)

#         st.markdown('<div class="dm-card">', unsafe_allow_html=True)
#         st.markdown(summary)
#         st.markdown("</div>", unsafe_allow_html=True)

#         c1, c2 = st.columns(2)
#         with c1:
#             st.download_button("⬇️ Download TXT", data=summary,
#                                file_name=f"summary_{selected}.txt", mime="text/plain")
#         with c2:
#             st.download_button("⬇️ Download MD", data=summary,
#                                file_name=f"summary_{selected}.md",
#                                mime="text/markdown")


# def _generate_api_summary(full_text: str, simple_mode: bool,
#                            groq_key: str, gemini_key: str) -> str:
#     """Generate summary using API — directly from full document text."""
#     tone = (
#         "Use very simple everyday language. Short sentences. No jargon. "
#         "Explain as if to someone with no background in the subject."
#         if simple_mode else
#         "Be thorough and well-structured. Use ## headings and bullet points."
#     )

#     # Take first 6000 chars of document for comprehensive coverage
#     context = full_text[:6000]

#     prompt = f"""You are DocuMind, an expert document analyst.

# Generate a COMPREHENSIVE SUMMARY of the document below.

# IMPORTANT: This is a SUMMARY request. Do NOT describe methodology steps only.
# Cover ALL of these sections:
# 1. **What is this document about** — topic, field, purpose
# 2. **Objectives** — what the authors aim to achieve
# 3. **Methodology** — brief overview of the approach used
# 4. **Key Results** — important findings, numbers, outcomes
# 5. **Conclusions** — what was concluded, future directions

# {tone}

# DOCUMENT TEXT:
# {context}

# COMPREHENSIVE SUMMARY:"""

#     # Try Groq first
#     if groq_key:
#         try:
#             from groq import Groq
#             client = Groq(api_key=groq_key)
#             resp   = client.chat.completions.create(
#                 model="llama-3.3-70b-versatile",
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=1500,
#                 temperature=0.1
#             )
#             result = resp.choices[0].message.content.strip()
#             if result and len(result) > 100:
#                 return result
#         except Exception as e:
#             print(f"[Summary] Groq error: {e}")

#     # Fallback to Gemini
#     if gemini_key:
#         try:
#             from google import genai
#             client = genai.Client(api_key=gemini_key)
#             resp   = client.models.generate_content(
#                 model="gemini-1.5-flash",
#                 contents=prompt
#             )
#             result = (resp.text or "").strip()
#             if result and len(result) > 100:
#                 return result
#         except Exception as e:
#             print(f"[Summary] Gemini error: {e}")

#     # Last resort — local
#     return _generate_local_summary(full_text, simple_mode)


# def _generate_local_summary(full_text: str, simple_mode: bool) -> str:
#     """Smart local summary — extracts from multiple document sections."""
#     import re

#     text = re.sub(r'\s+', ' ', full_text).strip()
#     sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
#                  if len(s.strip()) > 30]

#     if not sentences:
#         return "Could not extract content from document."

#     # Section-specific keyword groups
#     sections = {
#         "🎯 Overview & Objectives": [
#             'propose', 'present', 'aim', 'objective', 'goal', 'purpose',
#             'address', 'problem', 'research', 'study', 'paper', 'work',
#             'introduce', 'develop', 'system', 'framework', 'approach'
#         ],
#         "🔬 Methodology": [
#             'method', 'approach', 'algorithm', 'technique', 'model',
#             'architecture', 'network', 'train', 'implement', 'pipeline',
#             'process', 'extract', 'classify', 'detect', 'layer', 'module'
#         ],
#         "📊 Results & Findings": [
#             'accuracy', 'result', 'achieve', 'performance', 'score',
#             'percent', 'precision', 'recall', 'f1', 'show', 'demonstrate',
#             'obtain', 'improve', 'outperform', 'evaluation', 'metric'
#         ],
#         "🏁 Conclusions": [
#             'conclude', 'conclusion', 'future', 'limitation', 'contribute',
#             'significant', 'effective', 'novel', 'real-time', 'potential',
#             'recommend', 'suggest', 'direction', 'work'
#         ]
#     }

#     stopwords = {
#         'the','and','for','that','this','with','from','are','was',
#         'is','in','on','at','to','a','an','of','it','its','be',
#         'been','have','has','had','will','would','could','should'
#     }

#     def score_sents(kws):
#         kw_set = set(kws)
#         scored = []
#         for sent in sentences:
#             words   = set(re.findall(r'\b\w+\b', sent.lower())) - stopwords
#             overlap = len(words & kw_set)
#             scored.append((overlap, sent))
#         scored.sort(key=lambda x: -x[0])
#         return [s for sc, s in scored if sc > 0][:3]

#     if simple_mode:
#         intro = sentences[:3]
#         answer = "## 📄 Simple Summary\n\n"
#         answer += " ".join(intro) + "\n\n"
#         for sec_name, kws in list(sections.items())[:2]:
#             top = score_sents(kws)
#             if top:
#                 answer += f"**{sec_name.split()[-1]}:** {top[0]}\n\n"
#         return answer + "\n💡 *Add API key for better summaries*"

#     answer = "## 📄 Document Summary\n\n"
#     for sec_name, kws in sections.items():
#         top = score_sents(kws)
#         if top:
#             answer += f"### {sec_name}\n"
#             for s in top:
#                 answer += f"- {s}\n"
#             answer += "\n"

#     return answer + "\n💡 *Add API key for richer summaries*"


# # ══ Quiz tab — NEW FEATURE ═════════════════════════════════════════════════════

# def _render_quiz_tab(processed_docs, use_api):
#     st.markdown('<div class="dm-card">', unsafe_allow_html=True)
#     st.markdown("### 📝 Auto Quiz Generator")
#     st.markdown("Generate MCQ questions automatically from your document.")
#     st.markdown("</div>", unsafe_allow_html=True)

#     if not processed_docs:
#         st.info("📂 Upload documents first.")
#         return

#     import os
#     groq_key = os.environ.get("GROQ_API_KEY", "").strip()
#     if not (use_api and groq_key):
#         st.warning("⚠️ Quiz generation requires a Groq API key. Add it to your `.env` file.")
#         return

#     doc_names = [d["name"] for d in processed_docs]
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         selected = st.selectbox("Document:", doc_names, key="quiz_doc")
#     with col2:
#         num_q = st.selectbox("Questions:", [3, 5, 8, 10], index=1, key="quiz_num")
#     with col3:
#         difficulty = st.selectbox("Difficulty:", ["Easy", "Medium", "Hard"], index=1, key="quiz_diff")

#     if st.button("🎯 Generate Quiz", key="gen_quiz_btn"):
#         doc_data = next((d for d in processed_docs if d["name"] == selected), None)
#         if not doc_data:
#             return

#         text = doc_data.get("text", "")[:4000]

#         with st.spinner("🎯 Generating quiz questions…"):
#             quiz = _generate_quiz(text, num_q, difficulty, groq_key)

#         if quiz:
#             st.session_state["current_quiz"]   = quiz
#             st.session_state["quiz_answers"]    = {}
#             st.session_state["quiz_submitted"]  = False
#             st.session_state["quiz_score"]      = 0

#     # Display quiz
#     if "current_quiz" in st.session_state and st.session_state["current_quiz"]:
#         quiz     = st.session_state["current_quiz"]
#         answers  = st.session_state.get("quiz_answers", {})
#         submitted= st.session_state.get("quiz_submitted", False)

#         st.markdown("---")
#         score = 0

#         for i, q in enumerate(quiz):
#             st.markdown(f"**Q{i+1}. {q['question']}**")
#             options = q.get("options", [])
#             correct = q.get("correct", "")

#             if submitted:
#                 user_ans = answers.get(i, "")
#                 if user_ans == correct:
#                     st.success(f"✅ Your answer: {user_ans}")
#                     score += 1
#                 else:
#                     st.error(f"❌ Your answer: {user_ans}")
#                     st.info(f"✅ Correct: {correct}")
#             else:
#                 choice = st.radio(
#                     f"q_{i}",
#                     options,
#                     key=f"quiz_q_{i}",
#                     label_visibility="collapsed"
#                 )
#                 answers[i] = choice

#             st.markdown("---")

#         st.session_state["quiz_answers"] = answers

#         if not submitted:
#             if st.button("📤 Submit Quiz", key="submit_quiz"):
#                 st.session_state["quiz_submitted"] = True
#                 st.session_state["quiz_score"]     = score
#                 st.rerun()
#         else:
#             total = len(quiz)
#             pct   = int((score / total) * 100) if total else 0
#             color = "#86efac" if pct >= 70 else "#fbbf24" if pct >= 40 else "#f87171"
#             st.markdown(
#                 f'<div class="dm-card" style="text-align:center;">'
#                 f'<div style="font-size:2rem;font-weight:700;color:{color};">{score}/{total}</div>'
#                 f'<div style="color:#94a3b8;">Score: {pct}%</div>'
#                 f'</div>',
#                 unsafe_allow_html=True
#             )
#             if st.button("🔄 New Quiz", key="new_quiz"):
#                 for k in ["current_quiz","quiz_answers","quiz_submitted","quiz_score"]:
#                     st.session_state.pop(k, None)
#                 st.rerun()


# def _generate_quiz(text: str, num_q: int, difficulty: str, groq_key: str) -> list:
#     """Generate MCQ quiz using Groq."""
#     import json, re

#     prompt = f"""Generate exactly {num_q} multiple-choice questions from this document.
# Difficulty: {difficulty}

# DOCUMENT:
# {text}

# Return ONLY a valid JSON array. No explanation. No markdown. Example format:
# [
#   {{
#     "question": "What is the main purpose of the system?",
#     "options": ["Option A", "Option B", "Option C", "Option D"],
#     "correct": "Option A"
#   }}
# ]

# Generate {num_q} questions now:"""

#     try:
#         from groq import Groq
#         client = Groq(api_key=groq_key)
#         resp   = client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=2000,
#             temperature=0.3
#         )
#         raw  = resp.choices[0].message.content.strip()
#         # Extract JSON array
#         match = re.search(r'\[.*\]', raw, re.DOTALL)
#         if match:
#             return json.loads(match.group())
#     except Exception as e:
#         st.error(f"Quiz generation failed: {e}")

#     return []


# # ══ Analytics tab — NEW FEATURE ════════════════════════════════════════════════

# def _render_analytics_tab(processed_docs):
#     st.markdown('<div class="dm-card">', unsafe_allow_html=True)
#     st.markdown("### 📊 Document Analytics")
#     st.markdown("Visual analysis of your document content.")
#     st.markdown("</div>", unsafe_allow_html=True)

#     if not processed_docs:
#         st.info("📂 Upload documents first.")
#         return

#     import re
#     from collections import Counter

#     doc_names = [d["name"] for d in processed_docs]
#     selected  = st.selectbox("Choose document:", doc_names, key="analytics_doc")
#     doc_data  = next((d for d in processed_docs if d["name"] == selected), None)

#     if not doc_data:
#         return

#     text = doc_data.get("text", "")

#     # Basic stats
#     words     = re.findall(r'\b\w+\b', text.lower())
#     sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
#     paragraphs= [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]

#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("📝 Words",      f"{len(words):,}")
#     col2.metric("📄 Sentences",  f"{len(sentences):,}")
#     col3.metric("📑 Paragraphs", f"{len(paragraphs):,}")
#     col4.metric("✂️ Chunks",     f"{doc_data.get('chunk_count', 0):,}")

#     st.markdown("---")

#     # Top keywords
#     stopwords = {
#         'the','and','for','that','this','with','from','are','was','were',
#         'is','in','on','at','to','a','an','of','it','its','be','been',
#         'have','has','had','will','would','could','should','may','might',
#         'also','which','their','they','we','our','using','by','as','or',
#         'but','not','all','can','into','than','then','these','those','about'
#     }
#     filtered  = [w for w in words if w not in stopwords and len(w) > 3]
#     freq      = Counter(filtered)
#     top_words = freq.most_common(15)

#     if top_words:
#         col_chart, col_info = st.columns([3, 1])

#         with col_chart:
#             st.markdown("**🔑 Top Keywords**")
#             try:
#                 import plotly.graph_objects as go
#                 labels = [w for w, _ in top_words]
#                 counts = [c for _, c in top_words]
#                 fig = go.Figure(go.Bar(
#                     x=counts[::-1], y=labels[::-1],
#                     orientation='h',
#                     marker_color='#6366f1',
#                     marker_line_width=0
#                 ))
#                 fig.update_layout(
#                     height=350,
#                     margin=dict(l=0, r=0, t=10, b=10),
#                     paper_bgcolor='rgba(0,0,0,0)',
#                     plot_bgcolor='rgba(0,0,0,0)',
#                     font_color='#94a3b8',
#                     xaxis=dict(gridcolor='rgba(99,102,241,0.15)'),
#                     yaxis=dict(gridcolor='rgba(0,0,0,0)')
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
#             except ImportError:
#                 for word, count in top_words[:10]:
#                     bar_w = int((count / top_words[0][1]) * 100)
#                     st.markdown(
#                         f'`{word}` — {count} &nbsp;'
#                         f'<div style="height:6px;width:{bar_w}%;background:#6366f1;border-radius:3px;display:inline-block;"></div>',
#                         unsafe_allow_html=True
#                     )

#         with col_info:
#             st.markdown("**📈 Doc Stats**")
#             avg_sent_len = round(len(words) / max(len(sentences), 1), 1)
#             st.markdown(f"- Avg sentence: **{avg_sent_len}** words")
#             st.markdown(f"- Unique words: **{len(set(filtered)):,}**")
#             vocab_richness = round(len(set(filtered)) / max(len(filtered), 1) * 100, 1)
#             st.markdown(f"- Vocabulary richness: **{vocab_richness}%**")
#             doc_type = doc_data.get("metadata", {}).get("type", "unknown").upper()
#             st.markdown(f"- File type: **{doc_type}**")

#     # Sentence length distribution
#     st.markdown("---")
#     st.markdown("**📏 Sentence Length Distribution**")
#     sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 2]

#     if sent_lengths:
#         try:
#             import plotly.figure_factory as ff
#             fig2 = ff.create_distplot(
#                 [sent_lengths], ['Sentence lengths'],
#                 colors=['#8b5cf6'],
#                 show_rug=False
#             )
#             fig2.update_layout(
#                 height=200,
#                 margin=dict(l=0, r=0, t=10, b=10),
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 font_color='#94a3b8',
#                 showlegend=False
#             )
#             st.plotly_chart(fig2, use_container_width=True)
#         except Exception:
#             short = sum(1 for l in sent_lengths if l <= 10)
#             med   = sum(1 for l in sent_lengths if 10 < l <= 25)
#             long_ = sum(1 for l in sent_lengths if l > 25)
#             c1, c2, c3 = st.columns(3)
#             c1.metric("Short (≤10 words)",  short)
#             c2.metric("Medium (11-25)",      med)
#             c3.metric("Long (>25 words)",    long_)


# # ══ About tab ══════════════════════════════════════════════════════════════════

# def _render_about():
#     st.markdown("""
#     <div class="dm-card">
#         <h3>🧠 About DocuMind</h3>
#         <p style="color:#94a3b8;">
#             DocuMind is a <b>RAG-based intelligent document analysis system</b> that combines
#             semantic search, multimodal processing, and AI-powered Q&amp;A.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

#     features = [
#         ("📄 Multimodal Input",      "PDF, DOCX, PNG, JPG, BMP, TIFF with full OCR support."),
#         ("🔍 Semantic Search",        "Sentence-transformers + FAISS with RRF re-ranking."),
#         ("🤖 Multi-API Support",      "Groq, Gemini, Claude. Falls back to local offline mode."),
#         ("🗺️ Auto Flowcharts",        "Process flows, research maps, concept diagrams via Mermaid."),
#         ("📝 Quiz Generator",         "Auto MCQ quiz from any document using Groq Llama3."),
#         ("📊 Document Analytics",     "Keywords, word frequency, sentence distribution charts."),
#         ("🗣️ Layman Mode",           "Simple-language explanations for non-technical users."),
#         ("📎 Source Attribution",     "Ranked excerpts with relevance scores for transparency."),
#     ]

#     cols = st.columns(2)
#     for i, (title, desc) in enumerate(features):
#         with cols[i % 2]:
#             st.markdown(f"""
#             <div class="dm-card">
#                 <b style="color:#a5b4fc;">{title}</b>
#                 <p style="color:#94a3b8;font-size:0.88rem;margin-top:0.3rem;">{desc}</p>
#             </div>
#             """, unsafe_allow_html=True)