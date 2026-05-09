"""
DocuMind — Quiz Generator Tab
Auto-generates MCQ quizzes from any document using Groq.
"""

import os
import re
import json
import streamlit as st


def render_quiz_tab(processed_docs: list[dict], use_api: bool):
    st.markdown('<div class="dm-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Auto Quiz Generator")
    st.markdown("Generate MCQ questions automatically from your document using AI.")
    st.markdown("</div>", unsafe_allow_html=True)

    if not processed_docs:
        st.info("📂 Upload a document first to generate a quiz.")
        return

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not use_api or not groq_key:
        st.warning("⚠️ Quiz generation needs a **Groq API key** in your `.env` file.")
        st.code("GROQ_API_KEY=your_key_here", language="bash")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    doc_names = [d["name"] for d in processed_docs]
    c1, c2, c3 = st.columns(3)
    with c1:
        selected   = st.selectbox("📄 Document:", doc_names, key="quiz_doc")
    with c2:
        num_q      = st.selectbox("❓ Questions:", [3, 5, 8, 10], index=1)
    with c3:
        difficulty = st.selectbox("🎯 Difficulty:", ["Easy", "Medium", "Hard"], index=1)

    topic = st.text_input(
        "🔎 Focus topic (optional):",
        placeholder="e.g. methodology, results, CNN architecture…",
        key="quiz_topic"
    )

    if st.button("🎯 Generate Quiz", key="gen_quiz_btn", use_container_width=True):
        doc_data = next((d for d in processed_docs if d["name"] == selected), None)
        if not doc_data:
            return

        text = doc_data.get("text", "")
        if topic.strip():
            # Find topic section
            idx  = text.lower().find(topic.lower())
            text = text[max(0, idx - 100): idx + 3000] if idx != -1 else text[:3000]
        else:
            text = text[:4000]

        with st.spinner("🎯 Generating quiz questions…"):
            quiz = _generate_quiz(text, num_q, difficulty, topic, groq_key)

        if quiz:
            st.session_state["current_quiz"]  = quiz
            st.session_state["quiz_answers"]  = {}
            st.session_state["quiz_submitted"] = False
        else:
            st.error("Failed to generate quiz. Try again.")

    # ── Display quiz ──────────────────────────────────────────────────────────
    if "current_quiz" not in st.session_state:
        return

    quiz      = st.session_state["current_quiz"]
    answers   = st.session_state.get("quiz_answers", {})
    submitted = st.session_state.get("quiz_submitted", False)

    st.markdown("---")
    st.markdown(f"**📋 Quiz — {len(quiz)} Questions**")

    score = 0
    for i, q in enumerate(quiz):
        question = q.get("question", "")
        options  = q.get("options", [])
        correct  = q.get("correct", "")
        explain  = q.get("explanation", "")

        with st.container():
            st.markdown(
                f'<div class="dm-card"><b>Q{i+1}. {question}</b></div>',
                unsafe_allow_html=True
            )

            if submitted:
                user_ans = answers.get(i, "")
                if user_ans == correct:
                    st.success(f"✅ {user_ans}")
                    score += 1
                else:
                    st.error(f"❌ Your answer: {user_ans}")
                    st.success(f"✅ Correct: {correct}")
                if explain:
                    with st.expander("💡 Explanation"):
                        st.write(explain)
            else:
                choice = st.radio(
                    f"Select answer for Q{i+1}:",
                    options,
                    key=f"quiz_q_{i}",
                    label_visibility="collapsed"
                )
                answers[i] = choice

    st.session_state["quiz_answers"] = answers

    if not submitted:
        if st.button("📤 Submit Answers", key="submit_quiz", use_container_width=True):
            st.session_state["quiz_submitted"] = True
            st.rerun()
    else:
        # Score display
        total = len(quiz)
        pct   = int((score / total) * 100) if total else 0
        color = "#86efac" if pct >= 70 else "#fbbf24" if pct >= 40 else "#f87171"
        grade = "Excellent! 🏆" if pct >= 80 else "Good! 👍" if pct >= 60 else "Keep practising 📚"

        st.markdown(
            f"""<div class="dm-card" style="text-align:center;">
            <div style="font-size:2.5rem;font-weight:700;color:{color};">{score}/{total}</div>
            <div style="font-size:1.2rem;color:#94a3b8;">{pct}% — {grade}</div>
            </div>""",
            unsafe_allow_html=True
        )

        # Export quiz
        quiz_text = _export_quiz_text(quiz, answers, score, total)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download Quiz + Answers",
                data=quiz_text,
                file_name="quiz_results.txt",
                mime="text/plain"
            )
        with col2:
            if st.button("🔄 New Quiz", use_container_width=True):
                for k in ["current_quiz","quiz_answers","quiz_submitted"]:
                    st.session_state.pop(k, None)
                st.rerun()


def _generate_quiz(text: str, num_q: int, difficulty: str,
                   topic: str, groq_key: str) -> list:
    topic_note = f"Focus on: {topic}." if topic.strip() else ""

    prompt = f"""Generate exactly {num_q} multiple-choice questions from this document.
Difficulty: {difficulty}. {topic_note}

DOCUMENT:
{text}

Return ONLY a valid JSON array with NO extra text, NO markdown fences.
Each item must have: question, options (array of 4), correct (exact match to one option), explanation.

Example:
[
  {{
    "question": "What is the main purpose of the proposed system?",
    "options": ["To detect emotions only", "To detect drowsiness and emotions simultaneously", "To improve GPS accuracy", "To analyse traffic patterns"],
    "correct": "To detect drowsiness and emotions simultaneously",
    "explanation": "The paper proposes a dual-stage framework integrating both drowsiness and emotion detection."
  }}
]

Generate {num_q} questions now (JSON only):"""

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        resp   = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500,
            temperature=0.2
        )
        raw   = (resp.choices[0].message.content or "").strip()
        raw   = re.sub(r'^```(?:json)?\s*', '', raw)
        raw   = re.sub(r'```\s*$', '', raw).strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        st.error(f"Quiz error: {e}")
    return []

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
def _export_quiz_text(quiz: list, answers: dict, score: int, total: int) -> str:
    lines = [f"DocuMind Quiz Results — {score}/{total} ({int(score/total*100)}%)\n", "="*50, ""]
    for i, q in enumerate(quiz):
        lines.append(f"Q{i+1}. {q['question']}")
        for opt in q.get("options", []):
            lines.append(f"   {'✓' if opt == q['correct'] else '○'} {opt}")
        user = answers.get(i, "Not answered")
        lines.append(f"   Your answer: {user}")
        if q.get("explanation"):
            lines.append(f"   Explanation: {q['explanation']}")
        lines.append("")
    return "\n".join(lines)

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
