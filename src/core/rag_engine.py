"""
RAG Engine — Complete rebuild with guaranteed Gemini integration & High Accuracy
"""

# ══ Load .env ABSOLUTELY FIRST ════════════════════════════════════════════════
import sys, os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ENV  = _ROOT / ".env"

# Manual .env parsing — bypasses all dotenv import order issues
if _ENV.exists():
    with open(_ENV, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ[k] = v
                
print(f"[RAG] ENV loaded | GEMINI={'✓' if os.environ.get('GEMINI_API_KEY') else '✗'} | GROQ={'✓' if os.environ.get('GROQ_API_KEY') else '✗'}")

# ══ Now safe to import everything else ════════════════════════════════════════
import re
import streamlit as st
from src.core.vector_store import DocuMindVectorStore
from src.core.local_llm import generate_answer_local

TOP_K_RETRIEVAL = 8


# ══ Prompts ═══════════════════════════════════════════════════════════════════

def _get_system_prompt(context: str, question: str, simple_mode: bool) -> str:
    """Centralized prompt generator using XML tags for strict LLM adherence."""
    
    tone_ins = (
        "Use very simple everyday language. Short sentences. No technical jargon. "
        "Explain as if to someone with no technical background."
    ) if simple_mode else (
        "Be comprehensive and detailed. Use markdown ## headings and bullet points. "
        "Cover every relevant aspect found in the context."
    )

    return f"""You are DocuMind — an expert, highly accurate AI document analyst.

INSTRUCTIONS:
1. Read the <context> completely before answering.
2. Answer the <question> using ONLY the provided <context>. 
3. If the answer is NOT explicitly contained in the <context>, do not guess or use outside knowledge. Simply reply: "I cannot find the answer to this question in the uploaded documents."
4. {tone_ins}

FORMAT RULES:
- If asked about methodology/process/steps → list all steps in order with a ## heading.
- If asked for summary/overview → cover: Topic, Objectives, Methods, Results, Conclusions.
- If asked about results/findings → list all exact numbers, percentages, and comparisons.
- Always use bullet points for lists to maximize readability.
- Never truncate your thoughts — provide a COMPLETE answer.

<context>
{context}
</context>

<question>
{question}
</question>

ANSWER:"""


# ══ Gemini ════════════════════════════════════════════════════════════════════

def _get_gemini_answer(context: str, question: str, simple_mode: bool) -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("[RAG] Gemini skipped — no key")
        return None

    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=key)
        prompt = _get_system_prompt(context, question, simple_mode)

        print(f"[RAG] Calling Gemini | key_len={len(key)}")
        
        # Enforcing Temperature 0.0 for strict RAG accuracy
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, 
                top_p=0.95,
            )
        )
        text = (resp.text or "").strip()
        print(f"[RAG] Gemini answered | response_len={len(text)}")
        return text if len(text) > 30 else None

    except Exception as e:
        print(f"[RAG] Gemini FAILED: {e}")
        return None


# ══ Groq ══════════════════════════════════════════════════════════════════════

def _get_groq_answer(context: str, question: str, simple_mode: bool) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return None

    try:
        from groq import Groq
        client = Groq(api_key=key)
        prompt = _get_system_prompt(context, question, simple_mode)

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.0 # Strictly 0.0 to prevent hallucinations
        )
        text = (resp.choices[0].message.content or "").strip()
        print(f"[RAG] Groq answered | response_len={len(text)}")
        return text if len(text) > 30 else None

    except Exception as e:
        print(f"[RAG] Groq FAILED: {e}")
        return None


# ══ Query expansion ════════════════════════════════════════════════════════════

def _expand_query(question: str) -> list[str]:
    q = question.lower()
    base = [question]

    if any(w in q for w in ['method', 'approach', 'how', 'step', 'process', 'procedure', 'pipeline', 'workflow']):
        base += [
            "proposed methodology system pipeline steps procedure",
            "algorithm implementation technique preprocessing classification"
        ]
    elif any(w in q for w in ['about', 'summary', 'overview', 'what is', 'describe', 'contain', 'topic', 'paper', 'research']):
        base += [
            "introduction background motivation objective contribution",
            "abstract research goal problem statement scope"
        ]
    elif any(w in q for w in ['result', 'finding', 'performance', 'accuracy', 'score', 'conclusion', 'outcome']):
        base += [
            "results accuracy performance evaluation metrics score percentage",
            "findings conclusion experimental comparison table"
        ]
    elif any(w in q for w in ['dataset', 'data', 'train', 'test', 'sample']):
        base += [
            "dataset training testing validation samples split",
            "data collection preprocessing augmentation annotation"
        ]
    
    # Keep it simple, extract longer words as a pure keyword search fallback
    kws = re.findall(r'\b\w{5,}\b', q)
    if kws:
        base.append(" ".join(kws[:5]))

    return base[:4] # Don't overwhelm the retriever


# ══ Context builder ════════════════════════════════════════════════════════════

def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks):
        src  = c["metadata"].get("source", "Document")
        # Removing the score from the prompt so the LLM doesn't get confused by the math
        text = c["text"].strip()
        parts.append(f"--- Document Source {i+1}: {src} ---\n{text}")
    return "\n\n".join(parts)


# ══ Main pipeline ══════════════════════════════════════════════════════════════

def answer_question(
    question:     str,
    vector_store: DocuMindVectorStore,
    simple_mode:  bool = False,
    use_api:      bool = True
) -> dict:

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    groq_key   = os.environ.get("GROQ_API_KEY",   "").strip()
    print(f"[RAG] answer_question called | use_api={use_api} | gemini={bool(gemini_key)} | groq={bool(groq_key)}")

    if vector_store.total_chunks == 0:
        return {"answer": "⚠️ No documents indexed. Please upload a document first.",
                "sources": [], "context": "", "mode": "none"}

    # ── Multi-query retrieval with Reciprocal Rank Fusion (RRF) ────────────────
    queries  = _expand_query(question)
    rrf_scores = {}

    for q in queries:
        # Fetch results for each sub-query
        hits = vector_store.search(q, top_k=TOP_K_RETRIEVAL)
        
        # Apply Reciprocal Rank Fusion
        for rank, hit in enumerate(hits):
            idx = hit["index"]
            if idx not in rrf_scores:
                rrf_scores[idx] = {"hit": hit, "score": 0.0}
            
            # The standard RRF formula: 1 / (k + rank), k is usually 60
            rrf_scores[idx]["score"] += 1.0 / (60 + rank)

    # Sort the unified results by their new RRF score
    fused_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    # Take the top 8 distinct, highest-voted chunks
    retrieved = [x["hit"] for x in fused_results[:8]]

    if not retrieved:
        return {"answer": "Could not find relevant content for this question in the uploaded documents.",
                "sources": [], "context": "", "mode": "none"}

    context = _build_context(retrieved)

    # ── Generate ───────────────────────────────────────────────────────────────
    answer = None
    mode   = "local"

    if use_api:
        if gemini_key:
            answer = _get_gemini_answer(context, question, simple_mode)
            if answer:
                mode = "gemini"

        if answer is None and groq_key:
            answer = _get_groq_answer(context, question, simple_mode)
            if answer:
                mode = "groq"

    if answer is None:
        print("[RAG] Falling back to local LLM")
        answer = generate_answer_local(context, question, simple_mode)
        mode   = "local"

    return {"answer": answer, "sources": retrieved, "context": context, "mode": mode}