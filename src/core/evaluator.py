"""
DocuMind — RAG Answer Evaluator
Computes faithfulness, relevance, context precision,
semantic similarity, and ROUGE-L without any paid eval library.
Works with Groq as the judge LLM.
"""

import re
import os
import json
import math
from pathlib import Path

# Load env
_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV.exists():
    with open(_ENV) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


# ══ ROUGE-L ═══════════════════════════════════════════════════════════════════

def _lcs_length(a: list, b: list) -> int:
    """Longest common subsequence length."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Use space-efficient rolling array
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> dict:
    """
    Compute ROUGE-L precision, recall, and F1.
    Returns dict with keys: precision, recall, f1
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall    = lcs / len(ref_tokens)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4)
    }


# ══ Semantic similarity (cosine on sentence embeddings) ══════════════════════

def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Cosine similarity between two texts using the same
    MiniLM embeddings already loaded for the vector store.
    Returns float 0-1.
    """
    try:
        from src.core.embeddings import embed_texts
        vecs = embed_texts([text_a, text_b])
        a, b = vecs[0], vecs[1]
        # Vectors are already L2-normalized (normalize_embeddings=True)
        # so cosine = dot product
        dot = float(sum(x * y for x, y in zip(a, b)))
        return round(max(0.0, min(1.0, dot)), 4)
    except Exception:
        return 0.0


# ══ LLM-as-judge (Groq) ══════════════════════════════════════════════════════

def _call_groq(prompt: str, max_tokens: int = 300) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=key)
        resp   = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[Eval] Groq error: {e}")
        return ""


def _extract_score(text: str, key: str = "score") -> float:
    """Extract numeric score from JSON-like LLM output."""
    # Try JSON parse first
    try:
        m = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if m:
            d = json.loads(m.group())
            for k in [key, "score", "rating", "value"]:
                if k in d:
                    return float(d[k])
    except Exception:
        pass
    # Fallback: find any number 0-10 or 0.0-1.0 in the text
    nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    for n in nums:
        v = float(n)
        if 0 <= v <= 10:
            return v / 10 if v > 1 else v
    return 0.5


def eval_faithfulness(answer: str, context: str) -> dict:
    """
    LLM judge: is every claim in the answer supported by the context?
    Score 0-1.
    """
    prompt = f"""You are an expert evaluator. Score the faithfulness of an AI answer.

Faithfulness = every factual claim in the ANSWER is explicitly supported by the CONTEXT.
Penalise if the answer contains facts not present in the context.

CONTEXT:
{context[:2000]}

ANSWER:
{answer[:1000]}

Respond ONLY with JSON: {{"score": 0.85, "reason": "brief explanation"}}
Score must be between 0.0 and 1.0."""

    raw    = _call_groq(prompt, 150)
    score  = _extract_score(raw, "score")
    reason = ""
    try:
        m = re.search(r'"reason"\s*:\s*"([^"]+)"', raw)
        if m:
            reason = m.group(1)
    except Exception:
        pass

    return {"score": round(score, 3), "reason": reason, "raw": raw}


def eval_answer_relevance(question: str, answer: str) -> dict:
    """
    LLM judge: does the answer actually address the question?
    Score 0-1.
    """
    prompt = f"""You are an expert evaluator. Score the relevance of an AI answer.

Relevance = the answer directly addresses what was asked, stays on topic,
and does not give a generic or unrelated response.

QUESTION: {question}

ANSWER: {answer[:1000]}

Respond ONLY with JSON: {{"score": 0.90, "reason": "brief explanation"}}
Score must be between 0.0 and 1.0."""

    raw   = _call_groq(prompt, 150)
    score = _extract_score(raw, "score")
    reason = ""
    try:
        m = re.search(r'"reason"\s*:\s*"([^"]+)"', raw)
        if m:
            reason = m.group(1)
    except Exception:
        pass

    return {"score": round(score, 3), "reason": reason}


def eval_context_precision(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Of the retrieved chunks, what fraction are actually relevant to the question?
    Uses embedding similarity as proxy (no LLM call needed).
    """
    if not retrieved_chunks:
        return {"score": 0.0, "relevant_count": 0, "total": 0}

    try:
        from src.core.embeddings import embed_query, embed_texts
        import numpy as np

        q_vec      = embed_query(question)
        chunk_texts= [c["text"] for c in retrieved_chunks]
        c_vecs     = embed_texts(chunk_texts)

        threshold  = 0.35   # cosine threshold for "relevant"
        relevant   = sum(
            1 for cv in c_vecs
            if float(sum(q * c for q, c in zip(q_vec, cv))) >= threshold
        )
        score = relevant / len(retrieved_chunks)

    except Exception:
        # Fallback: use stored scores
        scores   = [c.get("score", 0.0) for c in retrieved_chunks]
        relevant = sum(1 for s in scores if s >= 0.35)
        score    = relevant / len(scores) if scores else 0.0

    return {
        "score":          round(score, 3),
        "relevant_count": relevant,
        "total":          len(retrieved_chunks)
    }


def eval_context_recall(answer: str, retrieved_chunks: list[dict]) -> dict:
    """
    Does the combined context contain enough information to produce the answer?
    Uses semantic similarity between answer and full context.
    """
    if not retrieved_chunks:
        return {"score": 0.0}

    full_context = " ".join(c["text"] for c in retrieved_chunks)
    sim          = semantic_similarity(answer, full_context[:3000])

    # Normalise — recall is typically high; we cap at 1.0
    score = min(sim * 1.5, 1.0)   # boost slightly since answer is shorter
    return {"score": round(score, 3)}


# ══ Full evaluation pipeline ══════════════════════════════════════════════════

def evaluate_answer(
    question:         str,
    answer:           str,
    context:          str,
    retrieved_chunks: list[dict],
    reference_answer: str = ""
) -> dict:
    """
    Run all metrics and return a unified result dict.

    Args:
        question:         User's question
        answer:           Generated answer
        context:          Context string sent to LLM
        retrieved_chunks: List of chunk dicts from vector store
        reference_answer: Optional gold-standard answer for reference-based metrics

    Returns:
        {
          faithfulness:       {score, reason}
          answer_relevance:   {score, reason}
          context_precision:  {score, relevant_count, total}
          context_recall:     {score}
          semantic_similarity: float  (only if reference_answer provided)
          rouge_l:            {precision, recall, f1}  (only if reference)
          overall_score:      float   (weighted average)
          grade:              str     (A/B/C/D/F)
        }
    """
    has_groq = bool(os.environ.get("GROQ_API_KEY", "").strip())
    result   = {}

    # ── LLM-judge metrics (need Groq) ──────────────────────────────────────
    if has_groq:
        result["faithfulness"]     = eval_faithfulness(answer, context)
        result["answer_relevance"] = eval_answer_relevance(question, answer)
    else:
        result["faithfulness"]     = {"score": None, "reason": "Groq key required"}
        result["answer_relevance"] = {"score": None, "reason": "Groq key required"}

    # ── Retrieval metrics (always available) ───────────────────────────────
    result["context_precision"] = eval_context_precision(question, retrieved_chunks)
    result["context_recall"]    = eval_context_recall(answer, retrieved_chunks)

    # ── Reference-based metrics ────────────────────────────────────────────
    if reference_answer and reference_answer.strip():
        result["semantic_similarity"] = semantic_similarity(answer, reference_answer)
        result["rouge_l"]             = rouge_l(answer, reference_answer)
    else:
        result["semantic_similarity"] = None
        result["rouge_l"]             = None

    # ── Overall score ──────────────────────────────────────────────────────
    scores = []
    weights = []

    faith_sc = result["faithfulness"].get("score")
    rel_sc   = result["answer_relevance"].get("score")
    prec_sc  = result["context_precision"]["score"]
    rec_sc   = result["context_recall"]["score"]

    if faith_sc is not None:
        scores.append(faith_sc);  weights.append(0.35)
    if rel_sc is not None:
        scores.append(rel_sc);    weights.append(0.30)
    scores.append(prec_sc);       weights.append(0.20)
    scores.append(rec_sc);        weights.append(0.15)

    if scores:
        total_w = sum(weights)
        overall = sum(s * w for s, w in zip(scores, weights)) / total_w
    else:
        overall = 0.0

    result["overall_score"] = round(overall, 3)
    result["grade"]         = _score_to_grade(overall)

    return result


def _score_to_grade(score: float) -> str:
    if score >= 0.85:  return "A"
    if score >= 0.70:  return "B"
    if score >= 0.55:  return "C"
    if score >= 0.40:  return "D"
    return "F"