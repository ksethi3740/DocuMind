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
# ══ BLEU Score ════════════════════════════════════════════════════════════════

def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> dict:
    """
    Compute BLEU-1 through BLEU-4 without any external library.
    Returns individual n-gram precisions and the geometric mean (BLEU-4).
    """
    import math
    from collections import Counter

    def ngrams(tokens: list, n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    hyp = hypothesis.lower().split()
    ref = reference.lower().split()

    if not hyp or not ref:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n+1)} | {"bleu": 0.0}

    # Brevity penalty
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref)/len(hyp))

    precisions = []
    for n in range(1, max_n+1):
        hyp_ng = ngrams(hyp, n)
        ref_ng = ngrams(ref, n)
        if not hyp_ng:
            precisions.append(0.0)
            continue
        clipped = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
        precisions.append(clipped / sum(hyp_ng.values()))

    # Geometric mean (log-space to avoid underflow)
    pos_p = [p for p in precisions if p > 0]
    if not pos_p:
        bleu = 0.0
    else:
        log_avg = sum(math.log(p) for p in pos_p) / max_n
        bleu    = bp * math.exp(log_avg)

    return {
        "bleu_1": round(precisions[0], 4),
        "bleu_2": round(precisions[1], 4) if len(precisions) > 1 else 0.0,
        "bleu_3": round(precisions[2], 4) if len(precisions) > 2 else 0.0,
        "bleu_4": round(precisions[3], 4) if len(precisions) > 3 else 0.0,
        "bleu":   round(bleu,          4),
        "bp":     round(bp,            4),
    }


# ══ METEOR Score ══════════════════════════════════════════════════════════════

def meteor_score(hypothesis: str, reference: str) -> float:
    """
    Simplified METEOR: precision + recall with stemming (suffix stripping).
    Returns harmonic mean weighted 1:9 toward recall.
    """
    def stem(word: str) -> str:
        for suffix in ['ing','tion','ed','er','ly','ness','ment','ful','able']:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

    hyp_tokens = [stem(w) for w in hypothesis.lower().split()]
    ref_tokens = [stem(w) for w in reference.lower().split()]

    if not hyp_tokens or not ref_tokens:
        return 0.0

    ref_set  = set(ref_tokens)
    hyp_set  = set(hyp_tokens)
    matches  = len(hyp_set & ref_set)

    precision = matches / len(hyp_tokens) if hyp_tokens else 0.0
    recall    = matches / len(ref_tokens)  if ref_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    # Weighted harmonic mean (METEOR weights recall 9x more than precision)
    meteor = (10 * precision * recall) / (9 * precision + recall)
    return round(meteor, 4)


# ══ Exact Match ═══════════════════════════════════════════════════════════════

def exact_match(hypothesis: str, reference: str) -> dict:
    """
    Binary exact match and normalised partial match.
    """
    hyp_norm = re.sub(r'\s+', ' ', hypothesis.lower().strip())
    ref_norm = re.sub(r'\s+', ' ', reference.lower().strip())

    exact    = hyp_norm == ref_norm
    contains = ref_norm in hyp_norm or hyp_norm in ref_norm

    # Token-level F1 (SQuAD style)
    hyp_toks = set(hyp_norm.split())
    ref_toks = set(ref_norm.split())
    common   = hyp_toks & ref_toks

    if not common:
        token_f1 = 0.0
    else:
        p = len(common) / len(hyp_toks)
        r = len(common) / len(ref_toks)
        token_f1 = 2 * p * r / (p + r)

    return {
        "exact":    exact,
        "contains": contains,
        "token_f1": round(token_f1, 4),
    }


# ══ BERTScore (optional — needs bert_score package) ═══════════════════════════

def bertscore_f1(hypothesis: str, reference: str) -> float | None:
    """
    BERTScore F1 using distilbert (fastest model).
    Returns None if bert_score not installed.
    """
    try:
        from bert_score import score as bs_score
        P, R, F = bs_score(
            [hypothesis], [reference],
            model_type="distilbert-base-uncased",
            verbose=False,
            device="cpu"
        )
        return round(float(F[0]), 4)
    except ImportError:
        return None
    except Exception as e:
        print(f"[BERTScore] Error: {e}")
        return None


# ══ Confidence Score ══════════════════════════════════════════════════════════

def confidence_score(
    answer:           str,
    retrieved_chunks: list[dict],
    question:         str
) -> dict:
    """
    Composite confidence score (0–1) based on:
    - Mean retrieval score of top chunks            (40%)
    - Answer length adequacy                        (20%)
    - Source-answer token overlap                   (25%)
    - Query-answer semantic alignment               (15%)
    """
    if not retrieved_chunks:
        return {"score": 0.0, "components": {}}

    # 1. Mean retrieval score (cosine similarity)
    scores       = [c.get("score", 0.0) for c in retrieved_chunks[:5]]
    mean_ret     = sum(scores) / len(scores) if scores else 0.0
    ret_conf     = min(mean_ret / 0.8, 1.0)   # normalise to 1.0 at score=0.8

    # 2. Answer length adequacy
    ans_words    = len(answer.split())
    length_conf  = min(ans_words / 80, 1.0)   # 80 words = full confidence

    # 3. Source-answer overlap
    context_text = " ".join(c["text"] for c in retrieved_chunks[:5]).lower()
    ans_words_l  = set(
        w for w in answer.lower().split()
        if len(w) > 4
    )
    ctx_words    = set(context_text.split())
    if ans_words_l:
        overlap_conf = len(ans_words_l & ctx_words) / len(ans_words_l)
    else:
        overlap_conf = 0.0

    # 4. Query-answer semantic alignment
    try:
        q_sim = semantic_similarity(question, answer)
    except Exception:
        q_sim = 0.3   # fallback neutral

    # Weighted composite
    comp = {
        "retrieval_strength": round(ret_conf,     3),
        "answer_length":      round(length_conf,  3),
        "source_overlap":     round(overlap_conf, 3),
        "query_alignment":    round(q_sim,        3),
    }
    weights = [0.40, 0.20, 0.25, 0.15]
    values  = list(comp.values())
    total   = sum(v * w for v, w in zip(values, weights))

    return {
        "score":      round(total, 3),
        "components": comp,
        "label":      (
            "Very High"  if total >= 0.80 else
            "High"       if total >= 0.65 else
            "Moderate"   if total >= 0.50 else
            "Low"        if total >= 0.35 else
            "Very Low"
        )
    }


# ══ Hallucination Rate ════════════════════════════════════════════════════════

def hallucination_rate(answer: str, retrieved_chunks: list[dict]) -> dict:
    """
    Estimate what fraction of answer sentences have NO support
    in the retrieved context.
    Uses sentence-level cosine similarity — threshold 0.30.
    """
    sentences = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', answer)
        if len(s.strip()) > 20
    ]

    if not sentences:
        return {"rate": 0.0, "unsupported": 0, "total": 0}

    context_text = " ".join(c["text"] for c in retrieved_chunks)

    try:
        from src.core.embeddings import embed_texts
        import numpy as np

        all_texts = sentences + [context_text[:2000]]
        vecs      = embed_texts(all_texts)
        ctx_vec   = vecs[-1]
        sent_vecs = vecs[:-1]

        unsupported = 0
        for sv in sent_vecs:
            sim = float(sum(a * b for a, b in zip(sv, ctx_vec)))
            if sim < 0.30:
                unsupported += 1

    except Exception:
        # Fallback: token overlap
        ctx_words   = set(context_text.lower().split())
        unsupported = 0
        for sent in sentences:
            s_words = set(sent.lower().split())
            overlap = len(s_words & ctx_words) / max(len(s_words), 1)
            if overlap < 0.20:
                unsupported += 1

    rate = unsupported / len(sentences)
    return {
        "rate":        round(rate, 3),
        "unsupported": unsupported,
        "total":       len(sentences),
        "label": (
            "Very Low"   if rate < 0.10 else
            "Low"        if rate < 0.25 else
            "Moderate"   if rate < 0.50 else
            "High"
        )
    }


# ══ Answer Completeness ═══════════════════════════════════════════════════════

def answer_completeness(question: str, answer: str) -> dict:
    """
    Does the answer cover the key topics implied by the question?
    Extracts content keywords from question and checks coverage in answer.
    """
    noise = {
        'what', 'is', 'are', 'the', 'a', 'an', 'tell', 'me', 'about',
        'explain', 'describe', 'how', 'why', 'does', 'do', 'this', 'that',
        'give', 'please', 'can', 'you', 'list', 'show', 'main', 'which',
        'where', 'when', 'who', 'has', 'was', 'were', 'will', 'would'
    }
    q_keywords = [
        w.lower() for w in re.findall(r'\b\w{3,}\b', question)
        if w.lower() not in noise
    ]

    if not q_keywords:
        return {"score": 1.0, "covered": [], "missing": []}

    ans_lower = answer.lower()
    covered   = [kw for kw in q_keywords if kw in ans_lower]
    missing   = [kw for kw in q_keywords if kw not in ans_lower]

    score = len(covered) / len(q_keywords)
    return {
        "score":   round(score, 3),
        "covered": covered,
        "missing": missing,
        "label": (
            "Complete"   if score >= 0.80 else
            "Partial"    if score >= 0.50 else
            "Incomplete"
        )
    }


# ══ Coherence Score ═══════════════════════════════════════════════════════════

def coherence_score(answer: str) -> dict:
    """
    Measures sentence-to-sentence semantic flow.
    Average cosine similarity between adjacent sentence embeddings.
    Higher = more coherent / on-topic.
    """
    sentences = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', answer)
        if len(s.strip()) > 15
    ]

    if len(sentences) < 2:
        return {"score": 1.0, "label": "N/A (single sentence)"}

    try:
        from src.core.embeddings import embed_texts
        vecs  = embed_texts(sentences)
        pairs = []
        for i in range(len(vecs) - 1):
            sim = float(sum(a * b for a, b in zip(vecs[i], vecs[i+1])))
            pairs.append(sim)
        score = sum(pairs) / len(pairs)
    except Exception:
        # Fallback: word overlap between adjacent sentences
        pairs = []
        for i in range(len(sentences) - 1):
            a = set(sentences[i].lower().split())
            b = set(sentences[i+1].lower().split())
            pairs.append(len(a & b) / max(len(a | b), 1))
        score = sum(pairs) / len(pairs) if pairs else 0.5

    return {
        "score": round(score, 3),
        "label": (
            "High"     if score >= 0.70 else
            "Moderate" if score >= 0.50 else
            "Low"
        )
    }


# ══ MRR — Mean Reciprocal Rank ════════════════════════════════════════════════

def mean_reciprocal_rank(question: str, retrieved_chunks: list[dict],
                          relevance_threshold: float = 0.45) -> dict:
    """
    MRR: 1/rank of the first relevant result.
    A chunk is "relevant" if its cosine score >= relevance_threshold.
    MRR=1.0 means the top result was relevant. MRR=0.5 means rank 2 was first relevant.
    """
    if not retrieved_chunks:
        return {"mrr": 0.0, "first_relevant_rank": None}

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk.get("score", 0.0) >= relevance_threshold:
            return {
                "mrr":                round(1.0 / rank, 4),
                "first_relevant_rank": rank,
                "label": (
                    "Excellent" if rank == 1 else
                    "Good"      if rank <= 2 else
                    "Fair"      if rank <= 4 else
                    "Poor"
                )
            }

    return {"mrr": 0.0, "first_relevant_rank": None, "label": "No relevant results"}


# ══ NDCG@K ════════════════════════════════════════════════════════════════════

def ndcg_at_k(retrieved_chunks: list[dict], k: int = 5) -> dict:
    """
    Normalised Discounted Cumulative Gain at K.
    Uses raw cosine scores as relevance grades.
    NDCG@K = 1.0 means chunks are perfectly ranked.
    """
    import math

    scores = [c.get("score", 0.0) for c in retrieved_chunks[:k]]

    if not scores:
        return {"ndcg": 0.0}

    def dcg(rel_list):
        return sum(
            r / math.log2(i + 2)
            for i, r in enumerate(rel_list)
        )

    actual_dcg  = dcg(scores)
    ideal_dcg   = dcg(sorted(scores, reverse=True))

    ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    return {
        "ndcg":  round(ndcg, 4),
        "k":     k,
        "label": (
            "Excellent" if ndcg >= 0.90 else
            "Good"      if ndcg >= 0.75 else
            "Fair"      if ndcg >= 0.55 else
            "Poor"
        )
    }


# ══ Updated full pipeline ══════════════════════════════════════════════════════

def evaluate_answer_full(
    question:         str,
    answer:           str,
    context:          str,
    retrieved_chunks: list[dict],
    reference_answer: str = ""
) -> dict:
    """
    Complete evaluation pipeline — all metrics.
    Returns a unified result dict.
    """
    has_groq = bool(os.environ.get("GROQ_API_KEY", "").strip())
    result   = {}

    # ── LLM-judge (Groq) ──────────────────────────────────────────────────────
    if has_groq:
        result["faithfulness"]     = eval_faithfulness(answer, context)
        result["answer_relevance"] = eval_answer_relevance(question, answer)
    else:
        result["faithfulness"]     = {"score": None, "reason": "Groq key required"}
        result["answer_relevance"] = {"score": None, "reason": "Groq key required"}

    # ── Always-available metrics ───────────────────────────────────────────────
    result["context_precision"]  = eval_context_precision(question, retrieved_chunks)
    result["context_recall"]     = eval_context_recall(answer, retrieved_chunks)
    result["confidence"]         = confidence_score(answer, retrieved_chunks, question)
    result["hallucination"]      = hallucination_rate(answer, retrieved_chunks)
    result["completeness"]       = answer_completeness(question, answer)
    result["coherence"]          = coherence_score(answer)
    result["mrr"]                = mean_reciprocal_rank(question, retrieved_chunks)
    result["ndcg"]               = ndcg_at_k(retrieved_chunks, k=min(5, len(retrieved_chunks)))

    # ── Reference-based ───────────────────────────────────────────────────────
    if reference_answer and reference_answer.strip():
        result["rouge_l"]             = rouge_l(answer, reference_answer)
        result["bleu"]                = bleu_score(answer, reference_answer)
        result["meteor"]              = meteor_score(answer, reference_answer)
        result["exact_match"]         = exact_match(answer, reference_answer)
        result["semantic_similarity"] = semantic_similarity(answer, reference_answer)
        result["bertscore"]           = bertscore_f1(answer, reference_answer)
    else:
        result["rouge_l"]             = None
        result["bleu"]                = None
        result["meteor"]              = None
        result["exact_match"]         = None
        result["semantic_similarity"] = None
        result["bertscore"]           = None

    # ── Overall score ──────────────────────────────────────────────────────────
    metric_weights = [
        (result["faithfulness"].get("score"),          0.20),
        (result["answer_relevance"].get("score"),      0.18),
        (result["context_precision"]["score"],         0.12),
        (result["context_recall"]["score"],            0.10),
        (result["confidence"]["score"],                0.15),
        (1.0 - result["hallucination"]["rate"],        0.13),   # lower hallucination = better
        (result["completeness"]["score"],              0.07),
        (result["coherence"]["score"],                 0.05),
    ]

    valid   = [(s, w) for s, w in metric_weights if s is not None]
    if valid:
        total_w = sum(w for _, w in valid)
        overall = sum(s * w for s, w in valid) / total_w
    else:
        overall = 0.0

    result["overall_score"] = round(overall, 3)
    result["grade"]         = _score_to_grade(overall)

    return result

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

# def evaluate_answer(
#     question:         str,
#     answer:           str,
#     context:          str,
#     retrieved_chunks: list[dict],
#     reference_answer: str = ""
# ) -> dict:
#     """
#     Run all metrics and return a unified result dict.

#     Args:
#         question:         User's question
#         answer:           Generated answer
#         context:          Context string sent to LLM
#         retrieved_chunks: List of chunk dicts from vector store
#         reference_answer: Optional gold-standard answer for reference-based metrics

#     Returns:
#         {
#           faithfulness:       {score, reason}
#           answer_relevance:   {score, reason}
#           context_precision:  {score, relevant_count, total}
#           context_recall:     {score}
#           semantic_similarity: float  (only if reference_answer provided)
#           rouge_l:            {precision, recall, f1}  (only if reference)
#           overall_score:      float   (weighted average)
#           grade:              str     (A/B/C/D/F)
#         }
#     """
#     has_groq = bool(os.environ.get("GROQ_API_KEY", "").strip())
#     result   = {}

#     # ── LLM-judge metrics (need Groq) ──────────────────────────────────────
#     if has_groq:
#         result["faithfulness"]     = eval_faithfulness(answer, context)
#         result["answer_relevance"] = eval_answer_relevance(question, answer)
#     else:
#         result["faithfulness"]     = {"score": None, "reason": "Groq key required"}
#         result["answer_relevance"] = {"score": None, "reason": "Groq key required"}

#     # ── Retrieval metrics (always available) ───────────────────────────────
#     result["context_precision"] = eval_context_precision(question, retrieved_chunks)
#     result["context_recall"]    = eval_context_recall(answer, retrieved_chunks)

#     # ── Reference-based metrics ────────────────────────────────────────────
#     if reference_answer and reference_answer.strip():
#         result["semantic_similarity"] = semantic_similarity(answer, reference_answer)
#         result["rouge_l"]             = rouge_l(answer, reference_answer)
#     else:
#         result["semantic_similarity"] = None
#         result["rouge_l"]             = None

#     # ── Overall score ──────────────────────────────────────────────────────
#     scores = []
#     weights = []

#     faith_sc = result["faithfulness"].get("score")
#     rel_sc   = result["answer_relevance"].get("score")
#     prec_sc  = result["context_precision"]["score"]
#     rec_sc   = result["context_recall"]["score"]

#     if faith_sc is not None:
#         scores.append(faith_sc);  weights.append(0.35)
#     if rel_sc is not None:
#         scores.append(rel_sc);    weights.append(0.30)
#     scores.append(prec_sc);       weights.append(0.20)
#     scores.append(rec_sc);        weights.append(0.15)

#     if scores:
#         total_w = sum(weights)
#         overall = sum(s * w for s, w in zip(scores, weights)) / total_w
#     else:
#         overall = 0.0

#     result["overall_score"] = round(overall, 3)
#     result["grade"]         = _score_to_grade(overall)

#     return result


def _score_to_grade(score: float) -> str:
    if score >= 0.85:  return "A"
    if score >= 0.70:  return "B"
    if score >= 0.55:  return "C"
    if score >= 0.40:  return "D"
    return "F"