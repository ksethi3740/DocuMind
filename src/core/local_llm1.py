"""
DocuMind — Universal Local Answer Engine
Works for ANY document: research papers, legal docs, manuals,
business reports, contracts, books, articles, notes, etc.
No API needed.
"""

import re
from collections import Counter


# ══ Master dispatcher ══════════════════════════════════════════════════════════

def generate_answer_local(context: str, question: str, simple_mode: bool = False) -> str:
    """
    Detects document type + question intent automatically.
    Works for ANY uploaded document — no hardcoding.
    """
    text = _clean_context(context)
    q    = question.lower().strip()

    if not text or len(text) < 30:
        return "❌ Could not extract content from the document. Try re-uploading it."

    # Auto-detect document type
    doc_type = _detect_document_type(text)

    # Detect question intent
    intent = _detect_intent(q)

    # Route to best handler
    handlers = {
        "summary":      _answer_summary,
        "methodology":  _answer_steps_process,
        "results":      _answer_results_numbers,
        "dataset":      _answer_specific_topic,
        "architecture": _answer_specific_topic,
        "future":       _answer_specific_topic,
        "definition":   _answer_definition,
        "comparison":   _answer_comparison,
        "list":         _answer_list_items,
        "general":      _answer_general,
    }

    handler = handlers.get(intent, _answer_general)
    return handler(text, question, simple_mode, doc_type)


# ══ Detection engines ══════════════════════════════════════════════════════════

def _detect_document_type(text: str) -> str:
    """
    Auto-detect what kind of document this is.
    No hardcoding — purely signal-based.
    """
    t = text.lower()
    signals = {
        "research":  ['abstract', 'methodology', 'conclusion', 'references',
                      'literature review', 'proposed', 'experiment', 'dataset',
                      'accuracy', 'precision', 'recall', 'ieee', 'arxiv'],
        "legal":     ['whereas', 'hereinafter', 'pursuant', 'agreement',
                      'clause', 'party', 'obligations', 'jurisdiction',
                      'contract', 'terms', 'liability', 'warranty'],
        "manual":    ['installation', 'configuration', 'setup', 'click',
                      'step 1', 'step 2', 'navigate', 'select', 'enter',
                      'menu', 'button', 'screen', 'settings', 'troubleshoot'],
        "business":  ['revenue', 'profit', 'quarter', 'fiscal', 'strategy',
                      'market', 'customer', 'growth', 'stakeholder', 'kpi',
                      'roi', 'budget', 'forecast', 'executive'],
        "medical":   ['patient', 'diagnosis', 'treatment', 'clinical',
                      'symptoms', 'dosage', 'therapy', 'prognosis',
                      'medication', 'disease', 'hospital'],
        "educational": ['chapter', 'lesson', 'exercise', 'example',
                        'definition', 'theorem', 'formula', 'solution',
                        'question', 'answer', 'practice'],
    }

    scores = {}
    for dtype, keywords in signals.items():
        scores[dtype] = sum(1 for kw in keywords if kw in t)

    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


def _detect_intent(q: str) -> str:
    """Detect what the user is asking for."""
    intents = {
        "summary": [
            'summarise', 'summarize', 'summary', 'overview', 'about',
            'what is this', 'what does this', 'describe', 'contain',
            'tell me about', 'explain the', 'what is the document',
            'what is the paper', 'what is the report', 'main topic',
            'key points', 'brief'
        ],
        "methodology": [
            'methodology', 'method', 'approach', 'how does', 'how is',
            'pipeline', 'process', 'procedure', 'steps', 'workflow',
            'how to', 'how do', 'mechanism', 'technique', 'algorithm',
            'implementation', 'framework', 'architecture of'
        ],
        "results": [
            'result', 'finding', 'accuracy', 'performance', 'score',
            'conclusion', 'outcome', 'achieve', 'percentage', 'metric',
            'f1', 'precision', 'recall', 'what was found', 'how well',
            'how accurate', 'evaluation', 'benchmark'
        ],
        "definition": [
            'what is', 'define', 'definition', 'meaning', 'explain what',
            'what does', 'what are', 'what means', 'concept of'
        ],
        "comparison": [
            'compare', 'comparison', 'difference', 'versus', 'vs',
            'better', 'worse', 'advantage', 'disadvantage', 'pros', 'cons',
            'contrast', 'similar', 'different from'
        ],
        "list": [
            'list', 'enumerate', 'what are all', 'types of', 'kinds of',
            'features', 'components', 'elements', 'items', 'categories',
            'examples of', 'name the', 'mention'
        ],
        "future": [
            'future', 'limitation', 'drawback', 'challenge', 'improvement',
            'scope', 'next steps', 'recommendations', 'suggest', 'propose'
        ],
    }

    for intent, keywords in intents.items():
        if any(kw in q for kw in keywords):
            return intent

    return "general"


# ══ Context cleaner ════════════════════════════════════════════════════════════

def _clean_context(context: str) -> str:
    """Remove RAG metadata and normalize text."""
    text = re.sub(r'\[SECTION \d+[^\]]*\]\n?', '', context)
    text = re.sub(r'─{3,}', '\n', text)
    text = re.sub(r'={3,}', '\n', text)
    text = re.sub(r'\[EXCERPT \d+[^\]]*\]\n?', '', text)
    text = re.sub(r'Document Source \d+:.*?---', '', text, flags=re.DOTALL)
    text = re.sub(r'---\s*Document Source.*', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# ══ Sentence utilities ══════════════════════════════════════════════════════════

def _get_sentences(text: str) -> list[str]:
    """Extract clean sentences from any document."""
    raw = re.split(r'(?<=[.!?])\s+|\n', text)
    good = []
    for s in raw:
        s = s.strip()
        if len(s) < 25:
            continue
        if len(s) > 500:
            parts = re.split(r';\s+|,\s+(?=[A-Z])', s)
            good.extend([p.strip() for p in parts if len(p.strip()) > 25])
            continue
        # Skip lines that are just numbers, headers, or figure captions
        if re.match(r'^[\d\s\.\-]+$', s):
            continue
        if re.match(r'^(fig|table|figure|eq|equation)\s*[\d\.]+', s, re.IGNORECASE):
            continue
        good.append(s)
    return good


def _get_keywords(question: str) -> list[str]:
    """Extract content keywords from the question."""
    noise = {
        'what', 'is', 'are', 'the', 'a', 'an', 'tell', 'me', 'about',
        'explain', 'describe', 'how', 'why', 'does', 'do', 'this', 'that',
        'give', 'please', 'can', 'you', 'list', 'show', 'main', 'key',
        'primary', 'paper', 'document', 'report', 'file', 'text', 'which',
        'where', 'when', 'who', 'have', 'has', 'was', 'were', 'will', 'would'
    }
    words = re.findall(r'\b\w{3,}\b', question.lower())
    return [w for w in words if w not in noise]


def _score_sentences(sentences: list[str], keywords: list[str],
                     boost: list[str] = None) -> list[tuple]:
    """Score sentences by relevance to keywords."""
    stopwords = {
        'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are', 'was',
        'were', 'is', 'in', 'on', 'at', 'to', 'a', 'an', 'of', 'it', 'its',
        'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should',
        'may', 'might', 'also', 'which', 'their', 'they', 'we', 'our', 'using',
        'by', 'as', 'or', 'but', 'not', 'all', 'can', 'its', 'into', 'than'
    }
    kw_set    = {w.lower() for w in keywords if w.lower() not in stopwords and len(w) > 2}
    boost_set = {w.lower() for w in (boost or [])}

    scored = []
    for sent in sentences:
        words    = set(re.findall(r'\b\w+\b', sent.lower()))
        overlap  = len(words & kw_set)
        boost_sc = len(words & boost_set) * 1.5
        # Reward medium-length sentences
        len_factor = min(len(sent.split()) / 12.0, 1.5)
        score = (overlap + boost_sc) * len_factor
        scored.append((score, sent))

    return sorted(scored, key=lambda x: -x[0])


def _extract_top(text: str, question: str, boost_words: list[str],
                 n: int = 6) -> list[str]:
    """Universal top-N sentence extractor."""
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question) + boost_words
    scored    = _score_sentences(sentences, keywords, boost_words)
    top       = [s for sc, s in scored if sc > 0][:n]
    if not top:
        top = sentences[:min(n, len(sentences))]
    return top


def _api_notice() -> str:
    return (
        "\n\n---\n"
        "💡 *Add `GEMINI_API_KEY` to your `.env` for full AI-powered answers "
        "([get free key](https://aistudio.google.com/app/apikey))*"
    )


# ══ Universal answer handlers ══════════════════════════════════════════════════

def _answer_summary(text: str, question: str,
                    simple_mode: bool, doc_type: str) -> str:
    """
    Generates a smart summary for ANY document type.
    Adapts structure based on detected document type.
    """
    sentences = _get_sentences(text)
    if not sentences:
        return "Could not extract content from document." + _api_notice()

    # Section keywords adapted per document type
    section_map = {
        "research":    {
            "intro":    ['propose', 'present', 'aim', 'objective', 'goal', 'address', 'study'],
            "method":   ['method', 'algorithm', 'model', 'framework', 'architecture', 'train'],
            "result":   ['accuracy', 'result', 'achieve', 'performance', 'show', 'obtain'],
            "conclude": ['conclude', 'future', 'contribute', 'effective', 'novel']
        },
        "legal":       {
            "intro":    ['agreement', 'contract', 'party', 'parties', 'purpose', 'whereas'],
            "method":   ['clause', 'obligation', 'requirement', 'term', 'condition'],
            "result":   ['penalty', 'breach', 'remedy', 'liability', 'damages'],
            "conclude": ['terminate', 'expiry', 'renewal', 'governing', 'jurisdiction']
        },
        "manual":      {
            "intro":    ['overview', 'introduction', 'purpose', 'feature', 'requirement'],
            "method":   ['install', 'configure', 'setup', 'step', 'click', 'select'],
            "result":   ['complete', 'successful', 'result', 'verify', 'confirm'],
            "conclude": ['troubleshoot', 'support', 'contact', 'faq', 'note']
        },
        "business":    {
            "intro":    ['executive', 'summary', 'objective', 'strategy', 'vision'],
            "method":   ['plan', 'initiative', 'approach', 'process', 'implement'],
            "result":   ['revenue', 'growth', 'profit', 'performance', 'kpi', 'roi'],
            "conclude": ['recommendation', 'next', 'action', 'conclusion', 'outlook']
        },
        "medical":     {
            "intro":    ['patient', 'diagnosis', 'condition', 'background', 'history'],
            "method":   ['treatment', 'therapy', 'procedure', 'medication', 'dosage'],
            "result":   ['outcome', 'result', 'improve', 'recovery', 'response'],
            "conclude": ['recommendation', 'follow-up', 'prognosis', 'conclusion']
        },
        "educational": {
            "intro":    ['introduction', 'objective', 'learn', 'topic', 'chapter'],
            "method":   ['method', 'example', 'explanation', 'formula', 'theorem'],
            "result":   ['result', 'solution', 'answer', 'conclusion', 'summary'],
            "conclude": ['exercise', 'practice', 'review', 'key takeaway']
        },
    }

    # Fall back to general if doc_type not in map
    smap = section_map.get(doc_type, section_map["research"])

    def top_n(kws, n=2):
        scored = _score_sentences(sentences, kws, kws)
        return [s for sc, s in scored if sc > 0][:n]

    intro    = top_n(smap["intro"],    3) or sentences[:2]
    methods  = top_n(smap["method"],   3)
    results  = top_n(smap["result"],   2)
    conclude = top_n(smap["conclude"], 2)

    if simple_mode:
        all_sents = (intro + methods + results)[:5]
        answer = "## 📄 Simple Summary\n\n"
        answer += " ".join(all_sents) + "\n"
        return answer + _api_notice()

    # Section labels per doc type
    labels = {
        "research":    ("🎯 Introduction & Objectives", "🔬 Methods", "📊 Results", "🏁 Conclusion"),
        "legal":       ("📋 Agreement Overview",        "📌 Key Terms", "⚖️ Key Provisions", "🔚 Termination"),
        "manual":      ("📖 Overview",                  "🔧 Steps",    "✅ Outcomes",        "ℹ️ Notes"),
        "business":    ("🎯 Executive Summary",         "📋 Strategy", "📈 Performance",     "🔮 Outlook"),
        "medical":     ("🏥 Background",                "💊 Treatment","📊 Outcomes",        "📝 Recommendations"),
        "educational": ("📚 Introduction",              "📖 Content",  "✅ Key Points",      "🎓 Summary"),
        "general":     ("📄 Overview",                  "📌 Content",  "📊 Key Points",      "🏁 Conclusion"),
    }

    l = labels.get(doc_type, labels["general"])
    answer = f"## 📄 Document Summary\n*Document type detected: **{doc_type.title()}***\n\n"

    if intro:
        answer += f"### {l[0]}\n" + " ".join(intro) + "\n\n"
    if methods:
        answer += f"### {l[1]}\n"
        for s in methods:
            answer += f"- {s}\n"
        answer += "\n"
    if results:
        answer += f"### {l[2]}\n"
        for s in results:
            answer += f"- {s}\n"
        answer += "\n"
    if conclude:
        answer += f"### {l[3]}\n" + " ".join(conclude) + "\n"

    return answer + _api_notice()


def _answer_steps_process(text: str, question: str,
                           simple_mode: bool, doc_type: str) -> str:
    """Extract procedural steps from ANY document."""
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question)

    process_signals = [
        'step', 'first', 'second', 'third', 'then', 'next', 'finally',
        'process', 'procedure', 'method', 'approach', 'phase', 'stage',
        'begin', 'start', 'continue', 'complete', 'implement', 'apply',
        'collect', 'preprocess', 'extract', 'classify', 'evaluate', 'detect',
        'install', 'configure', 'click', 'select', 'enter', 'navigate',
        'train', 'test', 'validate', 'deploy', 'calculate', 'analyse'
    ]

    scored = _score_sentences(sentences, keywords + process_signals, process_signals)
    top    = [s for sc, s in scored if sc > 0][:8]
    if not top:
        top = sentences[:6]

    answer = "## 🔬 Process / Methodology\n\n"

    # Check for explicit numbered steps in raw text
    numbered = re.findall(
        r'(?:^|\n)\s*(\d+[\.\)])\s*([A-Z][^\n]{20,200})',
        text, re.MULTILINE
    )

    if len(numbered) >= 3:
        answer += "**Steps found in document:**\n\n"
        for num, step in numbered[:10]:
            answer += f"**{num}** {step.strip()}\n\n"
    else:
        answer += "**Key process components:**\n\n"
        for i, sent in enumerate(top, 1):
            answer += f"**{i}.** {sent.rstrip('.')}.\n\n"

    if simple_mode:
        answer += "\n**Simple explanation:** The document describes a "
        answer += f"{doc_type} process broken into several steps as listed above.\n"

    return answer + _api_notice()


def _answer_results_numbers(text: str, question: str,
                             simple_mode: bool, doc_type: str) -> str:
    """Extract results, numbers, and findings from any document."""
    sentences   = _get_sentences(text)
    keywords    = _get_keywords(question)
    result_boost = [
        'result', 'achieve', 'performance', 'score', 'percent', 'show',
        'accuracy', 'found', 'demonstrate', 'conclude', 'obtain', 'improve',
        'revenue', 'profit', 'growth', 'outcome', 'finding', 'total',
        'increase', 'decrease', 'measure', 'compare', 'evaluate'
    ]

    scored = _score_sentences(sentences, keywords + result_boost, result_boost)
    top    = [s for sc, s in scored if sc > 0][:8]
    if not top:
        top = sentences[:5]

    # Separate numeric sentences from qualitative
    numeric = [s for s in top if re.search(r'\d+\.?\d*\s*%|\d+\.\d+|\d{4,}', s)]
    qualitative = [s for s in top if s not in numeric]

    answer = "## 📊 Results & Key Findings\n\n"

    if numeric:
        answer += "### 📈 Quantitative Results\n"
        for s in numeric:
            answer += f"- {s}\n"
        answer += "\n"

    if qualitative:
        answer += "### 📝 Key Observations\n"
        for s in qualitative[:5]:
            answer += f"- {s}\n"
        answer += "\n"

    return answer + _api_notice()


def _answer_definition(text: str, question: str,
                        simple_mode: bool, doc_type: str) -> str:
    """Answer 'what is X' questions from any document."""
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question)

    if not keywords:
        return _answer_general(text, question, simple_mode, doc_type)

    scored = _score_sentences(sentences, keywords)
    top    = [s for sc, s in scored if sc > 0][:5]

    if not top:
        top = sentences[:3]

    answer = f"## 💡 About: {' '.join(keywords[:3]).title()}\n\n"
    answer += " ".join(top[:2]) + "\n\n"

    if len(top) > 2:
        answer += "**Additional context:**\n"
        for s in top[2:5]:
            answer += f"- {s}\n"

    return answer + _api_notice()


def _answer_comparison(text: str, question: str,
                        simple_mode: bool, doc_type: str) -> str:
    """Answer comparison questions."""
    sentences   = _get_sentences(text)
    keywords    = _get_keywords(question)
    comp_boost  = [
        'compare', 'versus', 'better', 'worse', 'advantage', 'disadvantage',
        'higher', 'lower', 'faster', 'slower', 'more', 'less', 'than',
        'while', 'whereas', 'however', 'contrast', 'unlike', 'similar'
    ]

    scored = _score_sentences(sentences, keywords + comp_boost, comp_boost)
    top    = [s for sc, s in scored if sc > 0][:6]
    if not top:
        top = sentences[:4]

    answer = "## ⚖️ Comparison\n\n"
    for s in top:
        answer += f"- {s}\n"

    return answer + _api_notice()


def _answer_list_items(text: str, question: str,
                        simple_mode: bool, doc_type: str) -> str:
    """Extract list items from any document."""
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question)

    # Look for actual bullet/numbered list items
    raw_bullets = re.findall(
        r'(?:^|\n)\s*(?:[•\-\*]|\d+[\.\)])\s*(.{20,200})',
        text, re.MULTILINE
    )

    answer = f"## 📋 List: {' '.join(keywords[:4]).title()}\n\n"

    if len(raw_bullets) >= 3:
        for item in raw_bullets[:12]:
            answer += f"- {item.strip()}\n"
    else:
        scored = _score_sentences(sentences, keywords)
        top    = [s for sc, s in scored if sc > 0][:8]
        if not top:
            top = sentences[:6]
        for s in top:
            answer += f"- {s}\n"

    return answer + _api_notice()


def _answer_specific_topic(text: str, question: str,
                            simple_mode: bool, doc_type: str) -> str:
    """Answer questions about a specific topic mentioned in the question."""
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question)

    if not keywords:
        return _answer_general(text, question, simple_mode, doc_type)

    scored = _score_sentences(sentences, keywords)
    top    = [s for sc, s in scored if sc > 0][:7]
    if not top:
        top = sentences[:5]

    topic  = ' '.join(keywords[:3]).title()
    answer = f"## 🔍 {topic}\n\n"

    if len(top) <= 3:
        answer += " ".join(top) + "\n\n"
    else:
        answer += top[0] + "\n\n"
        answer += "**Key points:**\n"
        for s in top[1:7]:
            answer += f"- {s}\n"
        answer += "\n"

    return answer + _api_notice()


def _answer_general(text: str, question: str,
                     simple_mode: bool, doc_type: str = "general") -> str:
    """Fallback general answer for any question."""
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question)

    if keywords:
        scored = _score_sentences(sentences, keywords)
        top    = [s for sc, s in scored if sc > 0][:6]
    else:
        top = []

    if not top:
        top = sentences[:5]

    answer = "## 💡 Answer\n\n"

    if simple_mode:
        answer += " ".join(top[:3]) + "\n"
    elif len(top) == 1:
        answer += top[0] + "\n"
    else:
        answer += top[0] + "\n\n"
        if len(top) > 1:
            answer += "**Related information:**\n"
            for s in top[1:6]:
                answer += f"- {s}\n"
        answer += "\n"

    return answer + _api_notice()