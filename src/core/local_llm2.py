"""
DocuMind — Smart Local Answer Engine (No API needed)
Produces clean, well-structured answers using pure Python NLP.
"""

import re


def generate_answer_local(context: str, question: str, simple_mode: bool = False) -> str:
    """
    Master dispatcher — detects question type and routes to best handler.
    """
    # Clean the context first
    text = _clean_context(context)
    q    = question.lower().strip()

    if not text:
        return "❌ Could not extract content from the document."

    # Route by question intent
    if any(w in q for w in ['methodology', 'method', 'approach', 'how does', 'how is',
                              'pipeline', 'process', 'procedure', 'steps', 'workflow']):
        return _answer_methodology(text, question, simple_mode)

    elif any(w in q for w in ['summarise', 'summarize', 'summary', 'overview',
                               'about', 'what is this', 'describe', 'contain',
                               'what does', 'explain the paper', 'tell me about']):
        return _answer_summary(text, question, simple_mode)

    elif any(w in q for w in ['result', 'finding', 'accuracy', 'performance',
                               'score', 'conclusion', 'outcome', 'achieve',
                               'percentage', 'metric', 'f1', 'precision', 'recall']):
        return _answer_results(text, question, simple_mode)

    elif any(w in q for w in ['dataset', 'data', 'corpus', 'sample', 'training data',
                               'test set', 'benchmark']):
        return _answer_dataset(text, question, simple_mode)

    elif any(w in q for w in ['model', 'architecture', 'network', 'layer',
                               'component', 'module', 'structure']):
        return _answer_architecture(text, question, simple_mode)

    elif any(w in q for w in ['future', 'limitation', 'drawback', 'challenge',
                               'improvement', 'scope']):
        return _answer_future(text, question, simple_mode)

    else:
        return _answer_general(text, question, simple_mode)


# ══ Context cleaner ════════════════════════════════════════════════════════════

def _clean_context(context: str) -> str:
    """Strip metadata headers, separators, and normalize whitespace."""
    # Remove section headers added by RAG engine
    text = re.sub(r'\[SECTION \d+.*?\]\n', '', context)
    text = re.sub(r'─{3,}', '\n', text)
    text = re.sub(r'={3,}', '\n', text)
    # Remove PDF artifacts
    text = re.sub(r'\[EXCERPT \d+.*?\]', '', text)
    text = re.sub(r'Document Source \d+:.*?---', '', text, flags=re.DOTALL)
    text = re.sub(r'---\s*Document Source.*', '', text, flags=re.DOTALL)
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# ══ Sentence utilities ══════════════════════════════════════════════════════════

def _get_sentences(text: str) -> list[str]:
    """Split into clean sentences, filter noise."""
    raw = re.split(r'(?<=[.!?])\s+|\n', text)
    good = []
    for s in raw:
        s = s.strip()
        # Skip too short, too long, or mostly numbers/symbols
        if len(s) < 20:
            continue
        if len(s) > 400:
            # Split long sentences further
            parts = re.split(r'(?<=,)\s+(?=[A-Z])', s)
            good.extend([p.strip() for p in parts if len(p.strip()) > 20])
            continue
        # Skip lines that are just figure captions or table headers
        if re.match(r'^(fig|table|figure)\s*[\d\.]+', s, re.IGNORECASE):
            continue
        good.append(s)
    return good


def _score_sentences(sentences: list[str], keywords: list[str],
                     boost_words: list[str] = None) -> list[tuple]:
    """Score sentences by keyword relevance."""
    stopwords = {
        'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are', 'was',
        'were', 'what', 'does', 'tell', 'about', 'explain', 'describe', 'how',
        'why', 'is', 'in', 'on', 'at', 'to', 'a', 'an', 'of', 'it', 'its',
        'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should',
        'may', 'might', 'also', 'which', 'their', 'they', 'we', 'our', 'using'
    }

    kw_set    = {w.lower() for w in keywords if w.lower() not in stopwords and len(w) > 2}
    boost_set = {w.lower() for w in (boost_words or [])}

    scored = []
    for sent in sentences:
        words   = set(re.findall(r'\b\w+\b', sent.lower()))
        overlap = len(words & kw_set)
        boost   = len(words & boost_set) * 2
        # Penalise very short sentences
        length_factor = min(len(sent.split()) / 15.0, 1.0)
        score = (overlap + boost) * length_factor
        scored.append((score, sent))

    return sorted(scored, key=lambda x: -x[0])


def _get_keywords(question: str) -> list[str]:
    """Extract meaningful keywords from the question."""
    noise = {
        'what', 'is', 'are', 'the', 'a', 'an', 'tell', 'me', 'about', 'explain',
        'describe', 'how', 'why', 'does', 'do', 'this', 'that', 'give', 'please',
        'can', 'you', 'list', 'show', 'main', 'key', 'primary', 'paper', 'document'
    }
    return [w for w in re.findall(r'\b\w{3,}\b', question.lower()) if w not in noise]


def _api_notice() -> str:
    return (
        "\n\n---\n"
        "💡 *Add `GEMINI_API_KEY` to your `.env` file for full AI-powered answers "
        "from Google Gemini (free at [aistudio.google.com](https://aistudio.google.com/app/apikey)).*"
    )


# ══ Answer handlers ═════════════════════════════════════════════════════════════

def _answer_methodology(text: str, question: str, simple_mode: bool) -> str:
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question)

    method_boost = [
        'propose', 'method', 'step', 'approach', 'algorithm', 'technique',
        'framework', 'pipeline', 'process', 'implement', 'architecture',
        'detect', 'classify', 'extract', 'train', 'preprocess', 'collect',
        'capture', 'input', 'output', 'layer', 'module', 'component',
        'first', 'second', 'third', 'then', 'next', 'finally', 'phase'
    ]

    scored = _score_sentences(sentences, keywords + method_boost, method_boost)

    # Get top relevant sentences
    top = [s for score, s in scored if score > 0][:8]
    if not top:
        top = sentences[:6]

    answer = "## 🔬 Methodology\n\n"

    # Try to find explicit numbered steps in the raw text
    numbered = re.findall(
        r'(?:^|\n)\s*(\d+[\.\)])\s*([A-Z][^.\n]{20,150})',
        text, re.MULTILINE
    )

    if len(numbered) >= 3:
        answer += "**Process Steps:**\n\n"
        for num, step in numbered[:8]:
            step = step.strip().rstrip('.')
            answer += f"**{num}** {step}\n\n"
    else:
        # Use scored sentences as steps
        answer += "**Key methodology components:**\n\n"
        for i, sent in enumerate(top, 1):
            sent = sent.rstrip('.')
            answer += f"**{i}.** {sent}.\n\n"

    if simple_mode:
        answer += "\n**In simple terms:** This paper proposes a system that goes through "
        answer += "several steps to automatically analyze and detect patterns from the input data.\n"

    return answer + _api_notice()


def _answer_summary(text: str, question: str, simple_mode: bool) -> str:
    sentences = _get_sentences(text)

    # Score for different document sections
    intro_kws    = ['propose', 'present', 'paper', 'study', 'research', 'aim',
                    'objective', 'goal', 'address', 'problem', 'system', 'approach']
    method_kws   = ['method', 'algorithm', 'technique', 'framework', 'model',
                    'architecture', 'network', 'train', 'dataset', 'implement']
    result_kws   = ['accuracy', 'result', 'achieve', 'performance', 'show',
                    'demonstrate', 'obtain', 'percent', 'improve', 'score']
    conclude_kws = ['conclude', 'conclusion', 'future', 'limitation', 'contribute',
                    'significant', 'effective', 'novel', 'real-time', 'integrated']

    def top_sents(kws, n=2):
        scored = _score_sentences(sentences, kws, kws)
        return [s for sc, s in scored if sc > 0][:n]

    intro    = top_sents(intro_kws,    3)
    methods  = top_sents(method_kws,   3)
    results  = top_sents(result_kws,   2)
    conclude = top_sents(conclude_kws, 2)

    # Fallback if nothing scored well
    if not intro:
        intro = sentences[:2]

    answer = "## 📄 Document Summary\n\n"

    if intro:
        answer += "### 🎯 Overview\n"
        answer += " ".join(intro) + "\n\n"

    if methods:
        answer += "### 🔬 Methods & Approach\n"
        for s in methods:
            answer += f"- {s}\n"
        answer += "\n"

    if results:
        answer += "### 📊 Key Results\n"
        for s in results:
            answer += f"- {s}\n"
        answer += "\n"

    if conclude:
        answer += "### 🏁 Conclusion\n"
        answer += " ".join(conclude) + "\n\n"

    if simple_mode:
        all_top = (intro + methods + results + conclude)[:4]
        answer  = "## 📄 Simple Summary\n\n"
        answer += " ".join(all_top) + "\n\n"

    return answer + _api_notice()


def _answer_results(text: str, question: str, simple_mode: bool) -> str:
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question)

    result_boost = [
        'accuracy', 'result', 'achieve', 'performance', 'score', 'percent',
        'precision', 'recall', 'f1', 'loss', 'improve', 'obtain', 'show',
        'demonstrate', 'outperform', 'compare', 'baseline', 'metric',
        'evaluate', 'test', 'validation', 'epoch', 'dataset', 'model'
    ]

    scored = _score_sentences(sentences, keywords + result_boost, result_boost)
    top    = [s for sc, s in scored if sc > 0][:8]

    if not top:
        top = sentences[:5]

    answer = "## 📊 Results & Findings\n\n"

    # Find lines with numbers/percentages — likely result sentences
    numeric_sents = [s for s in top if re.search(r'\d+\.?\d*\s*%|\d+\.\d+', s)]
    other_sents   = [s for s in top if s not in numeric_sents]

    if numeric_sents:
        answer += "### 📈 Quantitative Results\n"
        for s in numeric_sents:
            answer += f"- {s}\n"
        answer += "\n"

    if other_sents:
        answer += "### 📝 Key Observations\n"
        for s in other_sents[:4]:
            answer += f"- {s}\n"
        answer += "\n"

    return answer + _api_notice()


def _answer_dataset(text: str, question: str, simple_mode: bool) -> str:
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question)

    dataset_boost = [
        'dataset', 'data', 'collect', 'sample', 'image', 'video', 'annotate',
        'label', 'split', 'train', 'test', 'validation', 'benchmark', 'corpus',
        'subject', 'participant', 'record', 'frame', 'class', 'category'
    ]

    scored = _score_sentences(sentences, keywords + dataset_boost, dataset_boost)
    top    = [s for sc, s in scored if sc > 0][:6]

    if not top:
        top = sentences[:4]

    answer = "## 🗂️ Dataset Information\n\n"
    for s in top:
        answer += f"- {s}\n"

    return answer + _api_notice()


def _answer_architecture(text: str, question: str, simple_mode: bool) -> str:
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question)

    arch_boost = [
        'model', 'layer', 'network', 'architecture', 'component', 'module',
        'input', 'output', 'feature', 'convolution', 'pooling', 'dense',
        'activation', 'dropout', 'batch', 'encoder', 'decoder', 'attention',
        'transformer', 'cnn', 'rnn', 'lstm', 'resnet', 'vgg', 'backbone'
    ]

    scored = _score_sentences(sentences, keywords + arch_boost, arch_boost)
    top    = [s for sc, s in scored if sc > 0][:7]

    if not top:
        top = sentences[:5]

    answer = "## 🏗️ Model Architecture\n\n"
    for i, s in enumerate(top, 1):
        answer += f"**{i}.** {s}\n\n"

    return answer + _api_notice()


def _answer_future(text: str, question: str, simple_mode: bool) -> str:
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question)

    future_boost = [
        'future', 'limitation', 'challenge', 'improve', 'extend', 'scope',
        'propose', 'further', 'plan', 'potential', 'drawback', 'constrain',
        'address', 'work', 'direction', 'next', 'upcoming', 'investigate'
    ]

    scored = _score_sentences(sentences, keywords + future_boost, future_boost)
    top    = [s for sc, s in scored if sc > 0][:6]

    if not top:
        top = sentences[-5:]  # often at the end of papers

    answer = "## 🔮 Future Work & Limitations\n\n"
    for s in top:
        answer += f"- {s}\n"

    return answer + _api_notice()


def _answer_general(text: str, question: str, simple_mode: bool) -> str:
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

    if len(top) <= 2:
        answer += " ".join(top) + "\n\n"
    else:
        answer += top[0] + "\n\n"
        answer += "**Key points:**\n"
        for s in top[1:5]:
            answer += f"- {s}\n"
        answer += "\n"

    return answer + _api_notice()