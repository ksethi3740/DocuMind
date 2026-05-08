"""
DocuMind — Universal Local Answer Engine
Works for ANY document. No API needed.
"""

import re


_STOPWORDS = {
    'the','and','for','that','this','with','from','are','was','were',
    'what','does','tell','about','explain','describe','how','why','is',
    'in','on','at','to','a','an','of','it','its','be','been','have',
    'has','had','will','would','could','should','may','might','also',
    'which','their','they','we','our','using','by','as','or','but',
    'not','all','can','into','than','then','there','these','those'
}


def generate_answer_local(context: str, question: str, simple_mode: bool = False) -> str:
    text = _clean_context(context)
    q    = question.lower().strip()

    if not text or len(text) < 30:
        return "❌ Could not extract content from the document."

    intent = _detect_intent(q)

    if intent == "methodology":
        return _answer_methodology(text, question, simple_mode)
    elif intent == "summary":
        return _answer_summary(text, question, simple_mode)
    elif intent == "results":
        return _answer_results(text, question, simple_mode)
    elif intent == "list":
        return _answer_list(text, question, simple_mode)
    elif intent == "definition":
        return _answer_definition(text, question, simple_mode)
    else:
        return _answer_general(text, question, simple_mode)


def _detect_intent(q: str) -> str:
    if any(w in q for w in [
        'methodology','method','approach','how does','how is','pipeline',
        'process','procedure','steps','workflow','how to','technique',
        'algorithm','implementation','framework','mechanism'
    ]):
        return "methodology"

    if any(w in q for w in [
        'summarise','summarize','summary','overview','about','what is this',
        'what does this','describe','contain','tell me about','explain the',
        'what is the document','what is the paper','main topic','brief'
    ]):
        return "summary"

    if any(w in q for w in [
        'result','finding','accuracy','performance','score','conclusion',
        'outcome','achieve','percentage','metric','f1','precision','recall',
        'what was found','how well','how accurate','evaluation','benchmark'
    ]):
        return "results"

    if any(w in q for w in [
        'list','enumerate','what are all','types of','kinds of','features',
        'components','elements','items','categories','examples of','name the'
    ]):
        return "list"

    if any(w in q for w in [
        'what is','define','definition','meaning','explain what',
        'what does','what are','concept of'
    ]):
        return "definition"

    return "general"


def _clean_context(context: str) -> str:
    text = re.sub(r'---\s*Document Source \d+:[^\n]*\n?', '\n', context)
    text = re.sub(r'\[SECTION \d+[^\]]*\]\n?', '', text)
    text = re.sub(r'─{3,}', '\n', text)
    text = re.sub(r'={3,}', '\n', text)
    text = re.sub(r'\[EXCERPT \d+[^\]]*\]\n?', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def _get_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\s+|\n', text)
    good = []
    for s in raw:
        s = s.strip()
        if len(s) < 20:
            continue
        if re.match(r'^[\d\s\.\-\|]+$', s):
            continue
        if re.match(r'^(fig|table|figure|eq)\s*[\d\.]+', s, re.IGNORECASE):
            continue
        if len(s) > 500:
            parts = re.split(r';\s+', s)
            good.extend([p.strip() for p in parts if len(p.strip()) > 20])
        else:
            good.append(s)
    return good


def _get_keywords(question: str) -> list[str]:
    noise = {
        'what','is','are','the','a','an','tell','me','about','explain',
        'describe','how','why','does','do','this','that','give','please',
        'can','you','list','show','main','key','primary','paper','document',
        'report','file','text','which','where','when','who','have','has',
        'was','were','will','would','does','did'
    }
    return [w for w in re.findall(r'\b\w{3,}\b', question.lower()) if w not in noise]


def _score(sentences: list[str], keywords: list[str],
           boost: list[str] = None) -> list[tuple]:
    kw_set    = {w for w in keywords if w not in _STOPWORDS and len(w) > 2}
    boost_set = {w for w in (boost or [])}
    scored = []
    for sent in sentences:
        words    = set(re.findall(r'\b\w+\b', sent.lower()))
        overlap  = len(words & kw_set)
        boost_sc = len(words & boost_set) * 1.5
        length_f = min(len(sent.split()) / 12.0, 1.5)
        scored.append(((overlap + boost_sc) * length_f, sent))
    return sorted(scored, key=lambda x: -x[0])


def _top(text: str, question: str, boost: list[str], n: int = 8) -> list[str]:
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question) + boost
    scored    = _score(sentences, keywords, boost)
    top       = [s for sc, s in scored if sc > 0][:n]
    return top if top else sentences[:min(n, len(sentences))]


def _notice() -> str:
    return (
        "\n\n---\n"
        "💡 *Add `GEMINI_API_KEY` to `.env` for full AI answers "
        "([free key here](https://aistudio.google.com/app/apikey))*"
    )


# ══ Answer handlers ═══════════════════════════════════════════════════════════

def _answer_methodology(text: str, question: str, simple_mode: bool) -> str:
    sentences = _get_sentences(text)
    keywords  = _get_keywords(question)

    boost = [
        'step','first','second','third','fourth','fifth','then','next',
        'finally','phase','stage','begin','propose','method','approach',
        'algorithm','technique','framework','pipeline','process','implement',
        'detect','classify','extract','train','preprocess','collect',
        'capture','input','output','layer','module','component','system',
        'image','video','frame','feature','model','network','classify',
        'evaluate','measure','compute','apply','use','perform','generate'
    ]

    scored = _score(sentences, keywords + boost, boost)

    # Get ALL sentences with any relevance score > 0
    all_relevant = [s for sc, s in scored if sc > 0]

    answer = "## 🔬 Methodology\n\n"

    # Strategy 1: find explicit numbered steps in raw text
    numbered = re.findall(
        r'(?:^|\n)\s*(\d+[\.\)])\s*([A-Z][^\n]{15,250})',
        text, re.MULTILINE
    )
    if len(numbered) >= 2:
        answer += "**Process Steps:**\n\n"
        for num, step in numbered[:10]:
            answer += f"**{num}** {step.strip().rstrip('.')}\n\n"
        return answer + _notice()

    # Strategy 2: find phase/step keywords with context
    phase_pattern = re.findall(
        r'(?:step|phase|stage|first|second|third|fourth|fifth|finally|then|next)'
        r'[,\s]+([A-Z][^\n.]{20,200})',
        text, re.IGNORECASE
    )
    if len(phase_pattern) >= 2:
        answer += "**Process Steps:**\n\n"
        seen = set()
        count = 1
        for step in phase_pattern[:10]:
            step = step.strip().rstrip('.')
            if step not in seen and len(step.split()) >= 3:
                seen.add(step)
                answer += f"**Step {count}.** {step}\n\n"
                count += 1
        if count > 2:
            return answer + _notice()

    # Strategy 3: use all relevant scored sentences
    if all_relevant:
        answer += "**Key methodology components:**\n\n"
        for i, sent in enumerate(all_relevant[:8], 1):
            answer += f"**{i}.** {sent.rstrip('.')}\n\n"
    else:
        # Last resort: first 5 sentences of document
        answer += "**Document content:**\n\n"
        for i, sent in enumerate(sentences[:5], 1):
            answer += f"**{i}.** {sent}\n\n"

    if simple_mode:
        answer += "\n**In simple terms:** The document describes a step-by-step system "
        answer += "as listed above.\n"

    return answer + _notice()


def _answer_summary(text: str, question: str, simple_mode: bool) -> str:
    sentences = _get_sentences(text)

    intro_kws    = ['propose','present','paper','study','aim','objective',
                    'goal','address','problem','system','approach','research']
    method_kws   = ['method','algorithm','model','framework','architecture',
                    'network','train','dataset','implement','technique']
    result_kws   = ['accuracy','result','achieve','performance','show',
                    'demonstrate','obtain','percent','improve','score']
    conclude_kws = ['conclude','conclusion','future','limitation','contribute',
                    'significant','effective','novel','real-time','integrated']

    def top_n(kws, n=3):
        sc = _score(sentences, kws, kws)
        return [s for score, s in sc if score > 0][:n]

    intro    = top_n(intro_kws,    3) or sentences[:2]
    methods  = top_n(method_kws,   3)
    results  = top_n(result_kws,   2)
    conclude = top_n(conclude_kws, 2)

    if simple_mode:
        all_top = (intro + methods + results)[:5]
        return "## 📄 Simple Summary\n\n" + " ".join(all_top) + "\n" + _notice()

    answer = "## 📄 Document Summary\n\n"
    if intro:
        answer += "### 🎯 Overview\n" + " ".join(intro) + "\n\n"
    if methods:
        answer += "### 🔬 Methods\n"
        for s in methods:
            answer += f"- {s}\n"
        answer += "\n"
    if results:
        answer += "### 📊 Results\n"
        for s in results:
            answer += f"- {s}\n"
        answer += "\n"
    if conclude:
        answer += "### 🏁 Conclusion\n" + " ".join(conclude) + "\n"

    return answer + _notice()


def _answer_results(text: str, question: str, simple_mode: bool) -> str:
    boost = [
        'accuracy','result','achieve','performance','score','percent',
        'precision','recall','f1','improve','obtain','show','demonstrate',
        'outperform','compare','metric','evaluate','test','validation'
    ]
    top = _top(text, question, boost, 8)

    numeric = [s for s in top if re.search(r'\d+\.?\d*\s*%|\d+\.\d+', s)]
    other   = [s for s in top if s not in numeric]

    answer = "## 📊 Results & Findings\n\n"
    if numeric:
        answer += "### 📈 Quantitative Results\n"
        for s in numeric:
            answer += f"- {s}\n"
        answer += "\n"
    if other:
        answer += "### 📝 Key Observations\n"
        for s in other[:4]:
            answer += f"- {s}\n"
        answer += "\n"

    return answer + _notice()


def _answer_list(text: str, question: str, simple_mode: bool) -> str:
    keywords = _get_keywords(question)

    # Look for actual bullet/numbered items first
    raw = re.findall(
        r'(?:^|\n)\s*(?:[•\-\*]|\d+[\.\)])\s*(.{20,200})',
        text, re.MULTILINE
    )

    answer = "## 📋 Key Points\n\n"
    if len(raw) >= 3:
        for item in raw[:12]:
            answer += f"- {item.strip()}\n"
    else:
        top = _top(text, question, keywords, 8)
        for s in top:
            answer += f"- {s}\n"

    return answer + _notice()


def _answer_definition(text: str, question: str, simple_mode: bool) -> str:
    keywords = _get_keywords(question)
    top      = _top(text, question, keywords, 5)

    topic  = ' '.join(keywords[:3]).title() if keywords else 'Topic'
    answer = f"## 💡 About: {topic}\n\n"
    answer += " ".join(top[:2]) + "\n\n"
    if len(top) > 2:
        answer += "**More context:**\n"
        for s in top[2:5]:
            answer += f"- {s}\n"

    return answer + _notice()


def _answer_general(text: str, question: str, simple_mode: bool) -> str:
    keywords = _get_keywords(question)
    top      = _top(text, question, keywords, 6) if keywords else []

    if not top:
        top = _get_sentences(text)[:5]

    answer = "## 💡 Answer\n\n"
    if simple_mode:
        answer += " ".join(top[:3]) + "\n"
    else:
        answer += top[0] + "\n\n"
        if len(top) > 1:
            answer += "**Related information:**\n"
            for s in top[1:6]:
                answer += f"- {s}\n"
        answer += "\n"

    return answer + _notice()