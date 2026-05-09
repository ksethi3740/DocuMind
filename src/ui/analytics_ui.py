# analytics_ui.py — DocuMind
# Fixed version: correct HTML rendering, better topic modeling,
# compact pie chart, working information-dense sentences section.

import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ── Optional heavy deps (graceful fallback if missing) ────────────────────────
try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

try:
    from keybert import KeyBERT
    HAS_KEYBERT = True
except ImportError:
    HAS_KEYBERT = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── Stopwords ──────────────────────────────────────────────────────────────────
_STOPWORDS = {
    "paper", "study", "method", "methods", "results", "conclusion",
    "proposed", "system", "model", "models", "using", "based",
    "approach", "analysis", "research", "ieee", "et", "al",
    "figure", "table", "dataset", "datasets", "accuracy", "performance",
    "achieved", "training", "testing", "data", "used", "use", "problem",
    "problems", "technique", "techniques", "framework", "application",
    "applications", "also", "show", "shown", "can", "will", "two",
    "one", "three", "four", "five", "first", "second", "third",
    "however", "therefore", "thus", "well", "this", "that", "with",
    "from", "they", "their", "which", "have", "been", "were", "more"
}

_COMMON_EN = {
    "the", "and", "for", "that", "with", "from", "are", "was", "were",
    "is", "in", "on", "at", "to", "a", "an", "of", "it", "its",
    "be", "been", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "as", "or", "but", "not", "all", "by",
    "into", "than", "then", "these", "those", "about", "after", "each"
}


# ── Text utilities ─────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r'\[[0-9]+\]', '', text)          # remove citation numbers
    text = re.sub(r'https?://\S+', '', text)         # remove URLs
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\s+', text)
    out = []
    for s in raw:
        s = s.strip()
        if len(s.split()) < 6:
            continue
        if s.count('|') > 3:       # table row
            continue
        if re.search(r'https?://', s):
            continue
        digits = sum(c.isdigit() for c in s)
        if digits / max(len(s), 1) > 0.25:  # mostly numbers
            continue
        out.append(s)
    return out


def _keywords_fallback(words: list[str], n: int = 15) -> list[tuple]:
    """Simple TF-based keyword extraction when KeyBERT unavailable."""
    filtered = [
        w.lower() for w in words
        if len(w) > 3
        and w.lower() not in _STOPWORDS
        and w.lower() not in _COMMON_EN
        and w.isalpha()
    ]
    freq = Counter(filtered)
    total = max(sum(freq.values()), 1)
    return [(w, round(c / total * 100, 1)) for w, c in freq.most_common(n)]


def _lda_topics(sentences: list[str]) -> list[tuple[str, int]]:
    """Run LDA topic modeling and return (topic_label, score) pairs."""
    if not HAS_SKLEARN or len(sentences) < 5:
        return []
    try:
        vec = CountVectorizer(
            stop_words='english',
            max_df=0.90,
            min_df=1,           # min_df=1 so short docs still work
            max_features=500
        )
        X = vec.fit_transform(sentences)
        if X.shape[1] < 3:
            return []

        n_topics = min(5, len(sentences) // 3, X.shape[1] // 2)
        if n_topics < 2:
            return []

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
        feature_names = vec.get_feature_names_out()

        topics = []
        for idx, topic in enumerate(lda.components_):
            top_idx   = topic.argsort()[-12:][::-1]
            top_words = []
            for i in top_idx:
                w = feature_names[i]
                if (len(w) > 3
                        and w.lower() not in _STOPWORDS
                        and w.lower() not in _COMMON_EN
                        and not w.isdigit()):
                    top_words.append(w)
            top_words = list(dict.fromkeys(top_words))[:3]
            if not top_words:
                continue
            label = ", ".join(top_words)
            score = max(100 - idx * 8, 60)
            topics.append((label, score))
        return topics
    except Exception:
        return []


def _tfidf_dense(sentences: list[str], n: int = 5) -> list[tuple[str, float]]:
    """Return top-N information-dense sentences via TF-IDF scoring."""
    if not HAS_SKLEARN or len(sentences) < 2:
        # Fallback: longest sentences
        scored = sorted(sentences, key=lambda s: len(s.split()), reverse=True)
        return [(s, 1.0) for s in scored[:n]]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        X     = tfidf.fit_transform(sentences)
        scores = []
        for i, sent in enumerate(sentences):
            tf_sc        = float(X[i].sum())
            unique_ratio = len(set(sent.split())) / max(len(sent.split()), 1)
            len_bonus    = min(len(sent.split()) / 20, 1.5)
            score        = tf_sc * 0.65 + unique_ratio * 0.25 + len_bonus * 0.10
            scores.append((sent, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:n]
    except Exception:
        return [(s, 1.0) for s in sentences[:n]]


# ── Metric card helper (pure Streamlit — no unsafe HTML) ──────────────────────

def _metric_card(col, icon: str, label: str, value: str):
    with col:
        st.markdown(f"**{icon} {label}**")
        st.markdown(f"### {value}")


# ══ MAIN RENDER ════════════════════════════════════════════════════════════════

def render_analytics_tab(processed_docs):
    st.title("📊 Document Analytics")
    st.markdown(
        "Deep semantic analysis of document structure, "
        "topics, readability, semantics, and information density."
    )

    if not processed_docs:
        st.warning("No documents uploaded yet.")
        return

    # ── Document selector ─────────────────────────────────────────────────────
    if isinstance(processed_docs, dict):
        doc_names = list(processed_docs.keys())
        selected  = st.selectbox("Choose document:", doc_names, key="an_doc")
        doc_data  = processed_docs[selected]
    else:
        doc_names = [
            d.get("name") or d.get("filename") or f"Document {i+1}"
            for i, d in enumerate(processed_docs)
        ]
        selected  = st.selectbox("Choose document:", doc_names, key="an_doc")
        doc_data  = processed_docs[doc_names.index(selected)]

    full_text = _clean(doc_data.get("text", ""))
    if not full_text:
        st.warning("Document appears empty.")
        return

    # ── NLP computations ──────────────────────────────────────────────────────
    words          = re.findall(r'\b[a-zA-Z]\w+\b', full_text)
    word_count     = len(words)
    unique_words   = len({w.lower() for w in words})
    char_count     = len(full_text)
    sents          = _sentences(full_text)
    sentence_count = len(sents)
    paragraphs     = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 20]
    para_count     = len(paragraphs)
    chunks         = doc_data.get("chunks", [])
    chunk_count    = len(chunks) or doc_data.get("chunk_count", 0)
    avg_sl         = round(word_count / max(sentence_count, 1), 1)
    avg_wl         = round(sum(len(w) for w in words) / max(word_count, 1), 1)
    vocab_r        = round(unique_words / max(word_count, 1) * 100, 1)
    reading_time   = round(word_count / 200, 1)

    readability = grade = None
    if HAS_TEXTSTAT:
        try:
            readability = round(textstat.flesch_reading_ease(full_text), 1)
            grade       = round(textstat.flesch_kincaid_grade(full_text), 1)
        except Exception:
            pass

    # ══ SECTION 1 — Metric cards ═══════════════════════════════════════════════
    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📝 Words",        f"{word_count:,}")
    c2.metric("📄 Sentences",    f"{sentence_count:,}")
    c3.metric("📚 Paragraphs",   f"{para_count:,}")
    c4.metric("🔤 Unique Words", f"{unique_words:,}")
    c5.metric("✂️ Chunks",       f"{chunk_count:,}")

    # ══ SECTION 2 — Keywords + Doc Stats ══════════════════════════════════════
    st.divider()
    left, right = st.columns([2.3, 1])

    with left:
        st.subheader("🔑 Top Semantic Keywords")

        if HAS_KEYBERT:
            try:
                kw_model = KeyBERT()
                raw_kws  = kw_model.extract_keywords(
                    full_text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=15
                )
                kw_pairs = [(kw, round(sc * 100, 1)) for kw, sc in raw_kws]
            except Exception:
                kw_pairs = _keywords_fallback(words, 15)
        else:
            kw_pairs = _keywords_fallback(words, 15)

        if kw_pairs:
            kw_df = pd.DataFrame(kw_pairs, columns=["Keyword", "Score"])
            fig_kw = px.bar(
                kw_df.iloc[::-1],
                x="Score", y="Keyword",
                orientation="h",
                color="Score",
                color_continuous_scale=["#4f46e5", "#7c3aed", "#06b6d4"],
                labels={"Score": "Relevance Score", "Keyword": ""}
            )
            fig_kw.update_coloraxes(showscale=False)
            fig_kw.update_layout(
                height=500,
                margin=dict(l=0, r=20, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1', size=12),
                xaxis=dict(gridcolor='rgba(99,102,241,0.15)', zeroline=False),
                yaxis=dict(gridcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("No keywords extracted.")

    with right:
        st.subheader("📈 Document Stats")
        stats_rows = [
            ("Avg sentence",    f"{avg_sl} words"),
            ("Avg word length", f"{avg_wl} chars"),
            ("Vocab richness",  f"{vocab_r}%"),
            ("Reading time",    f"{reading_time} min"),
            ("Characters",      f"{char_count:,}"),
        ]
        if readability is not None:
            stats_rows.insert(2, ("Readability",  str(readability)))
            stats_rows.insert(3, ("Grade level",  str(grade)))

        stats_df = pd.DataFrame(stats_rows, columns=["Metric", "Value"])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Reading ease indicator
        if readability is not None:
            ease_label = (
                "Very Easy"   if readability >= 80 else
                "Easy"        if readability >= 60 else
                "Moderate"    if readability >= 40 else
                "Difficult"   if readability >= 20 else
                "Very Difficult"
            )
            ease_color = (
                "#86efac" if readability >= 60 else
                "#fbbf24" if readability >= 40 else
                "#f87171"
            )
            st.markdown(
                f"**Reading ease:** "
                f"<span style='color:{ease_color};font-weight:600;'>"
                f"{ease_label}</span>",
                unsafe_allow_html=True
            )

    # ══ SECTION 3 — Pie chart + Topic modeling ════════════════════════════════
    st.divider()
    left2, right2 = st.columns([1.1, 1.3])

    with left2:
        st.subheader("📏 Sentence Length Distribution")
        short_c  = sum(1 for s in sents if len(s.split()) <= 10)
        medium_c = sum(1 for s in sents if 10 < len(s.split()) <= 25)
        long_c   = sum(1 for s in sents if len(s.split()) > 25)

        fig_pie = go.Figure(go.Pie(
            labels=["Short (≤10 words)", "Medium (11–25)", "Long (>25)"],
            values=[short_c, medium_c, long_c],
            hole=0.52,
            marker_colors=["#ef4444", "#6366f1", "#06b6d4"],
            textinfo="label+percent",
            textfont_size=13,
            insidetextorientation="radial"
        ))
        fig_pie.update_layout(
            height=340,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', size=12),
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with right2:
        st.subheader("🧩 Semantic Topics")
        topics = _lda_topics(sents)

        if topics:
            for label, score in topics:
                # Use native Streamlit — no unsafe HTML needed
                st.markdown(f"**{label}**")
                st.progress(score / 100, text=f"{score}%")
                st.markdown("")
        else:
            # Fallback: show top bigrams as pseudo-topics
            st.caption("(LDA topic modeling — using keyword clusters as fallback)")
            bigrams = []
            for i in range(len(words) - 1):
                w1, w2 = words[i].lower(), words[i+1].lower()
                if (w1 not in _COMMON_EN and w2 not in _COMMON_EN
                        and len(w1) > 3 and len(w2) > 3
                        and w1.isalpha() and w2.isalpha()):
                    bigrams.append(f"{w1} {w2}")
            top_bigrams = Counter(bigrams).most_common(6)
            max_c = top_bigrams[0][1] if top_bigrams else 1
            for bg, cnt in top_bigrams:
                pct = int(cnt / max_c * 100)
                st.markdown(f"**{bg}**")
                st.progress(pct / 100, text=f"{pct}%")

    # ══ SECTION 4 — Word frequency bar ════════════════════════════════════════
    st.divider()
    st.subheader("📌 Word Frequency Analysis")

    freq_words = [
        w.lower() for w in words
        if len(w) > 3
        and w.lower() not in _STOPWORDS
        and w.lower() not in _COMMON_EN
        and w.isalpha()
    ]
    freq_counts = Counter(freq_words).most_common(20)

    if freq_counts:
        freq_df  = pd.DataFrame(freq_counts, columns=["Word", "Frequency"])
        fig_freq = px.bar(
            freq_df,
            x="Word", y="Frequency",
            color="Frequency",
            color_continuous_scale=["#4f46e5", "#06b6d4"],
            labels={"Word": "", "Frequency": "Count"}
        )
        fig_freq.update_coloraxes(showscale=False)
        fig_freq.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', size=12),
            xaxis=dict(tickangle=-35, gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(gridcolor='rgba(99,102,241,0.15)')
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    # ══ SECTION 5 — Extracted tables ══════════════════════════════════════════
    st.divider()
    table_lines = [
        ln.strip() for ln in full_text.splitlines()
        if "|" in ln and len(ln.strip()) > 5
    ]

    if len(table_lines) >= 3:
        st.subheader("📋 Extracted Tables")
        # Group consecutive pipe-rows into tables
        groups, cur = [], []
        for ln in table_lines:
            if ln:
                cur.append(ln)
            else:
                if len(cur) >= 2:
                    groups.append(cur)
                cur = []
        if len(cur) >= 2:
            groups.append(cur)

        for grp in groups[:3]:   # show max 3 tables
            rows = []
            for row in grp:
                cols = [c.strip() for c in row.split("|") if c.strip()]
                if cols:
                    rows.append(cols)
            if rows:
                max_c = max(len(r) for r in rows)
                norm  = [r + [""] * (max_c - len(r)) for r in rows]
                df    = pd.DataFrame(norm)
                st.dataframe(df, use_container_width=True)

    # ══ SECTION 6 — Most information-dense sentences ══════════════════════════
    st.divider()
    st.subheader("🌟 Most Information-Dense Sentences")

    ranked = _tfidf_dense(sents, n=5)
    if ranked:
        for idx, (sent, score) in enumerate(ranked, start=1):
            # Use st.container(border=True) — pure Streamlit, no HTML injection
            with st.container(border=True):
                col_num, col_text = st.columns([0.06, 0.94])
                with col_num:
                    st.markdown(
                        f"<div style='font-size:1.4rem;font-weight:700;"
                        f"color:#6366f1;padding-top:4px;'>{idx}</div>",
                        unsafe_allow_html=True
                    )
                with col_text:
                    st.markdown(sent)
                st.caption(f"Information score: {round(score, 3)}")
    else:
        st.info("Not enough sentences to rank.")

    # ══ SECTION 7 — Download report ═══════════════════════════════════════════
    st.divider()
    report_lines = [
        f"DocuMind Analytics Report — {selected}",
        "=" * 60,
        f"Words:           {word_count:,}",
        f"Sentences:       {sentence_count:,}",
        f"Paragraphs:      {para_count:,}",
        f"Unique Words:    {unique_words:,}",
        f"Vocab Richness:  {vocab_r}%",
        f"Avg Sentence:    {avg_sl} words",
        f"Reading Time:    {reading_time} min",
    ]
    if readability is not None:
        report_lines += [
            f"Readability:     {readability}",
            f"Grade Level:     {grade}",
        ]
    report_lines += ["", "--- Top Keywords ---"]
    if kw_pairs:
        for kw, sc in kw_pairs[:10]:
            report_lines.append(f"  {kw}: {sc}%")
    report_lines += ["", "--- Information-Dense Sentences ---"]
    for i, (s, _) in enumerate(ranked, 1):
        report_lines.append(f"  {i}. {s}")

    st.download_button(
        "⬇️ Download Full Analytics Report",
        data="\n".join(report_lines),
        file_name=f"analytics_{selected}.txt",
        mime="text/plain",
        use_container_width=True
    )