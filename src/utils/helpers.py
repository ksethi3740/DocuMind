"""
DocuMind — General Utility Helpers
Shared functions used across the project.
"""

import re
import os
import hashlib
from pathlib import Path
from datetime import datetime


# ── Text Utilities ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excessive whitespace, special characters, and normalize text."""
    if not text:
        return ""
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 300, suffix: str = "…") -> str:
    """Truncate text to a maximum character count."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + suffix


def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split()) if text else 0


def count_sentences(text: str) -> int:
    """Approximate sentence count."""
    return len(re.split(r'[.!?]+', text)) if text else 0


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """
    Simple keyword extraction using word frequency.
    Filters out common stopwords.
    """
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'this', 'that',
        'these', 'those', 'it', 'its', 'we', 'our', 'they', 'their', 'he',
        'she', 'his', 'her', 'as', 'if', 'so', 'not', 'no', 'can', 'also',
        'such', 'than', 'then', 'into', 'about', 'which', 'all', 'each'
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    freq  = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in sorted_words[:top_n]]


def split_into_sentences(text: str) -> list[str]:
    """Split text into individual sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ── File Utilities ────────────────────────────────────────────────────────────

def get_file_hash(file_bytes: bytes) -> str:
    """Generate MD5 hash of file bytes — used to detect duplicates."""
    return hashlib.md5(file_bytes).hexdigest()


def get_file_size_str(file_bytes: bytes) -> str:
    """Return human-readable file size string."""
    size = len(file_bytes)
    if size < 1024:
        return f"{size} B"
    elif size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 ** 2):.1f} MB"


def sanitize_filename(name: str) -> str:
    """Make a string safe to use as a filename."""
    name = re.sub(r'[^\w\s\-_.]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name[:100]


def get_file_extension(filename: str) -> str:
    """Return lowercase file extension without dot."""
    return Path(filename).suffix.lstrip('.').lower()


def is_supported_file(filename: str) -> bool:
    """Check if a file type is supported by DocuMind."""
    supported = {'pdf', 'docx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'txt'}
    return get_file_extension(filename) in supported


# ── Formatting Utilities ──────────────────────────────────────────────────────

def format_timestamp(dt: datetime = None) -> str:
    """Return formatted timestamp string."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%d %b %Y, %H:%M")


def format_score_as_percent(score: float) -> str:
    """Convert a 0–1 cosine similarity score to a percentage string."""
    return f"{min(int(score * 100), 100)}%"


def highlight_query_terms(text: str, query: str, max_length: int = 400) -> str:
    """
    Return a truncated text snippet with query terms wrapped in
    markdown bold for display in source attribution.
    """
    text = truncate_text(text, max_length)
    for term in query.split():
        if len(term) > 3:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text    = pattern.sub(f"**{term}**", text)
    return text


def build_context_string(retrieved_chunks: list[dict], max_chars: int = 3000) -> str:
    """
    Build a clean context string from retrieved chunks for LLM input.
    Respects a character budget.
    """
    parts   = []
    budget  = max_chars
    for i, chunk in enumerate(retrieved_chunks):
        snippet = clean_text(chunk["text"])
        if len(snippet) > budget:
            snippet = snippet[:budget]
        parts.append(f"[Source {i+1}]\n{snippet}")
        budget -= len(snippet)
        if budget <= 0:
            break
    return "\n\n---\n\n".join(parts)


# ── Document Stats ────────────────────────────────────────────────────────────

def get_document_stats(doc: dict) -> dict:
    """
    Return a summary statistics dict for a processed document.
    """
    text = doc.get("text", "")
    return {
        "name":        doc.get("name", "Unknown"),
        "type":        doc.get("metadata", {}).get("type", "unknown"),
        "chunks":      doc.get("chunk_count", 0),
        "words":       count_words(text),
        "sentences":   count_sentences(text),
        "keywords":    extract_keywords(text, top_n=8),
        "char_count":  len(text),
    }


def format_stats_for_display(stats: dict) -> str:
    """Format document stats as a readable markdown string."""