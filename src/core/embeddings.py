"""
Sentence-Transformer embeddings — runs fully locally.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st
from config.settings import EMBEDDING_MODEL


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = load_embedding_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def embed_query(query: str) -> np.ndarray:
    model = load_embedding_model()
    return model.encode([query], normalize_embeddings=True)[0]