"""
FAISS-based vector store for semantic retrieval.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from config.settings import VECTORSTORE_DIR
from src.core.embeddings import embed_texts, embed_query


class DocuMindVectorStore:
    def __init__(self, session_id: str = "default"):
        self.session_id   = session_id
        self.index        = None
        self.chunks       = []        # raw text chunks
        self.metadatas    = []        # per-chunk metadata
        self.dimension    = 384       # MiniLM embedding dim

    def add_documents(self, chunks: list[str], metadata: dict = None):
        if not chunks:
            return

        vectors = embed_texts(chunks)
        vectors = np.array(vectors, dtype="float32")

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # inner-product = cosine on normalised

        self.index.add(vectors)
        for chunk in chunks:
            self.chunks.append(chunk)
            self.metadatas.append(metadata or {})

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []

        # Search more candidates then filter
        k = min(top_k * 3, self.index.ntotal)
        q_vec = embed_query(query).reshape(1, -1).astype("float32")
        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and float(score) > 0.15:  # filter low relevance
                results.append({
                    "text":     self.chunks[idx],
                    "score":    float(score),
                    "metadata": self.metadatas[idx],
                    "index":    int(idx)
                })

        # Return top_k after filtering
        return results[:top_k]

    def clear(self):
        self.index     = None
        self.chunks    = []
        self.metadatas = []

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)