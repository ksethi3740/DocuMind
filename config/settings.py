import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
UPLOADS_DIR     = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "").strip()

print(f"[Config] Gemini={'✓' if GEMINI_API_KEY else '✗'} | Groq={'✓' if GROQ_API_KEY else '✗'}")

EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM_MODEL   = "google/flan-t5-base"
LOCAL_LLM_MAX_NEW = 512
LOCAL_LLM_TEMP    = 0.3

# Increased for better answer coverage
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
TOP_K_RETRIEVAL = 10