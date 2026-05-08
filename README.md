# 🧠 DocuMind — Intelligent Document Analysis System

> RAG-powered document Q&A with flowchart generation, multimodal support, and offline capability.

## ✨ Features
- 📄 **Multimodal**: PDF, DOCX, PNG, JPG, scanned images with OCR
- 🔍 **Semantic Search**: FAISS + sentence-transformers
- 🤖 **Dual AI**: Works offline (flan-t5) OR with Claude API
- 🗺️ **Auto Flowcharts**: Process flows, research maps, mind maps
- 🗣️ **Layman Mode**: Simple-language explanations
- 📎 **Source Attribution**: Ranked excerpts with relevance scores
- 🔒 **Privacy-First**: 100% local mode available

## 🚀 Quick Start
```bash
# 1. Clone
git clone <repo-url> && cd DocuMind

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add Anthropic API key
cp .env.example .env
# Edit .env → ANTHROPIC_API_KEY=sk-ant-...

# 5. Install Tesseract OCR
# macOS:   brew install tesseract
# Ubuntu:  sudo apt install tesseract-ocr
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# 6. Run!
streamlit run app.py
```

## 📁 Project Structure
```
DocuMind/
├── app.py                     # Entry point
├── requirements.txt
├── .env.example
├── config/settings.py         # Configuration
├── src/
│   ├── core/
│   │   ├── document_processor.py  # PDF/DOCX/Image ingestion
│   │   ├── rag_engine.py          # RAG orchestration
│   │   ├── embeddings.py          # Vector embeddings
│   │   ├── vector_store.py        # FAISS store
│   │   └── local_llm.py           # Offline LLM
│   ├── ui/
│   │   ├── main_ui.py             # Layout
│   │   ├── sidebar.py             # Sidebar
│   │   ├── chat_ui.py             # Chat
│   │   ├── diagram_ui.py          # Flowcharts
│   │   └── styles.py              # CSS
│   └── utils/
│       ├── flowchart_generator.py # Mermaid diagrams
│       └── helpers.py
└── data/
    ├── uploads/
    └── vectorstore/
```

## ⚙️ Configuration
Edit `config/settings.py` or `.env`:

| Setting | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | `` | Optional — enables Claude API |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `LOCAL_LLM_MODEL` | `flan-t5-base` | Offline generation model |
| `CHUNK_SIZE` | `600` | Document chunk size |
| `TOP_K_RETRIEVAL` | `5` | Retrieved excerpts per query |

## 🎨 UI Theme
Deep space glassmorphism with indigo/violet/cyan neon accents.
```

---

## 📄 `.env.example`
```
# Optional: Add your Anthropic API key for Claude-powered answers
# Without this, DocuMind runs fully offline using local flan-t5
ANTHROPIC_API_KEY=

# Optional: Override default models
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# LOCAL_LLM_MODEL=google/flan-t5-base