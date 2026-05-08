# 🧠 DocuMind — Intelligent RAG-Based Document Analysis System

> AI-powered intelligent document understanding platform with semantic retrieval, multimodal processing, flowchart generation, and offline-capable architecture.

---

# 📌 Overview

DocuMind is a Retrieval-Augmented Generation (RAG) based intelligent document analysis platform designed to help users interact with complex documents through conversational AI, semantic search, automated summarization, and intelligent diagram generation.

The system supports multiple document formats including PDFs, Word documents, scanned images, and research papers. It combines vector embeddings, semantic retrieval, OCR, and large language models to provide accurate contextual answers and intelligent document insights.

The platform is designed with a privacy-first architecture and can operate completely offline using local language models, while also supporting external APIs for enhanced AI capabilities.

---

# 🚀 Core Features

## 📄 Multimodal Document Support

DocuMind supports ingestion and processing of:

* PDF documents
* DOCX files
* PNG / JPG / JPEG images
* Scanned documents
* Research papers
* OCR-based image text extraction

The system automatically extracts and preprocesses textual content from uploaded files.

---

## 🔍 Semantic Search with Vector Retrieval

The platform uses:

* FAISS Vector Database
* Sentence Transformers Embeddings
* Semantic Chunk Retrieval

This enables highly relevant contextual search instead of simple keyword matching.

Users can ask natural language questions and retrieve semantically related information from uploaded documents.

---

## 🤖 Dual AI Architecture

### 🔒 Offline AI Mode

Runs completely locally using lightweight transformer models such as:

* Google FLAN-T5
* Local HuggingFace pipelines

Benefits:

* Full privacy
* No internet dependency
* Secure document processing

---

### ⚡ API-Based AI Mode

Supports advanced LLM APIs including:

* Claude
* Gemini
* Groq
* OpenAI-compatible APIs

Benefits:

* Higher-quality responses
* Better summarization
* Enhanced reasoning capabilities

---

## 📋 Intelligent Document Summarization

DocuMind includes advanced summarization capabilities:

### Supported Summary Modes

* Detailed Summary
* Simple / Layman Explanation
* Executive Summary
* Research Paper Style Summary

The system uses chunk-based map-reduce summarization for handling long documents efficiently.

---

## 🗺️ Automated Diagram & Flowchart Generation

The platform can automatically generate:

* Process Flowcharts
* Research Architecture Diagrams
* Mind Maps
* Conceptual Workflows
* Mermaid Diagrams

This helps visualize complex information from technical documents and research papers.

---

## 🗣️ Layman Mode

Complex technical content can be automatically simplified into easy-to-understand language for non-technical users.

Ideal for:

* Students
* Beginners
* Business stakeholders
* Non-technical audiences

---

## 📎 Source Attribution & Transparency

Every AI-generated response includes:

* Retrieved document chunks
* Relevance-based ranking
* Semantic similarity references

This improves transparency and trustworthiness of generated answers.

---

## 🔒 Privacy-First Architecture

DocuMind is designed with secure local processing capabilities.

In offline mode:

* No document leaves the system
* No external API calls are required
* All embeddings and retrieval remain local

---

# 🏗️ System Architecture

## High-Level Pipeline

```text
Document Upload
       ↓
Text Extraction & OCR
       ↓
Chunking
       ↓
Embedding Generation
       ↓
FAISS Vector Storage
       ↓
Semantic Retrieval (RAG)
       ↓
LLM Response Generation
       ↓
Answer / Summary / Diagram Output
```

---

# ⚙️ Technology Stack

## Frontend / UI

* Streamlit
* Custom CSS
* Responsive Dark UI

---

## Backend / AI

* Python
* LangChain
* Transformers
* HuggingFace
* Sentence Transformers

---

## Vector Database

* FAISS

---

## OCR & Document Processing

* Tesseract OCR
* PyMuPDF
* python-docx
* Pillow

---

## LLM Support

* FLAN-T5
* Claude API
* Gemini API
* Groq API

---

# 📂 Project Structure

```text
DocuMind/
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
│
├── config/
│   └── settings.py
│
├── src/
│   ├── core/
│   │   ├── document_processor.py
│   │   ├── rag_engine.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── local_llm.py
│   │
│   ├── ui/
│   │   ├── main_ui.py
│   │   ├── sidebar.py
│   │   ├── chat_ui.py
│   │   ├── diagram_ui.py
│   │   └── styles.py
│   │
│   └── utils/
│       ├── flowchart_generator.py
│       └── helpers.py
│
├── data/
│   ├── uploads/
│   └── vectorstore/
│
└── assets/
```

---

# 🚀 Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd DocuMind
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create `.env`

```env
GEMINI_API_KEY=
GROQ_API_KEY=
ANTHROPIC_API_KEY=
```

---

## 5️⃣ Install Tesseract OCR

### Windows

Install:
https://github.com/UB-Mannheim/tesseract/wiki

### Ubuntu

```bash
sudo apt install tesseract-ocr
```

### macOS

```bash
brew install tesseract
```

---

## 6️⃣ Run Application

```bash
streamlit run app.py
```

---

# 🧠 RAG Workflow

The Retrieval-Augmented Generation workflow follows these stages:

## Step 1 — Document Ingestion

Uploaded documents are parsed and converted into raw text.

---

## Step 2 — Intelligent Chunking

Large documents are split into semantic chunks for efficient retrieval.

---

## Step 3 — Embedding Generation

Each chunk is converted into vector embeddings using sentence-transformer models.

---

## Step 4 — Vector Indexing

Embeddings are stored in FAISS for fast semantic similarity search.

---

## Step 5 — Semantic Retrieval

User queries retrieve the most contextually relevant chunks.

---

## Step 6 — AI Response Generation

Retrieved chunks are passed to the language model to generate contextual answers.

---

# 📋 Advanced Summarization Engine

DocuMind uses a chunk-based map-reduce summarization pipeline:

```text
Document
   ↓
Chunking
   ↓
Partial Summaries
   ↓
Combined Synthesis
   ↓
Final Structured Summary
```

This improves:

* Long-document handling
* Coherence
* Research paper summarization
* Output quality

---

# 🎨 UI Design

The platform uses a modern dark-themed UI inspired by:

* Glassmorphism
* Neon gradients
* AI dashboard aesthetics
* Research workflow interfaces

Color palette:

* Indigo
* Violet
* Cyan
* Deep Space Black

---

# 🔮 Future Enhancements

Planned future features include:

* Multi-document conversational memory
* Agentic workflow orchestration
* Audio document understanding
* PDF annotation
* Multi-user collaboration
* Cloud deployment support
* Real-time streaming responses
* Knowledge graph generation

---

# 📈 Potential Use Cases

DocuMind can be used for:

* Research paper analysis
* Academic document summarization
* Legal document review
* Technical documentation Q&A
* Enterprise knowledge management
* Healthcare document processing
* Educational AI assistants

---

# 👨‍💻 Author

Developed as an intelligent AI-powered document analysis platform focused on semantic understanding, retrieval-augmented generation, and offline-first AI workflows.

---

# 📜 License

This project is intended for educational, research, and experimental AI applications.

# 📜 Deployed on Streamlit
https://docsanalysisandragbased.streamlit.app/
