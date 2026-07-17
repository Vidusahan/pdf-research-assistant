# 📚 PDF Research Assistant

A Retrieval-Augmented Generation (RAG) app that lets you upload one or more PDFs and ask natural-language questions about them. Answers are grounded in your documents and come with page-level source citations, so you can verify exactly where each answer came from.

Built with **Streamlit**, **LangChain**, **Google Gemini**, and **FAISS**.

---

## ✨ Features

- **Multi-PDF upload** — drag and drop multiple PDFs at once and index them together.
- **Configurable chunking** — adjust chunk size and overlap from the sidebar to balance context vs. retrieval precision.
- **Adjustable retrieval depth (k)** — control how many chunks are passed to the LLM per query.
- **Grounded answers with citations** — every answer includes an expandable "Sources" section pointing back to the relevant chunks.
- **Persistent vector index** — the FAISS index is saved to disk (`faiss_index/`) and automatically reloaded on the next run, so you don't have to re-index the same PDFs every session.
- **Chat-style interface** — full conversation history is preserved in the session.
- **One-click reset** — clear the index and chat history from the sidebar at any time.

---

## 🛠️ Tech Stack

| Component            | Technology                                  |
|-----------------------|----------------------------------------------|
| UI                    | [Streamlit](https://streamlit.io/)            |
| Orchestration         | [LangChain](https://www.langchain.com/) / `langchain-community` |
| LLM & Embeddings      | Google Gemini via `langchain-google-genai` / `google-generativeai` |
| Vector Store          | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) |
| PDF Parsing           | `pypdf`                                       |
| Config / Secrets      | `python-dotenv`                               |

---

## 📁 Project Structure

```
pdf-research-assistant/
├── app.py                  # Streamlit app entry point (UI, upload, chat)
├── src/
│   ├── loader.py           # Loads and parses PDF files
│   ├── splitter.py         # Splits documents into overlapping chunks
│   ├── vector_store.py     # Builds/saves/loads the FAISS vector index
│   └── rag_chain.py        # Builds the RAG chain and extracts citations
├── data/
│   └── sample_pdfs/        # Sample PDFs for quick testing
├── tests/                  # Test suite
├── requirements.txt        # Python dependencies
├── .gitignore
└── LICENSE
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini)

### 1. Clone the repository

```bash
git clone https://github.com/Vidusahan/pdf-research-assistant.git
cd pdf-research-assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

Streamlit will open the app in your browser (typically at `http://localhost:8501`).

---

## 💡 Usage

1. **Upload PDFs** — drag and drop one or more PDF files under "1. Upload your PDFs."
2. **Index documents** — click **🔍 Index documents** to extract, chunk, and embed the text into a FAISS vector store.
3. **Ask questions** — type a question under "2. Ask questions." The app retrieves the most relevant chunks and generates an answer grounded in your documents.
4. **Check sources** — expand **📎 Sources** under any answer to see which chunks/pages the answer was based on.
5. **Tune retrieval** — use the sidebar sliders to adjust:
   - **Chunk size** (200–2000 tokens)
   - **Chunk overlap** (0–500 tokens)
   - **Chunks to retrieve (k)** (2–8)
6. **Reset** — click **🗑️ Clear index & chat** in the sidebar to wipe the FAISS index and start over.

---

## ⚙️ Configuration Reference

| Setting        | Default | Description                                                        |
|----------------|---------|----------------------------------------------------------------------|
| Chunk size     | 1000    | Larger chunks preserve more context per chunk.                       |
| Chunk overlap  | 200     | Overlap between consecutive chunks to avoid cutting sentences mid-thought. |
| k (chunks)     | 4       | Number of chunks retrieved and passed to the LLM per query.          |

---

## 🧪 Testing

Run the test suite from the project root:

```bash
pytest tests/
```

---

## 🗺️ How It Works

1. **Load** — PDFs are parsed into text using `pypdf` (`src/loader.py`).
2. **Split** — text is split into overlapping chunks using the configured chunk size/overlap (`src/splitter.py`).
3. **Embed & Store** — chunks are embedded and stored in a local FAISS index (`src/vector_store.py`), which is persisted to disk for reuse across sessions.
4. **Retrieve & Generate** — on each question, the top-k most similar chunks are retrieved and passed to Gemini via a LangChain RAG chain, which generates an answer along with source citations (`src/rag_chain.py`).

---


## 🤝 Contributing

Issues and pull requests are welcome. If you'd like to add a feature or fix a bug, feel free to open a PR.