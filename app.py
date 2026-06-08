import os
import tempfile
import shutil
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.loader import load_pdf
from src.splitter import split_documents
from src.vector_store import build_vector_store, save_vector_store, load_vector_store, FAISS_INDEX_DIR
from src.rag_chain import build_rag_chain, extract_citations

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Research Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # list of {role, content, citations}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "loaded_pdf_names" not in st.session_state:
    st.session_state.loaded_pdf_names = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    chunk_size = st.slider(
        "Chunk size (tokens)",
        min_value=200, max_value=2000, value=1000, step=100,
        help="Larger chunks preserve more context; smaller chunks improve retrieval precision."
    )
    chunk_overlap = st.slider(
        "Chunk overlap (tokens)",
        min_value=0, max_value=500, value=200, step=50,
        help="Overlap between consecutive chunks to avoid cutting sentences mid-thought."
    )
    k_chunks = st.slider(
        "Chunks to retrieve (k)",
        min_value=2, max_value=8, value=4, step=1,
        help="How many chunks are passed to the LLM as context per query."
    )

    st.markdown("---")

    if st.session_state.loaded_pdf_names:
        st.markdown("### 📂 Loaded PDFs")
        for name in st.session_state.loaded_pdf_names:
            st.markdown(f"- `{name}`")
    else:
        st.caption("No PDFs loaded yet.")

    st.markdown("---")

    if st.button("🗑️ Clear index & chat", use_container_width=True):
        # Wipe FAISS index from disk
        if os.path.exists(FAISS_INDEX_DIR):
            shutil.rmtree(FAISS_INDEX_DIR)
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.session_state.loaded_pdf_names = []
        st.session_state.chunk_count = 0
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────
st.title("📚 PDF Research Assistant")
st.caption("Upload PDFs, then ask questions. Answers are grounded in your documents with page citations.")

# Check for Google API key early
if not os.environ.get("GOOGLE_API_KEY"):
    st.error(
        "⚠️ **GOOGLE_API_KEY not found.** "
        "Create a `.env` file in the project root with `GOOGLE_API_KEY=your_key_here` and restart the app."
    )
    st.stop()

# ── Task 1: PDF Upload & Processing ───────────────────────────────────────────
st.markdown("### 1. Upload your PDFs")

uploaded_files = st.file_uploader(
    "Drag & drop PDFs here, or click to browse",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    if st.button("🔍 Index documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF before indexing.")
        else:
            all_docs = []
            pdf_names = []

            with st.spinner("Reading and chunking PDFs…"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for uf in uploaded_files:
                        tmp_path = os.path.join(tmp_dir, uf.name)
                        with open(tmp_path, "wb") as f:
                            f.write(uf.read())
                        try:
                            docs = load_pdf(tmp_path)
                            all_docs.extend(docs)
                            pdf_names.append(uf.name)
                        except Exception as e:
                            st.warning(f"Could not read **{uf.name}**: {e}")

            if not all_docs:
                st.error("No content could be extracted from the uploaded PDFs.")
            else:
                with st.spinner("Embedding chunks and building vector index…"):
                    chunks = split_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    vs = build_vector_store(chunks)
                    save_vector_store(vs)
                    st.session_state.vector_store = vs
                    st.session_state.loaded_pdf_names = pdf_names
                    st.session_state.chunk_count = len(chunks)

                st.success(
                    f"✅ Indexed **{len(pdf_names)} PDF(s)** into **{len(chunks)} chunks**. "
                    "You can now ask questions below."
                )

# ── Try to load existing index if none in session ─────────────────────────────
if st.session_state.vector_store is None and os.path.exists(FAISS_INDEX_DIR):
    try:
        st.session_state.vector_store = load_vector_store()
    except Exception:
        pass

# ── Task 2: Q&A Chat Interface ────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 2. Ask questions")

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📎 Sources"):
                for cite in msg["citations"]:
                    st.markdown(cite)

# Chat input
user_question = st.chat_input("Ask a question about your documents…")

if user_question:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and index at least one PDF before asking questions.")
    else:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question, "citations": []})

        # Run RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                try:
                    chain = build_rag_chain(st.session_state.vector_store, k=k_chunks)
                    result = chain.invoke(user_question)
                    answer = result["answer"]
                    citations = extract_citations(result.get("context", []))
                except Exception as e:
                    answer = f"❌ An error occurred: {e}"
                    citations = []

            st.markdown(answer)
            if citations:
                with st.expander("📎 Sources"):
                    for cite in citations:
                        st.markdown(cite)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
        })