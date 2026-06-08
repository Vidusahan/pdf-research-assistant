"""
src/vector_store.py
Build, save, load, and query a FAISS vector store using Google Gemini embeddings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
FAISS_INDEX_DIR = "faiss_index"

def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a Google Gemini embedding model instance."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Add it to your .env file: GOOGLE_API_KEY=your_key_here"
        )
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=api_key,
    )


# ── Build ─────────────────────────────────────────────────────────────────────
def build_vector_store(chunks: list) -> FAISS:
    """
    Embed a list of document chunks and return a FAISS vector store.

    Args:
        chunks: List of LangChain Document objects (from splitter.py)

    Returns:
        FAISS vector store object
    """
    if not chunks:
        raise ValueError("No chunks provided. Run the splitter first.")

    print(f"[vector_store] Embedding {len(chunks)} chunks with Google gemini-embedding-2 ...")
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("[vector_store] FAISS index built successfully.")
    return vector_store


# ── Save ──────────────────────────────────────────────────────────────────────
def save_vector_store(vector_store: FAISS, index_dir: str = FAISS_INDEX_DIR) -> None:
    """
    Persist the FAISS index to disk.

    Args:
        vector_store: A built FAISS vector store
        index_dir:    Directory to save the index (default: faiss_index/)
    """
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(index_dir)
    print(f"[vector_store] Index saved to '{index_dir}/'")


# ── Load ──────────────────────────────────────────────────────────────────────
def load_vector_store(index_dir: str = FAISS_INDEX_DIR) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Args:
        index_dir: Directory where the index was saved

    Returns:
        FAISS vector store object

    Raises:
        FileNotFoundError: If no index exists at index_dir
        RuntimeError:      If the index files are corrupted
    """
    if not Path(index_dir).exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{index_dir}/'. "
            "Upload and process PDFs first."
        )

    try:
        embeddings = _get_embeddings()
        vector_store = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,   # required by LangChain ≥0.2
        )
        print(f"[vector_store] Index loaded from '{index_dir}/'")
        return vector_store
    except Exception as e:
        raise RuntimeError(
            f"Failed to load FAISS index from '{index_dir}/'. "
            f"It may be corrupted. Delete the folder and re-index. Error: {e}"
        )


# ── Search ────────────────────────────────────────────────────────────────────
def similarity_search(vector_store: FAISS, query: str, k: int = 4) -> list:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        vector_store: A loaded or built FAISS vector store
        query:        The user's question string
        k:            Number of chunks to return (default: 4)

    Returns:
        List of LangChain Document objects with page_content and metadata
    """
    if not query.strip():
        raise ValueError("Query string cannot be empty.")

    results = vector_store.similarity_search(query, k=k)
    print(f"[vector_store] Retrieved {len(results)} chunks for query: '{query[:60]}...'")
    return results


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick smoke test — run from project root:
        python src/vector_store.py

    Requires:
        - .env with GOOGLE_API_KEY set
        - data/sample_pdfs/ containing at least one PDF
    """
    from loader import load_pdfs_from_folder
    from splitter import split_documents

    # 1. Load
    docs = load_pdfs_from_folder("data/sample_pdfs")
    print(f"Loaded {len(docs)} pages.")

    # 2. Split
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Build & save
    vs = build_vector_store(chunks)
    save_vector_store(vs)

    # 4. Load back
    vs2 = load_vector_store()

    # 5. Search
    query = "What is the main topic of this document?"
    results = similarity_search(vs2, query, k=3)

    print(f"\nTop {len(results)} results for: '{query}'")
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        print(f"\n--- Result {i} ---")
        print(f"Source : {meta.get('source', 'unknown')}")
        print(f"Page   : {meta.get('page_number', '?')}")
        print(f"Preview: {doc.page_content[:200]}...")
