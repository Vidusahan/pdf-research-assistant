import os
from pathlib import Path
from typing import List
 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
 
 
def load_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF file and return its pages as a list of Documents.
    Each Document has:
        - page_content : text content of the page
        - metadata     : { source: str, page: int }
 
    Args:
        file_path: Absolute or relative path to the PDF file.
 
    Returns:
        List of Document objects, one per page.
 
    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
        ValueError:        If the file is not a .pdf.
    """
    path = Path(file_path)
 
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
 
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")
 
    loader = PyPDFLoader(str(path))
    docs = loader.load()
 
    # Normalise metadata: ensure 'source' is the filename (not full path)
    # and 'page' is an integer (PyPDFLoader already sets page, but we make sure).
    for doc in docs:
        doc.metadata["source"] = path.name          # e.g. "paper.pdf"
        doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1  # 1-indexed
 
    return docs
 
 
def load_pdfs_from_folder(folder_path: str) -> List[Document]:
    """
    Recursively load all PDFs found under `folder_path`.
 
    Args:
        folder_path: Path to a directory containing PDF files.
 
    Returns:
        Combined list of Document objects from all PDFs, preserving
        per-file source metadata.
 
    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    folder = Path(folder_path)
 
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
 
    pdf_paths = sorted(folder.rglob("*.pdf"))
 
    if not pdf_paths:
        print(f"[loader] No PDFs found in: {folder_path}")
        return []
 
    all_docs: List[Document] = []
    for pdf_path in pdf_paths:
        print(f"[loader] Loading: {pdf_path.name} …")
        docs = load_pdf(str(pdf_path))
        all_docs.extend(docs)
        print(f"         → {len(docs)} page(s) loaded")
 
    print(f"\n[loader] Total pages loaded: {len(all_docs)} from {len(pdf_paths)} PDF(s)")
    return all_docs
 
 
# ─── quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
 
    target = sys.argv[1] if len(sys.argv) > 1 else "data/sample_pdfs"
 
    if target.endswith(".pdf"):
        docs = load_pdf(target)
    else:
        docs = load_pdfs_from_folder(target)
 
    print(f"\n{'─'*60}")
    print(f"Document count : {len(docs)}")
    if docs:
        print(f"First page content preview:\n")
        print(docs[0].page_content[:500])
        print(f"\nMetadata: {docs[0].metadata}")
 