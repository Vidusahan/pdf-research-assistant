from typing import List
 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
 
def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks while preserving metadata.
 
    Each chunk inherits the original document's metadata and additionally
    carries `chunk_index` (position within the source page).
 
    Args:
        docs:          List of Documents to split (typically from loader.py).
        chunk_size:    Maximum character length of each chunk.
        chunk_overlap: Character overlap between consecutive chunks (helps
                       preserve context across chunk boundaries).
 
    Returns:
        List of chunk Documents with enriched metadata:
            - source       : original filename  (e.g. "paper.pdf")
            - page         : 0-indexed page from PyPDFLoader
            - page_number  : 1-indexed page number
            - chunk_index  : position of this chunk within its source page
    """
    if not docs:
        return []
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer splitting at paragraph → sentence → word boundaries
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
 
    chunks = splitter.split_documents(docs)
 
    # Ensure required metadata fields survive the split and add chunk_index
    for i, chunk in enumerate(chunks):
        # source and page_number are set by loader.py; guard against missing values
        chunk.metadata.setdefault("source", "unknown")
        chunk.metadata.setdefault("page_number", 0)
        chunk.metadata["chunk_index"] = i   # global index across all chunks
 
    return chunks
 
 
# ─── quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from loader import load_pdf, load_pdfs_from_folder
 
    target = sys.argv[1] if len(sys.argv) > 1 else "data/sample_pdfs"
 
    docs = load_pdf(target) if target.endswith(".pdf") else load_pdfs_from_folder(target)
 
    chunks = split_documents(docs)
 
    print(f"\n{'─'*60}")
    print(f"Original pages : {len(docs)}")
    print(f"Total chunks   : {len(chunks)}")
 
    if chunks:
        print(f"\nFirst chunk content preview:\n")
        print(chunks[0].page_content[:400])
        print(f"\nFirst chunk metadata: {chunks[0].metadata}")
 
        # Verify all chunks carry required metadata
        missing_source = [c for c in chunks if not c.metadata.get("source")]
        missing_page   = [c for c in chunks if c.metadata.get("page_number") is None]
 
        print(f"\nMetadata check:")
        print(f"  Chunks missing 'source'      : {len(missing_source)}")
        print(f"  Chunks missing 'page_number' : {len(missing_page)}")
        if not missing_source and not missing_page:
            print("  ✓ All chunks have required metadata fields")
 