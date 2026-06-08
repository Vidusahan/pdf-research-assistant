from typing import List, Dict
from langchain_core.documents import Document


def extract_citations(source_documents: List[Document]) -> List[str]:
    """
    Parse source_documents from chain output and return
    formatted citation strings.
    
    Returns:
        List of unique citation strings, e.g.:
        ["research_paper.pdf — Page 4", "report.pdf — Page 12"]
    """
    seen = set()
    citations = []

    for doc in source_documents:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", None)

        # Normalise source to just filename
        filename = source.split("/")[-1].split("\\")[-1]

        key = (filename, page)
        if key not in seen:
            seen.add(key)
            if page is not None:
                citations.append(f"{filename} — Page {page + 1}")  # page is 0-indexed
            else:
                citations.append(filename)

    return citations