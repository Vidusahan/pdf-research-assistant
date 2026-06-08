import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_store import load_vector_store
from src.rag_chain import build_rag_chain, run_query
from src.citations import extract_citations

load_dotenv()

vs = load_vector_store()
chain = build_rag_chain(vs, k=4)

questions = [
    "What is the main methodology used?",        # in-scope
    "What is the capital of France?",             # out-of-scope
    "What are the key findings?",                 # partial / depends on PDF
]

for q in questions:
    print(f"\nQ: {q}")
    result = run_query(chain, q)
    print(f"A: {result['answer']}")
    print(f"Citations: {extract_citations(result['source_documents'])}")