from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from src.prompts import get_prompt

def build_rag_chain(vectorstore, k: int = 4):
    """
    Build and return a RetrievalQA chain.
    
    Args:
        vectorstore: A loaded FAISS vectorstore
        k: Number of chunks to retrieve per query
    
    Returns:
        A RetrievalQA chain instance
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt()},
    )

    return chain


def run_query(chain, question: str) -> dict:
    """
    Run a question through the RAG chain.
    
    Returns:
        dict with keys: 'answer', 'source_documents'
    """
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "source_documents": result["source_documents"],
    }