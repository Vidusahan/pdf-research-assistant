from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from src.prompts import get_prompt

def build_rag_chain(vectorstore, k: int = 4):
    """
    Build and return a RAG chain.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    prompt = get_prompt()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source

def run_query(chain, question: str) -> dict:
    """
    Run a question through the RAG chain.
    """
    result = chain.invoke(question)
    return {
        "answer": result["answer"],
        "source_documents": result["context"],
    }