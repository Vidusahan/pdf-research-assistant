from langchain_core.prompts import PromptTemplate

PROMPT_TEMPLATE = """You are a helpful research assistant. Use ONLY the context below to answer the question.

Rules:
- If the answer is in the context, answer clearly and cite the page number(s) like: (Source: filename.pdf, Page 3)
- If the answer is NOT in the context, say exactly: "I don't know based on the provided documents."
- Do not make up information or use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""


def get_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )