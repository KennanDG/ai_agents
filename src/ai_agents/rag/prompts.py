from langchain_core.prompts import ChatPromptTemplate

RAG_TEMPLATE = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""

def build_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(RAG_TEMPLATE)