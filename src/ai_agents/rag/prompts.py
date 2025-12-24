from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable


RAG_TEMPLATE = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""

@traceable
def build_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(RAG_TEMPLATE)