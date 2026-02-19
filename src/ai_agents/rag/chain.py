from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langsmith import traceable

from ai_agents.config.settings import settings
from .prompts import build_rag_prompt

@traceable
def build_rag_chain(retriever, chat_model: str):
    llm = ChatGroq(
        model=chat_model,
        api_key=settings.resolved_groq_api_key(),
        temperature=0.0,
    )
    prompt = build_rag_prompt()

    # Question -> Retrieve Context -> Prompt -> LLM -> String
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
