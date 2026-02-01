from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langsmith import traceable

from .prompts import build_rag_prompt

@traceable
def build_rag_chain(retriever, chat_model: str):
    llm = ChatOllama(
        model=chat_model,
        temperature=0,
        num_ctx=10_000
        )
    prompt = build_rag_prompt()

    # Question -> Retrieve Context -> Prompt -> LLM -> String
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
