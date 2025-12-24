from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from .prompts import build_rag_prompt


def build_rag_chain(retriever, chat_model: str):
    llm = ChatOllama(model=chat_model)
    prompt = build_rag_prompt()

    # Question -> Retrieve Context -> Prompt -> LLM -> String
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
