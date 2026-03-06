# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings.jina import JinaEmbeddings
from langsmith import traceable
import os

from ai_agents.config.settings import settings


# @traceable
# def build_fastembed_embeddings(model: str = "nomic-ai/nomic-embed-text-v1.5", chunk_size: int = 512) -> FastEmbedEmbeddings:
#     # This runs locally in your Python process using ONNX
#     return FastEmbedEmbeddings(model_name=model, max_length=chunk_size)


@traceable
def build_jina_embeddings(model: str = "jina-embeddings-v2-base-en") -> JinaEmbeddings:
    
    return JinaEmbeddings(model_name=model, jina_api_key=settings.resolved_jina_api_key())