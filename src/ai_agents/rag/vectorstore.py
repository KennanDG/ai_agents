from __future__ import annotations

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .settings import RagSettings


# Vector dimension sizes
NOMIC_EMBED_TEXT = 768
MXBAI_EMBED_LARGE_V1 = 1024
MISTRAL_EMBED = 1024

def build_qdrant(settings: RagSettings, embedding_fn):

    client = QdrantClient(url=settings.qdrant_url)
    collection_name = f"{settings.collection_name}-{settings.namespace}"

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        # IMPORTANT: vector size must match your embedding model.
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=NOMIC_EMBED_TEXT,
                distance=Distance.COSINE,
            ),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_fn,
    )


def build_retriever(vs: QdrantVectorStore, k: int):
    return vs.as_retriever(search_kwargs={"k": k})


def upsert_documents(vs: QdrantVectorStore, docs: List):
    # LangChain will generate IDs if not provided.
    # Later: pass stable IDs for idempotency.
    vs.add_documents(docs)







# def build_chroma(
#     embedding_fn,
#     persist_dir: str,
#     collection_name: str,
# ) -> Chroma:
    
#     # Persistent DB (creates/opens existing)
#     return Chroma(
#         collection_name=collection_name,
#         persist_directory=persist_dir,
#         embedding_function=embedding_fn,
#     )


# def upsert_documents(
#     vs: Chroma,
#     docs: List[Document],
#     ids: Optional[List[str]] = None,
# ) -> None:
#     # Pass ids to update existing chunks later.
#     vs.add_documents(docs, ids=ids)


# def build_retriever(vs: Chroma, k: int):
#     return vs.as_retriever(search_kwargs={"k": k})