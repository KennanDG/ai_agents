from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from .settings import RagSettings


# Vector dimension sizes
NOMIC_EMBED_TEXT = 768
MXBAI_EMBED_LARGE_V1 = 1024
MISTRAL_EMBED = 1024

def build_qdrant(
    settings: RagSettings,
    embedding_fn,
    *,
    collection_name_override: Optional[str] = None,
):

    client = QdrantClient(url=settings.qdrant_url)

    base_collection = collection_name_override or settings.collection_name
    collection_name = f"{base_collection}-{settings.namespace}"

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        # IMPORTANT: vector size must match the embedding model.
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


def delete_source(vs: QdrantVectorStore, source_uri: str):
    client = vs.client
    collection = vs.collection_name
    flt = Filter(
        must=[FieldCondition(key="source_uri", match=MatchValue(value=source_uri))]
    )
    client.delete(collection_name=collection, points_selector=flt)


def build_retriever(vs: QdrantVectorStore, k: int):
    return vs.as_retriever(search_kwargs={"k": k})


def upsert_documents(vs: QdrantVectorStore, docs, ids: list[str]):
    # LangChain Qdrant vectorstore supports ids for add_documents
    vs.add_documents(docs, ids=ids)


