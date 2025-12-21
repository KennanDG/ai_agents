from __future__ import annotations
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma

def build_chroma(
    embedding_fn,
    persist_dir: str,
    collection_name: str,
) -> Chroma:
    
    # Persistent DB (creates/opens existing)
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding_fn,
    )


def upsert_documents(
    vs: Chroma,
    docs: List[Document],
    ids: Optional[List[str]] = None,
) -> None:
    # Pass ids to update existing chunks later.
    vs.add_documents(docs, ids=ids)


def build_retriever(vs: Chroma, k: int):
    return vs.as_retriever(search_kwargs={"k": k})