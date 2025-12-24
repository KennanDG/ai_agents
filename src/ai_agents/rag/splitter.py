from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable


@traceable
def split_docs(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)

    # Attach chunk metadata
    for i, doc in enumerate(splits):
        doc.metadata = {
            **(doc.metadata or {}), 
            "chunk_index": i
        }
    return splits