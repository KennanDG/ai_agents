from __future__ import annotations
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langsmith import traceable


@traceable
def load_text_files(paths: Iterable[str | Path]) -> List[Document]:
    
    docs: List[Document] = []
    
    for p in paths:
        # Load text
        path = Path(p)
        loader = TextLoader(str(path))
        loaded = loader.load()

        # Add useful metadata 
        for doc in loaded:
            doc.metadata = {
                **(doc.metadata or {}),
                "source": "file",
                "path": str(path.resolve()),
            }
            
        docs.extend(loaded)
    return docs