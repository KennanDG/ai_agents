from __future__ import annotations
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langsmith import traceable


def _repo_relative(path: Path) -> str:
    """
    Try to compute a stable repo-relative path for idempotency across environments
    (WSL vs devcontainer). Falls back to absolute if needed.
    """
    try:
        repo_root = Path(__file__).resolve().parents[3]  # .../ai_agents (repo root)
        return str(path.resolve().relative_to(repo_root))
    
    except Exception:
        return str(path.resolve())


@traceable
def load_text_files(paths: Iterable[str | Path]) -> List[Document]:
    
    docs: List[Document] = []
    
    for p in paths:
        # Resolve path
        path = Path(p)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        
        # load text
        loader = TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()

        relative_path = _repo_relative(path)
        stat = path.stat() # (file size, mtime)

        # Add useful metadata 
        for doc in loaded:
            doc.metadata = {
                **(doc.metadata or {}),
                "source": "file",
                "path": str(path.resolve()),
                "source": "file",
                "source_uri": f"file:{relative_path}",  # stable across WSL/devcontainer
                "path": str(path.resolve()),
                "path_rel": relative_path,
                "file_name": path.name,
                "ext": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "mtime": int(stat.st_mtime),
            }
            
        docs.extend(loaded)
        
    return docs