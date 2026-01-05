from __future__ import annotations

from pathlib import Path
from typing import Iterable
import hashlib
from langsmith import traceable
from langchain_core.documents import Document

from .embeddings import build_ollama_embeddings
from .loaders import load_text_files, _repo_relative
from .settings import RagSettings
from .splitter import split_docs
from .vectorstore import build_qdrant, delete_source, upsert_documents
from ai_agents.db.session import SessionLocal
from ai_agents.rag.repo import upsert_source, replace_chunks
from sqlalchemy import select
from ai_agents.db.models import RagChunk
import uuid
from ai_agents.config.constants import QDRANT_ID_NAMESPACE

from .preprocess import (
    expand_inputs, is_text, is_pdf, is_image,
    pdf_to_derived_md, image_to_derived_md,
    parse_frontmatter_for_source_uri, parse_frontmatter_key
)


# ---------------- Create IDs for document embeddings ----------------
def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_point_id(source_uri: str, content_hash: str, chunk_index: int, chunk_hash: str) -> str:

    raw = f"{source_uri}|{content_hash}|{chunk_index}|{chunk_hash}"

    return str(uuid.uuid5(QDRANT_ID_NAMESPACE, raw))




# ---------------- Ingest docs in vector database ----------------


@traceable
def ingest_files(paths: Iterable[str | Path], settings: RagSettings) -> int:
    # 0) Expand directory/glob inputs
    input_paths = expand_inputs(paths)
    input_paths = [p for p in input_paths if p.suffix.lower() in {".md",".txt",".pdf",".png",".jpg",".jpeg",".webp"}]

    # 1) Split into text vs non-text
    text_paths: list[Path] = [p for p in input_paths if is_text(p)]
    pdf_paths: list[Path] = [p for p in input_paths if is_pdf(p)]
    img_paths: list[Path] = [p for p in input_paths if is_image(p)]

    # 2) Preprocess non-text into derived markdown
    # Directory layout you decided:
    # data/corpus/... originals
    # data/derived/... generated
    derived_root = Path("data/derived").resolve()
    pdf_md_dir = derived_root / "pdf_md"
    img_md_dir = derived_root / "images_md"
    ocr_pdf_dir = derived_root / "ocr_pdfs"

    derived_md_paths: list[Path] = []

    # helper to compute repo-relative path and source_uri matching your loader convention
    def rel_and_source(p: Path) -> tuple[str, str]:
        
        path_rel = _repo_relative(p)
        return path_rel, f"file:{path_rel}"


    for path in pdf_paths:
        path_rel, source_uri = rel_and_source(path)
        derived_md_paths.append(
            pdf_to_derived_md(
                path,
                source_uri=source_uri,
                path_rel=path_rel,
                derived_pdf_md_dir=pdf_md_dir,
                ocr_pdf_dir=ocr_pdf_dir,
            )
        )

    # VLM
    caption_model = settings.caption_model if hasattr(settings, "caption_model") else "llava:7b"


    for path in img_paths:
        path_rel, source_uri = rel_and_source(path)
        derived_md_paths.append(
            image_to_derived_md(
                path,
                source_uri=source_uri,
                path_rel=path_rel,
                derived_img_md_dir=img_md_dir,
                caption_model=caption_model,
            )
        )

    # 3) Load docs:
    # - text_paths are loaded normally
    # - derived_md_paths are loaded normally then we override their source_uri using frontmatter
    docs = []
    if text_paths:
        docs.extend(load_text_files(text_paths))

    if derived_md_paths:
        derived_docs = load_text_files(derived_md_paths)

        # Override source_uri & original path so stable IDs remain tied to ORIGINAL source, not derived file
        for d in derived_docs:
            md_text = d.page_content
            override = parse_frontmatter_for_source_uri(md_text)
            orig_rel = parse_frontmatter_key(md_text, "original_rel")

            if override:
                d.metadata["source_uri"] = override
            
            if orig_rel:
                # store original absolute path for hashing
                d.metadata["original_path"] = str((Path(__file__).resolve().parents[3] / orig_rel).resolve())

            # optionally keep derived path info too:
            d.metadata["derived_path"] = d.metadata.get("path")

        docs.extend(derived_docs)
    
    # paths = [Path(p) for p in paths]

    # docs = load_text_files(paths)
    splits = split_docs(docs, settings.chunk_size, settings.chunk_overlap)

    embeddings = build_ollama_embeddings(settings.embedding_model)
    vs = build_qdrant(settings=settings, embedding_fn=embeddings)


    # Pre-compute file-level hashes once per source_uri
    file_hashes: dict[str, str] = {}

    for doc in docs:
        source_uri = doc.metadata["source_uri"]
        original_path = doc.metadata.get("original_path")
        path_to_hash = Path(original_path) if original_path else Path(doc.metadata["path"]) # Derived path Fallback
        file_hashes[source_uri] = _sha256_file(path_to_hash)


    # Group splits by source_uri so chunk_index is per file
    by_source: dict[str, list] = {}

    for doc in splits:
        source_uri = doc.metadata["source_uri"]  # from loaders.py
        by_source.setdefault(source_uri, []).append(doc)

    
    total = 0

    with SessionLocal() as db:
        for source_uri, chunk_docs in by_source.items():
            content_hash = file_hashes[source_uri]

            # 1) Upsert source & check unchanged
            src_row, unchanged = upsert_source(
                db,
                source_uri=source_uri,
                content_hash=content_hash,
                collection_name=settings.collection_name,
                namespace=settings.namespace,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )


            # nothing changed, do not touch qdrant
            if unchanged:
                continue 

            # 2) Build chunk rows (stable point ids)
            chunks_payload: list[dict] = []
            ids: list[str] = []

            for idx, doc in enumerate(chunk_docs):
                chunk_hash = _sha256_text(doc.page_content)
                point_id = _make_point_id(source_uri, content_hash, idx, chunk_hash)

                # metadata for qdrant payload
                doc.metadata.update({
                    "source_uri": source_uri,
                    "content_hash": content_hash,
                    "chunk_index": idx,
                    "chunk_hash": chunk_hash,
                    "pg_source_id": src_row.id,
                })

                chunks_payload.append({
                    "chunk_index": idx,
                    "chunk_hash": chunk_hash,
                    "qdrant_point_id": point_id,
                })
                ids.append(point_id)

            # 3) Replace chunks in Postgres
            chunk_rows = replace_chunks(db, source_id=src_row.id, chunks=chunks_payload)

            # attach pg_chunk_id for traceability/debugging
            for doc, row in zip(chunk_docs, chunk_rows):
                doc.metadata["pg_chunk_id"] = row.id

            # 4) Qdrant index update (index follows Postgres)
            try:
                delete_source(vs, source_uri)
                upsert_documents(vs, chunk_docs, ids)
                db.commit()
                
            except Exception:
                db.rollback()
                raise

            total += len(chunk_docs)

    return total






# @traceable
# def ingest_files(paths: Iterable[str | Path], settings: RagSettings) -> int:

#     paths = [Path(p) for p in paths] # Ensures parameter is Path data type
    
#     # Split docs
#     docs = load_text_files(paths)
#     splits = split_docs(docs, settings.chunk_size, settings.chunk_overlap)

#     # Embed
#     embeddings = build_ollama_embeddings(settings.embedding_model)
#     vs = build_qdrant(settings=settings, embedding_fn=embeddings)

#     # Pre-compute file-level hashes (once per file)
#     file_hashes: dict[str, str] = {}
    
#     for doc in docs:
#         relative_path = doc.metadata["path_rel"]
#         file_hashes[relative_path] = _sha256_file(Path(doc.metadata["path"]))

    
#     ids: list[str] = [] # attach metadata + build stable IDs

#     chunk_counters: dict[str, int] = {} # chunk_index must be per source

#     for doc in splits:
#         relative_path = doc.metadata["path_rel"]
#         source_uri = f"file:{relative_path}"

#         chunk_index = chunk_counters.get(source_uri, 0)
#         chunk_counters[source_uri] = chunk_index + 1

#         content_hash = file_hashes[relative_path]
#         chunk_hash = _sha256_text(doc.page_content)

#         point_id = _make_point_id(
#             source_uri=source_uri,
#             content_hash=content_hash,
#             chunk_index=chunk_index,
#             chunk_hash=chunk_hash,
#         )

#         doc.metadata.update({
#             "source_uri": source_uri,
#             "content_hash": content_hash,
#             "chunk_index": chunk_index,
#             "chunk_hash": chunk_hash,
#         })

#         ids.append(point_id)

#     # Delete existing vectors per source (true idempotency)
#     for src in sorted(chunk_counters.keys()):
#         delete_source(vs, src)
    

#     upsert_documents(vs, splits, ids)
    
#     return len(splits)


#     # for i, doc in enumerate(splits):
#     #     # assuming loaders set metadata["path"] already
#     #     file_path = doc.metadata.get("path")
#     #     source_uri = f"file:{file_path}" if file_path else "unknown"

#     #     # compute full file hash if we can
#     #     content_hash = _sha256_file(Path(file_path)) if file_path else _sha256_text(doc.page_content)

#     #     chunk_hash = _sha256_text(doc.page_content)
#     #     point_id = _make_point_id(source_uri, content_hash, i, chunk_hash)

#     #     doc.metadata.update({
#     #         "source_uri": source_uri,
#     #         "content_hash": content_hash,
#     #         "chunk_index": i,
#     #         "chunk_hash": chunk_hash,
#     #     })
#     #     ids.append(point_id)

#     # # easiest true-idempotent behavior: delete all existing points for each source_uri
#     # # (dedupe sources)
#     # for src in sorted({doc.metadata["source_uri"] for doc in splits if "source_uri" in doc.metadata}):
#     #     delete_source(vs, src)

#     # upsert_documents(vs, splits, ids)
    
#     # return len(splits)