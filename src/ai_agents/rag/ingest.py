from __future__ import annotations

from pathlib import Path
from typing import Iterable
import hashlib
from langsmith import traceable
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from .embeddings import build_fastembed_embeddings
from .loaders import load_text_files, _repo_relative
from .settings import RagSettings
from .splitter import split_docs
from .vectorstore import build_qdrant, delete_source, upsert_documents
from ai_agents.db.session import SessionLocal
from ai_agents.rag.repo import upsert_source, replace_chunks
from sqlalchemy import select
from ai_agents.db.models import RagSource 
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



def infer_domain_key_from_source_uri(source_uri: str) -> str:
    """
    Returns a domain key like:
      - "engineering"
      - "robotics"
      - "cs"
    based on the folder layout under data/corpus/.
    """
    # expected: "file:data/corpus/<domain>/..."
    prefix = "file:data/corpus/"
    if not source_uri.startswith(prefix):
        return "default"

    rel = source_uri[len(prefix):]  # "<domain>/..."
    parts = rel.replace("\\", "/").split("/")
    if not parts:
        return "default"

    top = parts[0].lower()

    # # If top-level is "cs", also include the next folder if present (ai/aws/java)
    # if top == "cs" and len(parts) >= 2:
    #     return f"cs/{parts[1].lower()}"

    return top



def map_domain_to_collection(domain_key: str) -> str:
    """
    Map domain_key -> base collection name.

    """
    mapping = {
        "cs": "rag-cs",
        "cybersecurity": "rag-cybersecurity",
        "engineering": "rag-engineering",
        "robotics": "rag-robotics",
        "research": "rag-research",
        "notes": "rag-notes",
        "personal": "rag-personal",
        "other": "rag-other",
        "default": "rag-default",
    }

    return mapping.get(domain_key, mapping["default"])




# ---------------- Helper Functions ----------------

def _get_rel_and_source(p: Path) -> tuple[str, str]:
    """Helper to compute repo-relative path and source_uri."""
    path_rel = _repo_relative(p)
    return path_rel, f"file:{path_rel}"



def _categorize_paths(paths: Iterable[str | Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Expands inputs and filters them into text, pdf, and image paths."""
    input_paths = expand_inputs(paths)
    # Filter by allowed extensions
    valid_exts = {".md", ".txt", ".pdf", ".png", ".jpg", ".jpeg"}
    input_paths = [p for p in input_paths if p.suffix.lower() in valid_exts]

    text_paths = [p for p in input_paths if is_text(p)]
    pdf_paths = [p for p in input_paths if is_pdf(p)]
    img_paths = [p for p in input_paths if is_image(p)]

    return text_paths, pdf_paths, img_paths



def _detect_unchanged_sources(
    paths: list[Path], 
    settings: RagSettings
) -> tuple[set[str], dict[str, str]]:
    """
    Computes file hashes and checks the DB to identify sources that haven't changed.
    Returns a set of unchanged source URIs and a dict of current candidate hashes.
    """
    candidate_hashes: dict[str, str] = {}
    
    for p in paths:
        _, source_uri = _get_rel_and_source(p)
        candidate_hashes[source_uri] = _sha256_file(p)

    unchanged_sources: set[str] = set()

    with SessionLocal() as db:
        existing = db.scalars(
            select(RagSource).where(RagSource.source_uri.in_(list(candidate_hashes.keys())))
        ).all()
        
        existing_by_uri = {rag_source.source_uri: rag_source for rag_source in existing}

        for source_uri, content_hash in candidate_hashes.items():
            src = existing_by_uri.get(source_uri)
            if not src:
                continue

            expected_collection = map_domain_to_collection(infer_domain_key_from_source_uri(source_uri))

            if (
                src.content_hash == content_hash
                and src.collection_name == expected_collection
                and src.namespace == settings.namespace
                and src.chunk_size == settings.chunk_size
                and src.chunk_overlap == settings.chunk_overlap
            ):
                unchanged_sources.add(source_uri)
                
    return unchanged_sources, candidate_hashes



def _process_derived_files(
    pdf_paths: list[Path], 
    img_paths: list[Path], 
    unchanged_sources: set[str], 
    settings: RagSettings
) -> list[Path]:
    """
    Converts PDFs and Images to Markdown if they are not in the unchanged set.
    """
    derived_root = Path("data/derived").resolve()
    pdf_md_dir = derived_root / "pdf_md"
    img_md_dir = derived_root / "images_md"
    ocr_pdf_dir = derived_root / "ocr_pdfs"
    
    derived_md_paths: list[Path] = []

    # Process PDFs
    for path in pdf_paths:
        path_rel, source_uri = _get_rel_and_source(path)
        if source_uri in unchanged_sources:
            continue
            
        derived_md_paths.append(
            pdf_to_derived_md(
                path,
                source_uri=source_uri,
                path_rel=path_rel,
                derived_pdf_md_dir=pdf_md_dir,
                ocr_pdf_dir=ocr_pdf_dir,
            )
        )

    # Process Images
    caption_model = getattr(settings, "caption_model", "meta-llama/llama-4-scout-17b-16e-instruct")

    for path in img_paths:
        path_rel, source_uri = _get_rel_and_source(path)
        if source_uri in unchanged_sources:
            continue
            
        derived_md_paths.append(
            image_to_derived_md(
                path,
                source_uri=source_uri,
                path_rel=path_rel,
                derived_img_md_dir=img_md_dir,
                caption_model=caption_model,
            )
        )
        
    return derived_md_paths




def _load_documents(
    text_paths: list[Path], 
    derived_md_paths: list[Path], 
    unchanged_sources: set[str]
) -> list[Document]:
    """
    Loads text files and derived markdown files into Document objects.
    Applies metadata overrides (source_uri, original_path) for derived files.
    """
    changed_text_paths = []
    for p in text_paths:
        _, source_uri = _get_rel_and_source(p)
        if source_uri not in unchanged_sources:
            changed_text_paths.append(p)

    docs = []
    if changed_text_paths:
        docs.extend(load_text_files(changed_text_paths))

    if derived_md_paths:
        derived_docs = load_text_files(derived_md_paths)
        
        # Override source_uri & original path so stable IDs remain tied to ORIGINAL source
        for doc in derived_docs:
            md_text = doc.page_content
            override = parse_frontmatter_for_source_uri(md_text)
            orig_rel = parse_frontmatter_key(md_text, "original_rel")

            if override:
                doc.metadata["source_uri"] = override
            
            if orig_rel:
                # Store original absolute path for hashing. 
                # Assumes the script runs relative to the repo root usually.
                base_path = Path(__file__).resolve().parents[3] 
                doc.metadata["original_path"] = str((base_path / orig_rel).resolve())

            doc.metadata["derived_path"] = doc.metadata.get("path")

        docs.extend(derived_docs)
        
    return docs




def _group_and_upsert_docs(
    docs: list[Document], 
    settings: RagSettings, 
    embeddings: FastEmbedEmbeddings
) -> int:
    """
    Assigns domain keys, splits docs, groups by collection/source, 
    and performs the database/vector store upsert.
    """
    # 1. Assign metadata
    for doc in docs:
        source_uri = doc.metadata["source_uri"]
        domain_key = infer_domain_key_from_source_uri(source_uri)
        doc.metadata["domain_key"] = domain_key
        doc.metadata["target_collection"] = map_domain_to_collection(domain_key)

    # 2. Split docs
    splits = split_docs(docs, settings.chunk_size, settings.chunk_overlap)
    
    # 3. Compute/Verify file hashes for the loaded docs
    # Note: We re-hash here to ensure we use the 'original_path' for derived docs
    file_hashes = {}

    for doc in docs:
        source_uri = doc.metadata["source_uri"]
        original_path = doc.metadata.get("original_path")
        path_to_hash = Path(original_path) if original_path else Path(doc.metadata["path"])
        file_hashes[source_uri] = _sha256_file(path_to_hash)

    # 4. Group by Collection -> Source
    by_collection_then_source: dict[str, dict[str, list[Document]]] = {}

    for doc in splits:
        collection = doc.metadata.get("target_collection", settings.collection_name)
        source_uri = doc.metadata["source_uri"]
        by_collection_then_source.setdefault(collection, {}).setdefault(source_uri, []).append(doc)

    # 5. Upsert Loop
    total = 0
    # Initialize VS once to ensure connection/tables, though loop re-inits for collections
    vs = build_qdrant(settings=settings, embedding_fn=embeddings)

    with SessionLocal() as db:
        for collection_name, sources_map in by_collection_then_source.items():
            
            # Re-init VS for specific collection override
            vs = build_qdrant(
                settings=settings,
                embedding_fn=embeddings,
                collection_name_override=collection_name,
            )

            for source_uri, chunk_docs in sources_map.items():
                content_hash = file_hashes[source_uri]

                src_row, unchanged = upsert_source(
                    db,
                    source_uri=source_uri,
                    content_hash=content_hash,
                    collection_name=collection_name,
                    namespace=settings.namespace,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )

                if unchanged:
                    continue

                chunks_payload = []
                ids = []

                for idx, doc in enumerate(chunk_docs):
                    chunk_hash = _sha256_text(doc.page_content)
                    point_id = _make_point_id(source_uri, content_hash, idx, chunk_hash)

                    doc.metadata.update({
                        "source_uri": source_uri,
                        "content_hash": content_hash,
                        "chunk_index": idx,
                        "chunk_hash": chunk_hash,
                        "pg_source_id": src_row.id,
                        "collection_name": collection_name,  
                    })

                    chunks_payload.append({
                        "chunk_index": idx,
                        "chunk_hash": chunk_hash,
                        "qdrant_point_id": point_id,
                    })
                    ids.append(point_id)

                chunk_rows = replace_chunks(db, source_id=src_row.id, chunks=chunks_payload)

                for doc, row in zip(chunk_docs, chunk_rows):
                    doc.metadata["pg_chunk_id"] = row.id

                try:
                    delete_source(vs, source_uri)
                    upsert_documents(vs, chunk_docs, ids)
                    db.commit()
                except Exception:
                    db.rollback()
                    raise

                total += len(chunk_docs)
    
    return total




@traceable
def ingest_files(paths: Iterable[str | Path], settings: RagSettings) -> int:
    # 1. Categorize Inputs
    text_paths, pdf_paths, img_paths = _categorize_paths(paths)
    all_paths = text_paths + pdf_paths + img_paths
    
    # 2. Detect Changes
    unchanged_sources, _ = _detect_unchanged_sources(all_paths, settings)
    
    # 3. Process Derived Files (OCR/VLM)
    derived_md_paths = _process_derived_files(pdf_paths, img_paths, unchanged_sources, settings)

    # 4. Load Documents
    docs = _load_documents(text_paths, derived_md_paths, unchanged_sources)
    if not docs:
        return 0

    # 5. Embed and Upsert
    embeddings = build_fastembed_embeddings(settings.embedding_model, settings.chunk_size)
    
    return _group_and_upsert_docs(docs, settings, embeddings)







# ---------------- Ingest Function ----------------

# @traceable
# def ingest_files(paths: Iterable[str | Path], settings: RagSettings) -> int:

#     # 0) Expand directory/glob inputs
#     input_paths = expand_inputs(paths)
#     input_paths = [p for p in input_paths if p.suffix.lower() in {".md",".txt",".pdf",".png",".jpg",".jpeg",".webp"}]

#     # 1) Split into text vs non-text
#     #TODO: refacotr this into a function that runs O(n) time complexity
#     text_paths: list[Path] = [p for p in input_paths if is_text(p)]
#     pdf_paths: list[Path] = [p for p in input_paths if is_pdf(p)]
#     img_paths: list[Path] = [p for p in input_paths if is_image(p)]

#     # 2) Preprocess non-text into derived markdown
#     derived_root = Path("data/derived").resolve()
#     pdf_md_dir = derived_root / "pdf_md"
#     img_md_dir = derived_root / "images_md"
#     ocr_pdf_dir = derived_root / "ocr_pdfs"

#     derived_md_paths: list[Path] = []
#     candidate_hashes: dict[str, str] = {}

#     # helper to compute repo-relative path and source_uri matching your loader convention
#     def rel_and_source(p: Path) -> tuple[str, str]:
#         path_rel = _repo_relative(p)
#         return path_rel, f"file:{path_rel}"
    

#     for p in text_paths + pdf_paths + img_paths:
#         _, source_uri = rel_and_source(p)
#         candidate_hashes[source_uri] = _sha256_file(p)
    

#     unchanged_sources: set[str] = set()

#     with SessionLocal() as db:
#         existing = db.scalars(
#             select(RagSource).where(RagSource.source_uri.in_(list(candidate_hashes.keys())))
#         ).all()

#         existing_by_uri = {rag_source.source_uri: rag_source for rag_source in existing}

#         for source_uri, content_hash in candidate_hashes.items():
#             src = existing_by_uri.get(source_uri)
#             if not src:
#                 continue

#             expected_collection = map_domain_to_collection(infer_domain_key_from_source_uri(source_uri))

#             if (
#                 src.content_hash == content_hash
#                 and src.collection_name == expected_collection
#                 and src.namespace == settings.namespace
#                 and src.chunk_size == settings.chunk_size
#                 and src.chunk_overlap == settings.chunk_overlap
#             ):
#                 unchanged_sources.add(source_uri)


#     for path in pdf_paths:
#         path_rel, source_uri = rel_and_source(path)

#         if source_uri in unchanged_sources:
#             continue

#         derived_md_paths.append(
#             pdf_to_derived_md(
#                 path,
#                 source_uri=source_uri,
#                 path_rel=path_rel,
#                 derived_pdf_md_dir=pdf_md_dir,
#                 ocr_pdf_dir=ocr_pdf_dir,
#             )
#         )

#     # VLM
#     caption_model = settings.caption_model if hasattr(settings, "caption_model") else "llava:7b"


#     for path in img_paths:
#         path_rel, source_uri = rel_and_source(path)

#         if source_uri in unchanged_sources:
#             continue

#         derived_md_paths.append(
#             image_to_derived_md(
#                 path,
#                 source_uri=source_uri,
#                 path_rel=path_rel,
#                 derived_img_md_dir=img_md_dir,
#                 caption_model=caption_model,
#             )
#         )

#     # 3) Load docs:
#     # - text_paths are loaded normally
#     # - derived_md_paths are loaded normally then we override their source_uri using frontmatter

#     changed_text_paths: list[Path] = []

#     for p in text_paths:
#         _, source_uri = rel_and_source(p)
#         if source_uri not in unchanged_sources:
#             changed_text_paths.append(p)

#     docs = []
#     if changed_text_paths:
#         docs.extend(load_text_files(changed_text_paths))

#     if derived_md_paths:
#         derived_docs = load_text_files(derived_md_paths)

#         # Override source_uri & original path so stable IDs remain tied to ORIGINAL source, not derived file
#         for doc in derived_docs:
#             md_text = doc.page_content
#             override = parse_frontmatter_for_source_uri(md_text)
#             orig_rel = parse_frontmatter_key(md_text, "original_rel")

#             if override:
#                 doc.metadata["source_uri"] = override
            
#             if orig_rel:
#                 # store original absolute path for hashing
#                 doc.metadata["original_path"] = str((Path(__file__).resolve().parents[3] / orig_rel).resolve())

#             # optionally keep derived path info too:
#             doc.metadata["derived_path"] = doc.metadata.get("path")

#         docs.extend(derived_docs)
    
#     if not docs:
#         return 0
    
#     for doc in docs:
#         source_uri = doc.metadata["source_uri"]
#         domain_key = infer_domain_key_from_source_uri(source_uri)
#         doc.metadata["domain_key"] = domain_key
#         doc.metadata["target_collection"] = map_domain_to_collection(domain_key)
                                                               
#     splits = split_docs(docs, settings.chunk_size, settings.chunk_overlap)

#     embeddings = build_fastembed_embeddings(settings.embedding_model, settings.chunk_size)
    
#     vs = build_qdrant(settings=settings, embedding_fn=embeddings)


#     # Pre-compute file-level hashes once per source_uri
#     file_hashes: dict[str, str] = {}

#     for doc in docs:
#         source_uri = doc.metadata["source_uri"]
#         original_path = doc.metadata.get("original_path")
#         path_to_hash = Path(original_path) if original_path else Path(doc.metadata["path"]) # Derived path Fallback
#         file_hashes[source_uri] = _sha256_file(path_to_hash)


#     # Group splits by collection THEN by source_uri so chunk_index is per file
#     by_collection_then_source: dict[str, dict[str, list[Document]]] = {}

#     for doc in splits:
#         collection = doc.metadata.get("target_collection", settings.collection_name)
#         source_uri = doc.metadata["source_uri"]

#         by_collection_then_source.setdefault(collection, {})
#         by_collection_then_source[collection].setdefault(source_uri, [])
#         by_collection_then_source[collection][source_uri].append(doc)

    
#     total = 0

#     with SessionLocal() as db:
#         for collection_name, sources_map in by_collection_then_source.items():

#             vs = build_qdrant(
#                 settings=settings,
#                 embedding_fn=embeddings,
#                 collection_name_override=collection_name,
#             )

#             for source_uri, chunk_docs in sources_map.items():
#                 content_hash = file_hashes[source_uri]

#                 src_row, unchanged = upsert_source(
#                     db,
#                     source_uri=source_uri,
#                     content_hash=content_hash,
#                     collection_name=collection_name,   # IMPORTANT: store per-domain collection
#                     namespace=settings.namespace,
#                     chunk_size=settings.chunk_size,
#                     chunk_overlap=settings.chunk_overlap,
#                 )

#                 if unchanged:
#                     continue

#                 # stable ids per file
#                 chunks_payload: list[dict] = []
#                 ids: list[str] = []

#                 for idx, doc in enumerate(chunk_docs):
#                     chunk_hash = _sha256_text(doc.page_content)
#                     point_id = _make_point_id(source_uri, content_hash, idx, chunk_hash)

#                     doc.metadata.update({
#                         "source_uri": source_uri,
#                         "content_hash": content_hash,
#                         "chunk_index": idx,
#                         "chunk_hash": chunk_hash,
#                         "pg_source_id": src_row.id,
#                         "collection_name": collection_name,  
#                     })

#                     chunks_payload.append({
#                         "chunk_index": idx,
#                         "chunk_hash": chunk_hash,
#                         "qdrant_point_id": point_id,
#                     })

#                     ids.append(point_id)

#                 chunk_rows = replace_chunks(db, source_id=src_row.id, chunks=chunks_payload)

#                 for doc, row in zip(chunk_docs, chunk_rows):
#                     doc.metadata["pg_chunk_id"] = row.id

#                 try:
#                     delete_source(vs, source_uri)
#                     upsert_documents(vs, chunk_docs, ids)
#                     db.commit()
#                 except Exception:
#                     db.rollback()
#                     raise

#                 total += len(chunk_docs)

#     return total


