from __future__ import annotations

import os
import re
import boto3
from urllib.parse import urlparse
from pathlib import Path
from typing import Iterable
import hashlib
import uuid
from dataclasses import dataclass

from langsmith import traceable
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from .embeddings import build_fastembed_embeddings
from .loaders import load_text_files, _repo_relative
from .settings import RagSettings
from .splitter import split_docs
from .vectorstore import build_qdrant, delete_source, upsert_documents
from .dynamodb import upsert_source_if_changed
# from ai_agents.db.session import SessionLocal
# from ai_agents.rag.repo import upsert_source, replace_chunks
# from sqlalchemy import select
# from ai_agents.db.models import RagSource 
from ai_agents.config.constants import QDRANT_ID_NAMESPACE

from .preprocess import (
    expand_inputs, is_text, is_pdf, is_image,
    pdf_to_derived_md, image_to_derived_md,
    parse_frontmatter_for_source_uri, parse_frontmatter_key
)

# ---------------- s3 helper functions ----------------
s3 = boto3.client("s3")

RAW_BUCKET = os.environ.get("RAW_BUCKET", "")
DERIVED_BUCKET = os.environ.get("DERIVED_BUCKET", "")

def parse_s3_uri(uri: str) -> tuple[str, str]:
    # s3://bucket/key
    u = urlparse(uri)

    if u.scheme != "s3":
        raise ValueError(f"Not an s3 uri: {uri}")
    
    bucket = u.netloc
    key = u.path.lstrip("/")

    return bucket, key



def _normalize_s3_uri(uri: str) -> str:
    """
    Accepts both:
      - s3://bucket/key
      - s3:/bucket/key  (incorrect)
    Returns canonical s3://bucket/key
    """
    uri = uri.strip()

    if uri.startswith("s3:/") and not uri.startswith("s3://"):
        return "s3://" + uri[len("s3:/"):].lstrip("/")
    
    return uri


def _is_s3_uri(x: str) -> bool:
    x = x.strip()
    return x.startswith("s3://") or (x.startswith("s3:/") and not x.startswith("s3://"))


def _looks_like_prefix(key: str) -> bool:
    # Treat "folder-like" keys as prefixes
    return key.endswith("/") or not re.search(r"\.[A-Za-z0-9]{1,6}$", key)


def _resolve_inputs_to_local(paths: Iterable[str | Path]) -> tuple[list[Path], dict[str, str], dict[str, str]]:
    """
    Resolve any S3 URIs into real local /tmp files so the rest of the pipeline
    can operate on Path objects.

    Returns:
      - local_paths: list[Path]
      - source_uri_by_local_abs: dict[str(abs_local_path)] -> "s3://bucket/key" OR "file:..."
      - rel_by_local_abs: dict[str(abs_local_path)] -> s3 key OR repo-relative rel path
    """
    local_paths: list[Path] = []
    source_uri_by_local_abs: dict[str, str] = {}
    rel_by_local_abs: dict[str, str] = {}

    for x in paths:
        # Path input (local)
        if isinstance(x, Path):
            p = x.resolve()
            abs_p = str(p)

            path_rel, source_uri = _get_rel_and_source(p)

            local_paths.append(p)
            source_uri_by_local_abs[abs_p] = source_uri
            rel_by_local_abs[abs_p] = path_rel
            continue

        s = str(x)

        # Local string path
        if not _is_s3_uri(s):
            p = Path(s).resolve()
            abs_p = str(p)

            path_rel, source_uri = _get_rel_and_source(p)

            local_paths.append(p)
            source_uri_by_local_abs[abs_p] = source_uri
            rel_by_local_abs[abs_p] = path_rel
            continue

        # S3 uri
        s3_uri = _normalize_s3_uri(s)
        bucket, key = parse_s3_uri(s3_uri)

        # Expand prefixes into object keys
        if _looks_like_prefix(key):
            prefix = key
            if prefix and not prefix.endswith("/"):
                prefix += "/"

            for obj_key in list_s3_keys(bucket, prefix):
                local = Path(download_to_tmp(bucket, obj_key)).resolve()
                abs_local = str(local)

                local_paths.append(local)
                source_uri_by_local_abs[abs_local] = f"s3://{bucket}/{obj_key}"
                rel_by_local_abs[abs_local] = obj_key
        else:
            local = Path(download_to_tmp(bucket, key)).resolve()
            abs_local = str(local)

            local_paths.append(local)
            source_uri_by_local_abs[abs_local] = s3_uri
            rel_by_local_abs[abs_local] = key

    return local_paths, source_uri_by_local_abs, rel_by_local_abs



def _content_hash_for_source(source_uri: str, local_path: Path | None = None) -> str:
    # S3: use HEAD fingerprint
    if source_uri.startswith("s3://"):
        bucket, key = parse_s3_uri(source_uri)
        fp = head_content_fingerprint(bucket, key)
        return hashlib.sha256(fp.encode("utf-8")).hexdigest()

    # Local: hash bytes
    if local_path is None:
        raise ValueError("local_path required for non-s3 source")
    
    return _sha256_file(local_path)



def list_s3_keys(bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            keys.append(obj["Key"])

    return keys


def head_content_fingerprint(bucket: str, key: str) -> str:
    res = s3.head_object(Bucket=bucket, Key=key)
    etag = (res.get("ETag") or "").strip('"')
    size = int(res.get("ContentLength") or 0)
    version = res.get("VersionId")  
    lm = res.get("LastModified")

    # stable idempotency token:
    return f"etag:{etag}|size:{size}|lm:{lm}|ver:{version or ''}"


def download_to_tmp(bucket: str, key: str) -> str:
    local_path = f"/tmp/{key.replace('/', '_')}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)

    return local_path


def upload_text(bucket: str, key: str, text: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType="text/markdown")




# ---------------- dynamodb helper functions ----------------
ddb = boto3.resource("dynamodb")

SOURCES_TABLE = os.environ.get("SOURCES_TABLE", "rag_sources")


def _sources_table():
    return ddb.Table(SOURCES_TABLE)

def _ddb_source_pk(namespace: str) -> str:
    return f"NS#{namespace}"

def _ddb_source_sk(source_uri: str) -> str:
    return f"SRC#{source_uri}"

def _get_source_item(namespace: str, source_uri: str) -> dict | None:
    table = _sources_table()

    res = table.get_item(Key={
        "pk": _ddb_source_pk(namespace), 
        "sk": _ddb_source_sk(source_uri)
    })

    return res.get("Item")





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



def _categorize_paths(
        paths: Iterable[str | Path],
        expand: bool = False
    ) -> tuple[list[Path], list[Path], list[Path]]:
    """ 
    Filters inputs into text, pdf, and image paths.

    (Dev) If expand is set to True, it expands inputs according 
    to local file directory setup
    """

    if expand:
        input_paths = expand_inputs(paths)
    else:
        input_paths = [Path(p) for p in paths]


    # Filter by allowed extensions
    valid_exts = {".md", ".txt", ".pdf", ".png", ".jpg", ".jpeg"}
    input_paths = [p for p in input_paths if p.suffix.lower() in valid_exts]

    text_paths = [p for p in input_paths if is_text(p)]
    pdf_paths = [p for p in input_paths if is_pdf(p)]
    img_paths = [p for p in input_paths if is_image(p)]

    return text_paths, pdf_paths, img_paths







#############################################
################ Dev version ################
#############################################

# def _detect_unchanged_sources(
#     paths: list[Path], 
#     settings: RagSettings
# ) -> tuple[set[str], dict[str, str]]:
#     """
#     Computes file hashes and checks the DB to identify sources that haven't changed.
#     Returns a set of unchanged source URIs and a dict of current candidate hashes.
#     """
#     candidate_hashes: dict[str, str] = {}
    
#     for p in paths:
#         _, source_uri = _get_rel_and_source(p)
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
                
#     return unchanged_sources, candidate_hashes



#############################################
################ Prod version ################
#############################################
def _detect_unchanged_sources_dynamodb(
    paths: list[Path],
    settings: RagSettings,
    source_uri_by_local_abs: dict[str, str]
) -> tuple[set[str], dict[str, str]]:
    """
    Computes file hashes and checks DynamoDB to identify sources that haven't changed.
    Returns a set of unchanged source URIs and a dict of current candidate hashes.
    """
    
    candidate_hashes: dict[str, str] = {}
    unchanged_sources: set[str] = set()

    for p in paths:
        abs_p = str(p.resolve())
        source_uri = source_uri_by_local_abs.get(abs_p, _get_rel_and_source(p)[1])
        candidate_hashes[source_uri] = _content_hash_for_source(source_uri, local_path=p)

    for source_uri, content_hash in candidate_hashes.items():
        item = _get_source_item(settings.namespace, source_uri)

        if not item:
            continue

        expected_collection = map_domain_to_collection(
            infer_domain_key_from_source_uri(source_uri)
        )

        if (
            item.get("content_hash") == content_hash
            and item.get("collection_name") == expected_collection
            and item.get("namespace") == settings.namespace
            and int(item.get("chunk_size", 0)) == int(settings.chunk_size)
            and int(item.get("chunk_overlap", 0)) == int(settings.chunk_overlap)
        ):
            unchanged_sources.add(source_uri)

    return unchanged_sources, candidate_hashes





@dataclass
class IngestFailure:
    stage: str          # "pdf_to_md" | "image_to_md" 
    path: str
    source_uri: str
    error_type: str
    message: str


def _process_derived_files(
    pdf_paths: list[Path], 
    img_paths: list[Path], 
    unchanged_sources: set[str], 
    settings: RagSettings,
    source_uri_by_local_abs: dict[str, str] | None = None,
    rel_by_local_abs: dict[str, str] | None = None,
) -> tuple[list[Path], list[IngestFailure]]:
    """
    Converts PDFs and Images to Markdown if they are not in the unchanged set.
    """
    derived_root = Path("data/derived").resolve()
    pdf_md_dir = derived_root / "pdf_md"
    img_md_dir = derived_root / "images_md"
    ocr_pdf_dir = derived_root / "ocr_pdfs"
    
    derived_md_paths: list[Path] = []
    failures: list[IngestFailure] = []

    # Process PDFs
    for path in pdf_paths:
        
        # Prod
        if source_uri_by_local_abs and rel_by_local_abs:
            abs_path = str(path.resolve())
            path_rel = rel_by_local_abs.get(abs_path, _get_rel_and_source(path)[0])  
            source_uri = source_uri_by_local_abs.get(abs_path, _get_rel_and_source(path)[1])
              

        # Dev
        else:
            path_rel, source_uri = _get_rel_and_source(path)  



        if source_uri in unchanged_sources:
            continue
            
        try:
            output = pdf_to_derived_md(
                pdf_path=path,
                source_uri=source_uri,
                path_rel=path_rel,
                derived_pdf_md_dir=pdf_md_dir,
                ocr_pdf_dir=ocr_pdf_dir,
            )

            # Upload derived markdown to S3
            if DERIVED_BUCKET:
                md_text = Path(output).read_text(encoding="utf-8")
                derived_key = Path(path_rel).with_suffix(".md").as_posix()
                upload_text(DERIVED_BUCKET, derived_key, md_text)

            derived_md_paths.append(output)

        except Exception as e:
            failures.append(
                IngestFailure(
                    stage="pdf_to_md",
                    path=str(path),
                    source_uri=source_uri,
                    error_type=type(e).__name__,
                    message=str(e),
                )
            )

            # skip this file, continue ingest
            continue

    
    
    # VLM
    caption_model = getattr(settings, "caption_model", "meta-llama/llama-4-scout-17b-16e-instruct")
    
    # Process Images
    for path in img_paths:
        
        # Prod
        if source_uri_by_local_abs and rel_by_local_abs:
            abs_path = str(path.resolve())
            path_rel = rel_by_local_abs.get(abs_path, _get_rel_and_source(path)[0])  
            source_uri = source_uri_by_local_abs.get(abs_path, _get_rel_and_source(path)[1])
              

        # Dev
        else:
            path_rel, source_uri = _get_rel_and_source(path)  


        if source_uri in unchanged_sources:
            continue
            
        try:
            out = image_to_derived_md(
                image_path=path,
                source_uri=source_uri,
                path_rel=path_rel,
                derived_img_md_dir=img_md_dir,
                caption_model=caption_model,
            )


            # Upload derived markdown to S3
            if DERIVED_BUCKET:
                md_text = Path(out).read_text(encoding="utf-8")
                derived_key = Path(path_rel).with_suffix(".md").as_posix()
                upload_text(DERIVED_BUCKET, derived_key, md_text)


            derived_md_paths.append(out)

        except Exception as e:
            failures.append(
                IngestFailure(
                    stage="image_to_md",
                    path=str(path),
                    source_uri=source_uri,
                    error_type=type(e).__name__,
                    message=str(e),
                )
            )

            # skip this file, continue ingest
            continue

        
    return derived_md_paths, failures




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

            ######### Dev only #########
            # orig_rel = parse_frontmatter_key(md_text, "original_rel")   ######### Dev only #########

            if override:
                doc.metadata["source_uri"] = override
            

            ######### Dev only #########
            # if orig_rel:
            #     # Store original absolute path for hashing. 
            #     # Assumes the script runs relative to the repo root.
            #     base_path = Path(__file__).resolve().parents[3] 
            #     doc.metadata["original_path"] = str((base_path / orig_rel).resolve())

            doc.metadata["derived_path"] = doc.metadata.get("path")

        docs.extend(derived_docs)
        
    return docs




def _group_and_upsert_docs(
    docs: list[Document], 
    settings: RagSettings, 
    embeddings: FastEmbedEmbeddings,
    dev: bool = False
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
    file_hashes = {}

    if dev:
        for doc in docs:
            source_uri = doc.metadata["source_uri"]
            original_path = doc.metadata.get("original_path")
            path_to_hash = Path(original_path) if original_path else Path(doc.metadata["path"])
            file_hashes[source_uri] = _sha256_file(path_to_hash)
    else:
        source_uri = doc.metadata["source_uri"]
        local_path = Path(doc.metadata["path"])
        file_hashes[source_uri] = _content_hash_for_source(source_uri, local_path=local_path)

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

    for collection_name, sources_map in by_collection_then_source.items():

        vs = build_qdrant(
            settings=settings,
            embedding_fn=embeddings,
            collection_name_override=collection_name,
        )

        for source_uri, chunk_docs in sources_map.items():
            content_hash = file_hashes[source_uri]

            # DynamoDB: decides if unchanged; also records metadata if changed/new
            unchanged = upsert_source_if_changed(
                namespace=settings.namespace,
                source_uri=source_uri,
                content_hash=content_hash,
                collection_name=collection_name,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            if unchanged:
                continue

            ids: list[str] = []

            for idx, doc in enumerate(chunk_docs):
                chunk_hash = _sha256_text(doc.page_content)
                point_id = _make_point_id(source_uri, content_hash, idx, chunk_hash)

                doc.metadata.update({
                    "source_uri": source_uri,
                    "content_hash": content_hash,
                    "chunk_index": idx,
                    "chunk_hash": chunk_hash,
                    "collection_name": collection_name,
                })

                ids.append(point_id)

            # Qdrant: replace all chunks for this source
            delete_source(vs, source_uri)
            upsert_documents(vs, chunk_docs, ids)

            total += len(chunk_docs)


    #############################################
    ################ Dev version ################
    #############################################

    # with SessionLocal() as db:
    #     for collection_name, sources_map in by_collection_then_source.items():
            
    #         # Re-init VS for specific collection override
    #         vs = build_qdrant(
    #             settings=settings,
    #             embedding_fn=embeddings,
    #             collection_name_override=collection_name,
    #         )

    #         for source_uri, chunk_docs in sources_map.items():
    #             content_hash = file_hashes[source_uri]

    #             src_row, unchanged = upsert_source(
    #                 db,
    #                 source_uri=source_uri,
    #                 content_hash=content_hash,
    #                 collection_name=collection_name,
    #                 namespace=settings.namespace,
    #                 chunk_size=settings.chunk_size,
    #                 chunk_overlap=settings.chunk_overlap,
    #             )

    #             if unchanged:
    #                 continue

    #             chunks_payload = []
    #             ids = []

    #             for idx, doc in enumerate(chunk_docs):
    #                 chunk_hash = _sha256_text(doc.page_content)
    #                 point_id = _make_point_id(source_uri, content_hash, idx, chunk_hash)

    #                 doc.metadata.update({
    #                     "source_uri": source_uri,
    #                     "content_hash": content_hash,
    #                     "chunk_index": idx,
    #                     "chunk_hash": chunk_hash,
    #                     "pg_source_id": src_row.id,
    #                     "collection_name": collection_name,  
    #                 })

    #                 chunks_payload.append({
    #                     "chunk_index": idx,
    #                     "chunk_hash": chunk_hash,
    #                     "qdrant_point_id": point_id,
    #                 })
    #                 ids.append(point_id)

    #             chunk_rows = replace_chunks(db, source_id=src_row.id, chunks=chunks_payload)

    #             for doc, row in zip(chunk_docs, chunk_rows):
    #                 doc.metadata["pg_chunk_id"] = row.id

    #             try:
    #                 delete_source(vs, source_uri)
    #                 upsert_documents(vs, chunk_docs, ids)
    #                 db.commit()
    #             except Exception:
    #                 db.rollback()
    #                 raise

    #             total += len(chunk_docs)
    

    
    return total






#############################################
################ Dev version ################
#############################################

# @traceable
# def ingest_files(paths: Iterable[str | Path], settings: RagSettings) -> int:
#     # 1. Categorize Inputs
#     text_paths, pdf_paths, img_paths = _categorize_paths(paths, expand=True)
#     all_paths = text_paths + pdf_paths + img_paths
    
#     # 2. Detect Changes
#     unchanged_sources, _ = _detect_unchanged_sources_dynamodb(all_paths, settings)

#     # Dev version
#     # unchanged_sources, _ = _detect_unchanged_sources(all_paths, settings) 
    
#     # 3. Process Derived Files (OCR/VLM)
#     derived_md_paths, failures = _process_derived_files(pdf_paths, img_paths, unchanged_sources, settings)

#     for f in failures:
#         print(f"[ingest] WARN stage={f.stage} source={f.source_uri} err={f.error_type}: {f.message}")

#     # 4. Load Documents
#     docs = _load_documents(text_paths, derived_md_paths, unchanged_sources)
#     if not docs:
#         return 0

#     # 5. Embed and Upsert
#     embeddings = build_fastembed_embeddings(settings.embedding_model, settings.chunk_size)
    
#     return _group_and_upsert_docs(docs, settings, embeddings, True)




#############################################
################ Prod version ################
#############################################

@traceable
def ingest_files(paths: Iterable[str | Path], settings: RagSettings) -> int:
    # 0) Resolve S3 URIs -> local /tmp files (and keep mapping to original source_uri)
    local_inputs, source_uri_by_local_abs, rel_by_local_abs = _resolve_inputs_to_local(paths)

    # 1) Categorize Inputs (only local Paths now)
    text_paths, pdf_paths, img_paths = _categorize_paths(local_inputs, expand=False)
    all_paths = text_paths + pdf_paths + img_paths

    # 2) Detect Changes (hash local bytes)
    unchanged_sources, _ = _detect_unchanged_sources_dynamodb(all_paths, settings, source_uri_by_local_abs)

    # 3) Process Derived Files (OCR/VLM)
    derived_md_paths, failures = _process_derived_files(
        pdf_paths=pdf_paths, 
        img_paths=img_paths, 
        unchanged_sources=unchanged_sources, 
        settings=settings,
        source_uri_by_local_abs=source_uri_by_local_abs,
        rel_by_local_abs=rel_by_local_abs
    )

    for f in failures:
        print(f"[ingest] WARN stage={f.stage} source={f.source_uri} err={f.error_type}: {f.message}")

    # 4) Load Documents (only local Paths)
    docs = _load_documents(text_paths, derived_md_paths, unchanged_sources)
    if not docs:
        return 0

    # 4.5) Override doc metadata to use original S3 source_uri (not file:/tmp/...)
    for doc in docs:
        p = doc.metadata.get("path")
        if not p:
            continue
        abs_p = str(Path(p).resolve())
        if abs_p in source_uri_by_local_abs:
            doc.metadata["source_uri"] = source_uri_by_local_abs[abs_p]
            doc.metadata["path_rel"] = rel_by_local_abs.get(abs_p, doc.metadata.get("path_rel"))

    # 5) Embed and Upsert
    embeddings = build_fastembed_embeddings(settings.embedding_model, settings.chunk_size)
    return _group_and_upsert_docs(docs, settings, embeddings, False)





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


