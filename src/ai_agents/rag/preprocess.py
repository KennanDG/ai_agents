from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import hashlib
import re
import subprocess

from langchain_core.documents import Document

from ..tools.definitions.pdf_to_markdown import pdf_to_markdown, PdfToMarkdownRequest

TEXT_EXTS = {".md", ".txt"}
PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

DERIVED_SOURCE_URI_KEY = "rag_source_uri"  # stored inside derived md for override


def expand_inputs(inputs: Iterable[str | Path]) -> list[Path]:

    out: list[Path] = []

    for x in inputs:
        p = Path(x)

        # basic glob support
        if any(ch in str(p) for ch in ["*", "?", "["]):
            out.extend([q for q in p.parent.glob(p.name) if q.is_file()])

        elif p.is_dir():
            out.extend([q for q in p.rglob("*") if q.is_file()])
            
        else:
            out.append(p)

    return out



# ----------------- Determine file type -----------------
def is_text(p: Path) -> bool:
    return p.suffix.lower() in TEXT_EXTS

def is_pdf(p: Path) -> bool:
    return p.suffix.lower() in PDF_EXTS

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS




def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _derive_md_path(derived_dir: Path, original_rel: str, content_hash: str) -> Path:
    # deterministic name to avoid collisions
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", original_rel)

    return derived_dir / f"{safe}.{content_hash[:12]}.md"


def _ocr_pdf_if_needed(pdf_path: Path, ocr_out_path: Path) -> Path:
    """
    Uses OCRmyPDF to generate a searchable PDF.
    Requires system deps + ocrmypdf installed in container.
    """
    ocr_out_path.parent.mkdir(parents=True, exist_ok=True)

    # --skip-text keeps existing text if present
    cmd = ["ocrmypdf", "--skip-text", str(pdf_path), str(ocr_out_path)]
    subprocess.run(cmd, check=True)
    return ocr_out_path


def _should_ocr_pdf(extracted_md: str, min_chars: int = 800) -> bool:
    # strip frontmatter if present
    body = extracted_md
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) == 3:
            body = parts[2]
    return len(body.strip()) < min_chars


def pdf_to_derived_md(
    pdf_path: Path,
    *,
    source_uri: str,
    path_rel: str,
    derived_pdf_md_dir: Path,
    ocr_pdf_dir: Path,
) -> Path:
    """
    Always produces a derived markdown file for PDFs.
    OCR is performed if extracted text is too small.
    """
    content_hash = _sha256_file(pdf_path)
    out_md = _derive_md_path(derived_pdf_md_dir, path_rel, content_hash)

    # 1) extract
    res = pdf_to_markdown(PdfToMarkdownRequest(pdf_path=pdf_path, output_md_path=None))
    md = res.markdown

    # 2) OCR if needed
    if _should_ocr_pdf(md):
        ocr_pdf = ocr_pdf_dir / f"{pdf_path.stem}.{content_hash[:12]}.ocr.pdf"
        ocr_pdf = _ocr_pdf_if_needed(pdf_path, ocr_pdf)
        res2 = pdf_to_markdown(PdfToMarkdownRequest(pdf_path=ocr_pdf, output_md_path=None))
        md = res2.markdown
        has_ocr = True
    else:
        has_ocr = False

    # 3) prepend strong metadata frontmatter (for later override)
    frontmatter = (
        "---\n"
        f"{DERIVED_SOURCE_URI_KEY}: {source_uri}\n"
        f"original_rel: {path_rel}\n"
        "content_type: pdf\n"
        f"has_ocr: {str(has_ocr).lower()}\n"
        "has_caption: false\n"
        "---\n\n"
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(frontmatter + md, encoding="utf-8")

    return out_md


def ocr_image_text(image_path: Path) -> str:
    """
    OCR for images (Tesseract).
    """
    try:
        from PIL import Image
        import pytesseract
    except Exception as e:
        raise RuntimeError("Install deps: pillow pytesseract + system tesseract-ocr") from e

    img = Image.open(image_path)

    return pytesseract.image_to_string(img)


def caption_image(image_path: Path, *, caption_model: str) -> str:
    """
    Caption using an LLM/VLM. Implementation depends on your stack.
    Start with a stub that you can hook into Ollama later.
    """
    # TODO: implement with your chosen VLM. For now, return a basic placeholder.
    return f"Image file '{image_path.name}' (captioning not yet wired)."


def image_to_derived_md(
    image_path: Path,
    *,
    source_uri: str,
    path_rel: str,
    derived_img_md_dir: Path,
    caption_model: str,
) -> Path:
    """
    Always OCR + always caption for image files.
    """
    content_hash = _sha256_file(image_path)
    out_md = _derive_md_path(derived_img_md_dir, path_rel, content_hash)

    ocr_text = ocr_image_text(image_path)
    cap = caption_image(image_path, caption_model=caption_model)

    md = (
        "---\n"
        f"{DERIVED_SOURCE_URI_KEY}: {source_uri}\n"
        f"original_rel: {path_rel}\n"
        "content_type: image\n"
        "has_ocr: true\n"
        "has_caption: true\n"
        f"caption_model: {caption_model}\n"
        "---\n\n"
        "## Caption\n"
        f"{cap}\n\n"
        "## OCR Text\n"
        f"{ocr_text}\n"
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    return out_md


def parse_frontmatter_for_source_uri(md_text: str) -> Optional[str]:
    """
    Read rag_source_uri from derived markdown frontmatter.
    Lightweight parser to avoid extra deps.
    """
    if not md_text.startswith("---"):
        return None
    
    parts = md_text.split("---", 2)

    if len(parts) != 3:
        return None
    
    fm = parts[1]

    for line in fm.splitlines():
        if line.strip().startswith(f"{DERIVED_SOURCE_URI_KEY}:"):
            return line.split(":", 1)[1].strip()
        
    return None
