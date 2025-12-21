from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Request Model
class PdfToMarkdownRequest(BaseModel):
    pdf_path: Path = Field(..., description="Path to the source PDF")
    output_md_path: Optional[Path] = Field(
        default=None, description="Optional path to write the markdown file"
    )
    include_frontmatter: bool = True
    add_page_markers: bool = False

    @field_validator("pdf_path")
    @classmethod
    def validate_pdf_path(cls, v: Path) -> Path:
        v = v.expanduser().resolve()

        if not v.exists():
            raise ValueError(f"PDF not found: {v}")
        
        if v.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {v.name}")
        
        return v

    @field_validator("output_md_path")
    @classmethod
    def normalize_output_md_path(cls, v: Optional[Path]) -> Optional[Path]:

        if v is None:
            return None
        
        v = v.expanduser().resolve()

        return v



# Response Model
class PdfToMarkdownResult(BaseModel):
    source_pdf: Path
    markdown: str
    page_count: int
    wrote_file: bool = False
    output_md_path: Optional[Path] = None




def pdf_to_markdown(req: PdfToMarkdownRequest) -> PdfToMarkdownResult:
    try:
        import fitz  # PyMuPDF
        import pymupdf4llm
    except Exception as e:
        raise RuntimeError("Install deps: uv add pymupdf4llm pymupdf") from e

    doc = fitz.open(req.pdf_path)
    page_count = doc.page_count
    doc.close()

    md = pymupdf4llm.to_markdown(str(req.pdf_path))

    if req.include_frontmatter:
        frontmatter = (
            f"---\nsource: {req.pdf_path}\npage_count: {page_count}\n---\n\n"
        )
        md = frontmatter + md

    # Optional page markers (simple version: you’d need per-page extraction for accurate markers)
    # if req.add_page_markers:
    #     md = "<!-- page: 1 -->\n" + md  # placeholder strategy

    wrote = False
    out_path = req.output_md_path
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        wrote = True

    return PdfToMarkdownResult(
        source_pdf=req.pdf_path,
        markdown=md,
        page_count=page_count,
        wrote_file=wrote,
        output_md_path=out_path if wrote else None,
    )