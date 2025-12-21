from __future__ import annotations

import argparse
from pathlib import Path

from ai_agents.tools.definitions.pdf_to_markdown import (
    pdf_to_markdown,
    PdfToMarkdownRequest,
)

def main():

    parser = argparse.ArgumentParser(description="Convert PDF files to Markdown.")
    parser.add_argument("pdf", help="Path to the PDF file (or a directory of PDFs).")
    parser.add_argument("--out", default="./data/processed", help="Output directory.")
    args = parser.parse_args()

    pdf_input = Path(args.pdf).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)


    if pdf_input.is_dir():
        pdfs = sorted(pdf_input.glob("*.pdf"))

        if not pdfs:
            raise SystemExit(f"No PDFs found in directory: {pdf_input}")
        
        for pdf_path in pdfs:
            out_path = out_dir / (pdf_path.stem + ".md")
            req = PdfToMarkdownRequest(pdf_path=pdf_path, output_md_path=out_path)
            res = pdf_to_markdown(req)
            print(f"✅ {pdf_path.name} -> {out_path.name} ({res.page_count} pages)")
    else:
        out_path = out_dir / (pdf_input.stem + ".md")
        req = PdfToMarkdownRequest(pdf_path=pdf_input, output_md_path=out_path)
        res = pdf_to_markdown(req)
        print(f"✅ {pdf_input.name} -> {out_path.name} ({res.page_count} pages)")


if __name__ == "__main__":
    main()