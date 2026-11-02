from pathlib import Path
from typing import TypedDict

import pymupdf


def list_pdf_files(source_path: str | Path) -> list[str]:
    """Return the PDF files under *source_path*.

    If *source_path* is a file, verify that it is a PDF and return it inside a
    single-element list. If it is a directory, return every .pdf file directly
    inside it (non-recursive).
    """
    path = Path(source_path)

    if path.is_file():
        if not path.name.lower().endswith(".pdf"):
            raise ValueError(f"{source_path} is not a .pdf file")
        return [str(path)]

    return sorted(
        str(entry) for entry in path.iterdir() if entry.name.lower().endswith(".pdf")
    )


class PdfPage(TypedDict):
    """A single extracted page: its 1-based page number and raw text."""

    pageNumber: int
    text: str


def extract_pdf_pages(file_path: str | Path) -> list[PdfPage]:
    """Extract text page-by-page from a PDF.

    Each entry maps a 1-based pageNumber to the page's text, so every chunk can
    carry a page number in its metadata (useful for citations back to the source
    document). Pages whose text is empty or only whitespace are omitted; if no
    page produces any text, a single entry holding the whole document's text is
    returned instead.
    """
    pages: list[PdfPage] = []
    all_text: list[str] = []

    with pymupdf.open(str(file_path)) as doc:
        for page in doc:
            text = page.get_text()
            all_text.append(text)
            if text.strip():
                pages.append({"pageNumber": page.number + 1, "text": text})

    if not pages:
        return [{"pageNumber": 1, "text": "\n".join(all_text)}]

    return pages
