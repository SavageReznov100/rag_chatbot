from pathlib import Path
from typing import TypedDict

import docx


def list_docx_files(source_path: str | Path) -> list[str]:
    """Return the Word (.docx) files under *source_path*.

    If *source_path* is a file, verify that it is a .docx file and return it
    inside a single-element list. If it is a directory, return every .docx file
    directly inside it (non-recursive).
    """
    path = Path(source_path)

    if path.is_file():
        if not path.name.lower().endswith(".docx"):
            raise ValueError(f"{source_path} is not a .docx file")
        return [str(path)]

    return sorted(
        str(entry) for entry in path.iterdir() if entry.name.lower().endswith(".docx")
    )


class DocxParagraph(TypedDict):
    """A single extracted paragraph: its 1-based index and raw text."""

    paragraphNumber: int
    text: str


def extract_docx_paragraphs(file_path: str | Path) -> list[DocxParagraph]:
    """Extract text paragraph-by-paragraph from a Word (.docx) document.

    Word documents have no fixed page layout like PDFs, so text is split into
    paragraphs instead of pages. Each entry maps a 1-based ``paragraphNumber``
    to the paragraph's text. Empty paragraphs are omitted; if none have text, a
    single entry holding the whole document's text is returned instead.
    """
    document = docx.Document(str(file_path))

    paragraphs: list[DocxParagraph] = []
    all_text: list[str] = []

    for index, paragraph in enumerate(document.paragraphs):
        text = paragraph.text
        all_text.append(text)
        if text.strip():
            paragraphs.append({"paragraphNumber": index + 1, "text": text})

    if not paragraphs:
        return [{"paragraphNumber": 1, "text": "\n".join(all_text)}]

    return paragraphs
