import asyncio
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from app.ingestion.loaders.docx import extract_docx_paragraphs, list_docx_files
from app.ingestion.loaders.pdf import extract_pdf_pages, list_pdf_files

# Directory (or single file) scanned for PDFs during ingestion. Override by
# setting the PDF_SOURCE_PATH environment variable.
PDF_SOURCE_PATH: str = os.getenv("PDF_SOURCE_PATH", "./data/pdfs")

# Directory (or single file) scanned for Word (.docx) documents during
# ingestion. Override by setting the DOCX_SOURCE_PATH environment variable.
DOCX_SOURCE_PATH: str = os.getenv("DOCX_SOURCE_PATH", "./data/docx")


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only the JSON-serializable entries of *metadata*.

    Strings, numbers and booleans are kept as-is; lists are kept only when every
    element is a string. ``None`` values and anything else are dropped.
    """
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            clean[key] = value
    return clean


def page_to_documents(
    text: str, page_number: int, file_path: str | Path
) -> list[Document]:
    """Split a page's raw text into paragraphs and wrap each in a Document.

    Paragraphs are separated by one or more blank lines. Every Document carries
    sanitized metadata (source file and 1-based page number) for citations.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    metadata = sanitize_metadata({"page": page_number, "source": str(file_path)})

    return [
        Document(page_content=paragraph, metadata=dict(metadata))
        for paragraph in paragraphs
    ]


def paragraph_to_documents(
    text: str, paragraph_number: int, file_path: str | Path
) -> list[Document]:
    """Split a Word paragraph's raw text into blocks and wrap each in a Document.

    Word documents have no fixed page layout, so metadata carries the 1-based
    paragraph number (instead of a page number) for citations back to the
    source document. Blocks are separated by one or more blank lines; a single
    paragraph usually produces exactly one block. Every Document carries
    sanitized metadata (source file and paragraph number).
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return []

    metadata = sanitize_metadata(
        {"paragraphNumber": paragraph_number, "source": str(file_path)}
    )

    return [
        Document(page_content=block, metadata=dict(metadata))
        for block in blocks
    ]


async def document_final_pdf(pdf_path: str | Path = PDF_SOURCE_PATH) -> list[Document]:
    """Ingest every PDF under *pdf_path* into a list of Documents."""
    logger.info("Starting PDF ingestion pipeline")

    pdf_files = list_pdf_files(pdf_path)
    if not pdf_files:
        logger.warning("No PDF files found at {}", pdf_path)
        return []

    logger.info("Found {} PDF file(s) to ingest", len(pdf_files))

    all_docs: list[Document] = []

    for i, file_path in enumerate(pdf_files):
        logger.info("[{}/{}] Reading: {}", i + 1, len(pdf_files), file_path)

        try:
            pages = await asyncio.to_thread(extract_pdf_pages, file_path)
            logger.info("  - {} page(s) extracted", len(pages))

            for page in pages:
                docs = page_to_documents(page["text"], page["pageNumber"], file_path)
                all_docs.extend(docs)
        except Exception as err:  # keep going past an unreadable file
            logger.error("Failed to read {} - {}", file_path, err)

        # Small pause between files to keep logs readable / avoid disk thrashing.
        if i < len(pdf_files) - 1:
            await asyncio.sleep(0.2)

    logger.info("Raw paragraph chunks collected: {}", len(all_docs))
    return all_docs


async def document_final_word(
    docx_path: str | Path = DOCX_SOURCE_PATH,
) -> list[Document]:
    """Ingest every Word (.docx) document under *docx_path* into Documents.

    Mirrors :func:`document_final_pdf`, but extracts text paragraph-by-paragraph
    (Word has no page layout) via :func:`extract_docx_paragraphs`.
    """
    logger.info("Starting Word ingestion pipeline")

    docx_files = list_docx_files(docx_path)
    if not docx_files:
        logger.warning("No Word (.docx) files found at {}", docx_path)
        return []

    logger.info("Found {} Word file(s) to ingest", len(docx_files))

    all_docs: list[Document] = []

    for i, file_path in enumerate(docx_files):
        logger.info("[{}/{}] Reading: {}", i + 1, len(docx_files), file_path)

        try:
            paragraphs = await asyncio.to_thread(extract_docx_paragraphs, file_path)
            logger.info("  - {} paragraph(s) extracted", len(paragraphs))

            for paragraph in paragraphs:
                docs = paragraph_to_documents(
                    paragraph["text"], paragraph["paragraphNumber"], file_path
                )
                all_docs.extend(docs)
        except Exception as err:  # keep going past an unreadable file
            logger.error("Failed to read {} - {}", file_path, err)

        # Small pause between files to keep logs readable / avoid disk thrashing.
        if i < len(docx_files) - 1:
            await asyncio.sleep(0.2)

    logger.info("Raw paragraph chunks collected: {}", len(all_docs))
    return all_docs
