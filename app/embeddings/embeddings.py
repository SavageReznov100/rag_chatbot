import asyncio
import math
import os
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pinecone import Pinecone
from pinecone.db_data import Index

from app.ingestion.loaders.pipeline import (
    document_final_pdf,
    document_final_word,
    sanitize_metadata,
)
from app.ingestion.versioning.manager import prepare_new_version, update_current_version
from app.llm.models.models import embeddings

EMBEDDING_BATCH_SIZE: int = 20
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _default_embeddings() -> Embeddings:
    """Build the default embeddings model (Cohere)."""
    return embeddings
def _default_pinecone_index() -> Index:
    """Build the default Pinecone index handle (raw SDK client)."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME") or os.getenv("PINECONE_INDEX")
    return pc.Index(index_name)


async def create_parent_document(
    article_docs: list[Document], doc_id: str,version:str, department:str
) -> list[Document]:
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    parent_docs = parent_splitter.split_documents(article_docs)

    results: list[Document] = []
    for split in parent_docs:
        chunk_id = str(uuid.uuid4())
        metadata = dict(split.metadata)

        metadata["chunk_id"] = chunk_id
        metadata["parentId"] = chunk_id
        metadata["original"] = metadata.get("source")  # original local file path
        metadata["docType"] = "parent"
        metadata["source"] = chunk_id
        metadata["originalSource"] = chunk_id
        metadata["docId"] = doc_id
        metadata["department"] = department
        metadata["version"] = version

        split.metadata = sanitize_metadata(metadata)
        results.append(split)

    return results


async def create_child_documents(
    parent_doc: Document, doc_id: str ,version:str, department:str
) -> list[Document]:
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    child_docs = child_splitter.split_documents([parent_doc])

    parent_chunk_id = parent_doc.metadata["chunk_id"]

    results: list[Document] = []
    for index, split in enumerate(child_docs):
        child_chunk_id = str(uuid.uuid4())
        metadata = dict(split.metadata)

        metadata["parentId"] = parent_chunk_id
        metadata["chunk_id"] = f"child-{parent_chunk_id}-{index}"
        metadata["original"] = metadata.get("source")  # original local file path
        metadata["docType"] = "child"
        metadata["source"] = child_chunk_id
        metadata["originalSource"] = child_chunk_id
        # Inherit the parent document id so retrieval filters on ``docId``
        # match child chunks (chunk identity lives in ``chunk_id``/``source``).
        metadata["docId"] = doc_id
        metadata["department"] = department
        metadata["version"] = version

        split.metadata = sanitize_metadata(metadata)
        results.append(split)

    return results


async def _verify_vector_dimension(embeddings: Embeddings, pinecone_index: Index) -> None:
    """Fail fast when embedding vectors and the Pinecone index dimensions disagree."""
    probe = await embeddings.aembed_documents(["dimension check"])
    vector_dim = len(probe[0]) if probe else 0

    try:
        stats = pinecone_index.describe_index_stats()
        try:
            index_dim = stats["dimension"]
        except (TypeError, KeyError):
            index_dim = getattr(stats, "dimension", None)
    except Exception as err:
        logger.warning("Could not read Pinecone index dimension: {}", err)
        return

    if index_dim and vector_dim != index_dim:
        raise ValueError(
            f"Embedding dimension {vector_dim} does not match Pinecone index "
            f"dimension {index_dim}. Recreate the index or set COHERE_EMBED_MODEL "
            "to a model with matching output dimensions."
        )

    logger.info("Vector dimension check passed ({} dims)", vector_dim)


async def _collect_documents(document_path: str) -> list[Document]:
    """Load raw Documents from *document_path* with the right ingestion pipeline.

    Dispatches on the path so both file types flow through ``documentEmbedding``:

    * a ``.docx`` file -> :func:`document_final_word`
    * a ``.pdf`` file  -> :func:`document_final_pdf`
    * a directory      -> both loaders (each picks up its own file type)
    """
    path = Path(document_path)

    if path.is_file():
        if path.name.lower().endswith(".docx"):
            return await document_final_word(document_path)
        if path.name.lower().endswith(".pdf"):
            return await document_final_pdf(document_path)
        raise ValueError(
            f"{document_path} is not a .pdf or .docx file"
        )

    # Directory: ingest every PDF and every Word document directly inside it.
    pdf_docs = await document_final_pdf(document_path)
    word_docs = await document_final_word(document_path)
    return [*pdf_docs, *word_docs]


async def documentEmbedding(
    db: AsyncSession,
    doc_id: str,
    department:str,
    version:str,
    document_path: str,
    embeddings: Embeddings | None = None,
    pinecone_index: Index | None = None,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> None:
    """Load, chunk, and embed the documents under *document_path* into Pinecone."""
    # The caller (POST /embeddings) already created the version row via
    # prepare_new_version and passes its number here; preparing another
    # version inside this function would create a duplicate row.
    logger.info("Embedding version {} for doc_id {}", version, doc_id)

    article_docs = await _collect_documents(document_path)

    if not article_docs:
        logger.warning("No Documents collected. Aborting embedding process.")
        return

    logger.info("Creating Parent Docs")
    parent_docs = await create_parent_document(article_docs, doc_id,version,department)

    if not parent_docs:
        logger.warning("No parent documents created. Aborting embedding process.")
        return

    logger.info("Creating Child Docs")
    child_docs: list[Document] = []
    for parent in parent_docs:
        child_docs.extend(await create_child_documents(parent, doc_id,version,department))

    all_docs: list[Document] = [*parent_docs, *child_docs]
    logger.info("Total Docs is {}", len(all_docs))

    if not all_docs:
        logger.warning("No child or parent documents created. Aborting embedding process.")
        return

    logger.info("Embedding both Parent and Child Docs")

    if embeddings is None:
        embeddings = _default_embeddings()
    if pinecone_index is None:
        pinecone_index = _default_pinecone_index()

    await _verify_vector_dimension(embeddings, pinecone_index)

    total_batches = math.ceil(len(all_docs) / batch_size)
    failed_batches = 0

    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        batch_number = i // batch_size + 1

        valid_batch = [
            doc
            for doc in batch
            if doc
            and isinstance(doc.page_content, str)
            and doc.page_content.strip()
        ]

        if not valid_batch:
            logger.warning(
                "Skipping empty batch {}/{} at index {}",
                batch_number,
                total_batches,
                i,
            )
            continue

        try:
            logger.info(
                "Embedding batch {}/{} ({} documents)",
                batch_number,
                total_batches,
                len(valid_batch),
            )

            vectors = await embeddings.aembed_documents(
                [doc.page_content for doc in valid_batch]
            )

            records = [
                {
                    "id": doc.metadata.get("chunk_id")
                    or doc.metadata.get("source")
                    or str(uuid.uuid4()),
                    "values": vectors[index],
                    "metadata": {**doc.metadata, "text": doc.page_content},
                }
                for index, doc in enumerate(valid_batch)
            ]

            await asyncio.to_thread(pinecone_index.upsert, vectors=records)
            logger.info("Successfully embedded batch {}/{}", batch_number, total_batches)
        except Exception as err:
            logger.error("Failed embedding batch {}: {}", batch_number, err)
            failed_batches += 1
            continue

        await asyncio.sleep(0.1)

    if failed_batches:
        logger.error(
            "Finished with errors: {}/{} batch(es) failed.",
            failed_batches,
            total_batches,
        )
        raise RuntimeError(
            f"Embedding failed for {failed_batches}/{total_batches} batch(es); "
            "the document version was not marked current."
        )

    logger.info("Finished embeddings. Processed {} total documents.", len(all_docs))

    await update_current_version(db, doc_id, version)

'''
async def _main() -> None:
    """Local test harness: exercise the versioning step of the pipeline."""
    from dotenv import load_dotenv
    from sqlalchemy import text

    from app.db.db import async_session_maker, create_db_and_tables

    load_dotenv()
    await create_db_and_tables()

    doc_id = os.getenv("TEST_DOC_ID", "Reznov10")
    document_path = os.getenv("PDF_SOURCE_PATH", "./data/pdfs")

    async with async_session_maker() as db:
        # The versioning step expects the parent "documents" row to already
        # exist (normally created when the document is registered upstream).
        await db.execute(
            text(
                "INSERT OR IGNORE INTO documents (id, doc_id, current_version) "
                "VALUES (:id, :doc_id, 0)"
            ),
            {"id": str(uuid.uuid4()), "doc_id": doc_id},
        )
        await db.commit()

        await documentEmbedding(db=db, doc_id=doc_id, document_path=document_path)


if __name__ == "__main__":
    asyncio.run(_main())
'''