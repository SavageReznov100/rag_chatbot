import asyncio
import json
import os

from pinecone import Pinecone
from pinecone.db_data import Index
from app.llm.models.models import embeddings, reranker
from loguru import logger
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import BaseDocumentCompressor, Document



def _default_pinecone_index() -> Index:
    """Build the default Pinecone index handle (raw SDK client)."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME") or os.getenv("PINECONE_INDEX")
    if not index_name:
        raise ValueError(
            "PINECONE_INDEX_NAME (or PINECONE_INDEX) environment variable is not set"
        )
    return pc.Index(index_name)

def _default_embeddings() -> Embeddings:
    """Build the default embeddings model (Cohere)."""
    return embeddings  

def _default_reranker() -> BaseDocumentCompressor:
    """Build the default reranker model (Cohere)."""
    return reranker


async def VectorDB(
    query: str,
    doc_id: str | None = None,
    department: str | None = None,
    embeddings: Embeddings | None = None,
    pinecone_index: Index | None = None,
) -> dict[str, list[Document]]:
    """Multi-vector retrieval: search child chunks, fetch their parent chunks.

    ``doc_id`` and ``department`` are optional metadata filters: each one that
    is provided is combined (logical AND) with the mandatory
    ``docType == "child"`` filter, so callers can scope a search to a single
    document, a single department, both, or neither (whole index).
    """
    if embeddings is None:
        embeddings = _default_embeddings()
    if pinecone_index is None:
        pinecone_index = _default_pinecone_index()

    stats = await asyncio.to_thread(pinecone_index.describe_index_stats)
    logger.info("Index stats: {}", stats)

    vector_store = PineconeVectorStore(
        index=pinecone_index,
        embedding=embeddings,
        text_key="text",
    )

    search_filter: dict[str, str] = {"docType": "child"}
    if doc_id:
        search_filter["docId"] = doc_id
    if department:
        search_filter["department"] = department
    logger.info("child search filter: {}", search_filter)

    child_docs = await vector_store.asimilarity_search(
        query,
        k=10,
        filter=search_filter,
    )
    logger.info("childDocs found: {}", len(child_docs))

    parent_chunk_ids = sorted(
        {
            doc.metadata["parentId"]
            for doc in child_docs
            if doc.metadata.get("parentId") is not None
        }
    )
    logger.info("filteredChunksIds: {}", parent_chunk_ids)

    if not parent_chunk_ids:
        logger.warning(
            "No valid documents found for reranking, returning empty array"
        )
        return {"validParentDocs": []}

    fetch_result = await asyncio.to_thread(
        pinecone_index.fetch, ids=parent_chunk_ids
    )

    await asyncio.sleep(0.3)

    parent_docs: list[Document] = []
    for rec in (fetch_result.vectors or {}).values():
        metadata = dict(rec.metadata or {})
        text = metadata.get("text") or metadata.get("pageContent") or ""
        parent_docs.append(Document(page_content=str(text), metadata=metadata))

    logger.info("parentDocs found: {}", len(parent_docs))
    if parent_docs:
        logger.info(
            "parentDocs sample: {}",
            json.dumps(
                {
                    "pageContent": parent_docs[0].page_content,
                    "metadata": parent_docs[0].metadata,
                },
                indent=2,
                default=str,
            ),
        )
    else:
        logger.info("parentDocs sample: No docs")

    # Filter out documents with empty or whitespace-only content.
    valid_parent_docs = [doc for doc in parent_docs if doc.page_content.strip()]

    logger.info("validParentDocs after filtering: {}", len(valid_parent_docs))

    if not valid_parent_docs:
        logger.warning(
            "No valid documents found for reranking, returning empty array"
        )
        return {"validParentDocs": []}

    #reranked_docs = await reranker.acompress_documents(valid_parent_docs, query)
    #logger.info("rerankedDocs found: {}", len(reranked_docs))

    return {"validParentDocs": valid_parent_docs}
