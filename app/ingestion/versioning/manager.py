import uuid
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def prepare_new_version(
    db: AsyncSession,
    department: str,
    title: str,
    doc_id: str | None = None,
) -> tuple[str, str, int]:

    # -----------------------------------
    # NEW DOCUMENT
    # -----------------------------------
    if doc_id is None:

        # Generate a new document identity
        doc_id = str(uuid.uuid4())

        # First version
        version = 1

        # Create the document
        await db.execute(
            text("""
                INSERT INTO documents (
                    id,
                    doc_id,
                    title,
                    department,
                    current_version
                )
                VALUES (
                    :id,
                    :doc_id,
                    :title,
                    :department,
                    :current_version
                )
            """),
            {
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "title": title,
                "department": department,
                "current_version": 0,
            },
        )

    # -----------------------------------
    # EXISTING DOCUMENT
    # -----------------------------------
    else:

        result = await db.execute(
            text("""
                SELECT current_version
                FROM documents
                WHERE doc_id = :doc_id
            """),
            {"doc_id": doc_id},
        )

        current_version = result.scalar_one_or_none()

        if current_version is None:
            raise ValueError(
                f"Document with doc_id '{doc_id}' does not exist"
            )

        version = current_version + 1

    # -----------------------------------
    # CREATE VERSION
    # -----------------------------------

    await db.execute(
        text("""
            INSERT INTO documents_version (
                id,
                doc_id,
                version,
                status,
                department
            )
            VALUES (
                :id,
                :doc_id,
                :version,
                'processing',
                :department
            )
        """),
        {
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "version": version,
            "department": department,
        },
    )

    await db.commit()

    return doc_id, department, version

async def activate_version(
    db: AsyncSession,
    doc_id: str,
    version: int,
) -> None:

    await db.execute(
        text("""
            UPDATE documents_version
            SET status = 'archived'
            WHERE doc_id = :doc_id
              AND status = 'active'
        """),
        {"doc_id": doc_id},
    )

    await db.execute(
        text("""
            UPDATE documents_version
            SET status = 'active'
            WHERE doc_id = :doc_id
              AND version = :version
        """),
        {
            "doc_id": doc_id,
            "version": version,
        },
    )

    await db.execute(
        text("""
            UPDATE documents
            SET current_version = :version,
                updated_at = CURRENT_TIMESTAMP
            WHERE doc_id = :doc_id
        """),
        {
            "doc_id": doc_id,
            "version": version,
        },
    )

    await db.commit()





async def update_current_version(
    db: AsyncSession,
    doc_id: str,
    version: int,
) -> None:

    await db.execute(
        text("""
            UPDATE documents
            SET current_version = :version,
                updated_at = CURRENT_TIMESTAMP
            WHERE doc_id = :doc_id
        """),
        {
            "doc_id": doc_id,
            "version": version,
        },
    )

    await db.commit()