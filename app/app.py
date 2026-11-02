import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, exceptions
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.db import (
    DATABASE_URL,
    Department,
    Documents,
    Role,
    User,
    UserMemory,
    create_db_and_tables,
    get_async_session,
    get_user_db,
    user_departments,
)
from app.db.schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DepartmentCreate,
    DepartmentRead,
    QueryRequest,
    QueryResponse,
    QueryResult,
    RoleCreate,
    RoleRead,
    UserAdminRead,
    UserCreate,
    UserRead,
    UserUpdate,
)
from app.embeddings.embeddings import documentEmbedding
from app.retrieval.retrieval import VectorDB
from app.agents.graph import build_agent
from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from fastapi import UploadFile, File
from loguru import logger

from app.ingestion.versioning.manager import prepare_new_version

load_dotenv()

SECRET = os.getenv("JWT_SECRET",)

# Directory where uploaded source documents are persisted before ingestion.
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def create(self, user_create, safe=False, request=None):
        await self.validate_password(user_create.password, user_create)

        existing_user = await self.user_db.get_by_email(user_create.email)
        if existing_user is not None:
            raise exceptions.UserAlreadyExists()

        user_dict = (
            user_create.create_update_dict()
            if safe
            else user_create.create_update_dict_superuser()
        )
        password = user_dict.pop("password")
        user_dict["hashed_password"] = self.password_helper.hash(password)

        created_user = await self.user_db.create(user_dict)

        await self.on_after_register(created_user, request)

        return created_user


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)


bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)


fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

current_user = fastapi_users.current_user()
current_active_user = fastapi_users.current_user(active=True)
current_superuser = fastapi_users.current_user(active=True, superuser=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()

    # Persist LangGraph conversation state (keyed by ``thread_id``) in Postgres.
    checkpoint_uri = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    async with AsyncPostgresSaver.from_conn_string(checkpoint_uri) as checkpointer:
        await checkpointer.setup()
        app.state.agent = build_agent(checkpointer)
        yield


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate), prefix="/auth", tags=["auth"]
)
app.include_router(
    fastapi_users.get_reset_password_router(), prefix="/auth", tags=["auth"]
)
app.include_router(
    fastapi_users.get_verify_router(UserRead), prefix="/auth", tags=["auth"]
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/auth/users",
    tags=["users"],
)


@app.get("/protected-route", tags=["auth"])
async def protected_route(user: User = Depends(current_active_user)):
    return {"message": f"Hello, {user.email}!"}


# Departments (superuser-only management) ----------------------------------------

@app.get(
    "/departments",
    response_model=list[DepartmentRead],
    dependencies=[Depends(current_superuser)],
)
async def list_departments(session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(select(Department).order_by(Department.name))
    return list(result.scalars().all())




@app.post(
    "/departments",
    response_model=DepartmentRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(current_superuser)],
)
async def create_department(
    payload: DepartmentCreate, session: AsyncSession = Depends(get_async_session)
):
    if await session.scalar(select(Department).where(Department.name == payload.name)):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Department with this name already exists"
        )
    department = Department(name=payload.name, description=payload.description)
    session.add(department)
    await session.commit()
    await session.refresh(department)
    return department


@app.delete(
    "/departments/{department_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(current_superuser)],
)
async def delete_department(
    department_id: uuid.UUID, session: AsyncSession = Depends(get_async_session)
):
    department = await session.get(Department, department_id)
    if department is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Department not found")
    await session.delete(department)
    await session.commit()


# Roles (superuser-only management) ----------------------------------------------

@app.get(
    "/roles", response_model=list[RoleRead], dependencies=[Depends(current_superuser)]
)
async def list_roles(session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(select(Role).order_by(Role.name))
    return list(result.scalars().all())


@app.post(
    "/roles",
    response_model=RoleRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(current_superuser)],
)
async def create_role(
    payload: RoleCreate, session: AsyncSession = Depends(get_async_session)
):
    if await session.scalar(select(Role).where(Role.name == payload.name)):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Role with this name already exists"
        )
    role = Role(name=payload.name, description=payload.description)
    session.add(role)
    await session.commit()
    await session.refresh(role)
    return role


@app.delete(
    "/roles/{role_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(current_superuser)],
)
async def delete_role(
    role_id: uuid.UUID, session: AsyncSession = Depends(get_async_session)
):
    role = await session.get(Role, role_id)
    if role is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Role not found")
    await session.delete(role)
    await session.commit()


# User administration (superuser-only) -------------------------------------------

@app.get(
    "/users",
    response_model=list[UserAdminRead],
    dependencies=[Depends(current_superuser)],
)
async def list_users(session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(
        select(User)
        .options(selectinload(User.roles), selectinload(User.departments))
        .order_by(User.email)
    )
    users = result.unique().scalars().all()
    return [
        {
            "id": user.id,
            "email": user.email,
            "is_active": user.is_active,
            "is_superuser": user.is_superuser,
            "is_verified": user.is_verified,
            "roles": [role.name for role in user.roles],
            "departments": [department.name for department in user.departments],
        }
        for user in users
    ]


@app.post(
    "/users/{user_id}/roles/{role_id}",
    dependencies=[Depends(current_superuser)],
)
async def assign_role(
    user_id: uuid.UUID,
    role_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    user = await session.get(User, user_id, options=[selectinload(User.roles)])
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    role = await session.get(Role, role_id)
    if role is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Role not found")
    if role not in user.roles:
        user.roles.append(role)
        await session.commit()
    return {"detail": "Role assigned to user"}


@app.delete(
    "/users/{user_id}/roles/{role_id}",
    dependencies=[Depends(current_superuser)],
)
async def remove_role(
    user_id: uuid.UUID,
    role_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    user = await session.get(User, user_id, options=[selectinload(User.roles)])
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    role = await session.get(Role, role_id)
    if role is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Role not found")
    if role in user.roles:
        user.roles.remove(role)
        await session.commit()
    return {"detail": "Role removed from user"}




   #Embedding 
   #  Upload a document and embed it into the database
    
@app.post("/embeddings")
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session),
    user:User =  Depends(current_active_user)
):
    """Persist an uploaded document (.pdf or .docx) and run the embedding pipeline.

    ``document_final_pdf`` / ``document_final_word`` / ``documentEmbedding``
    read from the filesystem, so the incoming ``UploadFile`` (a
    ``SpooledTemporaryFile``) is first written to ``UPLOAD_DIR`` and the
    resulting path is handed to the pipeline.
    """
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file has no filename")

    filename = Path(file.filename).name
    if not filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Only .pdf and .docx files are supported",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / filename

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    # The versioning step references documents.doc_id via a foreign key, so the
    # parent row must exist before a version can be created.
    title = destination.stem

    # ``documents.department`` stores a single department name; use the user's
    # first department, falling back to a default until the AI learns it.
    department = user.departments[0].name if user.departments else "General"

    document_values = await prepare_new_version(session, department, title)
    doc_id, department, version = document_values

    existing = await session.scalar(
        select(Documents).where(Documents.doc_id == doc_id)
    )
    if existing is None:
        session.add(
            Documents(doc_id=doc_id, title=title, department=department)
        )
        await session.commit()
   
    await documentEmbedding(db=session, doc_id=doc_id, department=department, version=version, document_path=str(destination))

    return {"message": "Document uploaded and embedded", "doc_id": doc_id}

   #  Query / retrieval: semantic search over embedded child chunks.

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    user: User = Depends(current_active_user),
) -> QueryResponse:
    """Search embedded documents.

    ``doc_id`` and ``department`` are optional metadata filters; when omitted,
    the search spans the whole index (child chunks only).
    """
    try:
        result = await VectorDB(
            query=payload.query,
            doc_id=payload.doc_id,
            department=payload.department,
        )
    except Exception as err:
        logger.exception("Retrieval failed")
        raise HTTPException(
            _agent_error_status(err),
            f"Retrieval failed: {err}",
        ) from err

    parent_docs = result.get("validParentDocs", [])
    return QueryResponse(
        results=[
            QueryResult(page_content=doc.page_content, metadata=dict(doc.metadata))
            for doc in parent_docs
        ]
    )

   #  Chat: run a user message straight through the agent graph.

def _chat_history_to_langchain(payload: ChatRequest) -> list:
    """Convert a ``ChatRequest`` payload into LangChain message objects."""
    if payload.messages is not None:
        history = []
        for message in payload.messages:
            if message.role in ("user", "human"):
                history.append(HumanMessage(content=message.content))
            elif message.role in ("assistant", "ai"):
                history.append(AIMessage(content=message.content))
        if history:
            return history

    if payload.message:
        return [HumanMessage(content=payload.message)]

    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        "Provide either `messages` (a list) or `message` (a single string).",
    )


def _agent_error_status(err: Exception) -> int:
    """Map upstream provider failures to 502; keep everything else at 500.

    Provider SDKs (Ollama, Cohere, Pinecone, Jina, ...) commonly raise errors
    that expose an HTTP ``status_code`` attribute (directly or on a nested
    ``response``).  When the upstream provider itself rejected the request, a
    generic 500 hides the real cause, so surface it as 502 Bad Gateway instead.
    """
    status_code = getattr(err, "status_code", None)
    if status_code is None:
        response = getattr(err, "response", None)
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and status_code >= 400:
        return status.HTTP_502_BAD_GATEWAY
    return status.HTTP_500_INTERNAL_SERVER_ERROR


async def _remember_department(session: AsyncSession, user: User, department_name: str) -> None:
    """Persist an AI-inferred department to user memory and the user's department link."""
    memory = await session.scalar(
        select(UserMemory).where(
            UserMemory.user_id == user.id, UserMemory.key == "department"
        )
    )
    if memory is None:
        session.add(UserMemory(user_id=user.id, key="department", value=department_name))
    else:
        memory.value = department_name

    department = await session.scalar(
        select(Department).where(Department.name == department_name)
    )
    if department is None:
        department = Department(name=department_name)
        session.add(department)
        await session.flush()

    existing = await session.execute(
        select(user_departments).where(
            user_departments.c.user_id == user.id,
            user_departments.c.department_id == department.id,
        )
    )
    if existing.first() is None:
        await session.execute(
            user_departments.insert().values(
                user_id=user.id, department_id=department.id
            )
        )
    await session.commit()


@app.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> ChatResponse:
    """Run the conversation history through the compiled LangGraph agent.

    The graph receives the message history via ``MessagesState`` and handles
    all routing internally (categorization, query rewrite, retrieval,
    synthesis). The final state is echoed back: the full message history plus
    the ``nextNode`` chosen by ``initial_node``.
    """
    history = _chat_history_to_langchain(payload)

    # Seed the conversation with the department the AI already learned about
    # this user (memory persists across conversations via ``user_memory``).
    remembered_department = await session.scalar(
        select(UserMemory.value).where(
            UserMemory.user_id == user.id, UserMemory.key == "department"
        )
    )
    if not remembered_department and user.departments:
        remembered_department = user.departments[0].name

    invoke_input: dict = {"messages": history}
    if remembered_department:
        invoke_input["department"] = remembered_department

    try:
        final_state = await request.app.state.agent.ainvoke(
            invoke_input,
            {"configurable": {"thread_id": payload.conversation_id}},
        )
    except Exception as err:
        logger.exception("Agent graph failed")
        raise HTTPException(
            _agent_error_status(err),
            f"Agent graph failed: {err}",
        ) from err

    inferred_department = final_state.get("department")
    if inferred_department:
        await _remember_department(session, user, inferred_department)

    return ChatResponse(
        messages=[
            ChatMessage(role=msg.type, content=str(msg.text))
            for msg in final_state.get("messages", [])
        ],
        next_node=final_state.get("nextNode"),
    )

   #  Get all documents

'''
@app.get("/documents")
async def get_documents(session: AsyncSession = Depends(get_async_session)):
    documents = await session.execute(select(Document).order_by(Document.doc_id))
    return list(documents.scalars().all())  

   #  Get a document by id
@app.get("/documents/{document_id}")
async def get_document(document_id: uuid.UUID, session: AsyncSession = Depends(get_async_session)):
    document = await session.get(Document, document_id)
    return document
'''