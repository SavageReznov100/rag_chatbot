
import json

import requests
from loguru import logger
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from app.llm.prompt.prompts import CATEGORIZATION_SYSTEM_PROMPT, QUERY_REWRITE_PROMPT , SYSTEM_PROMPT , CONVERSATION_SYSTEM_PROMPT
from app.llm.models.models import jina_reranker, model , model2,model3
from app.retrieval.retrieval import VectorDB
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ModelCallLimitMiddleware




class InitialNodeResponse(BaseModel):
    """Determine the next node to route to and, if possible, the user's department."""

    nextNode: str = Field(
        description="The node to route to next"
    )
    department: str | None = Field(
        default=None,
        description="The user's department inferred from the conversation, if determinable",
    )


async def initialNode(state):
    message_history = state["messages"]
    sys_msg = SystemMessage(CATEGORIZATION_SYSTEM_PROMPT)

    agent = create_agent(
        model=model3,
        tools=[],
        response_format=InitialNodeResponse,
        checkpointer=InMemorySaver(),
         middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
    ],
    )

    response = await agent.ainvoke(
        {"messages":[sys_msg , *message_history] }
    )

    result = response["structured_response"]

    state_update = {"nextNode": result.nextNode}
    if result.department:
        state_update["department"] = result.department
    return state_update

'''
async def initialNode(state):
    """Classify the latest message as QUERY_REWRITE or OUT_OF_SCOPE.

    Uses a plain Cohere completion (not ``create_agent``) because LangChain's
    ``create_agent`` forces ``tool_choice="any"`` when a ``response_format`` is
    supplied, which the Cohere Chat API rejects with HTTP 422.
    """
    message_history = state["messages"]
    sys_msg = SystemMessage(CATEGORIZATION_SYSTEM_PROMPT)

    response = await model3.ainvoke([sys_msg, *message_history])

    content = str(response.content).strip()
    # The model may wrap the JSON in markdown fences; strip them if present.
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    try:
        payload = json.loads(content)
        next_node = str(payload.get("nextNode", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        next_node = ""

    if next_node not in ("QUERY_REWRITE", "OUT_OF_SCOPE"):
        raise ValueError(
            "initialNode produced an invalid nextNode "
            f"(got {next_node!r}). Raw response: {response.content!r}"
        )

    return {"nextNode": next_node}
'''


async def nextNode(state):
    if state["nextNode"] == "QUERY_REWRITE":
        return "queryRewriteNode"
    if state['nextNode'] == "OUT_OF_SCOPE":
        return "outOfScopeNode"
    if state['nextNode'] == "CONVERSATION":
        return "conversationNode"  
    raise ValueError(f"Unknown nextNode value: {state['nextNode']!r}")



async def conversationNode(state):
    message_history = state["messages"]
    sys_msg = SystemMessage(CONVERSATION_SYSTEM_PROMPT)
    response = await model3.ainvoke([sys_msg, *message_history])
    return {"messages": [response]}


async def queryRewriteNode(state):
    """Rewrite the user question into optimized search queries.

    Fills ``state["rewriteQueries"]`` with the rewritten queries produced by
    ``QUERY_REWRITE_PROMPT``. Retrieval/synthesis nodes will consume this
    state later.
    """
    question = state["messages"][-1].content

    response = await model3.ainvoke(
        QUERY_REWRITE_PROMPT.format(question=question)
    )

    content = str(response.content).strip()
    # The model may wrap the JSON in markdown fences; strip them if present.
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    try:
        payload = json.loads(content)
        questions = [str(q) for q in payload.get("questions", [])]
        logger.info(questions)
    except (json.JSONDecodeError, AttributeError):
        questions = []

    return {"rewriteQueries": questions or [question]}

async def retrieverNode(state):
    last_text = next(
    (m.content for m in reversed(state["messages"])
     if isinstance(m, HumanMessage)),
    ""
)
    retrievedDocs = []

    for q in state["rewriteQueries"]:
        result = await VectorDB(q, department=state.get("department"))
        retrievedDocs.append(result)

    flattened_docs = [
    doc
    for r in retrievedDocs
    for doc in r["validParentDocs"]
]    
    if not flattened_docs:
        logger.warning("No documents retrieved, skipping Jina rerank")
        return {"retrievedDocuments": []}

    # Jina rerank payload: base config + query (last user text) + flattened docs.
    data = {
        **jina_reranker["data"],
        "query": last_text,
        "documents": [doc.page_content for doc in flattened_docs],
    }

    # Fail fast with an actionable message instead of Jina's cryptic 401.
    if not jina_reranker["headers"].get("Authorization"):
        raise RuntimeError(
            "JINA_API_KEY is not set. Add JINA_API_KEY=<your key> to .env "
            "and restart the app."
        )

    response = requests.post(
        jina_reranker["url"],
        headers=jina_reranker["headers"],
        data=json.dumps(data),
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        if response.status_code in (401, 403):
            raise RuntimeError(
                "Jina rerank authentication failed "
                f"(HTTP {response.status_code}). Check that JINA_API_KEY in "
                ".env is a valid key and restart the app."
            ) from exc
        raise

    # Jina returns {"results": [{"index": ..., "relevance_score": ...}, ...]}
    # already sorted by relevance; map indices back to the original documents.
    results = response.json().get("results", [])
    reranked_docs = [
        flattened_docs[result["index"]]
        for result in results
        if isinstance(result.get("index"), int)
        and 0 <= result["index"] < len(flattened_docs)
    ]

    if not reranked_docs:
        logger.warning("Jina rerank returned no usable results, keeping original order")
        reranked_docs = flattened_docs

    return {"retrievedDocuments" : reranked_docs}


async def agentNode(state):
    def format_documents_as_string(documents: list) -> str:
        return "\n\n".join(
        doc.page_content
        for doc in documents
        if getattr(doc, "page_content", None)
        )
    
    retrieved_docs_to_string = format_documents_as_string(state["retrievedDocuments"])
    sys_msg = SystemMessage(SYSTEM_PROMPT)

    human_msg = HumanMessage(f'''
      User Question:
      <user_question>
      {"\n".join(state["rewriteQueries"]) }
      Retrieved Data:
      <retrieved_chunks>
      {retrieved_docs_to_string}
        ''')

    # A raw chat model accepts a list of messages (not a graph-state dict).
    response = await model3.ainvoke([sys_msg, human_msg])
    return {"messages": [response]}

async def outOfScopeNode(state):
    ai_response = AIMessage(
        content="Sorry, I can't help you with that."
    )

    return {
        "messages": [ai_response]
    }
