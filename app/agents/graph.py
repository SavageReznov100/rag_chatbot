
from typing import Annotated

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.errors import NodeError
from langgraph.types import RetryPolicy, TimeoutPolicy

from app.agents.node import initialNode ,nextNode,outOfScopeNode, queryRewriteNode , retrieverNode , agentNode , conversationNode


class State(MessagesState):
    rewriteQueries: Annotated[list[str], lambda x, y: x + y]
    nextNode: str | None
    retrievedDocuments: Annotated[list[str], lambda x, y: x + y]
    status: str | None
    department: str | None


async def default_error_handler(state: State, error: NodeError) -> State:
    return {"status": f"handled: {error.error}"}

def build_agent(checkpointer):
    agent_builder = StateGraph(State).set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=2),
        error_handler=default_error_handler,
        timeout=TimeoutPolicy(run_timeout=30),
    )

    agent_builder.add_node("initialNode", initialNode)
    agent_builder.add_node("outOfScopeNode", outOfScopeNode)
    agent_builder.add_node("conversationNode", conversationNode)
    agent_builder.add_node("queryRewriteNode", queryRewriteNode)
    agent_builder.add_node("retrieverNode", retrieverNode)
    agent_builder.add_node("agentNode", agentNode)

    agent_builder.add_edge(START, "initialNode")
    agent_builder.add_conditional_edges(
        "initialNode",
        nextNode,
        ["queryRewriteNode", "outOfScopeNode", "conversationNode"],
    )

    agent_builder.add_edge("outOfScopeNode", END)
    agent_builder.add_edge("conversationNode", END)
    agent_builder.add_edge("queryRewriteNode", "retrieverNode")
    agent_builder.add_edge("retrieverNode", "agentNode")
    agent_builder.add_edge("agentNode", END)

    return agent_builder.compile(checkpointer=checkpointer)






