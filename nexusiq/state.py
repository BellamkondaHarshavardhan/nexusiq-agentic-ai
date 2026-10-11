"""Shared state that flows through the LangGraph workflow."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

RESET = {"__reset__": True}
ROUTES = ("knowledge", "database", "hybrid", "tools", "chat")


def merge_trace(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """Reducer for the per-turn execution trace. A leading RESET marker starts a fresh trace for a new turn,
    which is needed because the checkpointer keeps state between turns of the same conversation."""
    left, right = left or [], right or []
    if right and right[0] == RESET:
        return right[1:]
    return left + right


class AgentState(TypedDict, total=False):
    # conversation memory (persisted per thread_id by the checkpointer)
    messages: Annotated[list[AnyMessage], add_messages]
    role: str
    # per-turn working state (reset by the first node every turn)
    question: str
    standalone_question: str
    route: str
    route_reasoning: str
    blocked: bool
    block_reason: str
    rag: dict
    sql: dict
    tools: dict
    draft_answer: str
    final_answer: str
    critic: dict
    retries: int
    redactions: list
    trace: Annotated[list[dict], merge_trace]
