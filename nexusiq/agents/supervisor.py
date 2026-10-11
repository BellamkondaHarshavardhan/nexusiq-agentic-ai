"""Supervisor layer: input guardrails -> follow-up rewriting -> routing to the right specialist agent(s)."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .. import prompts as P
from ..guardrails import check_input, get_policy
from ..state import RESET, ROUTES
from ..utils import extract_json, last_human_text, message_text
from .context import AgentContext


def guard_input(state: dict, ctx: AgentContext) -> dict:
    question = last_human_text(state["messages"])
    verdict = check_input(question)
    update = {
        "question": question,
        "standalone_question": question,
        "role": get_policy(state.get("role")).role,
        "route": "", "route_reasoning": "",
        "blocked": not verdict.allowed, "block_reason": verdict.reason,
        "rag": {}, "sql": {}, "tools": {}, "critic": {}, "retries": 0, "redactions": [],
        "draft_answer": "", "final_answer": "",
        "trace": [RESET],
    }
    if verdict.allowed:
        update["_detail"] = "input passed safety screening"
    else:
        update["draft_answer"] = f"I can't process that request. {verdict.reason}"
        update["_detail"] = f"BLOCKED - {verdict.reason}"
    return update


def rewrite(state: dict, ctx: AgentContext) -> dict:
    """Turn a follow-up such as 'and for Marketing?' into a standalone question using chat history."""
    question = state["question"]
    history = [m for m in state["messages"][:-1] if m.type in ("human", "ai")][-6:]
    if not history:
        return {"standalone_question": question, "_detail": "first turn - no rewrite needed"}
    transcript = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {message_text(m.content)[:400]}" for m in history
    )
    reply = ctx.llm.invoke([
        SystemMessage(content=P.REWRITE_SYSTEM),
        HumanMessage(content=f"Chat history:\n{transcript}\n\nFollow-up question: {question}"),
    ])
    rewritten = message_text(reply.content).strip().splitlines()[0].strip() if message_text(reply.content).strip() else question
    return {"standalone_question": rewritten or question, "_detail": f"standalone question: {rewritten}"}


def router(state: dict, ctx: AgentContext) -> dict:
    """The supervisor decides which specialist agents should work on the question."""
    reply = ctx.llm.invoke([
        SystemMessage(content=P.ROUTER_SYSTEM),
        HumanMessage(content=f"Question: {state['standalone_question']}"),
    ])
    data = extract_json(message_text(reply.content)) or {}
    route = str(data.get("route", "")).strip().lower()
    reasoning = str(data.get("reasoning", "")).strip()
    if route not in ROUTES:                       # never trust raw LLM output - fall back safely
        route, reasoning = "hybrid", "Router output was unclear, so all knowledge sources are used."
    return {"route": route, "route_reasoning": reasoning, "_detail": f"route = {route} ({reasoning})"}


def dispatch(state: dict) -> list[str]:
    """Conditional edge: fan out to one or several agents. 'hybrid' runs the RAG and SQL agents IN PARALLEL."""
    return {
        "knowledge": ["rag_agent"],
        "database": ["sql_agent"],
        "hybrid": ["rag_agent", "sql_agent"],
        "tools": ["tools_agent"],
    }.get(state.get("route", ""), ["synthesize"])


def after_guard(state: dict) -> str:
    return "finalize" if state.get("blocked") else "rewrite"
