"""The LangGraph workflow + the high-level NexusIQ facade used by the UI, CLI, MCP server and evaluation.

    guard_input -> rewrite -> router(supervisor) -+-> rag_agent  --+
                                                  +-> sql_agent  --+--> synthesize -> critic -> finalize
                                                  +-> tools_agent -+        ^           |
                                                  +-> (chat) ---------------+-----------+  (reflection loop)
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Iterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import answer, supervisor
from .agents.context import AgentContext
from .agents.rag_agent import rag_agent
from .agents.sql_agent import sql_agent
from .agents.tools_agent import tools_agent
from .config import Settings, get_settings
from .db import Database
from .llm import get_chat_model
from .observability import setup_tracing
from .rag import KnowledgeBase
from .state import AgentState

log = logging.getLogger("nexusiq")


def _node(name: str, fn, ctx: AgentContext):
    """Wrap an agent function: inject dependencies, time it, and record a human-readable trace entry."""

    def run(state: dict) -> dict:
        started = time.perf_counter()
        update = dict(fn(state, ctx) or {})
        detail = update.pop("_detail", "")
        entry = {"node": name, "ms": round((time.perf_counter() - started) * 1000), "detail": detail}
        update["trace"] = list(update.get("trace", [])) + [entry]
        return update

    run.__name__ = name
    return run


def build_graph(ctx: AgentContext, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("guard_input", _node("guard_input", supervisor.guard_input, ctx))
    g.add_node("rewrite", _node("rewrite", supervisor.rewrite, ctx))
    g.add_node("router", _node("router", supervisor.router, ctx))
    g.add_node("rag_agent", _node("rag_agent", rag_agent, ctx))
    g.add_node("sql_agent", _node("sql_agent", sql_agent, ctx))
    g.add_node("tools_agent", _node("tools_agent", tools_agent, ctx))
    g.add_node("synthesize", _node("synthesize", answer.synthesize, ctx))
    g.add_node("critic", _node("critic", answer.critic, ctx))
    g.add_node("finalize", _node("finalize", answer.finalize, ctx))

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", supervisor.after_guard, ["rewrite", "finalize"])
    g.add_edge("rewrite", "router")
    g.add_conditional_edges("router", supervisor.dispatch, ["rag_agent", "sql_agent", "tools_agent", "synthesize"])
    for agent in ("rag_agent", "sql_agent", "tools_agent"):
        g.add_edge(agent, "synthesize")
    g.add_edge("synthesize", "critic")
    g.add_conditional_edges("critic", answer.after_critic, ["synthesize", "finalize"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


class NexusIQ:
    """Facade: build once, then call `ask()` (blocking) or `stream()` (live agent steps)."""

    def __init__(self, settings: Settings | None = None, llm=None, kb: KnowledgeBase | None = None,
                 db: Database | None = None, checkpointer=None):
        self.settings = settings or get_settings()
        self.tracing = setup_tracing()
        self.llm = llm or get_chat_model(self.settings)
        self.db = db or Database(self.settings.db_path)
        self.db.ensure_ready()
        self.kb = kb or KnowledgeBase(self.settings)
        self.kb.ensure_ready()
        self.ctx = AgentContext(settings=self.settings, llm=self.llm, kb=self.kb, db=self.db)
        self.graph = build_graph(self.ctx, checkpointer)

    # ------------------------------------------------------------------
    @staticmethod
    def new_thread() -> str:
        return uuid.uuid4().hex[:12]

    def _config(self, thread_id: str, role: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "run_name": "NexusIQ",
            "tags": ["nexusiq", f"role:{role}"],
            "metadata": {"role": role, "llm": self.settings.llm_provider},
        }

    def stream(self, question: str, thread_id: str = "default", role: str = "employee") -> Iterator[dict]:
        """Yield {'type': 'step', node, ms, detail} for every agent step, then {'type': 'final', 'result': {...}}."""
        config = self._config(thread_id, role)
        payload = {"messages": [HumanMessage(content=question)], "role": role}
        started = time.perf_counter()
        try:
            for chunk in self.graph.stream(payload, config, stream_mode="updates"):
                for _node_name, update in chunk.items():
                    for entry in (update or {}).get("trace", []):
                        if "node" in entry:
                            yield {"type": "step", **entry}
            state = self.graph.get_state(config).values
        except Exception as exc:  # noqa: BLE001 - surface a friendly error instead of crashing the UI
            log.exception("NexusIQ run failed")
            yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
            return
        result = self._result(state)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000)
        yield {"type": "final", "result": result}

    def ask(self, question: str, thread_id: str = "default", role: str = "employee") -> dict:
        for event in self.stream(question, thread_id, role):
            if event["type"] == "final":
                return event["result"]
            if event["type"] == "error":
                return {"answer": "Sorry, something went wrong while processing your request. "
                                  f"Please check your API configuration. ({event['error']})",
                        "error": event["error"], "route": "", "sources": [], "sql": {}, "tools": {}, "trace": [],
                        "critic": {}, "redactions": [], "blocked": False, "role": role, "question": question}
        raise RuntimeError("graph produced no result")

    @staticmethod
    def _result(state: dict[str, Any]) -> dict:
        return {
            "answer": state.get("final_answer", ""),
            "question": state.get("question", ""),
            "standalone_question": state.get("standalone_question", ""),
            "route": state.get("route", ""),
            "route_reasoning": state.get("route_reasoning", ""),
            "blocked": state.get("blocked", False),
            "block_reason": state.get("block_reason", ""),
            "sources": (state.get("rag") or {}).get("hits", []),
            "sql": state.get("sql") or {},
            "tools": state.get("tools") or {},
            "critic": state.get("critic") or {},
            "redactions": state.get("redactions", []),
            "trace": state.get("trace", []),
            "role": state.get("role", "employee"),
        }
