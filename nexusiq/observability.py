"""LangSmith tracing switch. LangChain/LangGraph emit traces automatically once these env vars are set."""
from __future__ import annotations

import os


def tracing_enabled() -> bool:
    flag = (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "").strip().lower()
    key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    return flag in {"1", "true", "yes"} and bool(key)


def setup_tracing() -> bool:
    """Returns True when every LLM call, tool call and graph node will be traced in LangSmith."""
    if tracing_enabled():
        os.environ.setdefault("LANGSMITH_PROJECT", "nexusiq")
        return True
    return False
