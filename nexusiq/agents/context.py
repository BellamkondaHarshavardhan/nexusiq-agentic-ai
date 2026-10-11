from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import Settings
from ..db import Database
from ..rag import KnowledgeBase


@dataclass
class AgentContext:
    """Everything the agents need - injected once, which keeps every node easy to unit-test."""

    settings: Settings
    llm: BaseChatModel
    kb: KnowledgeBase
    db: Database
