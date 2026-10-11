"""Small helpers shared across modules."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import BaseMessage


def message_text(content: Any) -> str:
    """LangChain message content can be a str or a list of content blocks - normalise to str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply (tolerates ```json fences and chatter)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def last_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if msg.type == "human":
            return message_text(msg.content)
    return ""


def markdown_table(columns: list[str], rows: list[list[Any]], max_rows: int = 20) -> str:
    """Render query results as a compact markdown table (used inside LLM prompts)."""
    if not columns:
        return "(no columns)"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows not shown)")
    return "\n".join(lines)
