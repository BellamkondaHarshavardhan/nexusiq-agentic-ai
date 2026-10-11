"""NexusIQ as a Model Context Protocol (MCP) server.

Any MCP-compatible client (Claude Desktop, VS Code / GitHub Copilot agent mode, custom agents, ...) can plug into
the company knowledge base, the read-only SQL database and the full multi-agent copilot through these tools.

Run it (stdio transport):   python -m nexusiq.mcp_server
Try it with the demo client: python scripts/mcp_client_demo.py
"""
from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache

try:                                              # mcp >= 2.0 renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                               # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .config import get_settings
from .db import Database, validate_sql
from .guardrails import get_policy
from .rag import KnowledgeBase
from .tools import safe_eval

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)   # stdout is reserved for the MCP protocol

mcp = _Server("nexusiq")


@lru_cache(maxsize=1)
def _kb() -> KnowledgeBase:
    kb = KnowledgeBase(get_settings())
    kb.ensure_ready()
    return kb


@lru_cache(maxsize=1)
def _db() -> Database:
    db = Database(get_settings().db_path)
    db.ensure_ready()
    return db


@lru_cache(maxsize=1)
def _copilot():
    from .graph import NexusIQ

    return NexusIQ(kb=_kb(), db=_db())


# ----------------------------------------------------------------------------- plain functions (unit-testable)
def search_knowledge_base_impl(query: str, role: str = "employee", top_k: int = 4) -> str:
    hits = _kb().search(query, role=role, k=max(1, min(top_k, 8)))
    if not hits:
        return "No relevant documents found for this access level."
    return "\n\n".join(
        f"[{i}] {h.metadata.get('section')} ({h.metadata.get('source')})\n{h.text}"
        for i, h in enumerate(hits, start=1)
    )


def get_database_schema_impl(role: str = "employee") -> str:
    return _db().describe_schema(get_policy(role).blocked_columns)


def run_sql_query_impl(sql: str, role: str = "employee") -> str:
    settings, policy = get_settings(), get_policy(role)
    check = validate_sql(sql, _db().table_names(), policy.blocked_columns, settings.sql_max_rows)
    if not check.ok:
        return f"Query rejected: {check.error}"
    try:
        columns, rows = _db().execute_readonly(check.sql, max_rows=settings.sql_max_rows)
    except Exception as exc:  # noqa: BLE001
        return f"Query failed: {exc}"
    return json.dumps({"sql": check.sql, "columns": columns, "rows": rows, "row_count": len(rows)}, default=str)


def calculate_impl(expression: str) -> str:
    try:
        return str(safe_eval(expression))
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


# ----------------------------------------------------------------------------- MCP tool registrations
@mcp.tool()
def search_knowledge_base(query: str, role: str = "employee", top_k: int = 4) -> str:
    """Search the company knowledge base (HR, IT security, expenses, product FAQ) with hybrid vector + keyword retrieval.
    `role` controls which documents are visible: 'employee' or 'hr_admin'."""
    return search_knowledge_base_impl(query, role, top_k)


@mcp.tool()
def get_database_schema(role: str = "employee") -> str:
    """Return the company database schema (tables and columns visible to the given role)."""
    return get_database_schema_impl(role)


@mcp.tool()
def run_sql_query(sql: str, role: str = "employee") -> str:
    """Run ONE read-only SELECT query on the company database. The query is validated (single statement, known tables,
    no restricted columns for the role) and executed on a read-only connection with a row limit."""
    return run_sql_query_impl(sql, role)


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. '(1250 * 12) / 100'."""
    return calculate_impl(expression)


@mcp.tool()
def ask_nexusiq(question: str, role: str = "employee") -> str:
    """Ask the full multi-agent NexusIQ copilot (supervisor + RAG + SQL + tools agents with guardrails)."""
    result = _copilot().ask(question, thread_id=_copilot().new_thread(), role=role)
    sources = sorted({s["source"] for s in result.get("sources", [])})
    footer = f"\n\n(route: {result.get('route')}; sources: {', '.join(sources) or 'n/a'})"
    return result["answer"] + footer


@mcp.resource("nexusiq://schema/{role}")
def schema_resource(role: str) -> str:
    """The database schema as a readable resource."""
    return get_database_schema_impl(role)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
