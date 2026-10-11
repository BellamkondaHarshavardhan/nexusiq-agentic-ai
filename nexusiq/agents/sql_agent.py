"""SQL agent: text-to-SQL with schema grounding, static validation, read-only execution and self-correction."""
from __future__ import annotations

import sqlite3

from langchain_core.messages import HumanMessage, SystemMessage

from .. import prompts as P
from ..db import validate_sql
from ..guardrails import get_policy
from ..utils import message_text, strip_code_fences
from .context import AgentContext


def _generate(ctx: AgentContext, schema: str, question: str) -> str:
    system = P.SQL_SYSTEM.replace("{max_rows}", str(ctx.settings.sql_max_rows)).replace("{schema}", schema)
    reply = ctx.llm.invoke([SystemMessage(content=system), HumanMessage(content=f"Question: {question}")])
    return strip_code_fences(message_text(reply.content))


def _fix(ctx: AgentContext, schema: str, question: str, bad_sql: str, error: str) -> str:
    system = P.SQL_FIX_SYSTEM.replace("{schema}", schema)
    reply = ctx.llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Question: {question}\n\nFailed query:\n{bad_sql}\n\nError:\n{error}"),
    ])
    return strip_code_fences(message_text(reply.content))


def sql_agent(state: dict, ctx: AgentContext) -> dict:
    policy = get_policy(state.get("role"))
    question = state["standalone_question"]
    schema = ctx.db.describe_schema(policy.blocked_columns)
    allowed_tables = ctx.db.table_names()

    sql = _generate(ctx, schema, question)
    result: dict = {"query": sql, "columns": [], "rows": [], "row_count": 0, "error": "", "denied": False, "attempts": 1}

    for attempt in range(1, ctx.settings.sql_max_retries + 2):
        result["attempts"] = attempt
        result["query"] = sql
        if "UNANSWERABLE" in sql.upper():
            result["error"] = "UNANSWERABLE: the database schema has no data that answers this question."
            break
        check = validate_sql(sql, allowed_tables, policy.blocked_columns, ctx.settings.sql_max_rows)
        error = check.error
        if check.ok:
            result["query"] = check.sql
            try:
                columns, rows = ctx.db.execute_readonly(check.sql, max_rows=ctx.settings.sql_max_rows)
                result.update(columns=columns, rows=rows, row_count=len(rows), error="")
                break
            except sqlite3.Error as exc:
                error = f"SQLite error: {exc}"
        elif check.policy_violation:
            result.update(error=check.error, denied=True)       # policy denials are final - no "fix" attempts
            break
        result["error"] = error
        if attempt <= ctx.settings.sql_max_retries:
            sql = _fix(ctx, schema, question, sql, error)       # self-correction loop

    if result["denied"]:
        detail = f"DENIED by policy: {result['error']}"
    elif result["error"]:
        detail = f"failed after {result['attempts']} attempt(s): {result['error']}"
    else:
        detail = f"{result['row_count']} row(s) in {result['attempts']} attempt(s) | {result['query'][:140]}"
    return {"sql": result, "_detail": detail}
