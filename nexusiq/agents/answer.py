"""Answer layer: grounded synthesis -> self-critique (reflection loop) -> output guardrails."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .. import prompts as P
from ..guardrails import get_policy, redact_pii
from ..utils import extract_json, markdown_table, message_text
from .context import AgentContext


def build_context(state: dict) -> str:
    """Assemble every source the specialist agents produced into one labelled context block."""
    parts: list[str] = []
    route = state.get("route", "")
    rag = state.get("rag") or {}
    if rag.get("context"):
        parts.append("KNOWLEDGE CONTEXT:\n" + rag["context"])
    elif route in ("knowledge", "hybrid"):
        parts.append("KNOWLEDGE CONTEXT:\n(no relevant documents were found for this user's access level)")

    sql = state.get("sql") or {}
    if sql:
        if sql.get("denied"):
            parts.append(f"POLICY NOTE:\n{sql['error']} (the user's role is '{state.get('role')}').")
        elif sql.get("error"):
            parts.append(f"DATABASE RESULT:\nThe query could not be answered. {sql['error']}")
        else:
            parts.append(
                f"DATABASE RESULT:\nSQL: {sql['query']}\nRows ({sql['row_count']}):\n"
                + markdown_table(sql["columns"], sql["rows"], max_rows=20)
            )

    tools = state.get("tools") or {}
    if tools:
        calls = "\n".join(f"- {c['tool']}({c['args']}) = {c['output']}" for c in tools.get("calls", []))
        parts.append(f"TOOL RESULT:\n{calls or tools.get('answer', '')}")

    if route == "chat":
        parts.append("ROUTE: chat\n(The user is making small talk. Reply warmly and briefly mention what you can do.)")
    return "\n\n".join(parts)


def synthesize(state: dict, ctx: AgentContext) -> dict:
    user = f"QUESTION: {state['standalone_question']}\n\n{build_context(state)}"
    critic = state.get("critic") or {}
    if critic.get("retry") and critic.get("issues"):
        user += f"\n\nREVISION NOTE:\nYour previous draft was rejected by a fact-checker: {critic['issues']}\nWrite a corrected answer."
    reply = ctx.llm.invoke([SystemMessage(content=P.SYNTH_SYSTEM), HumanMessage(content=user)])
    draft = message_text(reply.content).strip()
    return {"draft_answer": draft, "_detail": f"drafted answer ({len(draft)} chars)"}


def critic(state: dict, ctx: AgentContext) -> dict:
    """Reflection: a second LLM pass checks the draft against the retrieved context and can force one rewrite."""
    sql = state.get("sql") or {}
    rag = state.get("rag") or {}
    verifiable = state.get("route") in ("knowledge", "database", "hybrid") and not sql.get("denied") and (
        bool(rag.get("context")) or (bool(sql) and not sql.get("error"))
    )
    if not verifiable:
        return {"critic": {"skipped": True, "retry": False, "grounded": True, "issues": ""},
                "_detail": "skipped - nothing retrieved to verify against"}

    context = build_context(state)[:6000]
    reply = ctx.llm.invoke([
        SystemMessage(content=P.CRITIC_SYSTEM),
        HumanMessage(content=f"QUESTION: {state['standalone_question']}\n\nCONTEXT:\n{context}\n\nDRAFT ANSWER:\n{state['draft_answer']}"),
    ])
    data = extract_json(message_text(reply.content)) or {}
    ok = bool(data.get("grounded", True)) and bool(data.get("complete", True))
    issues = str(data.get("issues", "")).strip()
    retries = state.get("retries", 0)
    retry = (not ok) and retries < ctx.settings.critic_max_retries
    return {
        "critic": {"skipped": False, "grounded": ok, "issues": issues, "retry": retry},
        "retries": retries + 1 if retry else retries,
        "_detail": "answer verified as grounded" if ok else f"rejected ({issues or 'unsupported claims'}) - {'rewriting' if retry else 'retry limit reached'}",
    }


def after_critic(state: dict) -> str:
    return "synthesize" if (state.get("critic") or {}).get("retry") else "finalize"


def finalize(state: dict, ctx: AgentContext) -> dict:
    """Output guardrail: redact PII according to the caller's role, then commit the reply to memory."""
    policy = get_policy(state.get("role"))
    clean, kinds = redact_pii(state.get("draft_answer", ""), policy)
    detail = f"redacted: {', '.join(kinds)}" if kinds else "no sensitive data detected"
    return {"final_answer": clean, "redactions": kinds, "messages": [AIMessage(content=clean)], "_detail": detail}
