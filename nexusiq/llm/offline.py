"""Offline demo chat model - a deterministic, rule-based stand-in for a real LLM.

Why it exists
  * the whole multi-agent pipeline runs (and is unit-tested in CI) with NO API key and NO cost
  * anyone who clones the repo can see the system work in 30 seconds
It reads the [TASK:...] tag in the system prompt and answers with simple keyword rules and SQL templates.
It is intentionally limited - for real reasoning configure Azure OpenAI / OpenAI in `.env`.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .. import prompts as P
from ..rag.text import tokenize
from ..utils import markdown_table, message_text

_GREETING = re.compile(r"^\s*(hi|hello|hey|hola|namaste|good (morning|afternoon|evening)|thanks|thank you|who are you|what can you do|help)\b", re.I)
_MATH = re.compile(r"(\d+(\.\d+)?\s*[-+*/x×^]\s*\d+)|(\d+(\.\d+)?\s*%\s*of\s*\d+)|\bsqrt\b|\bcalculate\b|\bcompute\b", re.I)
_TIME = re.compile(
    r"what(?:'s| is) (?:the )?(?:current |today'?s )?(?:date|time|day)\b|what time is it|today'?s date|current (?:date|time)", re.I)
_DB_INTENT = ("how many", "number of", "count", "total", "average", "avg", "top ", "highest", "lowest", "most ", "least ",
              "revenue", "sales", "per ", " by ", "list ", "show me", "which customer", "which product", "recent",
              "latest", "newest", "oldest", "trend", "monthly", "growth", "best selling")
_DB_ENTITY = ("employee", "customer", "order", "product", "department", "revenue", "sales", "headcount", "hired", "salary")
_KB_WORDS = ("policy", "policies", "leave", "vacation", "holiday", "remote", "work from", "wfh", "password", "mfa", "vpn",
             "security", "expense", "reimburs", "onboarding", "benefit", "insurance", "notice period", "probation", "sla",
             "support plan", "trial", "faq", "data residency", "uptime", "pricing", "how do i", "how to", "allowance",
             "guideline", "procedure", "bonus", "salary band", "review cycle", "esop", "travel", "hotel", "provident")
_COUNTRIES = {"india": "India", "usa": "USA", "uk": "UK", "germany": "Germany", "uae": "UAE", "singapore": "Singapore", "australia": "Australia"}
_DEPTS = {"engineering": "Engineering", "sales": "Sales", "marketing": "Marketing", "human resources": "Human Resources",
          "hr": "Human Resources", "finance": "Finance", "customer success": "Customer Success", "product": "Product"}


def _classify(q: str) -> tuple[str, str]:
    ql = q.lower()
    if _TIME.search(ql) or (_MATH.search(ql) and not any(w in ql for w in ("revenue", "customers", "orders", "employees"))):
        return "tools", "Needs an exact calculation or the current date/time."
    if _GREETING.match(ql) and len(ql.split()) <= 8:
        return "chat", "Greeting or small talk."
    db = any(w in ql for w in _DB_INTENT) and any(w in ql for w in _DB_ENTITY)
    kb = any(w in ql for w in _KB_WORDS)
    if db and kb:
        return "hybrid", "Needs both policy documents and numbers from the database."
    if db:
        return "database", "Asks for figures that live in the company database."
    return "knowledge", "Answerable from the company documents."


def _where_word(ql: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", ql) is not None


def _sql_for(question: str) -> str:
    """Template-based text-to-SQL for the demo database (a real LLM replaces this)."""
    ql = question.lower()
    n_match = re.search(r"\btop\s+(\d+)", ql)
    n = int(n_match.group(1)) if n_match else 5
    year = re.search(r"\b(2025|2026)\b", ql)
    country = next((v for k, v in _COUNTRIES.items() if _where_word(ql, k)), None)
    dept = next((v for k, v in sorted(_DEPTS.items(), key=lambda kv: -len(kv[0])) if _where_word(ql, k)), None)
    status = next((s for s in ("completed", "pending", "cancelled", "refunded") if s in ql), None)
    grouped = any(w in ql for w in ("each", "per ", " by ", "every", "breakdown", "split"))

    if "salary" in ql:
        if any(w in ql for w in ("highest", "top", "max")):
            return f"SELECT name, job_title, salary FROM employees ORDER BY salary DESC LIMIT {n}"
        where = f" WHERE d.name = '{dept}'" if dept else ""
        return ("SELECT d.name AS department, ROUND(AVG(e.salary)) AS avg_salary FROM employees e "
                f"JOIN departments d ON d.id = e.department_id{where} GROUP BY d.name ORDER BY avg_salary DESC")

    if "revenue" in ql or "sales" in ql:
        base = ("FROM orders o JOIN customers c ON c.id = o.customer_id JOIN order_items oi ON oi.order_id = o.id "
                "JOIN products p ON p.id = oi.product_id WHERE o.status = 'completed'")
        if year:
            base += f" AND strftime('%Y', o.order_date) = '{year.group(1)}'"
        if country:
            base += f" AND c.country = '{country}'"
        rev = "ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue_usd"
        if "month" in ql:
            return f"SELECT strftime('%Y-%m', o.order_date) AS month, {rev} {base} GROUP BY month ORDER BY month"
        if "categor" in ql:
            return f"SELECT p.category, {rev} {base} GROUP BY p.category ORDER BY revenue_usd DESC"
        if "product" in ql:
            return f"SELECT p.name AS product, {rev} {base} GROUP BY p.name ORDER BY revenue_usd DESC LIMIT {n}"
        if "customer" in ql:
            return f"SELECT c.name AS customer, {rev} {base} GROUP BY c.name ORDER BY revenue_usd DESC LIMIT {n}"
        if "countr" in ql and not country:
            return f"SELECT c.country, {rev} {base} GROUP BY c.country ORDER BY revenue_usd DESC"
        return f"SELECT {rev} {base}"

    if any(w in ql for w in ("employee", "headcount", "hired", "staff", "people")):
        if any(w in ql for w in ("recent", "latest", "newest", "hired")):
            return f"SELECT name, job_title, hire_date FROM employees ORDER BY hire_date DESC LIMIT {n}"
        if dept and not grouped:
            return ("SELECT COUNT(*) AS employee_count FROM employees e JOIN departments d ON d.id = e.department_id "
                    f"WHERE d.name = '{dept}'")
        if grouped or "department" in ql:
            return ("SELECT d.name AS department, COUNT(*) AS employee_count FROM employees e "
                    "JOIN departments d ON d.id = e.department_id GROUP BY d.name ORDER BY employee_count DESC")
        return "SELECT COUNT(*) AS employee_count FROM employees"

    if "customer" in ql:
        if "industr" in ql:
            return "SELECT industry, COUNT(*) AS customers FROM customers GROUP BY industry ORDER BY customers DESC"
        if country and any(w in ql for w in ("which", "list", "show", "who", "names")):
            return f"SELECT name, industry FROM customers WHERE country = '{country}' ORDER BY name LIMIT 50"
        if country and not grouped:
            return f"SELECT COUNT(*) AS customers FROM customers WHERE country = '{country}'"
        if grouped or "countr" in ql:
            return "SELECT country, COUNT(*) AS customers FROM customers GROUP BY country ORDER BY customers DESC"
        return "SELECT COUNT(*) AS customers FROM customers"

    if "order" in ql:
        where = []
        if status:
            where.append(f"status = '{status}'")
        if year:
            where.append(f"strftime('%Y', order_date) = '{year.group(1)}'")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        if "month" in ql:
            return f"SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS orders FROM orders{clause} GROUP BY month ORDER BY month"
        if (grouped or "status" in ql) and not status:
            return f"SELECT status, COUNT(*) AS orders FROM orders{clause} GROUP BY status ORDER BY orders DESC"
        return f"SELECT COUNT(*) AS orders FROM orders{clause}"

    if "product" in ql:
        if any(w in ql for w in ("best", "top", "selling", "popular", "most")):
            return ("SELECT p.name AS product, SUM(oi.quantity) AS units_sold FROM order_items oi "
                    "JOIN products p ON p.id = oi.product_id JOIN orders o ON o.id = oi.order_id "
                    f"WHERE o.status = 'completed' GROUP BY p.name ORDER BY units_sold DESC LIMIT {n}")
        return "SELECT name, category, unit_price FROM products ORDER BY category, name"

    return "SELECT 'UNANSWERABLE' AS status"


def _section(text: str, header: str) -> str:
    match = re.search(rf"^{re.escape(header)}:\s*\n?(.*?)(?=^\s*(?:QUESTION|KNOWLEDGE CONTEXT|DATABASE RESULT|TOOL RESULT|POLICY NOTE|REVISION NOTE):|\Z)",
                      text, flags=re.S | re.M)
    return match.group(1).strip() if match else ""


def _best_sentences(question: str, context: str, limit: int = 3) -> list[str]:
    """Extractive answer: pick the context sentences that overlap most with the question (demo mode only)."""
    q_tokens = set(tokenize(question))
    min_overlap = 1 if len(q_tokens) <= 1 else 2
    scored = []
    for block in re.split(r"\n(?=\[\d+\])", context):
        m = re.match(r"\[(\d+)\]\s*\(source:[^,)]*(?:,\s*section:\s*([^)]*))?\)\s*(.*)", block, flags=re.S)
        if not m:
            continue
        num, section, body = m.group(1), m.group(2) or "", m.group(3)
        section_tokens = set(tokenize(section))
        for raw in re.split(r"\n+|(?<!Rs\.)(?<=[.!?])\s+", body):
            if raw.lstrip().startswith("#"):          # headings are labels, not answers
                continue
            sent = re.sub(r"^[\-\*\s]+", "", raw).replace("**", "").strip()
            if len(sent) < 25 or sent.startswith("This document is restricted"):
                continue
            overlap = len(q_tokens & (set(tokenize(sent)) | section_tokens))
            if overlap >= min_overlap:
                scored.append((overlap, sent, num))
    scored.sort(key=lambda t: -t[0])
    chosen, seen = [], set()
    for _, sent, num in scored:
        if sent not in seen:
            seen.add(sent)
            chosen.append(f"{sent} [{num}]")
        if len(chosen) == limit:
            break
    return chosen


class OfflineChatModel(BaseChatModel):
    """Rule-based model that satisfies LangChain's chat-model interface (including tool calling)."""

    @property
    def _llm_type(self) -> str:
        return "offline-demo"

    def bind_tools(self, tools: Any, **kwargs: Any):  # tool schemas are irrelevant to the rule engine
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])

    # ------------------------------------------------------------------
    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        system = next((message_text(m.content) for m in messages if m.type == "system"), "")
        human = next((message_text(m.content) for m in reversed(messages) if m.type == "human"), "")

        if P.TASK_ROUTE in system:
            m = re.search(r"Question:\s*(.*)\Z", human, flags=re.S)
            route, why = _classify((m.group(1) if m else human).strip())
            return AIMessage(content=json.dumps({"route": route, "reasoning": why}))
        if P.TASK_REWRITE in system:
            m = re.search(r"Follow-up question:\s*(.*)\Z", human, flags=re.S)
            return AIMessage(content=(m.group(1) if m else human).strip())
        if P.TASK_SQL in system:
            m = re.search(r"Question:\s*(.*)\Z", human, flags=re.S)
            return AIMessage(content=_sql_for((m.group(1) if m else human).strip()))
        if P.TASK_SQL_FIX in system:
            return AIMessage(content="SELECT 'UNANSWERABLE' AS status")
        if P.TASK_CRITIC in system:
            return AIMessage(content=json.dumps({"grounded": True, "complete": True, "issues": ""}))
        if P.TASK_TOOLS in system:
            return self._tools_turn(messages, human)
        if P.TASK_SYNTH in system:
            return AIMessage(content=self._synthesize(human))
        return AIMessage(content="Offline demo model: no task recognised.")

    # ------------------------------------------------------------------
    def _tools_turn(self, messages: list[BaseMessage], question: str) -> AIMessage:
        tool_msgs = [m for m in messages if m.type == "tool"]
        if tool_msgs:
            return AIMessage(content="Result: " + "; ".join(message_text(m.content) for m in tool_msgs))
        ql = question.lower()
        if _TIME.search(ql):
            return AIMessage(content="", tool_calls=[{"name": "get_current_datetime", "args": {}, "id": "call_offline_1", "type": "tool_call"}])
        expr = None
        pct = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", ql)
        sqrt = re.search(r"sqrt\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?", ql)
        if pct:
            expr = f"({pct.group(2)} * {pct.group(1)}) / 100"
        elif sqrt:
            expr = f"sqrt({sqrt.group(1)})"
        else:
            m = re.search(r"[\d(][\d.\s()+\-*/x×^]*[\d)]", ql)
            if m:
                expr = m.group(0).replace("x", "*").replace("×", "*").strip()
        if expr:
            return AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": expr}, "id": "call_offline_1", "type": "tool_call"}])
        return AIMessage(content="I can use my tools for calculations and for the current date and time.")

    def _synthesize(self, user_text: str) -> str:
        question = _section(user_text, "QUESTION")
        knowledge = _section(user_text, "KNOWLEDGE CONTEXT")
        database = _section(user_text, "DATABASE RESULT")
        tool = _section(user_text, "TOOL RESULT")
        policy = _section(user_text, "POLICY NOTE")

        if policy:
            return f"I'm sorry, I can't share that. {policy}"
        parts: list[str] = []
        if knowledge:
            sentences = _best_sentences(question, knowledge)
            if sentences:
                parts.append("**From the company documents:**\n" + "\n".join(f"- {s}" for s in sentences))
        if database:
            if "UNANSWERABLE" in database or "could not be answered" in database:
                parts.append("The company database does not contain data that can answer this question.")
            else:
                table = [ln for ln in database.splitlines() if ln.startswith("|") or ln.startswith("...")]
                body = [ln for ln in table if not ln.startswith("|---")]
                if len(body) == 2 and body[0].count("|") == 2:      # one column, one row -> a single figure
                    label, value = body[0].strip("| ").strip(), body[1].strip("| ").strip()
                    parts.append(f"**From the company database:** {label.replace('_', ' ')} = **{value}**.")
                else:
                    parts.append("**From the company database:**\n" + "\n".join(table))
        if tool:
            parts.append(tool)
        if not parts:
            if "ROUTE: chat" in user_text:
                return P.CHAT_CAPABILITIES
            return "I couldn't find that in the company knowledge base or data. Please check with the relevant team."
        return "\n\n".join(parts)
