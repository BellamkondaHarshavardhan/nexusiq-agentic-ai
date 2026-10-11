"""Tools the agents can call (LangChain tool-calling)."""
from __future__ import annotations

import ast
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max, "log": math.log, "log10": math.log10}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float | int:
    """Evaluate an arithmetic expression WITHOUT eval(): only numbers, + - * / // % ** and a few math functions."""
    if len(expression) > 200:
        raise ValueError("Expression too long.")
    tree = ast.parse(expression.strip().replace("^", "**"), mode="eval")

    def ev(node: ast.AST):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent too large.")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("Unsupported expression.")

    return ev(tree)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. '(1250 * 12) / 100' or 'sqrt(144)'. Use for ANY maths."""
    try:
        result = safe_eval(expression)
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:  # noqa: BLE001 - tool errors must be returned to the model as text
        return f"Error: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(round(result, 6) if isinstance(result, float) else result)


@tool
def get_current_datetime(timezone: str = "Asia/Kolkata") -> str:
    """Get the current date and time in an IANA timezone (default Asia/Kolkata)."""
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:  # noqa: BLE001
        now = datetime.now(ZoneInfo("UTC"))
        timezone = "UTC"
    return now.strftime(f"%A, %d %B %Y, %H:%M:%S ({timezone})")


TOOLS = [calculator, get_current_datetime]
TOOL_MAP = {t.name: t for t in TOOLS}
