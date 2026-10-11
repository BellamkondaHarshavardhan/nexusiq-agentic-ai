"""Tool-calling agent: a ReAct-style loop where the model decides which tool to call and reads the results."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .. import prompts as P
from ..tools import TOOL_MAP, TOOLS
from ..utils import message_text
from .context import AgentContext

MAX_STEPS = 4


def tools_agent(state: dict, ctx: AgentContext) -> dict:
    model = ctx.llm.bind_tools(TOOLS)
    messages = [SystemMessage(content=P.TOOLS_SYSTEM), HumanMessage(content=state["standalone_question"])]
    calls: list[dict] = []
    answer = ""
    for _ in range(MAX_STEPS):
        ai = model.invoke(messages)
        messages.append(ai)
        if not getattr(ai, "tool_calls", None):
            answer = message_text(ai.content)
            break
        for call in ai.tool_calls:
            tool = TOOL_MAP.get(call["name"])
            output = tool.invoke(call["args"]) if tool else f"Error: unknown tool {call['name']}"
            calls.append({"tool": call["name"], "args": call["args"], "output": str(output)})
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
    summary = "; ".join(f"{c['tool']}({c['args']}) -> {c['output']}" for c in calls) or "no tool needed"
    return {"tools": {"calls": calls, "answer": answer}, "_detail": summary}
