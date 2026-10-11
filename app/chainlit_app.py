"""NexusIQ - Chainlit chat UI (optional).   Run:  chainlit run app/chainlit_app.py -w
Install first:  pip install -r requirements-chainlit.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chainlit as cl
from chainlit.input_widget import Select

from nexusiq.graph import NexusIQ

engine = NexusIQ()


@cl.on_chat_start
async def start() -> None:
    settings = await cl.ChatSettings([
        Select(id="role", label="Access role", values=["employee", "hr_admin"], initial_index=0),
    ]).send()
    cl.user_session.set("role", settings["role"])
    cl.user_session.set("thread_id", engine.new_thread())
    mode = "offline demo mode" if engine.settings.is_offline else engine.settings.llm_label
    await cl.Message(content=f"👋 I'm **NexusIQ**. Ask me about company policies, company data, or both.\n\n_Running on: {mode}_").send()


@cl.on_settings_update
async def update_settings(settings: dict) -> None:
    cl.user_session.set("role", settings["role"])


@cl.on_message
async def on_message(message: cl.Message) -> None:
    role = cl.user_session.get("role") or "employee"
    thread_id = cl.user_session.get("thread_id")
    result = await cl.make_async(engine.ask)(message.content, thread_id, role)

    for entry in result.get("trace", []):                       # show each agent step like a "thinking" panel
        async with cl.Step(name=entry["node"], type="tool") as step:
            step.output = f"{entry['detail']}  ({entry['ms']} ms)"

    elements = []
    sql = result.get("sql") or {}
    if sql.get("query"):
        elements.append(cl.Text(name="SQL used", content=f"```sql\n{sql['query']}\n```", display="inline"))
    for src in result.get("sources", [])[:3]:
        elements.append(cl.Text(name=f"[{src['id']}] {src['source']}", content=src["text"], display="inline"))
    await cl.Message(content=result["answer"], elements=elements).send()
