"""Demo MCP client: starts the NexusIQ MCP server over stdio, lists its tools and calls a few of them.

    python scripts/mcp_client_demo.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


def _text(result) -> str:
    return "\n".join(getattr(block, "text", str(block)) for block in result.content)


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "nexusiq.mcp_server"], cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("MCP tools exposed by NexusIQ:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {(tool.description or '').splitlines()[0]}")

            print("\n1) search_knowledge_base('notice period')")
            print(_text(await session.call_tool("search_knowledge_base", {"query": "notice period", "top_k": 2}))[:500])

            print("\n2) run_sql_query (employee role asks for salaries -> must be rejected)")
            print(_text(await session.call_tool("run_sql_query", {"sql": "SELECT name, salary FROM employees"})))

            print("\n3) run_sql_query (allowed)")
            print(_text(await session.call_tool("run_sql_query", {"sql": "SELECT COUNT(*) AS employees FROM employees"})))

            print("\n4) ask_nexusiq (full multi-agent graph)")
            print(_text(await session.call_tool("ask_nexusiq", {"question": "Top 3 customers by revenue"})))


if __name__ == "__main__":
    asyncio.run(main())
