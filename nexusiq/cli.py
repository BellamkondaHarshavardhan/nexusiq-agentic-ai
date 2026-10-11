"""Command line interface.

    python -m nexusiq.cli init                      # create the demo database and index the documents
    python -m nexusiq.cli ask "How many leaves do I get?" --role employee
    python -m nexusiq.cli chat --role hr_admin      # interactive multi-turn chat
    python -m nexusiq.cli info                      # show which providers are active
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import get_settings
from .db import Database
from .graph import NexusIQ
from .rag import KnowledgeBase


def _print_result(result: dict, verbose: bool) -> None:
    print("\n" + result["answer"] + "\n")
    if verbose:
        print(f"route : {result.get('route')}  ({result.get('route_reasoning', '')})")
        if result.get("sql", {}).get("query"):
            print(f"sql   : {result['sql']['query']}")
        for s in result.get("sources", []):
            print(f"source: [{s['id']}] {s['source']} > {s['section']}")
        print("trace : " + " -> ".join(f"{t['node']}({t['ms']}ms)" for t in result.get("trace", [])))
        print(f"total : {result.get('latency_ms')} ms\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nexusiq", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="seed the database and index the knowledge base")
    sub.add_parser("info", help="show active configuration")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--role", default="employee", choices=["employee", "hr_admin"])
    ask.add_argument("-v", "--verbose", action="store_true")
    chat = sub.add_parser("chat", help="interactive chat with memory")
    chat.add_argument("--role", default="employee", choices=["employee", "hr_admin"])
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.cmd == "info":
        print(json.dumps({"llm": settings.llm_label, "embeddings": settings.embedding_label}, indent=2))
        return 0
    if args.cmd == "init":
        Database(settings.db_path).seed()
        kb = KnowledgeBase(settings)
        kb.reset()
        chunks = kb.ingest_directory(settings.kb_dir)
        print(f"Database seeded at {settings.db_path}\nIndexed {chunks} chunks into ChromaDB ({settings.embedding_label})")
        return 0

    app = NexusIQ(settings)
    if settings.is_offline:
        print("[offline demo mode - add an Azure OpenAI / OpenAI key to .env for full LLM reasoning]", file=sys.stderr)
    if args.cmd == "ask":
        _print_result(app.ask(args.question, role=args.role), args.verbose)
        return 0
    thread = app.new_thread()
    print(f"NexusIQ chat as '{args.role}'. Type 'exit' to quit.")
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", ""}:
            break
        _print_result(app.ask(question, thread_id=thread, role=args.role), verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
