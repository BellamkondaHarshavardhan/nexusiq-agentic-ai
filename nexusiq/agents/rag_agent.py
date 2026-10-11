"""RAG agent: access-controlled hybrid retrieval from the ChromaDB knowledge base."""
from __future__ import annotations

from .context import AgentContext


def rag_agent(state: dict, ctx: AgentContext) -> dict:
    hits = ctx.kb.search(state["standalone_question"], role=state["role"], k=ctx.settings.top_k)
    sources, blocks = [], []
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        sources.append({
            "id": i, "source": meta.get("source", ""), "title": meta.get("title", ""),
            "section": meta.get("section", ""), "access": meta.get("access", "public"),
            "score": hit.score, "vector_rank": hit.vector_rank, "keyword_rank": hit.keyword_rank, "text": hit.text,
        })
        blocks.append(f"[{i}] (source: {meta.get('source', '')}, section: {meta.get('section', '')})\n{hit.text}")
    names = ", ".join(sorted({s["source"] for s in sources})) or "nothing relevant"
    return {"rag": {"hits": sources, "context": "\n\n".join(blocks)},
            "_detail": f"retrieved {len(sources)} chunks (hybrid vector+BM25, role={state['role']}) from {names}"}
