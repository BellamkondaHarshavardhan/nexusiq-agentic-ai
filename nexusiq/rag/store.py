"""ChromaDB-backed knowledge base with ingestion, access-controlled hybrid search and stats."""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma

from ..config import Settings
from ..guardrails import get_policy
from .embeddings import get_embeddings
from .loaders import chunk_document, load_document
from .retriever import BM25Index, Hit, reciprocal_rank_fusion

COLLECTION = "nexusiq_kb"


class KnowledgeBase:
    def __init__(self, settings: Settings, embeddings=None):
        self.settings = settings
        self.embeddings = embeddings or get_embeddings(settings)
        # one folder per embedding provider: vectors from different models are NOT interchangeable
        self.persist_dir = settings.vector_dir / settings.embedding_provider
        self._store: Chroma | None = None
        self._bm25: tuple[int, BM25Index, list[str], list[dict], list[str]] | None = None

    # ------------------------------------------------------------------ plumbing
    @property
    def store(self) -> Chroma:
        if self._store is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._store = Chroma(
                collection_name=COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir),
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._store

    def count(self) -> int:
        return self.store._collection.count()

    def ensure_ready(self) -> None:
        """Auto-index the bundled sample documents the first time the app runs."""
        if self.count() == 0 and self.settings.kb_dir.exists():
            self.ingest_directory(self.settings.kb_dir)

    # ------------------------------------------------------------------ ingestion
    def ingest_bytes(self, name: str, data: bytes, access: str = "public") -> int:
        text, meta = load_document(name, data, default_access=access)
        chunks = chunk_document(text, meta)
        if not chunks:
            return 0
        # deterministic ids -> re-ingesting the same file updates it instead of duplicating it
        ids = [f"{meta['source']}::{c.metadata['chunk']}" for c in chunks]
        self.delete_source(meta["source"])
        self.store.add_documents(chunks, ids=ids)
        self._bm25 = None
        return len(chunks)

    def ingest_directory(self, directory: Path) -> int:
        total = 0
        for path in sorted(Path(directory).glob("*")):
            if path.suffix.lower() in {".md", ".txt", ".pdf"}:
                total += self.ingest_bytes(path.name, path.read_bytes())
        return total

    def delete_source(self, source: str) -> None:
        self.store._collection.delete(where={"source": source})
        self._bm25 = None

    def reset(self) -> None:
        self.store.delete_collection()
        self._store = None
        self._bm25 = None

    # ------------------------------------------------------------------ stats
    def sources(self) -> list[dict]:
        data = self.store.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in data["metadatas"]:
            entry = seen.setdefault(meta["source"], {"source": meta["source"], "title": meta.get("title", ""),
                                                      "access": meta.get("access", "public"), "chunks": 0})
            entry["chunks"] += 1
        return sorted(seen.values(), key=lambda e: e["source"])

    # ------------------------------------------------------------------ search
    def _keyword_index(self):
        n = self.count()
        if self._bm25 is None or self._bm25[0] != n:
            data = self.store.get(include=["documents", "metadatas"])
            self._bm25 = (n, BM25Index(data["documents"]), data["ids"], data["metadatas"], data["documents"])
        return self._bm25

    def search(self, query: str, role: str = "employee", k: int | None = None, fetch_k: int = 8) -> list[Hit]:
        """Hybrid search restricted to documents the role may read (access control happens BEFORE ranking)."""
        self.ensure_ready()
        k = k or self.settings.top_k
        allowed = set(get_policy(role).doc_access)

        # 1) dense retrieval (metadata filter pushed down into Chroma)
        vector_hits = self.store.similarity_search_with_score(
            query, k=fetch_k, filter={"access": {"$in": sorted(allowed)}}
        )
        vec_ids, by_id = [], {}
        for doc, distance in vector_hits:
            cid = f"{doc.metadata['source']}::{doc.metadata['chunk']}"
            vec_ids.append(cid)
            by_id[cid] = Hit(cid, doc.page_content, dict(doc.metadata), 0.0, extra={"distance": round(float(distance), 4)})

        # 2) keyword retrieval (BM25) over the same permitted chunks
        _, index, ids, metas, docs = self._keyword_index()
        scores = index.scores(query)
        ranked = sorted(
            (i for i in range(len(ids)) if metas[i].get("access", "public") in allowed and scores[i] > 0),
            key=lambda i: scores[i], reverse=True,
        )[:fetch_k]
        kw_ids = []
        for i in ranked:
            kw_ids.append(ids[i])
            by_id.setdefault(ids[i], Hit(ids[i], docs[i], dict(metas[i]), 0.0))

        # 3) fuse the two rankings
        fused = reciprocal_rank_fusion([vec_ids, kw_ids])
        hits = []
        for cid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]:
            hit = by_id[cid]
            hit.score = round(score, 5)
            hit.vector_rank = vec_ids.index(cid) + 1 if cid in vec_ids else None
            hit.keyword_rank = kw_ids.index(cid) + 1 if cid in kw_ids else None
            hits.append(hit)
        return hits
