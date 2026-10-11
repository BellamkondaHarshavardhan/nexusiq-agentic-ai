"""Hybrid retrieval: dense vector search + BM25 keyword search, merged with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .text import tokenize


class BM25Index:
    """Minimal, dependency-free BM25 (Okapi) implementation."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(t) for t in texts]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = (sum(self.doc_len) / len(self.docs)) if self.docs else 0.0
        df: Counter = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tfs = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> list[float]:
        q_tokens = tokenize(query)
        out = []
        for tf, length in zip(self.tfs, self.doc_len):
            score = 0.0
            for tok in q_tokens:
                if tok not in tf:
                    continue
                num = tf[tok] * (self.k1 + 1)
                den = tf[tok] + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                score += self.idf.get(tok, 0.0) * num / den
            out.append(score)
        return out


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float                 # fused RRF score
    vector_rank: int | None = None
    keyword_rank: int | None = None
    extra: dict = field(default_factory=dict)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Merge several ranked id-lists: score(d) = sum over lists of 1 / (k + rank)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused
