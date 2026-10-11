"""Embedding providers: Azure OpenAI, OpenAI, or a dependency-free local hashing embedder for offline use."""
from __future__ import annotations

import math
import zlib

from langchain_core.embeddings import Embeddings

from ..config import Settings
from .text import tokenize


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words (+bigrams) embeddings using the hashing trick.

    Not semantic like a neural model, but fast, free and fully offline - perfect for demos, CI and tests.
    Production use should switch to Azure OpenAI / OpenAI embeddings through the .env file.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        counts: dict[str, int] = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        for gram, tf in counts.items():
            h = zlib.crc32(gram.encode("utf-8"))
            sign = 1.0 if (h >> 31) & 1 else -1.0
            weight = (1.0 + math.log(tf)) * (0.6 if "_" in gram else 1.0)
            vec[h % self.dim] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_endpoint,
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_deployment=settings.azure_embedding_deployment,
        )
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
        )
    return HashingEmbeddings()
