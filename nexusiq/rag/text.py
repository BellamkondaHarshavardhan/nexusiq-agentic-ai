"""Tokenisation shared by the offline embeddings and the BM25 keyword index."""
from __future__ import annotations

import re

_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were", "be", "been",
    "it", "this", "that", "as", "at", "by", "from", "do", "does", "did", "i", "my", "me", "we", "our", "you", "your",
    "can", "will", "what", "how", "which", "who", "when", "where", "any", "if", "than", "then", "so", "not", "no",
}


def _stem(tok: str) -> str:
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
            return tok[: len(tok) - len(suffix)] + repl
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]
