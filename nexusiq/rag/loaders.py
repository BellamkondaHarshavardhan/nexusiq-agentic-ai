"""Load documents (.md / .txt / .pdf), parse optional front-matter, and split into section-aware chunks."""
from __future__ import annotations

import io
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

VALID_ACCESS = {"public", "hr_only"}
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip()
    return meta, raw[match.end():]


def read_pdf_bytes(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_document(name: str, data: bytes, default_access: str = "public") -> tuple[str, dict]:
    """Return (text, metadata) for an uploaded or on-disk file."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        text, meta = read_pdf_bytes(data), {}
    elif suffix in {".md", ".txt", ".markdown"}:
        meta, text = parse_front_matter(data.decode("utf-8", errors="replace"))
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .md, .txt or .pdf")
    access = meta.get("access", default_access).lower()
    if access not in VALID_ACCESS:
        access = "public"
    title = meta.get("title") or Path(name).stem.replace("_", " ").replace("-", " ").title()
    return text, {"source": Path(name).name, "title": title, "access": access}


def chunk_document(text: str, meta: dict, chunk_size: int = 700, chunk_overlap: int = 90) -> list[Document]:
    """Markdown-header split first (keeps each section's heading), then size-limited recursive split."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
    )
    sections = header_splitter.split_text(text) or [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = []
    for section in sections:
        heading = " > ".join(section.metadata[h] for h in ("h1", "h2", "h3") if h in section.metadata)
        for piece in splitter.split_text(section.page_content):
            if piece.strip():
                chunks.append(Document(
                    page_content=piece.strip(),
                    metadata={**meta, "section": heading or meta["title"]},
                ))
    for i, doc in enumerate(chunks):
        doc.metadata["chunk"] = i
    return chunks
