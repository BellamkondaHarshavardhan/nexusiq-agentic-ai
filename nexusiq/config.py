"""Central configuration. Everything is driven by environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # keep ChromaDB quiet (no usage pings)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _temperature() -> float | None:
    """0 by default (deterministic). Set LLM_TEMPERATURE=default for models that only accept their own default."""
    raw = _env("LLM_TEMPERATURE", "0").lower()
    if raw in {"", "none", "default"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return 0.0


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    llm_provider: str            # azure | openai | offline  (already resolved from "auto")
    embedding_provider: str      # azure | openai | offline
    # Azure OpenAI
    azure_endpoint: str
    azure_api_key: str
    azure_api_version: str
    azure_chat_deployment: str
    azure_embedding_deployment: str
    # OpenAI / compatible
    openai_api_key: str
    openai_model: str
    openai_embedding_model: str
    openai_base_url: str
    llm_temperature: float | None
    # Retrieval / SQL / critic tuning
    top_k: int
    sql_max_rows: int
    sql_max_retries: int
    critic_max_retries: int
    # Paths
    data_dir: Path
    kb_dir: Path
    db_path: Path
    vector_dir: Path

    @property
    def is_offline(self) -> bool:
        return self.llm_provider == "offline"

    @property
    def llm_label(self) -> str:
        if self.llm_provider == "azure":
            return f"Azure OpenAI ({self.azure_chat_deployment})"
        if self.llm_provider == "openai":
            host = f" @ {self.openai_base_url}" if self.openai_base_url else ""
            return f"OpenAI-compatible ({self.openai_model}{host})"
        return "Offline demo model (no LLM)"

    @property
    def embedding_label(self) -> str:
        return {
            "azure": f"Azure OpenAI ({self.azure_embedding_deployment})",
            "openai": f"OpenAI ({self.openai_embedding_model})",
        }.get(self.embedding_provider, "Local hashing embeddings (offline)")


def get_settings() -> Settings:
    """Read the environment fresh each call (keeps tests and the UI honest)."""
    azure_endpoint = _env("AZURE_OPENAI_ENDPOINT")
    azure_key = _env("AZURE_OPENAI_API_KEY")
    azure_chat = _env("AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_emb = _env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    openai_key = _env("OPENAI_API_KEY")
    openai_base = _env("OPENAI_BASE_URL")

    azure_chat_ready = bool(azure_endpoint and azure_key and azure_chat)
    azure_emb_ready = bool(azure_endpoint and azure_key and azure_emb)
    openai_ready = bool(openai_key)

    # --- resolve the chat provider ---
    llm = _env("LLM_PROVIDER", "auto").lower()
    if llm == "auto":
        llm = "azure" if azure_chat_ready else "openai" if openai_ready else "offline"
    elif llm == "azure" and not azure_chat_ready:
        llm = "offline"
    elif llm == "openai" and not openai_ready:
        llm = "offline"

    # --- resolve the embedding provider ---
    emb = _env("EMBEDDING_PROVIDER", "auto").lower()
    if emb == "auto":
        if azure_emb_ready:
            emb = "azure"
        elif openai_ready and not openai_base:   # custom base URLs often have no embeddings endpoint
            emb = "openai"
        else:
            emb = "offline"
    elif emb == "azure" and not azure_emb_ready:
        emb = "offline"
    elif emb == "openai" and not openai_ready:
        emb = "offline"

    data_dir = ROOT / "data"
    return Settings(
        llm_provider=llm,
        embedding_provider=emb,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_key,
        azure_api_version=_env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_chat_deployment=azure_chat,
        azure_embedding_deployment=azure_emb,
        openai_api_key=openai_key,
        openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embedding_model=_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_base_url=openai_base,
        llm_temperature=_temperature(),
        top_k=_int("RETRIEVAL_TOP_K", 4),
        sql_max_rows=_int("SQL_MAX_ROWS", 50),
        sql_max_retries=_int("SQL_MAX_RETRIES", 2),
        critic_max_retries=_int("CRITIC_MAX_RETRIES", 1),
        data_dir=data_dir,
        kb_dir=data_dir / "knowledge_base",
        db_path=data_dir / "company.db",
        vector_dir=data_dir / "chroma",
    )
