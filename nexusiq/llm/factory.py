"""Chat model factory: Azure OpenAI -> OpenAI(-compatible) -> offline demo model."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import Settings
from .offline import OfflineChatModel


def get_chat_model(settings: Settings) -> BaseChatModel:
    kwargs = {"max_retries": 3, "timeout": 60}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature

    if settings.llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=settings.azure_endpoint,
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_deployment=settings.azure_chat_deployment,
            **kwargs,
        )
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            **kwargs,
        )
    return OfflineChatModel()
