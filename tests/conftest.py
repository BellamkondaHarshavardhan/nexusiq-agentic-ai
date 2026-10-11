"""Shared fixtures. Tests always run in OFFLINE mode (no API keys, no network, no cost)."""
from __future__ import annotations

import dataclasses

import pytest

from nexusiq.config import get_settings
from nexusiq.graph import NexusIQ


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("nexusiq")
    return dataclasses.replace(
        get_settings(),
        llm_provider="offline",
        embedding_provider="offline",
        data_dir=tmp,
        db_path=tmp / "company.db",
        vector_dir=tmp / "chroma",
    )


@pytest.fixture(scope="session")
def engine(settings):
    return NexusIQ(settings)


@pytest.fixture()
def thread(engine):
    return engine.new_thread()
