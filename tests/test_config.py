from nexusiq.config import ROOT, get_settings


def test_runtime_data_directory_can_be_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("NEXUSIQ_DATA_DIR", str(tmp_path))

    settings = get_settings()

    assert settings.data_dir == tmp_path
    assert settings.db_path == tmp_path / "company.db"
    assert settings.vector_dir == tmp_path / "chroma"
    assert settings.kb_dir == ROOT / "data" / "knowledge_base"
