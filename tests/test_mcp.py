import json

import pytest

from nexusiq import mcp_server


@pytest.fixture(autouse=True)
def _use_test_engine(engine, monkeypatch):
    monkeypatch.setattr(mcp_server, "_kb", lambda: engine.kb)
    monkeypatch.setattr(mcp_server, "_db", lambda: engine.db)
    monkeypatch.setattr(mcp_server, "_copilot", lambda: engine)


def test_search_tool_respects_roles():
    assert "hr_leave_policy" in mcp_server.search_knowledge_base_impl("annual leave days")
    assert "compensation_bands" not in mcp_server.search_knowledge_base_impl("salary bands bonus", role="employee")
    assert "compensation_bands" in mcp_server.search_knowledge_base_impl("salary bands bonus", role="hr_admin")


def test_sql_tool_validates_and_executes():
    ok = json.loads(mcp_server.run_sql_query_impl("SELECT COUNT(*) AS n FROM employees"))
    assert ok["rows"] == [[48]]
    assert mcp_server.run_sql_query_impl("SELECT salary FROM employees").startswith("Query rejected")
    assert mcp_server.run_sql_query_impl("DROP TABLE employees").startswith("Query rejected")


def test_schema_and_calculator_tools():
    assert "salary" not in mcp_server.get_database_schema_impl("employee")
    assert mcp_server.calculate_impl("12 * 12") == "144"
    assert mcp_server.calculate_impl("__import__('os')").startswith("Error")
