import pytest

from nexusiq.tools import calculator, get_current_datetime, safe_eval


@pytest.mark.parametrize("expr,expected", [
    ("2 + 2", 4), ("(1250 * 12) / 100", 150), ("2 ** 10", 1024), ("10 % 4", 2), ("-5 + 8", 3),
    ("sqrt(144)", 12), ("round(3.14159, 2)", 3.14), ("2 ^ 3", 8),
])
def test_safe_eval(expr, expected):
    assert safe_eval(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hacked')", "open('/etc/passwd').read()", "9 ** 9 ** 9",
    "().__class__", "lambda: 1", "x + 1", "1 +", "a" * 500,
])
def test_safe_eval_rejects_anything_unsafe(expr):
    with pytest.raises(Exception):
        safe_eval(expr)


def test_calculator_tool_returns_text_and_never_raises():
    assert calculator.invoke({"expression": "15 * 160"}) == "2400"
    assert calculator.invoke({"expression": "1/0"}).startswith("Error")
    assert calculator.invoke({"expression": "import os"}).startswith("Error")


def test_datetime_tool():
    assert "2026" in get_current_datetime.invoke({"timezone": "Asia/Kolkata"}) or "20" in get_current_datetime.invoke({})
    assert "UTC" in get_current_datetime.invoke({"timezone": "Not/AZone"})
