import pytest

from nexusiq.db import Database, validate_sql
from nexusiq.guardrails import get_policy

TABLES = {"employees", "departments", "customers", "products", "orders", "order_items"}
EMPLOYEE = get_policy("employee").blocked_columns
HR = get_policy("hr_admin").blocked_columns


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM employees",
    "SELECT d.name, COUNT(*) FROM employees e JOIN departments d ON d.id = e.department_id GROUP BY d.name",
    "WITH t AS (SELECT * FROM orders) SELECT COUNT(*) FROM t",
    "SELECT name FROM customers UNION SELECT name FROM products",
])
def test_valid_read_only_queries_pass(sql):
    assert validate_sql(sql, TABLES, EMPLOYEE).ok


@pytest.mark.parametrize("sql", [
    "DROP TABLE employees",
    "DELETE FROM orders",
    "UPDATE employees SET salary = 1",
    "INSERT INTO products VALUES (99, 'x', 'y', 1)",
    "SELECT 1; DROP TABLE orders",
    "SELECT * FROM sqlite_master",
    "SELECT load_extension('evil')",
])
def test_dangerous_queries_are_rejected_as_policy_violations(sql):
    result = validate_sql(sql, TABLES, EMPLOYEE)
    assert not result.ok and result.policy_violation


@pytest.mark.parametrize("sql", [
    "SELECT salary FROM employees",
    "SELECT name FROM employees ORDER BY salary DESC",
    "SELECT * FROM employees",
    "SELECT e.email FROM employees e",
    "SELECT x.s FROM (SELECT salary AS s FROM employees) x",
])
def test_restricted_columns_blocked_for_employee_but_allowed_for_hr(sql):
    assert not validate_sql(sql, TABLES, EMPLOYEE).ok
    assert validate_sql(sql, TABLES, HR).ok


def test_count_star_is_allowed_on_restricted_tables():
    assert validate_sql("SELECT COUNT(*) FROM employees", TABLES, EMPLOYEE).ok


def test_limit_is_enforced_and_capped():
    assert "LIMIT 50" in validate_sql("SELECT name FROM customers", TABLES, {}, 50).sql
    assert "LIMIT 50" in validate_sql("SELECT name FROM customers LIMIT 9999", TABLES, {}, 50).sql
    assert "LIMIT 5" in validate_sql("SELECT name FROM customers LIMIT 5", TABLES, {}, 50).sql


def test_typos_are_retryable_not_policy_denials():
    result = validate_sql("SELEC name FROM customers", TABLES, {})
    assert not result.ok and not result.policy_violation


def test_database_connection_is_read_only(tmp_path):
    db = Database(tmp_path / "t.db")
    db.seed()
    with pytest.raises(Exception):
        db.execute_readonly("DELETE FROM orders")
    columns, rows = db.execute_readonly("SELECT COUNT(*) AS n FROM employees")
    assert columns == ["n"] and rows == [[48]]


def test_schema_hides_restricted_columns(tmp_path):
    db = Database(tmp_path / "t.db")
    db.seed()
    assert "salary" not in db.describe_schema(EMPLOYEE)
    assert "salary" in db.describe_schema(HR)
