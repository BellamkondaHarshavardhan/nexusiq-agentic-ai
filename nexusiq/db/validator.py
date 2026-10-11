"""Static SQL safety validator (sqlglot AST based) - the guardrail between the LLM and the database.

An LLM-generated query is only executed if it is:
  1. exactly ONE statement, and a SELECT (CTEs / UNION allowed)
  2. free of dangerous functions (load_extension, readfile, ...)
  3. using only known tables
  4. not touching columns the caller's role may not see (including via SELECT *)
  5. bounded by a LIMIT
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

_FORBIDDEN_FUNCS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


@dataclass
class SQLValidation:
    ok: bool
    sql: str = ""
    error: str = ""
    policy_violation: bool = False     # True -> final denial (do not ask the LLM to "fix" it)


def validate_sql(
    sql: str,
    allowed_tables: set[str],
    blocked_columns: dict[str, frozenset[str]] | None = None,
    max_rows: int = 50,
) -> SQLValidation:
    blocked_columns = blocked_columns or {}
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return SQLValidation(False, error="Empty query.")

    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as exc:
        return SQLValidation(False, error=f"SQL syntax error: {str(exc)[:200]}")
    if len(statements) != 1:
        return SQLValidation(False, error="Exactly one SQL statement is allowed.", policy_violation=True)

    tree = statements[0]
    dangerous = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command)
    if isinstance(tree, dangerous) or tree.find(*dangerous):
        return SQLValidation(False, error="Data-modifying statements are not allowed.", policy_violation=True)
    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
        # e.g. a typo such as "SELEC ..." - the model can fix this, so it is NOT a policy denial
        return SQLValidation(False, error="The query must be a single SELECT statement.")

    for func in tree.find_all(exp.Anonymous):
        if str(func.name).lower() in _FORBIDDEN_FUNCS:
            return SQLValidation(False, error=f"Function '{func.name}' is not allowed.", policy_violation=True)

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables}
    used_tables: set[str] = set()
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if name in cte_names:
            continue
        if name not in allowed:
            return SQLValidation(False, error=f"Unknown or forbidden table '{table.name}'.", policy_violation=True)
        used_tables.add(name)

    # --- column-level access control ---
    blocked_flat = {c.lower() for cols in blocked_columns.values() for c in cols}
    if blocked_flat:
        for col in tree.find_all(exp.Column):
            if col.name.lower() in blocked_flat:
                return SQLValidation(
                    False,
                    error=f"Access denied: column '{col.name}' is restricted for your role.",
                    policy_violation=True,
                )
        for star in tree.find_all(exp.Star):
            if isinstance(star.parent, exp.Count):
                continue
            touched = used_tables & {t.lower() for t in blocked_columns}
            if touched:
                return SQLValidation(
                    False,
                    error="Access denied: SELECT * on a table with restricted columns is not allowed; list columns explicitly.",
                    policy_violation=True,
                )

    # --- enforce a row limit on plain SELECTs ---
    if isinstance(tree, exp.Select):
        limit = tree.args.get("limit")
        current = None
        if limit is not None:
            try:
                current = int(limit.expression.name)
            except (ValueError, AttributeError):
                current = None
        if current is None or current > max_rows:
            tree = tree.limit(max_rows)

    return SQLValidation(True, sql=tree.sql(dialect="sqlite"))
