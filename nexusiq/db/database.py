"""SQLite company database: deterministic seed data, schema description for the LLM, and safe read-only execution."""
from __future__ import annotations

import random
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT NOT NULL
);
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    department_id INTEGER NOT NULL REFERENCES departments(id),
    job_title TEXT NOT NULL,
    salary INTEGER NOT NULL,
    hire_date TEXT NOT NULL,
    city TEXT NOT NULL
);
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    industry TEXT NOT NULL,
    signup_date TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
"""

# Human-written hints that make the LLM far more accurate at text-to-SQL.
COLUMN_NOTES = {
    "employees.department_id": "FK -> departments.id",
    "employees.salary": "annual salary in INR (restricted column)",
    "employees.email": "work email (restricted column)",
    "employees.hire_date": "ISO date YYYY-MM-DD",
    "customers.country": "e.g. India, USA, UK, Germany, UAE, Singapore, Australia",
    "customers.industry": "Retail, Banking, Healthcare, Manufacturing, EdTech, Logistics",
    "products.category": "CRM, Analytics, AI, Security, Automation",
    "products.unit_price": "list price in USD",
    "orders.status": "one of: completed, pending, cancelled, refunded",
    "orders.order_date": "ISO date YYYY-MM-DD",
    "order_items.unit_price": "price actually charged in USD at time of sale",
}
SCHEMA_NOTES = (
    "Notes:\n"
    "- Revenue = SUM(order_items.quantity * order_items.unit_price) over orders with status = 'completed'.\n"
    "- Join path: customers -> orders -> order_items -> products; employees -> departments.\n"
    "- Data covers orders from 2025-01-01 to 2026-09-30."
)


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    # ------------------------------------------------------------------ setup
    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def ensure_ready(self) -> None:
        if not self.exists():
            self.seed()

    def seed(self, seed: int = 42) -> None:
        """(Re)create the demo database with deterministic fake data."""
        rng = random.Random(seed)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        con = sqlite3.connect(self.path)
        try:
            con.executescript(SCHEMA_SQL)

            departments = [
                (1, "Engineering", "Hyderabad"), (2, "Sales", "Mumbai"), (3, "Marketing", "Bengaluru"),
                (4, "Human Resources", "Hyderabad"), (5, "Finance", "Hyderabad"),
                (6, "Customer Success", "Pune"), (7, "Product", "Hyderabad"),
            ]
            con.executemany("INSERT INTO departments VALUES (?,?,?)", departments)

            first = ["Aarav", "Vivaan", "Aditya", "Ishaan", "Rohan", "Karthik", "Arjun", "Sai", "Harsha", "Ravi",
                     "Ananya", "Diya", "Meera", "Priya", "Sneha", "Kavya", "Neha", "Pooja", "Lakshmi", "Divya",
                     "Rahul", "Sanjay", "Vikram", "Suresh", "Anil", "Deepak", "Naveen", "Manoj", "Swathi", "Bhavana"]
            last = ["Sharma", "Reddy", "Rao", "Iyer", "Nair", "Gupta", "Patel", "Kumar", "Singh", "Verma",
                    "Chowdary", "Naidu", "Menon", "Joshi", "Mehta", "Das", "Kulkarni", "Bhat", "Pillai", "Yadav"]
            titles = {
                1: [("Software Engineer", 1100000), ("Senior Software Engineer", 2000000), ("AI Engineer", 1500000),
                    ("DevOps Engineer", 1400000), ("Engineering Manager", 3200000)],
                2: [("Sales Executive", 700000), ("Account Manager", 1200000), ("Sales Manager", 2200000)],
                3: [("Marketing Analyst", 650000), ("Content Strategist", 800000), ("Marketing Manager", 1900000)],
                4: [("HR Executive", 600000), ("Talent Partner", 900000), ("HR Manager", 1800000)],
                5: [("Financial Analyst", 900000), ("Accountant", 750000), ("Finance Manager", 2300000)],
                6: [("Support Engineer", 650000), ("Customer Success Manager", 1300000)],
                7: [("Product Analyst", 1000000), ("Product Manager", 2400000)],
            }
            headcount = {1: 16, 2: 8, 3: 5, 4: 4, 5: 5, 6: 6, 7: 4}
            cities = {1: "Hyderabad", 2: "Mumbai", 3: "Bengaluru", 4: "Hyderabad", 5: "Hyderabad",
                      6: "Pune", 7: "Hyderabad"}
            employees, emp_id, used = [], 1, set()
            for dept_id, count in headcount.items():
                for _ in range(count):
                    while True:
                        name = f"{rng.choice(first)} {rng.choice(last)}"
                        if name not in used:
                            used.add(name)
                            break
                    title, base = rng.choice(titles[dept_id])
                    salary = int(base * rng.uniform(0.9, 1.15) / 1000) * 1000
                    hire = date(2019, 1, 1) + timedelta(days=rng.randint(0, (date(2026, 8, 31) - date(2019, 1, 1)).days))
                    email = name.lower().replace(" ", ".") + "@nexustech.example"
                    employees.append((emp_id, name, email, dept_id, title, salary, hire.isoformat(), cities[dept_id]))
                    emp_id += 1
            con.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?,?)", employees)

            countries = ["India", "India", "India", "USA", "USA", "UK", "Germany", "UAE", "Singapore", "Australia"]
            industries = ["Retail", "Banking", "Healthcare", "Manufacturing", "EdTech", "Logistics"]
            prefixes = ["Apex", "Bluewave", "Cedar", "Delta", "Ember", "Falcon", "Granite", "Horizon", "Indigo",
                        "Juniper", "Keystone", "Lumen", "Meridian", "Nimbus", "Orion", "Pinnacle", "Quantum",
                        "Redwood", "Summit", "Titan", "Umbra", "Vertex", "Willow", "Xenon", "Yonder", "Zenith"]
            suffixes = ["Retail", "Bank", "Health", "Industries", "Learning", "Logistics", "Systems", "Group"]
            customers, cust_id, used_c = [], 1, set()
            while cust_id <= 32:
                name = f"{rng.choice(prefixes)} {rng.choice(suffixes)}"
                if name in used_c:
                    continue
                used_c.add(name)
                signup = date(2024, 6, 1) + timedelta(days=rng.randint(0, 600))
                customers.append((cust_id, name, rng.choice(countries), rng.choice(industries), signup.isoformat()))
                cust_id += 1
            con.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

            products = [
                (1, "Nexus CRM Pro", "CRM", 1200.0), (2, "Nexus CRM Lite", "CRM", 450.0),
                (3, "Nexus Analytics Studio", "Analytics", 1800.0), (4, "Nexus Insight Dashboards", "Analytics", 650.0),
                (5, "Nexus AI Copilot", "AI", 2500.0), (6, "Nexus Vision API", "AI", 900.0),
                (7, "Nexus Shield", "Security", 1500.0), (8, "Nexus Identity", "Security", 800.0),
                (9, "Nexus Flow Automation", "Automation", 1100.0), (10, "Nexus Connect Integrations", "Automation", 500.0),
            ]
            con.executemany("INSERT INTO products VALUES (?,?,?,?)", products)

            statuses = ["completed"] * 14 + ["pending"] * 2 + ["cancelled"] * 2 + ["refunded"]
            start, span = date(2025, 1, 1), (date(2026, 9, 30) - date(2025, 1, 1)).days
            orders, items, item_id = [], [], 1
            for order_id in range(1, 421):
                cust = rng.choice(customers)
                order_day = start + timedelta(days=rng.randint(0, span))
                if order_day < date.fromisoformat(cust[4]):
                    order_day = date.fromisoformat(cust[4]) + timedelta(days=rng.randint(1, 30))
                orders.append((order_id, cust[0], order_day.isoformat(), rng.choice(statuses)))
                for prod in rng.sample(products, rng.randint(1, 3)):
                    price = round(prod[3] * rng.choice([0.9, 0.95, 1.0, 1.0, 1.0]), 2)
                    items.append((item_id, order_id, prod[0], rng.randint(1, 8), price))
                    item_id += 1
            con.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
            con.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------------ schema for prompts
    def describe_schema(self, blocked_columns: dict[str, frozenset[str]] | None = None) -> str:
        """Compact schema text for the LLM. Restricted columns are hidden entirely (least privilege)."""
        blocked_columns = blocked_columns or {}
        self.ensure_ready()
        con = self._connect_ro()
        try:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            lines = []
            for table in tables:
                hidden = {c.lower() for c in blocked_columns.get(table, frozenset())}
                cols = []
                for _cid, cname, ctype, *_ in con.execute(f"PRAGMA table_info({table})"):
                    if cname.lower() in hidden:
                        continue
                    note = COLUMN_NOTES.get(f"{table}.{cname}")
                    cols.append(f"{cname} {ctype}" + (f"  -- {note}" if note else ""))
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                lines.append(f"TABLE {table} ({count} rows)\n  " + "\n  ".join(cols))
            return "\n".join(lines) + "\n\n" + SCHEMA_NOTES
        finally:
            con.close()

    def table_names(self) -> set[str]:
        self.ensure_ready()
        con = self._connect_ro()
        try:
            return {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        finally:
            con.close()

    # ------------------------------------------------------------------ execution
    def _connect_ro(self) -> sqlite3.Connection:
        uri = f"{self.path.resolve().as_uri()}?mode=ro"     # read-only at the SQLite level
        return sqlite3.connect(uri, uri=True)

    def execute_readonly(self, sql: str, timeout_s: float = 3.0, max_rows: int = 200):
        """Run a (pre-validated) query on a READ-ONLY connection with a wall-clock timeout."""
        self.ensure_ready()
        con = self._connect_ro()
        deadline = time.monotonic() + timeout_s
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            cur = con.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = [list(r) for r in cur.fetchmany(max_rows)]
            return columns, rows
        finally:
            con.close()
