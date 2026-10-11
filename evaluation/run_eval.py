"""Automated evaluation of the NexusIQ pipeline against a golden question set.

    python evaluation/run_eval.py                 # uses whatever provider is configured in .env (offline by default)
    python evaluation/run_eval.py --min-pass 0.9  # exit code 1 if the overall pass-rate is below 90 %

Metrics: routing accuracy, answer-fact check, SQL execution success, guardrail (RBAC / injection) pass-rate, latency.
Reports are written to evaluation/reports/latest.json and latest.md - handy evidence for a README or an interview.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexusiq.graph import NexusIQ  # noqa: E402


def grade(case: dict, result: dict) -> dict:
    checks: dict[str, bool] = {}
    answer = result.get("answer", "")
    sql = result.get("sql") or {}
    if "expected_route" in case:
        checks["route"] = result.get("route") == case["expected_route"]
    for needle in case.get("must_contain", []):
        checks[f"contains '{needle}'"] = needle.lower() in answer.lower()
    for needle in case.get("must_not_contain", []):
        checks[f"omits '{needle}'"] = needle.lower() not in answer.lower()
    if case.get("expect_sql"):
        checks["sql executed"] = bool(sql.get("query")) and not sql.get("error") and sql.get("row_count", 0) > 0
    for needle in case.get("sql_must_contain", []):
        checks[f"sql uses '{needle}'"] = needle.lower() in (sql.get("query", "")).lower()
    if case.get("expect_denied"):
        checks["access denied"] = bool(sql.get("denied"))
    if case.get("expect_blocked"):
        checks["input blocked"] = bool(result.get("blocked"))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pass", type=float, default=0.8)
    args = parser.parse_args()

    cases = json.loads((ROOT / "evaluation" / "golden_set.json").read_text(encoding="utf-8"))
    app = NexusIQ()
    print(f"Evaluating {len(cases)} cases | LLM: {app.settings.llm_label} | embeddings: {app.settings.embedding_label}\n")

    rows, latencies = [], []
    buckets = {"routing": [], "answer facts": [], "sql success": [], "guardrails": []}
    for case in cases:
        result = app.ask(case["question"], thread_id=app.new_thread(), role=case["role"])
        checks = grade(case, result)
        passed = all(checks.values())
        latencies.append(result.get("latency_ms", 0))
        for name, ok in checks.items():
            if name == "route":
                buckets["routing"].append(ok)
            elif name.startswith(("contains", "omits")):
                buckets["answer facts"].append(ok)
            elif name.startswith("sql"):
                buckets["sql success"].append(ok)
            else:
                buckets["guardrails"].append(ok)
        failed = [n for n, ok in checks.items() if not ok]
        rows.append({"id": case["id"], "passed": passed, "route": result.get("route"), "ms": result.get("latency_ms"),
                     "failed_checks": failed})
        print(f"{'PASS' if passed else 'FAIL'}  {case['id']:<16} route={result.get('route') or '-':<10} {result.get('latency_ms'):>5} ms"
              + (f"   failed: {failed}" if failed else ""))

    overall = sum(r["passed"] for r in rows) / len(rows)
    summary = {name: (sum(v) / len(v) if v else None) for name, v in buckets.items()}
    summary.update(overall_pass_rate=overall, cases=len(rows), avg_latency_ms=round(statistics.mean(latencies)),
                   p95_latency_ms=sorted(latencies)[int(0.95 * (len(latencies) - 1))],
                   llm=app.settings.llm_label, embeddings=app.settings.embedding_label)

    print("\n--- summary ---")
    for name, value in summary.items():
        print(f"{name:<18} {value:.0%}" if isinstance(value, float) and value <= 1 else f"{name:<18} {value}")

    out = ROOT / "evaluation" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    md = ["# NexusIQ evaluation report", "", f"- LLM: {summary['llm']}", f"- Embeddings: {summary['embeddings']}", "",
          "| Metric | Result |", "|---|---|"]
    for name in ("routing", "answer facts", "sql success", "guardrails"):
        md.append(f"| {name} | {summary[name]:.0%} |" if summary[name] is not None else f"| {name} | n/a |")
    md += [f"| overall pass-rate | {overall:.0%} |", f"| avg latency | {summary['avg_latency_ms']} ms |", "",
           "| Case | Result | Route | Latency |", "|---|---|---|---|"]
    md += [f"| {r['id']} | {'PASS' if r['passed'] else 'FAIL'} | {r['route'] or '-'} | {r['ms']} ms |" for r in rows]
    (out / "latest.md").write_text("\n".join(md), encoding="utf-8")
    return 0 if overall >= args.min_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
