"""Offline evals for the deterministic parts of the agent: retrieval and guardrail.

Run from the repo root: python -m evals.run_evals [--min-pass 0.8]
No API key or model needed. Exits 1 if the overall pass rate is below --min-pass.
"""
import argparse
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph import check
from retrieval import retrieve

ROLES = {"human": HumanMessage, "ai": AIMessage, "tool": lambda t: ToolMessage(t, tool_call_id="eval")}


def run_case(case: dict) -> tuple[bool, str]:
    """Return (passed, what the system actually did)."""
    if case["kind"] == "retrieval":
        got = retrieve(case["query"])
        # The agent puts the top 2 docs in context, so a hit anywhere in them counts.
        # expect null means "nothing relevant", i.e. retrieval should return no docs.
        passed = case["expect"] in got if case["expect"] else not got
        return passed, str(got)
    msgs = [ROLES[role](text) for role, text in case["messages"]]
    got = "pass" if check({"messages": msgs}) is None else "block"
    return got == case["expect"], got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.jsonl"))
    ap.add_argument("--min-pass", type=float, default=0.8)
    args = ap.parse_args()

    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    results: dict[str, list[bool]] = {}
    for case in cases:
        passed, got = run_case(case)
        results.setdefault(case["kind"], []).append(passed)
        if not passed:
            label = case.get("query") or case["messages"][-1][1]
            print(f"FAIL [{case['kind']}] {label!r}: expected {case['expect']!r}, got {got}")

    for kind, rs in results.items():
        print(f"{kind:<10} {sum(rs):>3}/{len(rs):<3} {sum(rs) / len(rs):.0%}")
    total = [r for rs in results.values() for r in rs]
    rate = sum(total) / len(total)
    print(f"{'overall':<10} {sum(total):>3}/{len(total):<3} {rate:.0%}  (min {args.min_pass:.0%})")
    return 0 if rate >= args.min_pass else 1


if __name__ == "__main__":
    sys.exit(main())
