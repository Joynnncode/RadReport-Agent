"""Re-score saved answers against the current gold set, without re-running the agent.

    python -m evals.rescore evals/results/groq_partial.jsonl

Why this exists. Running the agent is expensive (rate limits, minutes) and
scoring is free. Coupling them means every fix to a *metric* costs another full
sweep, which in practice means you stop fixing metrics. Separating them means the
saved answers become a reusable dataset: change a check, re-score history,
compare old runs against the new definition.

It also caught its own justification on day one. The first CTR case failed
because my gold set required the literal word "caveat" in the answer. The agent
had written a textbook-perfect caveat without using the word. Fixing the check
and re-scoring took a second; re-running would have taken twenty minutes and
another slice of the daily quota.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.metrics import (                              # noqa: E402
    score_deterministic_checks, score_groundedness, score_tool_selection,
)
from evals.run import load_gold_set, print_breakdown, print_failures, print_summary, summarise  # noqa: E402


def rescore(records: list[dict], cases_by_id: dict[str, dict]) -> list[dict]:
    out = []
    for rec in records:
        case = cases_by_id.get(rec["id"])
        if case is None:
            print(f"  ! {rec['id']} is no longer in the gold set; dropping")
            continue

        updated = dict(rec)
        # Apply the exclusion rule retroactively: records written before the rule
        # existed still carry the quota error in their error field.
        from evals.run import is_quota_error
        updated["excluded"] = is_quota_error(rec.get("error"))
        updated["tool_selection"] = score_tool_selection(
            case, rec["tool_selection"]["called"])
        updated["deterministic"] = score_deterministic_checks(case, rec["answer"])
        # Groundedness needs the tool results, which live in the trace. Re-read
        # them if the trace still exists; otherwise keep the original verdict
        # rather than silently scoring it as a pass.
        trace = rec.get("trace_path", "")
        if trace and Path(trace).is_file():
            from evals.run import tool_results_from_trace
            updated["groundedness"] = score_groundedness(
                rec["answer"], tool_results_from_trace(trace))
        out.append(updated)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="a *_partial.jsonl or a run's .json file")
    ap.add_argument("--label", default="")
    ns = ap.parse_args()

    path = Path(ns.results)
    if path.suffix == ".jsonl":
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    else:
        records = json.loads(path.read_text())["records"]

    cases_by_id = {c["id"]: c for c in load_gold_set()}
    before = summarise(records)
    rescored = rescore(records, cases_by_id)
    after = summarise(rescored)

    print_summary(ns.label or f"{path.name} (rescored)", after)
    print_breakdown(rescored)
    print_failures(rescored)

    print(f"\n  {'metric':<26} {'before':>9} {'after':>9}")
    print(f"  {'-' * 26} {'-' * 9} {'-' * 9}")
    for key in ("tool_selection_accuracy", "deterministic_pass"):
        b, a = before[key], after[key]
        fmt = lambda v: "n/a" if v != v else f"{v * 100:.1f}%"
        print(f"  {key:<26} {fmt(b):>9} {fmt(a):>9}")

    dest = path.with_name(path.stem + "_rescored.json")
    dest.write_text(json.dumps({"summary": after, "records": rescored}, indent=2))
    print(f"\n  written to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
