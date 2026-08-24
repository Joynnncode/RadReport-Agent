"""Write a committable summary of an eval run: numbers, no corpus text.

The raw records are gitignored because they embed radiologist report text, and
the corpus is gitignored. But a project whose README cites eval numbers should
commit the evidence for them, so this strips answers and tool output and keeps
the scores.

    python -m evals.export_summary evals/results/groq_partial.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.rescore import rescore          # noqa: E402
from evals.run import load_gold_set, summarise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--provider", default="")
    ap.add_argument("--model", default="")
    ns = ap.parse_args()

    path = Path(ns.results)
    if path.suffix == ".jsonl":
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    else:
        records = json.loads(path.read_text())["records"]

    # Always re-score against the CURRENT gold set before exporting. Exporting
    # raw records published 75.0% while the README, computed after a gold-set
    # fix, said 97.2%. The committed evidence and the claimed numbers must come
    # from the same computation or the evidence is worse than useless.
    records = rescore(records, {c["id"]: c for c in load_gold_set()})
    summary = summarise(records)

    by_cat = defaultdict(lambda: {"n": 0, "tool_pass": 0})
    for r in records:
        if r.get("excluded"):
            continue
        c = by_cat[r["category"]]
        c["n"] += 1
        c["tool_pass"] += bool(r["tool_selection"]["pass"])

    per_case = [{
        "id": r["id"],
        "category": r["category"],
        "difficulty": r["difficulty"],
        "excluded": bool(r.get("excluded")),
        "tool_selection_pass": r["tool_selection"]["pass"],
        "tools_called": r["tool_selection"]["called"],
        "deterministic_pass": r["deterministic"]["pass"],
        "grounded": (r["groundedness"]["pass"]
                     if r["groundedness"]["applicable"] else None),
        # deliberately no "answer" field: that is where the report text lives
    } for r in records]

    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": ns.provider or "unknown",
        "model": ns.model or "unknown",
        "gold_set_size": len(records),
        "summary": summary,
        "by_category": {k: v for k, v in sorted(by_cat.items())},
        "per_case": per_case,
    }

    dest = Path(__file__).parent / "results_summary.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest} ({len(per_case)} cases, no answer text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
