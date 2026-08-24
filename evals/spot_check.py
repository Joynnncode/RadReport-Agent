"""Format a sample of results for human review, and record the verdicts.

Why this exists. Three of the four metrics are mechanical and the fourth is an
LLM. Nothing in that set is an independent check on whether the answers are
actually any good: the deterministic checks only test what I thought to test, and
the judge is a model with blind spots correlated with the model it grades. Ten
cases read by a human is the only signal in this harness that does not come from
a system I built.

    python -m evals.spot_check evals/results/groq_partial.jsonl          # print
    python -m evals.spot_check evals/results/groq_partial.jsonl --record # interactive

Sampling is stratified across categories and seeded, so the same run always
yields the same ten cases and a second reviewer sees what I saw. Verdicts are
written to evals/spot_checks/<file>.json alongside the reviewer's notes.

Deliberately biased toward adversarial cases: a spot check of ten easy lookups
would confirm what the deterministic metrics already cover.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HERE = Path(__file__).parent
SEED = 20260821


def sample(records: list[dict], n: int = 10) -> list[dict]:
    """Stratified, adversarial-weighted, reproducible."""
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    rng = random.Random(SEED)
    adversarial = [r for r in records if r["difficulty"] == "adversarial"]
    other_cats = sorted(c for c in by_cat if not c.startswith("adversarial"))

    picked: list[dict] = []
    # Half the sample from adversarial cases, because that is where a human
    # reader adds the most over the mechanical checks.
    rng.shuffle(adversarial)
    picked.extend(adversarial[: n // 2])

    # The rest spread one-per-category so no single task type dominates.
    for cat in other_cats:
        if len(picked) >= n:
            break
        pool = [r for r in by_cat[cat] if r not in picked]
        if pool:
            picked.append(rng.choice(pool))

    i = 0
    while len(picked) < n and i < len(records):
        if records[i] not in picked:
            picked.append(records[i])
        i += 1
    return picked[:n]


def render(rec: dict, index: int, total: int) -> str:
    ts = rec["tool_selection"]
    lines = [
        "=" * 72,
        f"[{index}/{total}]  {rec['id']}   ({rec['category']} / {rec['difficulty']})",
        "=" * 72,
        f"QUESTION:  {rec['question']}",
        "",
        f"TOOLS:     {' -> '.join(ts['called']) or '(none)'}",
        f"AUTOMATED: tools={'PASS' if ts['pass'] else 'FAIL'}  "
        f"checks={rec['deterministic']['pass']}  "
        f"grounded={rec['groundedness']['pass'] if rec['groundedness']['applicable'] else 'n/a'}  "
        f"judge={rec['judge'].get('score')}",
        "",
        "ANSWER:",
    ]
    for line in (rec["answer"] or "(empty)").splitlines():
        lines.append(f"  {line}")
    lines += ["", "WHAT THIS CASE IS TESTING:", f"  {rec.get('notes', '(see gold set)')}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--record", action="store_true",
                    help="prompt for a verdict on each case and save them")
    ns = ap.parse_args()

    path = Path(ns.results)
    if path.suffix == ".jsonl":
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    else:
        records = json.loads(path.read_text())["records"]

    # Attach the gold-set notes so the reviewer knows what each case is for.
    from evals.run import load_gold_set
    notes = {c["id"]: c.get("notes", "") for c in load_gold_set()}
    for r in records:
        r["notes"] = notes.get(r["id"], "")

    picked = sample(records, ns.n)
    verdicts = []

    for i, rec in enumerate(picked, start=1):
        print(render(rec, i, len(picked)))
        if not ns.record:
            continue
        try:
            verdict = input("  verdict [g]ood / [a]cceptable / [b]ad / [s]kip: ").strip().lower()
            note = input("  note (optional): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  stopped early")
            break
        verdicts.append({"id": rec["id"], "verdict": verdict, "note": note,
                         "automated_judge": rec["judge"].get("score")})
        print()

    if verdicts:
        out_dir = HERE / "spot_checks"
        out_dir.mkdir(exist_ok=True)
        dest = out_dir / f"{path.stem}.json"
        dest.write_text(json.dumps(verdicts, indent=2))

        good = sum(1 for v in verdicts if v["verdict"] == "g")
        acceptable = sum(1 for v in verdicts if v["verdict"] == "a")
        bad = sum(1 for v in verdicts if v["verdict"] == "b")
        print(f"\n  {len(verdicts)} reviewed: {good} good, {acceptable} acceptable, {bad} bad")

        # The number worth reporting: where the human and the judge disagree.
        # Agreement is reassuring; disagreement tells you what the judge misses.
        disagreements = [
            v for v in verdicts
            if v["automated_judge"] is not None
            and ((v["verdict"] == "b" and v["automated_judge"] == 2)
                 or (v["verdict"] == "g" and v["automated_judge"] == 0))
        ]
        print(f"  human/judge disagreements: {len(disagreements)}")
        for d in disagreements:
            print(f"    {d['id']}: human={d['verdict']} judge={d['automated_judge']}")
        print(f"\n  written to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
