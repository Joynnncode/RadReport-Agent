"""Judge saved answers as a separate, resumable pass.

    python -m evals.judge evals/results/groq_partial.jsonl --judge gemini
    python -m evals.judge evals/results/groq_partial.jsonl --judge gemini --resume

WHY THIS IS ITS OWN COMMAND. `run.py` judged each case inline, immediately after
producing the answer. That couples two independent things to each other's
failures, and both failures happened:

  - The first full sweep reported `judge_n = 0` and `judge_mean_score = NaN`.
    The system under test was Groq and the judge was Gemini (deliberately: a
    model grading its own output grades it generously). Gemini's free tier was
    spent. Every judge call failed, and the harness published a summary with two
    NaN columns in it.

  - Re-running with quota available fixed the NaN and introduced the opposite
    problem: judge rate-limiting stretched cases from 20s to 170s, because the
    agent run now waited on the judge's retry budget. An hour of wall time, most
    of it sleeping, to produce answers that were already correct.

The answers are the expensive artefact and they do not change. Judging is a
separate opinion about them that can be formed later, on a different provider,
or twice. Same argument as `rescore.py`: separating the run from the scoring
turns saved answers into a reusable dataset instead of a thing you must redo.

A quota wall now costs you the judge column on some rows -- reported as an
explicit count -- rather than an hour, or a NaN.

THE BUDGET IS 20. Gemini's free tier allows

    generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
    quotaId: GenerateRequestsPerDayPerProjectPerModel

twenty requests PER DAY. Against a 43-case gold set, cross-provider judging can
therefore never cover the whole set in one sitting. That is a structural fact
about the tier, not a run that went badly, and it is why `--limit` exists and
why it samples the way it does.

Judging the first N cases in file order would spend the budget on the easy block
-- the gold set is ordered lookups, measurements, classification, then the
adversarial cases last -- and produce a judge score of 2.0 that means nothing.
`--limit` therefore takes a STRATIFIED, adversarial-weighted, seeded sample, so
twenty requests buy an opinion about the whole distribution and the same twenty
cases are chosen every time. Run it again tomorrow with --resume to extend the
coverage; judge_n is reported alongside every judged metric so a reader always
knows how much of the set the number rests on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.run import is_quota_error, judge as judge_one, load_gold_set  # noqa: E402
from radreport.llm import get_provider                                  # noqa: E402


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())["records"]


def needs_judging(record: dict, resume: bool) -> bool:
    if record.get("excluded") or record.get("error"):
        return False
    if not (record.get("answer") or "").strip():
        return False
    return not (resume and record.get("judge", {}).get("score") is not None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--judge", default="gemini", choices=["gemini", "groq"],
                    help="provider doing the grading; use the OTHER one from the "
                         "system under test")
    ap.add_argument("--resume", action="store_true",
                    help="skip records that already carry a judge score")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--limit", type=int,
                    help="judge at most N cases, chosen as a stratified "
                         "adversarial-weighted sample rather than in file order "
                         "(Gemini's free tier is 20 requests per DAY)")
    ap.add_argument("-o", "--out", help="default: <results stem>_judged.jsonl")
    ns = ap.parse_args()

    path = Path(ns.results)
    records = load_records(path)
    dest = Path(ns.out) if ns.out else path.with_name(path.stem + "_judged.jsonl")

    # --resume has to read the file this command WRITES, not the one it reads
    # from. The sweep is run with --no-judge, so the input never carries scores;
    # resuming against it re-judged everything and then overwrote the previous
    # pass's verdicts with the new (empty) ones. A resumable command that
    # discards the work it is resuming from is just a slower non-resumable one.
    if ns.resume and dest.exists() and dest != path:
        previous = {r["id"]: r.get("judge") for r in load_records(dest)}
        carried = 0
        for record in records:
            verdict = previous.get(record["id"])
            if verdict and verdict.get("score") is not None:
                record["judge"] = verdict
                carried += 1
        if carried:
            print(f"carrying {carried} verdict(s) forward from {dest.name}")

    cases_by_id = {c["id"]: c for c in load_gold_set()}
    provider = get_provider(ns.judge)

    todo = [r for r in records if needs_judging(r, ns.resume)]
    if ns.limit and len(todo) > ns.limit:
        from evals.spot_check import sample
        # Reuses spot_check's sampler on purpose: same seed, same stratification,
        # so the cases a human reviews and the cases the judge grades overlap and
        # their verdicts can be compared. Two independent opinions about
        # different cases cannot disagree, which makes them useless as a check on
        # each other.
        todo = sample(todo, ns.limit)
    print(f"{len(records)} record(s); judging {len(todo)} with {provider.model}")

    judged = failed = 0
    stopped_early = False
    for i, record in enumerate(todo, start=1):
        case = cases_by_id.get(record["id"])
        if case is None:
            print(f"  [{i:>3}/{len(todo)}] {record['id']:<36} not in gold set; skipped")
            continue

        verdict = judge_one(case, record["answer"], provider)
        record["judge"] = verdict
        if verdict.get("score") is not None:
            judged += 1
            print(f"  [{i:>3}/{len(todo)}] {record['id']:<36} score={verdict['score']} "
                  f"safe={verdict.get('safety_respected')}")
        else:
            failed += 1
            print(f"  [{i:>3}/{len(todo)}] {record['id']:<36} JUDGE FAILED: "
                  f"{verdict.get('error', '')[:70]}")
            # A spent daily quota will not recover inside this loop, and burning
            # the remaining cases against it just buries the real message under
            # forty identical errors. Stop, say how far it got, and let --resume
            # pick it up tomorrow.
            if is_quota_error(verdict.get("error")):
                print("\n  provider quota exhausted; stopping. "
                      "Re-run with --resume when it resets.")
                stopped_early = True
                break

        if ns.delay:
            time.sleep(ns.delay)

    with dest.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    scored = sum(1 for r in records
                 if r.get("judge", {}).get("score") is not None)
    print(f"\n  judged {judged}, failed {failed}"
          f"{' (stopped early)' if stopped_early else ''}")
    print(f"  {scored}/{len(records)} record(s) now carry a judge score")
    print(f"  written to {dest}")
    # A partial judge pass is a real result, not an error: the deterministic
    # metrics stand on their own and the summary reports judge_n alongside them.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
