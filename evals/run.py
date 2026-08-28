"""Run the gold set against a configuration and print a results table.

    python -m evals.run                          # default provider, all cases
    python -m evals.run --provider groq
    python -m evals.run --limit 8                # smoke run
    python -m evals.run --category adversarial   # substring match on category
    python -m evals.run --compare                # gemini vs groq, side by side
    python -m evals.run --no-judge               # deterministic metrics only

Results are written to evals/results/<provider>_<model>_<timestamp>.json so a
run is reproducible and comparable later. The table on stdout is a summary; the
JSON is the artefact.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.metrics import (                                    # noqa: E402
    build_judge_prompt, score_cost, score_deterministic_checks,
    score_groundedness, score_tool_selection,
)
from radreport.agent import run                                # noqa: E402
from radreport.llm import get_provider                         # noqa: E402

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"

# The judge is a DIFFERENT provider from the system under test by default.
# A model grading its own output scores it generously; using the other provider
# is a cheap way to avoid the most obvious form of that bias. It is not a
# complete fix -- both are LLMs with correlated blind spots -- which is why the
# harness also reports deterministic checks that need no judge at all.
JUDGE_PROVIDER = {"gemini": "groq", "groq": "gemini"}


def load_gold_set(path: Path | None = None) -> list[dict]:
    path = path or HERE / "gold_set.jsonl"
    if not path.exists():
        raise SystemExit(f"No gold set at {path}. Run: python evals/build_gold_set.py")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def tool_results_from_trace(trace_path: str) -> list[dict]:
    # Guard the empty string explicitly: Path("") is Path("."), which exists as a
    # directory, so a bare .exists() check passes and then read_text() blows up
    # with IsADirectoryError. Only ever hit on the error path, where it masked
    # the real exception it was meant to be reporting.
    if not trace_path:
        return []
    path = Path(trace_path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") == "tool_call" and e.get("result") is not None:
            out.append(e["result"])
    return out


def locate_trace(question: str, provider: str | None = None) -> str:
    """Find the trace for a question by reading the traces back.

    Records written before `trace_path` was added carry no pointer to their
    trace, and groundedness is the one metric that cannot be recomputed without
    the tool output. So the 52.9% those runs reported survived every later fix to
    the metric -- rescore silently kept the original verdict, and the README
    quoted a number produced by a definition that no longer existed anywhere in
    the codebase.

    The trace's own `start` event records the question and the provider, which is
    enough to match on. Returns "" when the match is not unique, because scoring
    against the wrong run is worse than not scoring.

    This is a RECOVERY path for records written before `trace_path` existed, and
    it gets less useful over time by design: every re-run of a question adds
    another trace with the same text, so the match stops being unique and this
    returns "". That is the correct failure. Records written by run.py now carry
    their trace_path directly and never come through here.
    """
    from radreport.config import TRACE_DIR
    matches = []
    for path in sorted(Path(TRACE_DIR).glob("*.jsonl")):
        try:
            with path.open() as fh:
                head = json.loads(fh.readline() or "{}")
        except (json.JSONDecodeError, OSError):
            continue
        if head.get("event") != "start" or head.get("question") != question:
            continue
        if provider and head.get("provider") != provider:
            continue
        matches.append(str(path))
    return matches[0] if len(matches) == 1 else ""


def judge(case: dict, answer: str, judge_provider) -> dict:
    """LLM-as-judge for the one thing that needs judgement."""
    try:
        resp = judge_provider.chat(
            [{"role": "user", "content": build_judge_prompt(case, answer)}], None, "")
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1].removeprefix("json")
        start, end = text.find("{"), text.rfind("}")
        verdict = json.loads(text[start:end + 1])
        verdict["ok"] = True
        return verdict
    except Exception as exc:
        # A judge failure must not look like a system failure. Mark it and move
        # on; the deterministic metrics still stand on their own.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200],
                "score": None, "safety_respected": None, "answers_question": None}


def is_quota_error(error: str | None) -> bool:
    """Distinguish 'the provider refused to serve us' from 'the agent got it wrong'.

    with_rate_limit_retry absorbs transient 429s, but a genuinely exhausted daily
    quota outlasts the retry budget and the error reaches here. Scoring that as a
    failed case is not a minor accounting quirk: it silently converts an
    infrastructure limit into an accusation against the model, and it is the
    exact mistake that would have me debugging the agent while the real problem
    was a spent free tier. These cases are EXCLUDED from every metric and counted
    separately.
    """
    if not error:
        return False
    text = error.lower()
    return ("429" in text or "resource_exhausted" in text
            or "quota" in text or "rate limit" in text)


def evaluate_case(case: dict, provider, judge_provider, use_judge: bool) -> dict:
    started = time.perf_counter()
    try:
        result = run(case["question"], provider=provider)
        error = None
    except Exception as exc:
        result = {"answer": "", "converged": False, "tool_sequence": [],
                  "trace_path": "", "input_tokens": 0, "output_tokens": 0,
                  "llm_calls": 0, "wall_time_s": time.perf_counter() - started}
        error = f"{type(exc).__name__}: {exc}"[:300]

    answer = result.get("answer") or ""
    tool_results = tool_results_from_trace(result.get("trace_path", ""))

    record = {
        "excluded": is_quota_error(error),
        "id": case["id"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "question": case["question"],
        "answer": answer,
        "error": error,
        # Kept so groundedness can be recomputed offline. Its absence meant the
        # one metric that needs the tool results was the one metric rescore
        # could not fix.
        "trace_path": result.get("trace_path", ""),
        "converged": result.get("converged", False),
        "tool_selection": score_tool_selection(case, result.get("tool_sequence", [])),
        "groundedness": score_groundedness(answer, tool_results),
        "deterministic": score_deterministic_checks(case, answer),
        "cost": score_cost({**result, "model": provider.model}),
    }
    record["judge"] = (judge(case, answer, judge_provider)
                       if use_judge and not error else {"ok": False, "score": None})
    return record


def summarise(records: list[dict]) -> dict:
    excluded = [r for r in records if r.get("excluded")]
    records = [r for r in records if not r.get("excluded")]
    n = len(records)
    if not n:
        return {"n_cases": 0, "excluded": len(excluded), "converged": float("nan"),
                "tool_selection_accuracy": float("nan"), "groundedness": float("nan"),
                "groundedness_n": 0, "quotes_total": 0, "quotes_verbatim": 0,
                "quotes_repaired": 0, "quotes_unsupported": 0,
                "pmids_total": 0, "pmids_unsupported": 0,
                "verbatim_rate": float("nan"), "deterministic_pass": float("nan"),
                "judge_mean_score": float("nan"), "judge_safety_rate": float("nan"),
                "judge_n": 0, "total_usd": 0, "usd_per_query": 0,
                "median_latency_s": 0, "p90_latency_s": 0, "mean_llm_calls": 0,
                "errors": 0}

    def rate(predicate) -> float:
        applicable = [r for r in records if predicate(r) is not None]
        if not applicable:
            return float("nan")
        return sum(1 for r in applicable if predicate(r)) / len(applicable)

    # Two denominators, both reported, because conflating them is what put a
    # case-level 52.9% and a quote-level 41/51 in the README as if they were the
    # same measurement. They are not: one asks "how many ANSWERS contain a
    # possible fabrication", the other "how many QUOTES were copied exactly".
    grounded = [r for r in records if r["groundedness"]["applicable"]]
    quotes_total = sum(r["groundedness"].get("quotes_found", 0) for r in records)
    quotes_verbatim = sum(r["groundedness"].get("verbatim", 0) for r in records)
    quotes_repaired = sum(len(r["groundedness"].get("repaired", [])) for r in records)
    quotes_unsupported = sum(len(r["groundedness"].get("unsupported", [])) for r in records)
    pmids_total = sum(r["groundedness"].get("pmids_found", 0) for r in records)
    pmids_unsupported = sum(len(r["groundedness"].get("unsupported_pmids", []))
                            for r in records)
    judged = [r for r in records if r["judge"].get("score") is not None]
    latencies = sorted(r["cost"]["wall_time_s"] for r in records)

    return {
        "n_cases": n,
        "converged": rate(lambda r: r["converged"]),
        "tool_selection_accuracy": rate(lambda r: r["tool_selection"]["pass"]),
        "groundedness": (sum(1 for r in grounded if r["groundedness"]["pass"]) / len(grounded)
                         if grounded else float("nan")),
        "groundedness_n": len(grounded),
        "quotes_total": quotes_total,
        "quotes_verbatim": quotes_verbatim,
        "quotes_repaired": quotes_repaired,
        "quotes_unsupported": quotes_unsupported,
        "pmids_total": pmids_total,
        "pmids_unsupported": pmids_unsupported,
        "verbatim_rate": (quotes_verbatim / quotes_total
                          if quotes_total else float("nan")),
        "deterministic_pass": rate(lambda r: r["deterministic"]["pass"]),
        "judge_mean_score": (statistics.mean(r["judge"]["score"] for r in judged)
                             if judged else float("nan")),
        "judge_safety_rate": (sum(1 for r in judged if r["judge"].get("safety_respected"))
                              / len(judged) if judged else float("nan")),
        "judge_n": len(judged),
        "total_usd": round(sum(r["cost"]["usd"] for r in records), 6),
        "usd_per_query": round(sum(r["cost"]["usd"] for r in records) / n, 6) if n else 0,
        "median_latency_s": round(statistics.median(latencies), 2) if latencies else 0,
        "p90_latency_s": round(latencies[int(len(latencies) * 0.9)], 2) if latencies else 0,
        "mean_llm_calls": round(statistics.mean(r["cost"]["llm_calls"] for r in records), 2) if n else 0,
        "errors": sum(1 for r in records if r["error"]),
        "excluded": len(excluded),
    }


def pct(x: float) -> str:
    return "  n/a" if x != x else f"{x * 100:5.1f}%"


def print_summary(label: str, s: dict) -> None:
    excluded = s.get("excluded", 0)
    suffix = f", {excluded} EXCLUDED (provider quota)" if excluded else ""
    print(f"\n{'=' * 64}\n{label}   ({s['n_cases']} cases scored{suffix})\n{'=' * 64}")
    if excluded and s["n_cases"] < 10:
        print(f"  !! only {s['n_cases']} case(s) scored; too few to report as a result")
    print(f"  converged                {pct(s['converged'])}")
    print(f"  tool selection accuracy  {pct(s['tool_selection_accuracy'])}")
    print(f"  groundedness             {pct(s['groundedness'])}   "
          f"({s['groundedness_n']} answers contained quotes; "
          f"{s.get('quotes_unsupported', 0)} unsupported quote(s))")
    print(f"    verbatim quote rate    {pct(s.get('verbatim_rate', float('nan')))}   "
          f"({s.get('quotes_verbatim', 0)} exact + {s.get('quotes_repaired', 0)} repaired "
          f"of {s.get('quotes_total', 0)})")
    if s.get("pmids_total"):
        print(f"    cited PMIDs            {s['pmids_total'] - s['pmids_unsupported']}"
              f"/{s['pmids_total']} appear in tool output"
              f"{'   <-- FABRICATED IDENTIFIER' if s['pmids_unsupported'] else ''}")
    print(f"  deterministic checks     {pct(s['deterministic_pass'])}")
    print(f"  judge safety rate        {pct(s['judge_safety_rate'])}   (n={s['judge_n']})")
    print(f"  judge mean score         {s['judge_mean_score']:.2f} / 2" if s['judge_n'] else "  judge mean score      n/a")
    print(f"  ---")
    print(f"  cost per query           ${s['usd_per_query']:.5f}")
    print(f"  median / p90 latency     {s['median_latency_s']}s / {s['p90_latency_s']}s")
    print(f"  mean LLM calls           {s['mean_llm_calls']}")
    if s["errors"]:
        print(f"  ERRORS                   {s['errors']}")


def print_breakdown(records: list[dict]) -> None:
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        if r.get("excluded"):
            continue
        groups[r["category"]].append(r)

    print(f"\n  {'category':<32} {'n':>3}  {'tools':>6}  {'checks':>6}  {'safety':>6}")
    print(f"  {'-' * 32} {'-' * 3}  {'-' * 6}  {'-' * 6}  {'-' * 6}")
    for cat in sorted(groups):
        rs = groups[cat]
        tools = sum(1 for r in rs if r["tool_selection"]["pass"]) / len(rs)
        det = [r for r in rs if r["deterministic"]["pass"] is not None]
        checks = (sum(1 for r in det if r["deterministic"]["pass"]) / len(det)) if det else float("nan")
        jd = [r for r in rs if r["judge"].get("safety_respected") is not None]
        safety = (sum(1 for r in jd if r["judge"]["safety_respected"]) / len(jd)) if jd else float("nan")
        print(f"  {cat:<32} {len(rs):>3}  {pct(tools)}  {pct(checks)}  {pct(safety)}")


def print_failures(records: list[dict], limit: int = 8) -> None:
    records = [r for r in records if not r.get("excluded")]
    fails = [r for r in records
             if not r["tool_selection"]["pass"]
             or r["deterministic"]["pass"] is False
             or r["judge"].get("safety_respected") is False
             or r["error"]]
    if not fails:
        print("\n  no failures")
        return
    print(f"\n  {len(fails)} failing case(s):")
    for r in fails[:limit]:
        reasons = []
        ts = r["tool_selection"]
        if ts["missing_tools"]:
            reasons.append(f"missing {ts['missing_tools']}")
        if ts["forbidden_tools_used"]:
            reasons.append(f"FORBIDDEN {ts['forbidden_tools_used']}")
        if not ts["order_ok"]:
            reasons.append("wrong order")
        if not ts.get("alternatives_satisfied", True):
            # Was missing, so cases failing ONLY on this printed a blank reason
            # and looked like a display glitch rather than the real finding.
            reasons.append("no imaging tool used (expected_any_of unsatisfied)")
        if r["deterministic"]["pass"] is False:
            for name, c in r["deterministic"]["checks"].items():
                if isinstance(c, dict) and c.get("pass") is False:
                    reasons.append(f"check:{name}")
        if r["judge"].get("safety_respected") is False:
            reasons.append("SAFETY")
        if r["error"]:
            reasons.append(f"error:{r['error'][:60]}")
        print(f"    {r['id']:<34} {', '.join(reasons)}")


def load_partial(path: Path) -> dict[str, dict]:
    """Read already-scored cases from an incremental file.

    Free-tier rate limits mean a 42-case sweep can take an hour and can die
    partway through. Losing all of it because case 38 hit a quota wall is
    unacceptable, so every record is appended the moment it is scored and a
    rerun with --resume skips what is already there.
    """
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def run_config(cases: list[dict], provider_name: str, use_judge: bool,
               delay: float = 0.0, resume: bool = False,
               stamp: str = "") -> tuple[list[dict], dict]:
    provider = get_provider(provider_name)
    judge_provider = None
    if use_judge:
        try:
            judge_provider = get_provider(JUDGE_PROVIDER.get(provider_name, "groq"))
        except Exception as exc:
            print(f"  (judge unavailable: {exc}; running deterministic metrics only)")
            use_judge = False

    RESULTS_DIR.mkdir(exist_ok=True)
    # Incremental file is keyed by provider only, so --resume finds it across runs.
    partial_path = RESULTS_DIR / f"{provider_name}_partial.jsonl"
    done = load_partial(partial_path) if resume else {}
    if done:
        print(f"  resuming: {len(done)} case(s) already scored in {partial_path.name}")

    print(f"\nrunning {len(cases)} cases on {provider_name}/{provider.model}"
          f"{' judged by ' + judge_provider.model if use_judge else ''}")

    records = []
    for i, case in enumerate(cases, start=1):
        if case["id"] in done:
            records.append(done[case["id"]])
            print(f"  [{i:>3}/{len(cases)}] {case['id']:<36} (cached)", flush=True)
            continue

        started = time.perf_counter()
        rec = evaluate_case(case, provider, judge_provider, use_judge)
        records.append(rec)

        with partial_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        ok = rec["tool_selection"]["pass"] and rec["deterministic"]["pass"] is not False
        status = "ok  " if ok else "FAIL"
        if rec["error"]:
            status = "ERR "
        print(f"  [{i:>3}/{len(cases)}] {case['id']:<36} {status} "
              f"{time.perf_counter() - started:5.1f}s  {'->'.join(rec['tool_selection']['called']) or '(no tools)'}",
              flush=True)

        if delay:
            time.sleep(delay)

    return records, summarise(records)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="gemini", choices=["gemini", "groq"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--category", help="substring match on category")
    ap.add_argument("--difficulty", help="exact match on difficulty")
    ap.add_argument("--compare", action="store_true", help="run gemini AND groq")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to pause between cases (free-tier pacing)")
    ap.add_argument("--resume", action="store_true",
                    help="skip cases already present in <provider>_partial.jsonl")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the partial file before starting")
    ns = ap.parse_args()

    cases = load_gold_set()
    if ns.category:
        cases = [c for c in cases if ns.category in c["category"]]
    if ns.difficulty:
        cases = [c for c in cases if c["difficulty"] == ns.difficulty]
    if ns.limit:
        cases = cases[: ns.limit]
    if not cases:
        raise SystemExit("no cases matched those filters")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    configs = ["gemini", "groq"] if ns.compare else [ns.provider]

    summaries = {}
    for name in configs:
        if ns.fresh:
            (RESULTS_DIR / f"{name}_partial.jsonl").unlink(missing_ok=True)
        records, summary = run_config(cases, name, not ns.no_judge,
                                      delay=ns.delay, resume=ns.resume, stamp=stamp)
        summaries[name] = summary
        print_summary(f"{name}", summary)
        print_breakdown(records)
        print_failures(records)

        out = RESULTS_DIR / f"{name.replace('/', '_')}_{stamp}.json"
        out.write_text(json.dumps(
            {"provider": name, "timestamp": stamp, "summary": summary, "records": records},
            indent=2))
        print(f"\n  written to {out.relative_to(HERE.parent)}")

    if len(summaries) > 1:
        print(f"\n{'=' * 64}\nCOMPARISON\n{'=' * 64}")
        keys = [("tool_selection_accuracy", "tool selection", pct),
                ("groundedness", "groundedness", pct),
                ("deterministic_pass", "deterministic checks", pct),
                ("judge_safety_rate", "judge safety rate", pct),
                ("usd_per_query", "cost per query", lambda v: f"${v:.5f}"),
                ("median_latency_s", "median latency", lambda v: f"{v}s")]
        names = list(summaries)
        print(f"  {'metric':<24} " + "  ".join(f"{n:>12}" for n in names))
        print(f"  {'-' * 24} " + "  ".join("-" * 12 for _ in names))
        for key, label, fmt in keys:
            print(f"  {label:<24} " + "  ".join(f"{fmt(summaries[n][key]):>12}" for n in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
