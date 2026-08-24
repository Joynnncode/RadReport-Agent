"""`python -m radreport ask "..."` — the CLI from the Weekend 2 done-when.

Thin on purpose: parse args, call agent.run, print. All behaviour lives in
agent.py so that the CLI, the eval harness and the Streamlit app all exercise
exactly the same code path. If the CLI has its own logic, your eval is measuring
something your users never run.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m radreport")
    sub = ap.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask the agent a question")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--provider", default="gemini", choices=["gemini", "groq"])
    ask.add_argument("--max-iterations", type=int, default=8)
    ask.add_argument("--structured", action="store_true",
                     help="return a validated AgentAnswer with grounding check")
    ask.add_argument("--no-cache", action="store_true",
                     help="bypass the LLM response cache")

    ns = ap.parse_args()

    if ns.no_cache:
        import os
        os.environ["RADREPORT_CACHE"] = "0"

    from radreport.agent import run, run_structured
    entry = run_structured if ns.structured else run
    try:
        result = entry(" ".join(ns.question),
                       provider_name=ns.provider,
                       max_iterations=ns.max_iterations)
    except RuntimeError as exc:
        # Missing API key is the overwhelmingly common case here, and a
        # traceback helps nobody diagnose it.
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(json.dumps(result, indent=2))
    # A non-converged run is a failure and the exit code should say so, so that
    # a shell loop or CI step can detect it without parsing JSON.
    return 0 if result["converged"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
