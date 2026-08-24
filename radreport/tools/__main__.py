"""CLI for exercising tools without the agent: `python -m radreport.tools <tool> [args]`

This exists so that on Weekend 1 you can see real numbers come out of real
models before any LLM is involved, and so that when the agent misbehaves later
you can reproduce a single tool call in isolation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from radreport.tools import REGISTRY, ToolError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m radreport.tools")
    parser.add_argument("tool", choices=sorted(REGISTRY), help="which tool to run")
    parser.add_argument("args", nargs="*", help="positional args, key=value for named")
    ns = parser.parse_args(argv)

    pos, kw = [], {}
    for a in ns.args:
        if "=" in a and not a.startswith("/"):
            key, _, val = a.partition("=")
            kw[key] = int(val) if val.isdigit() else val
        else:
            pos.append(a)

    started = time.perf_counter()
    try:
        result = REGISTRY[ns.tool](*pos, **kw)
    except ToolError as exc:
        print(json.dumps(exc.as_tool_result(), indent=2))
        return 1
    except TypeError as exc:
        print(f"Wrong arguments for {ns.tool}: {exc}", file=sys.stderr)
        return 2

    elapsed = time.perf_counter() - started
    print(json.dumps(result, indent=2))
    print(f"\n[{ns.tool} took {elapsed:.2f}s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
