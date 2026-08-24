"""Run the real agent loop against the real tools, with a scripted LLM.

Everything here is production code except the model itself: real dispatch, real
DenseNet and PSPNet inference, real BM25 over 3,826 reports, real trace file.
Only the "which tool shall I call next" decision is scripted.

This is how you verify the whole chain works before spending a single token, and
how you reproduce an agent run deterministically when you are debugging one.

    python scripts/demo_offline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radreport.agent import run, run_structured                    # noqa: E402
from radreport.config import IMAGE_DIR                            # noqa: E402
from radreport.llm import FakeProvider, LLMResponse, ToolCall     # noqa: E402


def tc(name, **args):
    return ToolCall(id=f"c_{name}", name=name, arguments=args)


def main() -> int:
    image = next(IMAGE_DIR.glob("*.dcm.png"), None) or (IMAGE_DIR / "sample_cxr.png")
    if not image.exists():
        print("No images. Run: python scripts/fetch_data.py")
        return 1
    case_id = image.name.replace(".dcm.png", "")

    print(f"Case: {case_id}\n")

    # The tool sequence a competent model should choose for a cardiomegaly
    # question: look at the image, measure it, then check what was written.
    script = [
        LLMResponse(tool_calls=[tc("classify_xray", image_path=case_id)],
                    input_tokens=900, output_tokens=18, latency_s=0.6),
        LLMResponse(tool_calls=[tc("segment_lungs", image_path=case_id)],
                    input_tokens=1400, output_tokens=20, latency_s=0.5),
        LLMResponse(tool_calls=[tc("compute_ctr", mask_handle="PLACEHOLDER")],
                    input_tokens=1900, output_tokens=22, latency_s=0.5),
        LLMResponse(tool_calls=[tc("get_report_by_image", image_id=case_id)],
                    input_tokens=2300, output_tokens=19, latency_s=0.4),
        LLMResponse(text="(scripted final answer)",
                    input_tokens=2800, output_tokens=180, latency_s=1.1),
    ]

    # The placeholder stands in for the real chaining a live model does: it reads
    # mask_handle out of the previous tool result. We patch it in the same way,
    # by watching the run. Simplest honest approach: run segmentation first to
    # learn the handle, then script it.
    from radreport.tools import segment_lungs, get_report_by_image, compute_ctr
    handle = segment_lungs(str(image))["mask_handle"]
    script[2].tool_calls[0].arguments["mask_handle"] = handle

    result = run(f"Does {case_id} show cardiomegaly, and what does the report say?",
                 provider=FakeProvider(script))

    print("tool sequence :", " -> ".join(result["tool_sequence"]))
    print("converged     :", result["converged"])
    print("llm calls     :", result["llm_calls"])
    print("tokens        :", result["input_tokens"], "in /", result["output_tokens"], "out")
    print("wall time     :", result["wall_time_s"], "s")
    print("trace         :", result["trace_path"])

    print("\n--- what the model actually received from the tools ---")
    for line in open(result["trace_path"]):
        e = json.loads(line)
        if e["event"] == "tool_call":
            status = "ok" if e["ok"] else f"FAILED: {e['error']}"
            print(f"  {e['tool']:<22} {e['latency_s']:>6.2f}s  {status}")

    # ---- structured output, with a deliberately fabricated quote -----------
    print("\n=== structured path (note: quote below is INVENTED on purpose) ===")
    report = get_report_by_image(case_id)
    real_quote = report["report"]["impression"] if report["found"] else ""

    ctr = compute_ctr(handle)["ctr"]
    honest = json.dumps({
        "summary": f"CTR measured {ctr}. The report impression is quoted below.",
        "case_id": case_id,
        "findings": [{
            "label": "Cardiothoracic ratio",
            "present": "not_suggested" if ctr <= 0.5 else "suggested",
            "ctr_value": ctr,
            "evidence": [
                {"source": "measurement", "detail": f"compute_ctr: ctr={ctr}"},
                {"source": "report", "detail": "get_report_by_image", "quote": real_quote},
            ],
        }],
    })
    fabricated = json.dumps({
        "summary": "The report describes marked cardiac enlargement.",
        "findings": [{
            "label": "Cardiomegaly", "present": "suggested",
            "evidence": [{"source": "report", "detail": "get_report_by_image",
                          "quote": "There is marked cardiomegaly with pulmonary oedema."}],
        }],
    })

    for name, payload in [("honest answer", honest), ("fabricated quote", fabricated)]:
        script2 = [
            LLMResponse(tool_calls=[tc("get_report_by_image", image_id=case_id)]),
            LLMResponse(tool_calls=[tc("compute_ctr", mask_handle=handle)]),
            LLMResponse(text="(prose answer)"),
            LLMResponse(text=payload),
        ]
        out = run_structured(f"Summarise {case_id}", provider=FakeProvider(script2))
        g = out["grounding"]
        print(f"  {name:<20} valid={out['structured'] is not None}  "
              f"grounded={g['grounded']}  checked={g['quotes_checked']}")
        for u in g["unsupported_quotes"]:
            print(f"      REJECTED: {u['quote'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
