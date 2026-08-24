"""The agent loop.

An agent is a `while` loop around a chat API. The model cannot run code, open
files, or touch anything. All it can do is emit text. Tool calling is a
convention where, instead of prose, it emits a structured request -- "call
compute_ctr with mask_handle='...'" -- and THIS code decides whether and how to
honour it. Everything the agent can do is something we explicitly gave it.

Five parts, in the order they appear below:

  1. TOOL SCHEMAS   the model's only documentation for what it can call
  2. SYSTEM PROMPT  the rules that cannot be enforced in Python
  3. dispatch()     runs one tool, never raises, turns errors into results
  4. Trace          one JSONL line per event; Weekend 4's metrics read this
  5. run()          the loop
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from radreport.config import DISCLAIMER, IMAGE_DIR, TRACE_DIR
from radreport.llm import LLMProvider, LLMResponse, ToolCall, get_provider
from radreport.schema import AgentAnswer, json_schema_for_prompt, verify_grounding
from radreport.tools import REGISTRY, ToolError

MAX_ITERATIONS = 8


# ===========================================================================
# 1. TOOL SCHEMAS
# ===========================================================================
# The model has never seen our source. It knows exactly what these say and
# nothing more. When an agent misuses a tool, suspect the schema before the
# model: nine times out of ten something was under-described.
#
# Each description carries four things beyond naming the function:
#   - WHEN to use it and when NOT to (this is what stops tool confusion)
#   - where each argument comes from
#   - what the output MEANS, including units and thresholds
#   - the boundary: what the tool does not establish
#
# Kept as plain JSON Schema. radreport.llm wraps it for each vendor.

TOOLS: list[dict] = [
    {
        "name": "classify_xray",
        "description": (
            "Run a DenseNet-121 screening classifier on a chest X-ray and return "
            "a probability for each of 18 pathologies. Probabilities are "
            "calibrated so that 0.5 is the model's operating threshold for that "
            "label; they are relative screening scores, NOT per-patient risks and "
            "NOT a diagnosis. Use this to see which pathologies the image "
            "suggests. Do not use it to answer questions about what a "
            "radiologist wrote; use get_report_by_image for that."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": (
                        "Path to the chest X-ray image. If the user gave a case "
                        "id such as '702_IM-2267-1001', pass that id and it will "
                        "be resolved against the local image directory."
                    ),
                },
                "threshold": {
                    "type": "number",
                    "description": "Probability cutoff for the 'above_threshold' summary. Default 0.5.",
                },
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "segment_lungs",
        "description": (
            "Segment the lungs and heart on a chest X-ray using PSPNet. Returns "
            "the pixel area of each structure and a 'mask_handle' path. Call this "
            "BEFORE compute_ctr: the mask_handle it returns is compute_ctr's only "
            "input. Also returns 'overlay_png', an image for a human to inspect. "
            "This tool does not measure anything clinical on its own."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path or case id of the chest X-ray, as for classify_xray.",
                },
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "compute_ctr",
        "description": (
            "Compute the cardiothoracic ratio (CTR) from segmentation masks. "
            "REQUIRES a mask_handle produced by segment_lungs, so call that "
            "first. Returns a ratio: above 0.50 on an erect PA film may be "
            "consistent with cardiomegaly, but the film's view (PA vs AP) is "
            "unknown here and AP films magnify the heart, so this value must "
            "never be reported as a diagnosis of cardiomegaly. If the result has "
            "plausible=false the segmentation failed and the number must not be "
            "reported at all. Always report the caveats in the 'method' field "
            "alongside the number."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mask_handle": {
                    "type": "string",
                    "description": "The 'mask_handle' value from a segment_lungs result.",
                },
            },
            "required": ["mask_handle"],
        },
    },
    {
        "name": "get_report_by_image",
        "description": (
            "Look up the radiologist report for ONE SPECIFIC case id, by exact "
            "match. This is the correct tool whenever the user names a case, for "
            "example 'what does the report for 702_IM-2267-1001 say'. If the case "
            "is not in the corpus it returns found=false; when that happens you "
            "must tell the user the case is not available and you must NOT "
            "substitute a similar case from search_reports."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "Exact case id, e.g. '702_IM-2267-1001' or the study uid.",
                },
            },
            "required": ["image_id"],
        },
    },
    {
        "name": "search_reports",
        "description": (
            "Search 3,800 de-identified radiologist reports by free text and "
            "return the best lexical (BM25) matches. Use this ONLY to find "
            "similar or example cases by description, e.g. 'reports mentioning a "
            "large pleural effusion'. Never use it to answer a question about a "
            "specific named case; that is get_report_by_image. Reports retrieved "
            "here belong to OTHER patients. Matching is keyword-based, so it will "
            "not match 'enlarged heart' to 'cardiomegaly'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text clinical query."},
                "k": {"type": "integer", "description": "How many reports to return. Default 3."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_literature",
        "description": (
            "Search PubMed for published literature and return citations with "
            "PMIDs and URLs. Use this to ground a general clinical or "
            "methodological statement, for example what CTR threshold is "
            "conventionally used. Returns titles and metadata, not full "
            "abstracts, and says nothing about the patient in the image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "PubMed search query."},
                "k": {"type": "integer", "description": "How many citations, max 10. Default 3."},
            },
            "required": ["query"],
        },
    },
]

TOOL_NAMES = [t["name"] for t in TOOLS]


# ===========================================================================
# 2. SYSTEM PROMPT
# ===========================================================================
# Short on purpose. A long system prompt usually means a rule belongs in a tool
# instead. Anything enforceable in Python is enforced in Python: the CTR
# plausibility check and the exact-lookup/fuzzy-search split are code, not
# instructions, because a deterministic constraint should never be delegated to
# a language model. What is left here is genuinely only expressible in prose.

SYSTEM_PROMPT = f"""\
You are a radiology research assistant operating on public, de-identified chest \
X-rays. You are a research prototype, not a diagnostic device.

EVIDENCE RULE. Every clinical claim you make must be traceable to a tool result \
in this conversation: a named model output with its number, or a quoted line \
from a retrieved report. You have no independent knowledge of these patients. \
If no tool supports a claim, do not make it.

QUOTING. When you cite a report, quote it verbatim, including any 'XXXX' tokens \
(these mark where identifiers were removed during de-identification). Never \
paraphrase a quotation into something cleaner, and never insert emphasis or \
change punctuation inside quotation marks.

LITERATURE. search_literature returns titles, journals, years, authors and \
PMIDs. It does NOT return abstracts or full text. You have therefore not read \
these papers. You may cite a title and its PMID; you must NEVER put words in \
quotation marks and attribute them to a paper, never write "X et al. reported \
that ...", and never invent citation markers. If asked why something is true \
and you can only supply titles, say that: give the citations and state plainly \
that you are pointing to relevant literature rather than quoting findings from \
it.

SPECIFIC CASES. When the user names a case id, use get_report_by_image. If it \
returns found=false, say plainly that the case is not in the corpus. Never \
present another patient's report as if it were theirs.

MEASUREMENTS. Report CTR with its caveats: the view is unknown and AP films \
magnify the heart. Say what the number would mean IF the film is PA. Never state \
that a patient has cardiomegaly. If a result is marked plausible=false, say the \
measurement failed and do not report the value.

SCOPE. You do not diagnose, and you do not give management, treatment or \
prescribing advice. If asked for any of those, say that is outside what this \
tool does, and offer what you can support instead: the model outputs, the \
measurements and the report text.

Close every answer that contains a clinical statement with: {DISCLAIMER}
"""


# ===========================================================================
# 3. DISPATCH
# ===========================================================================

def resolve_image_path(value: str) -> str:
    """Turn whatever the model passed into a real path, if we can.

    The model will pass a bare case id far more often than a filesystem path,
    because that is how users refer to cases. Doing this resolution here rather
    than inside each tool keeps the tools honest (they take real paths and are
    testable as such) while making the agent forgiving of a natural input.
    """
    p = Path(value)
    if p.exists():
        return str(p)

    candidate = IMAGE_DIR / value
    if candidate.exists():
        return str(candidate)

    # A case id like '702_IM-2267-1001' with the .dcm.png suffix omitted.
    for suffix in (".dcm.png", ".png", ".jpg"):
        candidate = IMAGE_DIR / f"{value}{suffix}"
        if candidate.exists():
            return str(candidate)

    return value    # let the tool raise a proper ToolError with a clear message


def dispatch(name: str, arguments: dict) -> dict:
    """Run one tool call. ALWAYS returns a JSON-serialisable dict, never raises.

    This is the boundary that converts a Python exception into something the
    model can read and act on. If this function raised, one bad tool call would
    kill the whole run; because it does not, a bad call becomes a message the
    model can recover from. Four cases, most specific first.
    """
    # (a) The model invented a tool that does not exist.
    if name not in REGISTRY:
        return {
            "ok": False,
            "error": f"No tool named {name!r}. Available tools: {', '.join(TOOL_NAMES)}.",
            "recoverable": True,
        }

    arguments = dict(arguments or {})
    if "image_path" in arguments:
        arguments["image_path"] = resolve_image_path(str(arguments["image_path"]))

    try:
        return REGISTRY[name](**arguments)

    # (b) A failure we anticipated. The message is already written for the model
    #     and carries a recoverable flag telling it whether a retry is worth it.
    except ToolError as exc:
        exc.tool = exc.tool or name
        return exc.as_tool_result()

    # (c) Right tool, wrong arguments. Echo the real signature so the model can
    #     correct itself instead of guessing again.
    except TypeError as exc:
        import inspect
        try:
            sig = str(inspect.signature(REGISTRY[name]))
        except (TypeError, ValueError):
            sig = "(unavailable)"
        return {
            "ok": False,
            "error": f"Wrong arguments for {name}: {exc}. Expected signature: {name}{sig}",
            "recoverable": True,
        }

    # (d) Genuinely unexpected. Do NOT put a traceback in the model's context:
    #     it costs hundreds of tokens and tells the model nothing it can act on.
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{name} failed unexpectedly: {type(exc).__name__}: {exc}",
            "recoverable": False,
        }


# ===========================================================================
# 4. TRACE
# ===========================================================================
# Not logging. Logging is for a human when something breaks; a trace is
# structured data the evaluation harness reads. Weekend 4 computes tool-selection
# accuracy, groundedness, cost and latency by parsing these files, so the schema
# below is an interface. Keep it stable.
#
# JSONL rather than JSON because we append as we go, which means a run that
# crashes halfway still leaves a readable file -- exactly when you most want it.

class Trace:
    def __init__(self, run_id: str, trace_dir: Path | None = None):
        self.run_id = run_id
        self.path = Path(trace_dir or TRACE_DIR) / f"{run_id}.jsonl"
        self.events: list[dict] = []

    def write(self, event: str, iteration: int, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "iteration": iteration,
            "event": event,
            **fields,
        }
        self.events.append(record)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    @property
    def tool_sequence(self) -> list[str]:
        """The ordered list of tools called. This IS the Weekend 4 metric."""
        return [e["tool"] for e in self.events if e["event"] == "tool_call"]

    @property
    def totals(self) -> dict:
        llm = [e for e in self.events if e["event"] == "llm_call"]
        return {
            "llm_calls": len(llm),
            "input_tokens": sum(e.get("input_tokens", 0) for e in llm),
            "output_tokens": sum(e.get("output_tokens", 0) for e in llm),
            "latency_s": round(sum(e.get("latency_s", 0.0) for e in self.events), 3),
        }


# ===========================================================================
# 5. THE LOOP
# ===========================================================================

def run(user_message: str, *,
        provider: LLMProvider | None = None,
        provider_name: str = "gemini",
        max_iterations: int = MAX_ITERATIONS,
        trace_dir: Path | None = None) -> dict:
    """Run the agent to completion. Returns the answer, the tool sequence, the trace.

    The loop, in one sentence: send the conversation, and if the model asked for
    tools, run them, append both the request and the results, and send again;
    stop when it answers with text instead.

    THE SUBTLE PART. We append the model's tool-CALL turn as well as the tool
    RESULTS. Skipping the call turn is the classic bug: the model then sees an
    answer to a question it has no record of asking, and calls the same tool
    again, and again, until max_iterations fires. Both go in, in order.
    """
    provider = provider or get_provider(provider_name)
    run_id = uuid.uuid4().hex[:8]
    tracer = Trace(run_id, trace_dir)
    started = time.perf_counter()

    tracer.write("start", -1, question=user_message,
                 provider=provider.name, model=provider.model,
                 max_iterations=max_iterations)

    # "Memory" is just this list. The API is stateless; every call sends it all.
    messages: list[dict] = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        response: LLMResponse = provider.chat(messages, TOOLS, SYSTEM_PROMPT)

        tracer.write("llm_call", iteration,
                     model=response.model,
                     input_tokens=response.input_tokens,
                     output_tokens=response.output_tokens,
                     latency_s=round(response.latency_s, 3),
                     cached=response.cached,
                     n_tool_calls=len(response.tool_calls))

        # --- exit condition: the model answered instead of calling a tool ---
        if not response.wants_tools:
            tracer.write("final", iteration, answer=response.text,
                         total_iterations=iteration + 1, converged=True)
            return _result(response.text, tracer, started, converged=True)

        # --- append the model's REQUEST, then run the tools ---
        messages.append({
            "role": "assistant",
            "content": response.text,
            "tool_calls": response.tool_calls,
        })

        for call in response.tool_calls:
            t0 = time.perf_counter()
            result = dispatch(call.name, call.arguments)
            elapsed = time.perf_counter() - t0

            tracer.write("tool_call", iteration,
                         tool=call.name,
                         arguments=call.arguments,
                         ok=bool(result.get("ok", False)),
                         error=result.get("error"),
                         latency_s=round(elapsed, 3),
                         # The full result goes in the trace on purpose. It makes
                         # the trace self-contained, so Weekend 4 can score
                         # groundedness offline from trace files alone without
                         # re-running inference. Costs a few KB per run.
                         result=result)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": result,
            })

    # --- the guard fired: this is a FAILURE, not a result ---------------------
    # Do not quietly return whatever text happened to be lying around. Weekend 4
    # must be able to count non-convergence as a failure rather than score
    # whatever the last partial output was.
    tracer.write("final", max_iterations,
                 answer=None, total_iterations=max_iterations, converged=False,
                 reason="max_iterations exhausted")
    return _result(
        f"The agent did not reach an answer within {max_iterations} steps. "
        f"Tools attempted: {' -> '.join(tracer.tool_sequence) or 'none'}.",
        tracer, started, converged=False,
    )


def _result(answer: str, tracer: Trace, started: float, *, converged: bool) -> dict:
    return {
        "answer": answer,
        "converged": converged,
        "run_id": tracer.run_id,
        "tool_sequence": tracer.tool_sequence,
        "trace_path": str(tracer.path),
        "wall_time_s": round(time.perf_counter() - started, 3),
        **tracer.totals,
    }


if __name__ == "__main__":
    import sys
    print(json.dumps(run(" ".join(sys.argv[1:])), indent=2))


# ===========================================================================
# 6. STRUCTURED OUTPUT  (Weekend 3)
# ===========================================================================
# run() returns prose. run_structured() returns a validated AgentAnswer.
#
# Why a second function rather than changing run(): the loop and the output
# contract are separate concerns, and keeping them separate means the loop stays
# testable on its own and the eval can measure them independently -- "did it pick
# the right tools" is a different question from "did it format and ground the
# answer correctly".

STRUCTURE_INSTRUCTION = """\
Now produce your final answer as a single JSON object matching this schema. \
Output ONLY the JSON, with no markdown fence and no commentary.

Rules that the validator enforces, so getting them wrong costs you a retry:
  - every finding needs at least one evidence entry
  - evidence with source "report" MUST include the exact verbatim quote
  - evidence with source "literature" MUST include the PMID as citation
  - a ctr_value requires evidence with source "measurement"
  - "present" is one of: suggested, not_suggested, indeterminate
  - no diagnostic or prescriptive language anywhere
  - if you have no findings, set unanswerable=true or give a refusal_reason

SCHEMA:
%s
"""


def _extract_json(text: str) -> str:
    """Pull a JSON object out of a model response.

    Models fence JSON in ```json blocks roughly half the time however clearly you
    ask them not to. Stripping it here is cheaper than spending a retry on it.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def run_structured(user_message: str, *,
                   provider: LLMProvider | None = None,
                   provider_name: str = "gemini",
                   max_iterations: int = MAX_ITERATIONS,
                   trace_dir: Path | None = None,
                   max_repairs: int = 1) -> dict:
    """Run the agent, then force the answer into a validated AgentAnswer.

    Three phases:
      1. the normal tool-calling loop, until the model answers in prose
      2. one more turn asking for that answer as JSON against our schema
      3. validate; on failure, hand the VALIDATION ERROR back and retry once

    Phase 3 matters more than it looks. Pydantic's error messages name the field
    and the rule that failed, which is precisely the feedback a model needs to
    fix its own output. Retrying with the same prompt would just reroll the dice;
    retrying with the error is a correction.
    """
    provider = provider or get_provider(provider_name)
    base = run(user_message, provider=provider,
               max_iterations=max_iterations, trace_dir=trace_dir)

    tracer = Trace(base["run_id"], trace_dir)

    if not base["converged"]:
        return {**base, "structured": None, "grounding": None,
                "validation_error": "run did not converge; no answer to structure"}

    # Rebuild the tool results from the trace so grounding can be verified and
    # so the structuring turn can see the evidence it must quote.
    tool_results = _tool_results_from_trace(Path(base["trace_path"]))

    messages: list[dict] = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": base["answer"]},
        {"role": "user", "content": STRUCTURE_INSTRUCTION
            % json.dumps(json_schema_for_prompt(), indent=2)},
    ]

    last_error = None
    for attempt in range(max_repairs + 1):
        # No tools on these turns: we want JSON, not another tool call.
        response = provider.chat(messages, None, SYSTEM_PROMPT)
        tracer.write("structure_attempt", attempt,
                     model=response.model,
                     input_tokens=response.input_tokens,
                     output_tokens=response.output_tokens,
                     latency_s=round(response.latency_s, 3))

        raw = _extract_json(response.text)
        try:
            answer = AgentAnswer.model_validate_json(raw)
        except Exception as exc:
            last_error = str(exc)
            tracer.write("structure_invalid", attempt, error=last_error[:1000])
            if attempt == max_repairs:
                break
            # Feed the actual validation error back. This is the repair.
            messages.append({"role": "assistant", "content": response.text})
            messages.append({"role": "user", "content":
                f"That JSON failed validation:\n{last_error}\n\n"
                "Fix ONLY what the error names and return the corrected JSON."})
            continue

        grounding = verify_grounding(answer, tool_results)
        tracer.write("structure_valid", attempt,
                     grounded=grounding["grounded"],
                     quotes_checked=grounding["quotes_checked"],
                     unsupported=len(grounding["unsupported_quotes"]))

        return {**base,
                "structured": answer.model_dump(mode="json"),
                "grounding": grounding,
                "validation_error": None,
                "repair_attempts": attempt}

    tracer.write("structure_failed", max_repairs, error=last_error)
    return {**base, "structured": None, "grounding": None,
            "validation_error": last_error, "repair_attempts": max_repairs}


def _tool_results_from_trace(path: Path) -> list[dict]:
    """Read back what the tools returned, for grounding verification.

    Reads the recorded results rather than replaying the calls. Replaying would
    re-run DenseNet just to check a quote, and worse, it would verify the answer
    against a FRESH tool run rather than against what the model actually saw.
    Those can differ, and when they do, the grounding check is meaningless.
    """
    results = []
    if not path.exists():
        return results
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "tool_call" and event.get("result") is not None:
            results.append(event["result"])
    return results
