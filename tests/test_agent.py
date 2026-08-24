"""Tests for the agent loop, using FakeProvider.

Why not test against a real model: that tests the model, not the loop. It is
slow, burns quota, and is non-deterministic, so a red test tells you nothing.
With a scripted provider we can construct exactly the situations that matter and
that a real model produces only occasionally -- a hallucinated tool name, a bad
argument, a model that never stops calling tools.
"""

from __future__ import annotations

import json

import pytest

from radreport.agent import (
    TOOLS, Trace, dispatch, resolve_image_path, run,
)
from radreport.llm import FakeProvider, LLMResponse, ToolCall


def call(name, **args):
    return ToolCall(id=f"c_{name}", name=name, arguments=args)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def test_every_schema_is_well_formed():
    for t in TOOLS:
        assert set(t) == {"name", "description", "parameters"}
        assert t["parameters"]["type"] == "object"
        for req in t["parameters"]["required"]:
            assert req in t["parameters"]["properties"], f"{t['name']}: {req} not described"


def test_confusable_tools_say_when_not_to_use_them():
    """search_reports and get_report_by_image are the pair most likely to be
    confused, and confusing them means reporting another patient's report. Both
    descriptions must contain an explicit negative instruction."""
    by_name = {t["name"]: t["description"] for t in TOOLS}
    assert "Never use it" in by_name["search_reports"]
    assert "get_report_by_image" in by_name["search_reports"]
    assert "must NOT" in by_name["get_report_by_image"]


def test_compute_ctr_schema_states_its_precondition():
    desc = next(t for t in TOOLS if t["name"] == "compute_ctr")["description"]
    assert "segment_lungs" in desc
    assert "plausible=false" in desc


# ---------------------------------------------------------------------------
# dispatch: it must never raise
# ---------------------------------------------------------------------------

def test_unknown_tool_lists_the_real_ones():
    result = dispatch("diagnose_patient", {})
    assert result["ok"] is False
    assert "No tool named" in result["error"]
    assert "classify_xray" in result["error"]     # tells the model what exists


def test_tool_error_becomes_a_result_not_an_exception(tmp_path):
    result = dispatch("compute_ctr", {"mask_handle": str(tmp_path / "nope.npz")})
    assert result["ok"] is False
    assert "Call segment_lungs first" in result["error"]   # actionable message


def test_wrong_arguments_echo_the_signature():
    # The model invented a keyword that does not exist on the function. This is
    # a real and common failure: it read "k" from the schema as "top_k".
    result = dispatch("search_reports", {"query": "effusion", "top_k": 3})
    assert result["ok"] is False
    assert "Wrong arguments" in result["error"]
    assert "search_reports(" in result["error"]


def test_dispatch_never_raises_on_garbage():
    for name, args in [("classify_xray", {}), ("search_reports", {"query": None}),
                       ("compute_ctr", {"mask_handle": 12345}), ("nope", {"x": 1})]:
        result = dispatch(name, args)
        assert result["ok"] is False, (name, args)
        assert isinstance(json.dumps(result), str)   # must stay serialisable


def test_image_path_resolution_falls_through_unchanged():
    assert resolve_image_path("definitely-not-here") == "definitely-not-here"


def test_image_path_resolves_bare_case_id():
    from radreport.config import IMAGE_DIR
    real = next(IMAGE_DIR.glob("*.dcm.png"), None)
    if real is None:
        pytest.skip("no dataset images present")
    stem = real.name.replace(".dcm.png", "")
    assert resolve_image_path(stem) == str(real)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def test_answers_without_tools(tmp_path):
    fake = FakeProvider([LLMResponse(text="Hello.")])
    out = run("hi", provider=fake, trace_dir=tmp_path)
    assert out["answer"] == "Hello."
    assert out["converged"] is True
    assert out["tool_sequence"] == []
    assert out["llm_calls"] == 1


def test_single_tool_round_trip(tmp_path):
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("search_reports", query="pleural effusion", k=1)]),
        LLMResponse(text="Found a matching report."),
    ])
    out = run("find effusion cases", provider=fake, trace_dir=tmp_path)
    assert out["tool_sequence"] == ["search_reports"]
    assert out["converged"] is True


def test_model_sees_both_its_request_and_the_result(tmp_path):
    """The classic bug this guards against: appending only the tool RESULT. The
    model then has no record of having asked, and loops."""
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("search_reports", query="pneumonia")]),
        LLMResponse(text="done"),
    ])
    run("q", provider=fake, trace_dir=tmp_path)

    second_request = fake.calls[1]["messages"]
    roles = [m["role"] for m in second_request]
    assert roles == ["user", "assistant", "tool"]
    assert second_request[1]["tool_calls"][0]["name"] == "search_reports"
    assert second_request[2]["name"] == "search_reports"


def test_system_prompt_is_sent_every_call(tmp_path):
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("search_reports", query="x")]),
        LLMResponse(text="done"),
    ])
    run("q", provider=fake, trace_dir=tmp_path)
    assert all("EVIDENCE RULE" in c["system"] for c in fake.calls)


def test_parallel_tool_calls_in_one_turn(tmp_path):
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("search_reports", query="a"),
                                call("search_literature", query="b")]),
        LLMResponse(text="done"),
    ])
    out = run("q", provider=fake, trace_dir=tmp_path)
    assert out["tool_sequence"] == ["search_reports", "search_literature"]


def test_max_iterations_is_a_failure_not_an_answer(tmp_path):
    """A model stuck in a loop must produce converged=False, so the Weekend 4
    eval counts it as a failure rather than scoring a partial output."""
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("search_reports", query="loop")]) for _ in range(10)
    ])
    out = run("q", provider=fake, trace_dir=tmp_path, max_iterations=3)
    assert out["converged"] is False
    assert "did not reach an answer" in out["answer"]
    assert len(out["tool_sequence"]) == 3
    assert fake.calls and len(fake.calls) == 3   # it stopped calling the API


def test_agent_recovers_from_a_tool_error(tmp_path):
    """compute_ctr before segment_lungs: the error message tells the model what
    to do, and the loop lets it act on that."""
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("compute_ctr", mask_handle="invented.npz")]),
        LLMResponse(tool_calls=[call("search_reports", query="recovered")]),
        LLMResponse(text="Recovered and answered."),
    ])
    out = run("q", provider=fake, trace_dir=tmp_path)
    assert out["converged"] is True
    assert out["tool_sequence"] == ["compute_ctr", "search_reports"]

    error_msg = fake.calls[1]["messages"][2]["content"]
    assert error_msg["ok"] is False
    assert "segment_lungs" in error_msg["error"]


def test_hallucinated_tool_does_not_crash_the_run(tmp_path):
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("diagnose", patient="x")]),
        LLMResponse(text="I cannot diagnose."),
    ])
    out = run("diagnose this", provider=fake, trace_dir=tmp_path)
    assert out["converged"] is True


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

def test_trace_file_is_valid_jsonl(tmp_path):
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("search_reports", query="x")], input_tokens=100,
                    output_tokens=10, latency_s=0.5),
        LLMResponse(text="done", input_tokens=200, output_tokens=20, latency_s=0.3),
    ])
    out = run("q", provider=fake, trace_dir=tmp_path)

    lines = [json.loads(l) for l in open(out["trace_path"]) if l.strip()]
    assert [l["event"] for l in lines] == ["start", "llm_call", "tool_call", "llm_call", "final"]
    assert all("ts" in l and "run_id" in l for l in lines)


def test_trace_totals_feed_the_cost_metric(tmp_path):
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("search_reports", query="x")],
                    input_tokens=100, output_tokens=10),
        LLMResponse(text="done", input_tokens=200, output_tokens=20),
    ])
    out = run("q", provider=fake, trace_dir=tmp_path)
    assert out["input_tokens"] == 300
    assert out["output_tokens"] == 30
    assert out["llm_calls"] == 2


def test_trace_records_failed_tool_calls(tmp_path):
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("compute_ctr", mask_handle="nope.npz")]),
        LLMResponse(text="done"),
    ])
    out = run("q", provider=fake, trace_dir=tmp_path)
    lines = [json.loads(l) for l in open(out["trace_path"]) if l.strip()]
    tool_event = next(l for l in lines if l["event"] == "tool_call")
    assert tool_event["ok"] is False
    assert tool_event["error"]


# ---------------------------------------------------------------------------
# Structured output (Weekend 3)
# ---------------------------------------------------------------------------

from radreport.agent import _extract_json, run_structured   # noqa: E402


def test_extract_json_strips_markdown_fences():
    for wrapped in ['```json\n{"a": 1}\n```', '```\n{"a": 1}\n```',
                    '{"a": 1}', 'Here it is:\n{"a": 1}\nhope that helps']:
        assert json.loads(_extract_json(wrapped)) == {"a": 1}


VALID_JSON = json.dumps({
    "summary": "CTR is 0.53, above the conventional 0.50 threshold.",
    "findings": [{
        "label": "Cardiomegaly", "present": "suggested", "ctr_value": 0.528,
        "evidence": [{"source": "measurement", "detail": "compute_ctr: ctr=0.528"}],
    }],
})


def test_structured_answer_validates(tmp_path):
    fake = FakeProvider([
        LLMResponse(text="CTR is 0.53."),          # the prose loop ends
        LLMResponse(text=VALID_JSON),              # the structuring turn
    ])
    out = run_structured("q", provider=fake, trace_dir=tmp_path)
    assert out["validation_error"] is None
    assert out["structured"]["findings"][0]["ctr_value"] == 0.528
    assert "Not a medical device" in out["structured"]["disclaimer"]
    assert out["repair_attempts"] == 0


def test_invalid_json_is_repaired_using_the_validation_error(tmp_path):
    """The repair must feed the ACTUAL Pydantic error back. Retrying with the
    same prompt would just reroll the dice."""
    broken = json.dumps({"summary": "Here is what I found."})   # no findings, no refusal
    fake = FakeProvider([
        LLMResponse(text="prose answer"),
        LLMResponse(text=broken),
        LLMResponse(text=VALID_JSON),
    ])
    out = run_structured("q", provider=fake, trace_dir=tmp_path)
    assert out["validation_error"] is None
    assert out["repair_attempts"] == 1

    repair_prompt = fake.calls[-1]["messages"][-1]["content"]
    assert "failed validation" in repair_prompt
    assert "unanswerable" in repair_prompt      # the real error, not a generic nudge


def test_repair_budget_is_finite(tmp_path):
    broken = json.dumps({"summary": "nothing"})
    fake = FakeProvider([LLMResponse(text="prose")] + [LLMResponse(text=broken)] * 5)
    out = run_structured("q", provider=fake, trace_dir=tmp_path, max_repairs=1)
    assert out["structured"] is None
    assert out["validation_error"]


def test_diagnostic_language_is_rejected_then_repaired(tmp_path):
    unsafe = json.dumps({
        "summary": "The patient has cardiomegaly.",
        "findings": [{"label": "Cardiomegaly", "present": "suggested",
                      "evidence": [{"source": "measurement", "detail": "ctr=0.53"}]}],
    })
    fake = FakeProvider([
        LLMResponse(text="prose"),
        LLMResponse(text=unsafe),
        LLMResponse(text=VALID_JSON),
    ])
    out = run_structured("q", provider=fake, trace_dir=tmp_path)
    assert out["structured"]["summary"] == "CTR is 0.53, above the conventional 0.50 threshold."


def test_non_convergent_run_is_not_structured(tmp_path):
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("search_reports", query="loop")]) for _ in range(5)
    ])
    out = run_structured("q", provider=fake, trace_dir=tmp_path, max_iterations=2)
    assert out["converged"] is False
    assert out["structured"] is None


def test_grounding_runs_against_recorded_tool_results(tmp_path):
    """The whole point: a quote is checked against what the model ACTUALLY saw."""
    fabricated = json.dumps({
        "summary": "The report describes an enlarged heart.",
        "findings": [{
            "label": "Cardiomegaly", "present": "suggested",
            "evidence": [{"source": "report", "detail": "get_report_by_image",
                          "quote": "There is gross cardiomegaly."}],
        }],
    })
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("get_report_by_image", image_id="CXR-does-not-exist")]),
        LLMResponse(text="prose"),
        LLMResponse(text=fabricated),
    ])
    out = run_structured("q", provider=fake, trace_dir=tmp_path)
    assert out["structured"] is not None
    assert out["grounding"]["grounded"] is False
    assert out["grounding"]["unsupported_quotes"][0]["quote"] == "There is gross cardiomegaly."


def test_trace_records_tool_results_for_offline_scoring(tmp_path):
    """The claim is that the trace records the RESULT, not that the tool
    succeeded. Asserting ok=True made this depend on data/reports.csv, which is
    gitignored, so it passed locally and failed on a fresh clone. A tool that
    errors must have its result recorded too -- that is precisely the case
    Weekend 4's offline scoring needs to see."""
    fake = FakeProvider([
        LLMResponse(tool_calls=[call("search_reports", query="effusion", k=1)]),
        LLMResponse(text="done"),
    ])
    out = run("q", provider=fake, trace_dir=tmp_path)
    lines = [json.loads(l) for l in open(out["trace_path"]) if l.strip()]
    tool_event = next(l for l in lines if l["event"] == "tool_call")

    result = tool_event["result"]
    assert isinstance(result, dict) and "ok" in result
    assert result["ok"] == tool_event["ok"]        # trace flag matches the result
    if result["ok"]:
        assert "hits" in result                    # corpus present
    else:
        assert result["error"]                     # corpus absent: still recorded
