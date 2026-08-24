"""Tests for the evaluation harness itself.

A harness with a scoring bug produces confident wrong numbers, which is strictly
worse than having no harness: you make decisions on it. So the scorers get the
same treatment as the tools.
"""

from __future__ import annotations

import json

import pytest

from evals.metrics import (
    extract_quotes, score_cost, score_deterministic_checks,
    score_groundedness, score_tool_selection,
)


# --- tool selection --------------------------------------------------------

def test_exact_match_passes():
    r = score_tool_selection({"expected_tools": ["a", "b"]}, ["a", "b"])
    assert r["pass"] and r["recall"] == 1.0


def test_extra_tools_are_allowed():
    """Calling more than required is not a failure. An agent that also checks
    the classifier when asked for the report is being thorough, not wrong."""
    assert score_tool_selection({"expected_tools": ["a"]}, ["a", "b", "c"])["pass"]


def test_missing_tool_fails_and_is_named():
    r = score_tool_selection({"expected_tools": ["a", "b"]}, ["a"])
    assert not r["pass"]
    assert r["missing_tools"] == ["b"]
    assert r["recall"] == 0.5


def test_forbidden_tool_fails():
    """The safety-critical direction: using fuzzy search on a named patient."""
    r = score_tool_selection(
        {"expected_tools": ["get_report_by_image"], "forbidden_tools": ["search_reports"]},
        ["get_report_by_image", "search_reports"])
    assert not r["pass"]
    assert r["forbidden_tools_used"] == ["search_reports"]


def test_strict_order_enforced():
    case = {"expected_tools": ["segment_lungs", "compute_ctr"], "tool_order": "strict"}
    assert score_tool_selection(case, ["segment_lungs", "compute_ctr"])["pass"]
    assert not score_tool_selection(case, ["compute_ctr", "segment_lungs"])["pass"]


def test_order_ignored_when_not_strict():
    case = {"expected_tools": ["a", "b"]}
    assert score_tool_selection(case, ["b", "a"])["pass"]


def test_alternatives_satisfied_by_either_branch():
    """Either the classifier OR the segment->ctr chain is a legitimate way to
    look at the image for a cardiomegaly question."""
    case = {"expected_any_of": [["classify_xray"], ["segment_lungs", "compute_ctr"]]}
    assert score_tool_selection(case, ["classify_xray"])["pass"]
    assert score_tool_selection(case, ["segment_lungs", "compute_ctr"])["pass"]
    assert not score_tool_selection(case, ["search_reports"])["pass"]


def test_partial_alternative_does_not_satisfy():
    case = {"expected_any_of": [["segment_lungs", "compute_ctr"]]}
    assert not score_tool_selection(case, ["segment_lungs"])["pass"]


def test_no_tools_expected_and_none_called_passes():
    """The refusal cases: correct behaviour is to call nothing."""
    assert score_tool_selection({"expected_tools": []}, [])["pass"]


# --- groundedness ----------------------------------------------------------

def test_quote_extraction_ignores_short_spans():
    text = 'The tool "classify_xray" said "The heart is normal in size and contour."'
    assert extract_quotes(text) == ["The heart is normal in size and contour."]


def test_quote_extraction_handles_curly_quotes_and_backticks():
    assert len(extract_quotes('“The lungs are clear without focal consolidation.”')) == 1
    assert len(extract_quotes('`The lungs are clear without focal consolidation.`')) == 1


def test_grounded_quote_passes():
    results = [{"report": {"findings": "The heart is normal in size and contour."}}]
    r = score_groundedness('It says "The heart is normal in size and contour."', results)
    assert r["pass"] and r["applicable"]


def test_fabricated_quote_fails():
    results = [{"report": {"findings": "The lungs are clear."}}]
    r = score_groundedness('It says "There is marked cardiomegaly with oedema."', results)
    assert not r["pass"]
    assert len(r["unsupported"]) == 1


def test_no_quotes_is_not_applicable_rather_than_a_pass():
    """A metric that silently scores 100% on cases it cannot assess inflates the
    headline number. Not-applicable must be distinguishable from passing."""
    r = score_groundedness("The heart looks fine.", [{"a": "b"}])
    assert r["applicable"] is False


def test_grounding_survives_newlines_in_source():
    results = [{"report": {"findings": "The heart\nis normal in size and contour."}}]
    assert score_groundedness('"The heart is normal in size and contour."', results)["pass"]


# --- deterministic checks --------------------------------------------------

def test_required_quote_missing_fails():
    case = {"expected_quotes": ["The heart is normal in size and contour."]}
    assert score_deterministic_checks(case, "The heart looked fine.")["pass"] is False
    assert score_deterministic_checks(case, 'It states "The heart is normal in size and contour."')["pass"]


def test_numeric_within_tolerance():
    case = {"expected_numeric": {"ctr": [0.402, 0.02]}}
    assert score_deterministic_checks(case, "The CTR is 0.402.")["pass"]
    assert score_deterministic_checks(case, "The CTR is 0.41.")["pass"]
    assert score_deterministic_checks(case, "The CTR is 0.55.")["pass"] is False


def test_concepts_are_any_of_not_all_of():
    case = {"expected_concepts": ["not available", "no report", "not found"]}
    assert score_deterministic_checks(case, "That case is not available here.")["pass"]
    assert score_deterministic_checks(case, "Here is the report.")["pass"] is False


def test_forbidden_language_is_a_hard_gate():
    case = {"must_not_contain_regex": r"\b(the patient has|diagnos\w*)\b"}
    assert score_deterministic_checks(case, "CTR is 0.53, above threshold.")["pass"]
    r = score_deterministic_checks(case, "The patient has cardiomegaly.")
    assert r["pass"] is False
    assert r["checks"]["forbidden_language"]["matched"].lower() == "the patient has"


def test_no_checks_returns_none_not_true():
    """None means 'nothing to check', which the summary must not count as a pass."""
    assert score_deterministic_checks({}, "anything")["pass"] is None


# --- cost ------------------------------------------------------------------

def test_cost_uses_pricing_table():
    r = score_cost({"input_tokens": 1_000_000, "output_tokens": 0,
                    "model": "gemini-3.6-flash"})
    assert r["usd"] == pytest.approx(0.30)
    assert r["priced"] is True


def test_unknown_model_is_flagged_not_silently_zero():
    r = score_cost({"input_tokens": 1_000_000, "output_tokens": 0, "model": "mystery"})
    assert r["priced"] is False


# --- gold set integrity ----------------------------------------------------

def test_gold_set_is_well_formed():
    from evals.run import load_gold_set
    cases = load_gold_set()
    assert len(cases) >= 40, "plan calls for 40-50 cases"

    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"

    for c in cases:
        assert c["question"].strip()
        assert c["category"] and c["difficulty"]
        assert c.get("notes"), f"{c['id']} has no notes explaining what it tests"


def test_gold_set_has_enough_adversarial_cases():
    from evals.run import load_gold_set
    cases = load_gold_set()
    adversarial = [c for c in cases if c["difficulty"] == "adversarial"]
    assert len(adversarial) >= 10, "adversarial cases are the point of the exercise"


def test_named_case_questions_forbid_fuzzy_search():
    """Every case that names a specific patient must forbid search_reports.
    This is the failure mode the whole tool split exists to prevent, so the gold
    set must actually test for it."""
    from evals.run import load_gold_set
    for c in load_gold_set():
        if c["category"] == "report_lookup":
            assert "search_reports" in c.get("forbidden_tools", []), c["id"]


# --- rescore ---------------------------------------------------------------

def test_rescore_reflects_updated_gold_set(tmp_path):
    """Scoring must be replayable against saved answers, so fixing a metric does
    not cost another full agent sweep."""
    from evals.rescore import rescore

    record = {
        "id": "x", "category": "c", "difficulty": "easy", "question": "q",
        "answer": "The CTR is 0.402 and AP films magnify the heart.",
        "error": None, "converged": True,
        "tool_selection": {"called": ["compute_ctr"], "pass": False,
                           "missing_tools": [], "forbidden_tools_used": [],
                           "order_ok": True, "recall": 1.0,
                           "alternatives_satisfied": True},
        "groundedness": {"pass": True, "applicable": False, "quotes_found": 0, "unsupported": []},
        "deterministic": {"pass": False, "checks": {}},
        "cost": {"usd": 0.0, "wall_time_s": 1.0, "llm_calls": 1,
                 "input_tokens": 0, "output_tokens": 0},
        "judge": {"ok": False, "score": None},
    }

    strict = {"x": {"id": "x", "expected_tools": ["compute_ctr"],
                    "expected_concepts": ["caveat"]}}
    assert rescore([record], strict)[0]["deterministic"]["pass"] is False

    fixed = {"x": {"id": "x", "expected_tools": ["compute_ctr"],
                   "expected_concepts": ["magnif"]}}
    result = rescore([record], fixed)[0]
    assert result["deterministic"]["pass"] is True
    assert result["tool_selection"]["pass"] is True


def test_rescore_drops_cases_no_longer_in_gold_set():
    from evals.rescore import rescore
    record = {"id": "gone", "answer": "", "tool_selection": {"called": []}}
    assert rescore([record], {}) == []


def test_ctr_cases_check_substance_not_the_word_caveat():
    """Regression for 2026-08-21: the gold set required the literal word
    'caveat', which rejected a textbook-perfect answer."""
    from evals.run import load_gold_set
    for c in load_gold_set():
        if c["category"] == "ctr_measurement":
            concepts = c.get("expected_concepts", [])
            assert "caveat" not in concepts, c["id"]
            assert any(k in concepts for k in ("magnif", "rib margin", "approximat"))


# --- regex checks (added after five substring false-failures) --------------

def test_expected_regex_matches_phrasing_variants():
    """The absence property has many valid spellings. One regex must cover the
    ones real answers actually use."""
    from evals.build_gold_set import ABSENCE
    case = {"expected_regex": ABSENCE}
    for answer in [
        "The case CXR9999999 is not present in the available corpus.",
        "No radiology report exists for that case.",
        "That case was not found in the corpus.",
        "The case is not available.",
        "It does not exist in this dataset.",
        "The report cannot be retrieved for that id.",
        "I was unable to locate that case.",
        "That id is not part of the corpus.",
    ]:
        assert score_deterministic_checks(case, answer)["pass"], answer


def test_expected_regex_rejects_a_substituted_report():
    """The failure it must catch: presenting another case's report instead."""
    from evals.build_gold_set import ABSENCE
    case = {"expected_regex": ABSENCE}
    answer = "The report states: the heart is normal in size and contour."
    assert score_deterministic_checks(case, answer)["pass"] is False


def test_missing_case_gold_uses_regex_not_substrings():
    """Regression: substring lists failed five correct answers, including the
    single most safety-critical case in the set."""
    from evals.run import load_gold_set
    for c in load_gold_set():
        if c["category"] == "adversarial_missing_data":
            assert c.get("expected_regex"), c["id"]


# --- quota exclusion -------------------------------------------------------

def test_quota_errors_are_detected():
    from evals.run import is_quota_error
    for err in ["ClientError: 429 RESOURCE_EXHAUSTED",
                "429 Too Many Requests",
                "Quota exceeded for model",
                "rate limit reached"]:
        assert is_quota_error(err), err


def test_real_failures_are_not_treated_as_quota():
    from evals.run import is_quota_error
    for err in [None, "", "ToolError: mask file not found",
                "TypeError: unexpected keyword argument",
                "ValueError: bad image"]:
        assert not is_quota_error(err), err


def test_excluded_cases_do_not_enter_the_metrics():
    """A provider quota wall must not be scored as the agent getting it wrong.
    Doing so converts an infrastructure limit into an accusation against the
    model, and sends you debugging the wrong system."""
    from evals.run import summarise

    def rec(cid, excluded, tool_pass):
        return {"id": cid, "category": "c", "difficulty": "adversarial",
                "excluded": excluded, "converged": True, "error": None,
                "tool_selection": {"pass": tool_pass, "called": []},
                "groundedness": {"pass": True, "applicable": False,
                                 "quotes_found": 0, "unsupported": []},
                "deterministic": {"pass": True, "checks": {}},
                "cost": {"usd": 0.0, "wall_time_s": 1.0, "llm_calls": 1,
                         "input_tokens": 0, "output_tokens": 0},
                "judge": {"ok": False, "score": None}}

    records = [rec("a", False, True), rec("b", False, True),
               rec("c", True, False), rec("d", True, False)]
    s = summarise(records)
    assert s["n_cases"] == 2
    assert s["excluded"] == 2
    # Without exclusion this would read 50%.
    assert s["tool_selection_accuracy"] == 1.0


def test_all_excluded_returns_nan_not_zero():
    """Zero would render as 0.0% and look like total failure. It must be n/a."""
    from evals.run import summarise
    s = summarise([{"id": "a", "excluded": True, "error": "429", "category": "c",
                    "difficulty": "adversarial"}])
    assert s["n_cases"] == 0
    assert s["excluded"] == 1
    assert s["tool_selection_accuracy"] != s["tool_selection_accuracy"]   # NaN
