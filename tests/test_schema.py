"""Tests for structured output. These are safety rules, so they get tested as
rules: each test names the clinical failure it prevents."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from radreport.schema import AgentAnswer, Evidence, Finding, verify_grounding


def measurement(detail="compute_ctr: ctr=0.528"):
    return Evidence(source="measurement", detail=detail)


def valid_answer(**overrides):
    base = dict(
        summary="CTR is 0.53, above the conventional 0.50 threshold.",
        findings=[Finding(label="Cardiomegaly", present="suggested",
                          ctr_value=0.528, evidence=[measurement()])],
    )
    return AgentAnswer(**{**base, **overrides})


def test_valid_answer_round_trips():
    assert valid_answer().findings[0].ctr_value == 0.528


def test_disclaimer_cannot_be_omitted_or_overridden():
    a = AgentAnswer(summary="x", unanswerable=True, disclaimer="no disclaimer here")
    assert "Not a medical device" in a.disclaimer


# --- the safety rules ------------------------------------------------------

def test_claim_without_evidence_is_rejected():
    with pytest.raises(ValidationError):
        Finding(label="Cardiomegaly", present="suggested", evidence=[])


def test_diagnostic_language_in_summary_is_rejected():
    for bad in ["The patient has cardiomegaly.",
                "Diagnosis: cardiomegaly.",
                "You should start treatment with a diuretic.",
                "I recommend therapy for this finding."]:
        with pytest.raises(ValidationError, match="diagnostic or prescriptive"):
            valid_answer(summary=bad)


def test_observational_language_is_allowed():
    """The rule must not be so broad it blocks legitimate description."""
    for ok in ["CTR is 0.53, above the conventional threshold.",
               "The classifier ranked Cardiomegaly highest at 0.62.",
               "The report states the cardiac silhouette is enlarged."]:
        assert valid_answer(summary=ok)


def test_declining_to_diagnose_is_allowed():
    """Regression, 2026-08-21. The first pattern was `diagnos\\w*`, which matched
    the word regardless of polarity and rejected the exact sentences we most want
    the agent to write. Found by the eval, not by these tests -- which is why
    each of those real phrasings is now pinned here.

    A safety check that punishes safe behaviour trains you to switch it off,
    which is worse than not having it."""
    for ok in ["This does not constitute a formal diagnosis.",
               "This is not a diagnostic device.",
               "Not for diagnostic use.",
               "I cannot provide a diagnosis.",
               "A diagnosis would require clinical correlation.",
               "This measurement alone cannot confirm cardiomegaly.",
               # negation: normal radiological phrasing, must not trip the check
               "The patient has no acute cardiopulmonary disease.",
               "The patient has not been assessed for this."]:
        assert valid_answer(summary=ok), ok


def test_assertive_diagnosis_still_caught():
    for bad in ["The patient has cardiomegaly.",
                "The patient shows definite cardiomegaly.",
                "This patient is diagnosed with pneumonia.",
                "Diagnosis: cardiomegaly.",
                "The diagnosis is congestive heart failure.",
                "Prescribe a beta blocker.",
                "Treatment with diuretics is indicated."]:
        with pytest.raises(ValidationError, match="diagnostic or prescriptive"):
            valid_answer(summary=bad)


def test_present_cannot_be_confirmed():
    """There is no 'confirmed' state. The system observes; it does not confirm."""
    with pytest.raises(ValidationError):
        Finding(label="X", present="confirmed", evidence=[measurement()])


def test_report_evidence_must_carry_a_verbatim_quote():
    """'The report mentions cardiomegaly' is unverifiable, and unverifiable is
    the exact shape a hallucination takes."""
    with pytest.raises(ValidationError, match="verbatim quote"):
        Evidence(source="report", detail="report says so")

    assert Evidence(source="report", detail="report", quote="The heart is enlarged.")


def test_literature_evidence_must_carry_a_pmid():
    with pytest.raises(ValidationError, match="PMID"):
        Evidence(source="literature", detail="a paper said so")


def test_implausible_ctr_is_rejected():
    with pytest.raises(ValidationError, match="physiologically plausible"):
        Finding(label="Cardiomegaly", present="suggested", ctr_value=0.95,
                evidence=[measurement()])


def test_ctr_requires_measurement_evidence():
    """Stops a CTR number being attributed to a report or to nothing."""
    with pytest.raises(ValidationError, match="must cite measurement evidence"):
        Finding(label="Cardiomegaly", present="suggested", ctr_value=0.5,
                evidence=[Evidence(source="report", detail="r", quote="big heart")])


def test_empty_answer_must_explain_itself():
    """No findings, no refusal, no admission: how a model says nothing while
    appearing to answer."""
    with pytest.raises(ValidationError, match="unanswerable"):
        AgentAnswer(summary="Here is what I found.")

    assert AgentAnswer(summary="Not in corpus.", unanswerable=True)
    assert AgentAnswer(summary="Out of scope.", refusal_reason="asked for treatment advice")


# --- grounding -------------------------------------------------------------

def test_real_quote_is_grounded():
    tool_results = [{"ok": True, "report": {"findings": "The heart is mildly enlarged. XXXX."}}]
    answer = valid_answer(findings=[Finding(
        label="Cardiomegaly", present="suggested",
        evidence=[Evidence(source="report", detail="get_report_by_image",
                           quote="The heart is mildly enlarged.")])])
    report = verify_grounding(answer, tool_results)
    assert report["grounded"] is True
    assert report["quotes_checked"] == 1


def test_fabricated_quote_is_caught():
    tool_results = [{"ok": True, "report": {"findings": "The lungs are clear."}}]
    answer = valid_answer(findings=[Finding(
        label="Cardiomegaly", present="suggested",
        evidence=[Evidence(source="report", detail="get_report_by_image",
                           quote="There is marked cardiomegaly.")])])
    report = verify_grounding(answer, tool_results)
    assert report["grounded"] is False
    assert report["unsupported_quotes"][0]["quote"] == "There is marked cardiomegaly."


def test_grounding_ignores_whitespace_and_case():
    tool_results = [{"report": {"findings": "The  heart\nis enlarged."}}]
    answer = valid_answer(findings=[Finding(
        label="C", present="suggested",
        evidence=[Evidence(source="report", detail="d", quote="the heart is enlarged.")])])
    assert verify_grounding(answer, tool_results)["grounded"] is True


def test_a_fabricated_pmid_in_structured_evidence_breaks_grounding():
    """The structured path has a schema and validators, so it is supposed to be
    the STRICTER of the two. It was not: only quotes were checked, so the prose
    path caught fabricated PMIDs and this one waved them through.

    An Evidence entry citing PMID 12456789 makes a factual claim about the
    world, and a more checkable one than any sentence: the identifier either
    came out of search_literature or it did not.
    """
    answer = AgentAnswer(
        summary="AP films magnify the cardiac silhouette.",
        findings=[Finding(
            label="AP magnification",
            present="indeterminate",
            evidence=[Evidence(source="literature",
                               detail="search_literature: projection effect",
                               citation="12456789")],
        )],
    )
    tool_results = [{"ok": True, "hits": [{"pmid": "7541234", "title": "A real paper"}]}]

    report = verify_grounding(answer, tool_results)
    assert report["grounded"] is False
    assert report["unsupported_citations"][0]["citation"] == "12456789"


def test_a_real_pmid_in_structured_evidence_is_accepted():
    answer = AgentAnswer(
        summary="AP films magnify the cardiac silhouette.",
        findings=[Finding(
            label="AP magnification",
            present="indeterminate",
            evidence=[Evidence(source="literature",
                               detail="search_literature: projection effect",
                               citation="7541234")],
        )],
    )
    tool_results = [{"ok": True, "hits": [{"pmid": "7541234", "title": "A real paper"}]}]
    assert verify_grounding(answer, tool_results)["grounded"] is True
