"""Structured output: the shape a clinical answer must take.

Free prose is unevaluable. "The heart looks somewhat enlarged" cannot be scored
for groundedness by a harness, and it cannot be rendered as a UI component. A
Pydantic model forces the agent to separate the claim from the evidence for the
claim, which is exactly what Weekend 4's groundedness metric measures.

The validators below are the interesting part. They are not type checks; they
are CLINICAL SAFETY RULES expressed as code:

  - a claim with no evidence is rejected outright
  - a quote must actually appear in the cited source, so the model cannot
    invent a plausible-sounding report line
  - diagnostic language is rejected, because this is not a diagnostic device
  - a CTR outside the physiological range cannot be attached to a finding

Anything checkable here is checked here rather than requested in the prompt. A
prompt is a request; a validator is a guarantee.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from radreport.config import DISCLAIMER
from radreport.grounding import Corpus, check_quote, extract_pmids


class EvidenceSource(str, Enum):
    """Where a claim came from. There is no 'model_knowledge' member on purpose:
    the agent's own recall is not an admissible source in this system."""
    CLASSIFIER = "classifier"        # classify_xray probabilities
    MEASUREMENT = "measurement"      # compute_ctr
    REPORT = "report"                # a quoted radiologist report
    LITERATURE = "literature"        # a PubMed citation


# Phrases that ASSERT a diagnosis, as opposed to describing an observation or
# declining to diagnose. A research prototype may say "CTR is 0.53, above the
# conventional threshold". It may not say "the patient has cardiomegaly".
#
# The first version of this was `diagnos\w*`, which matched the word regardless
# of polarity and so rejected "this does not constitute a formal diagnosis" --
# precisely the sentence we want the agent to write. Found by the eval on
# 2026-08-21. A safety check that punishes safe behaviour trains you to disable
# it, which is worse than not having it.
DIAGNOSTIC_LANGUAGE = re.compile(
    r"(?:"
    # Asserting a condition about the patient -- but NOT negating one. The
    # negative lookahead matters: "the patient has no acute disease" is normal
    # radiological phrasing and must be allowed.
    r"\bthe patient (?:has|shows|is)(?!\s+(?:no|not|never|nothing))\b"
    r"|\bpatient is (?:suffering|diagnosed)\b"
    r"|\b(?:the|my|final|definitive) diagnosis is\b"
    r"|\bdiagnosis:\s*\w"
    r"|\bdiagnosed with\b"
    r"|\bI (?:diagnose|can confirm|am confident)\b"
    r"|\byou (?:should|must|need to) (?:take|start|begin|be given|receive)\b"
    r"|\bprescrib(?:e|ing)\b"
    r"|\btreat(?:ment)? (?:with|plan)\b"
    r"|\brecommend\w* (?:therapy|medication|treatment)\b"
    r")",
    re.IGNORECASE,
)


class Evidence(BaseModel):
    source: EvidenceSource
    detail: str = Field(..., min_length=1,
                        description="Tool name and specific value, e.g. 'compute_ctr: ctr=0.528'")
    quote: str | None = Field(None, description="Verbatim text, required when source is 'report'")
    citation: str | None = Field(None, description="PMID or case id the quote came from")

    @model_validator(mode="after")
    def report_evidence_must_quote(self):
        """A claim attributed to a report must carry the actual words. Without
        this, 'the report mentions cardiomegaly' is unverifiable and is exactly
        the shape a hallucination takes."""
        if self.source == EvidenceSource.REPORT and not (self.quote and self.quote.strip()):
            raise ValueError("evidence with source='report' must include a verbatim quote")
        if self.source == EvidenceSource.LITERATURE and not self.citation:
            raise ValueError("evidence with source='literature' must include a PMID citation")
        return self


class Finding(BaseModel):
    label: str = Field(..., min_length=1, description="What was observed, e.g. 'Cardiomegaly'")
    present: Literal["suggested", "not_suggested", "indeterminate"] = Field(
        ..., description="Never 'confirmed'. This system does not confirm findings.")
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(..., min_length=1)
    ctr_value: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("label")
    @classmethod
    def label_is_not_a_diagnosis(cls, v: str) -> str:
        if DIAGNOSTIC_LANGUAGE.search(v):
            raise ValueError(f"label {v!r} uses diagnostic language")
        return v

    @model_validator(mode="after")
    def ctr_must_be_physiological(self):
        """0.402 is a chest. 0.95 is a broken segmentation mask. The tool already
        refuses to report implausible values; this stops one slipping through by
        another route."""
        if self.ctr_value is not None and not (0.20 <= self.ctr_value <= 0.85):
            raise ValueError(
                f"ctr_value {self.ctr_value} is outside the physiologically "
                "plausible range 0.20-0.85 and indicates a segmentation failure"
            )
        return self

    @model_validator(mode="after")
    def ctr_claims_need_measurement_evidence(self):
        if self.ctr_value is not None:
            if not any(e.source == EvidenceSource.MEASUREMENT for e in self.evidence):
                raise ValueError("a finding with a ctr_value must cite measurement evidence")
        return self


class AgentAnswer(BaseModel):
    """The agent's complete structured response."""

    summary: str = Field(..., min_length=1, description="Two or three sentences, no diagnosis.")
    findings: list[Finding] = Field(default_factory=list)
    case_id: str | None = None
    unanswerable: bool = Field(
        False, description="True when the tools cannot support an answer. Prefer this to guessing.")
    refusal_reason: str | None = Field(
        None, description="Set when the request was outside scope, e.g. asking for treatment.")
    disclaimer: str = DISCLAIMER

    @field_validator("summary")
    @classmethod
    def summary_is_not_a_diagnosis(cls, v: str) -> str:
        match = DIAGNOSTIC_LANGUAGE.search(v)
        if match:
            raise ValueError(
                f"summary contains diagnostic or prescriptive language: {match.group(0)!r}"
            )
        return v

    @model_validator(mode="after")
    def must_say_something_or_explain_why_not(self):
        """Rejects the empty answer: no findings, no refusal, no admission that
        it could not answer. That combination is how a model quietly says
        nothing while appearing to have responded."""
        if not self.findings and not self.unanswerable and not self.refusal_reason:
            raise ValueError(
                "an answer with no findings must set unanswerable=true or give a refusal_reason"
            )
        return self

    @model_validator(mode="after")
    def force_disclaimer(self):
        # Not a validation failure: just make it impossible to omit.
        self.disclaimer = DISCLAIMER
        return self


def json_schema_for_prompt() -> dict:
    """The schema to show the model. Pydantic generates it, so it can never
    drift from what we actually validate against."""
    return AgentAnswer.model_json_schema()


# ---------------------------------------------------------------------------
# Grounding verification
# ---------------------------------------------------------------------------
# This cannot be a Pydantic validator: checking a quote against its source needs
# the tool results, which the model does not carry. So it runs after validation,
# against the trace. This is the Weekend 4 groundedness metric, implemented once
# and used both to gate live answers and to score the eval.

def verify_grounding(answer: AgentAnswer, tool_results: list[dict]) -> dict:
    """Check every quoted piece of evidence against what the tools returned.

    A model that invents a plausible report line is the single most dangerous
    failure mode in this system, because the invention will read exactly like a
    real radiology sentence. The only defence is mechanical.

    The comparison itself lives in radreport.grounding, shared with the eval
    harness, because the runtime check and the metric measuring it must agree by
    construction. When they were two copies they disagreed, and the metric was
    the stricter of the two -- so the harness reported fabrication where the
    product had (correctly) seen a real citation with a curly apostrophe in it.

    Each quote comes back as verbatim, repaired or unsupported. Only unsupported
    breaks grounding; a repaired quote is a real span of tool output that the
    model retyped imperfectly, and it carries the true span so the caller can
    substitute it.

    Returns a report rather than raising, because the caller decides what to do:
    the agent repairs and retries, the eval harness scores.
    """
    corpus = Corpus(tool_results)

    checked, unsupported, repaired, bad_citations = 0, [], [], []
    for finding in answer.findings:
        for ev in finding.evidence:
            # A literature citation is checked even though it carries no quote.
            # An Evidence entry claiming PMID 12456789 is making a factual claim
            # about the world, and it is a more checkable one than any sentence:
            # the identifier either came out of search_literature or it did not.
            # Only quotes were checked here, so the prose path caught fabricated
            # PMIDs and the structured path -- the one with a schema and
            # validators, the one that is supposed to be STRICTER -- did not.
            for pmid in extract_pmids(f"PMID: {ev.citation}" if ev.citation else ""):
                if pmid not in corpus.raw:
                    bad_citations.append({"label": finding.label,
                                          "source": ev.source.value,
                                          "citation": ev.citation})

            if not ev.quote:
                continue
            checked += 1
            verdict = check_quote(ev.quote, corpus)
            if verdict.status == "unsupported":
                unsupported.append({
                    "label": finding.label,
                    "source": ev.source.value,
                    "quote": ev.quote,
                    "closest_similarity": verdict.similarity,
                })
            elif verdict.status == "repaired":
                repaired.append({
                    "label": finding.label,
                    "quote": ev.quote,
                    "actual_source_text": verdict.source,
                    "similarity": verdict.similarity,
                })

    problems = len(unsupported) + len(bad_citations)
    return {
        "grounded": not problems,
        "quotes_checked": checked,
        "quotes_verbatim": checked - len(unsupported) - len(repaired),
        "repaired_quotes": repaired,
        "unsupported_quotes": unsupported,
        "unsupported_citations": bad_citations,
        "note": (
            "All quoted evidence appears in tool output."
            if not problems
            else f"{problems} claim(s) do not appear in any tool result "
                 "and may be fabricated."
        ),
    }
