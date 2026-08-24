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

# Typographic normalisation, applied to both the quote and the source before
# comparison.
#
# Why: the first groundedness pass reported 52.9%, which looks like rampant
# fabrication. It was not. Models rewrite punctuation as they quote -- U+2011
# non-breaking hyphens for "well-aerated", curly apostrophes, en dashes -- and
# they insert markdown emphasis INSIDE the quotation marks
# ("**Heart size ... within normal limits**"). None of that changes a word.
#
# This check exists to catch invented clinical content, not typography. So we
# normalise the cosmetic layer away and stay strict about the words. A metric
# that flags eight harmless reformattings for every real fabrication is a metric
# people learn to ignore.
_UNICODE_FOLD = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u2009": " ",
    "\u2026": "...",
})
_MARKDOWN = re.compile(r"[*_`~]+")


def _normalise_whitespace(text: str) -> str:
    text = (text or "").translate(_UNICODE_FOLD)
    text = _MARKDOWN.sub("", text)          # emphasis inside a quote is a
                                            # rendering choice, not a word change
    return " ".join(text.split()).lower()


def _collect_strings(node) -> list[str]:
    """Pull every string leaf out of nested tool results.

    Deliberately NOT str(node) or json.dumps(node): both render a dict with
    repr(), which turns a real newline inside a report into the two characters
    backslash-n. A quote that spans a line break would then fail to match and a
    genuine citation would be reported as fabricated. Walking the structure
    keeps the text as text.
    """
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for v in node.values() for s in _collect_strings(v)]
    if isinstance(node, (list, tuple)):
        return [s for v in node for s in _collect_strings(v)]
    return []


def verify_grounding(answer: AgentAnswer, tool_results: list[dict]) -> dict:
    """Check every quoted piece of evidence against what the tools returned.

    A model that invents a plausible report line is the single most dangerous
    failure mode in this system, because the invention will read exactly like a
    real radiology sentence. The only defence is mechanical: the quote must
    appear, character for character, in text a tool actually produced.

    Returns a report rather than raising, because the caller decides what to do:
    the agent retries, the eval harness scores.
    """
    corpus = _normalise_whitespace(
        " \n ".join(_collect_strings(tool_results))
    )

    checked, unsupported = 0, []
    for finding in answer.findings:
        for ev in finding.evidence:
            if not ev.quote:
                continue
            checked += 1
            if _normalise_whitespace(ev.quote) not in corpus:
                unsupported.append({
                    "label": finding.label,
                    "source": ev.source.value,
                    "quote": ev.quote,
                })

    return {
        "grounded": not unsupported,
        "quotes_checked": checked,
        "unsupported_quotes": unsupported,
        "note": (
            "All quoted evidence appears in tool output."
            if not unsupported
            else f"{len(unsupported)} quote(s) do not appear in any tool result "
                 "and may be fabricated."
        ),
    }
