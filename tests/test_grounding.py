"""Tests for the quote-grounding check.

This is the check that stands between the system and its worst failure mode: a
model writing a radiology sentence that no radiologist ever wrote. It has to be
tolerant of typography and completely intolerant of changed words, and those two
requirements pull in opposite directions, so both directions are tested here.

The false-positive half is not politeness. The first version of this check was
strict about characters and reported 52.9% groundedness on a run whose real
fabrication rate was two answers in seventeen. A check that cries wolf eight
times for every wolf gets switched off.
"""

from __future__ import annotations

import pytest

from radreport.grounding import (
    Corpus, check_answer, check_quote, content, extract_quotes, normalise,
    repair_answer,
)

REPORT = (
    "The heart size is normal. There is no pleural effusion and no pneumothorax. "
    "Well-aerated lungs bilaterally. Followup imaging in 6 months is suggested. "
    "Right chest XXXX tip in the high SVC."
)
RESULTS = [{"ok": True, "found": True, "report": REPORT}]


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(RESULTS)


# --- what must be accepted: same words, different typing --------------------

@pytest.mark.parametrize("quote, why", [
    ("There is no pleural effusion and no pneumothorax",   "exact"),
    ("there is NO pleural effusion and no pneumothorax",   "case"),
    ("There  is no  pleural effusion and no pneumothorax", "collapsed whitespace"),
    ("There is **no pleural effusion** and no pneumothorax", "markdown inside the quote"),
    ("Right chest XXXX tip in the high SVC",               "de-identification token kept"),
])
def test_accepted_as_verbatim(corpus, quote, why):
    assert check_quote(quote, corpus).status == "verbatim", why


@pytest.mark.parametrize("quote, why", [
    ("The heart size is normal.",              "full stop the source field lacks"),
    ("Followup imaging in 6 months is suggested", "hyphen absent in both"),
    ("Follow-up imaging in 6 months is suggested", "model inserted a hyphen"),
    ("Follow up imaging in 6 months is suggested", "model split the word"),
    ("Wellaerated lungs bilaterally",           "model dropped the source's hyphen"),
    ("Well‑aerated lungs bilaterally",     "non-breaking hyphen"),
    ("There is no pleural effusion and no pneumothorax,", "trailing comma"),
])
def test_accepted_after_repair(corpus, quote, why):
    verdict = check_quote(quote, corpus)
    assert verdict.grounded, why
    # and the true span comes back, so the answer can be corrected rather than
    # merely forgiven
    assert verdict.source and content(verdict.source) == content(quote)


def test_repair_returns_the_sources_own_characters(corpus):
    verdict = check_quote("Follow-up imaging in 6 months is suggested", corpus)
    assert verdict.status == "repaired"
    assert verdict.source == "Followup imaging in 6 months is suggested"


# --- what must be rejected: changed words -----------------------------------

def test_dropped_negation_is_not_grounded(corpus):
    """The case that rules out scoring repair on similarity alone.

    "no pleural effusion" -> "pleural effusion" is a clinical inversion, and in a
    long quote it scores above 0.96 against its own source. Only comparing the
    letters and digits catches it.
    """
    verdict = check_quote("There is pleural effusion and no pneumothorax", corpus)
    assert verdict.status == "unsupported"
    assert verdict.similarity > 0.9, "similarity alone would have accepted this"


@pytest.mark.parametrize("quote, why", [
    ("Followup imaging in 8 months is suggested", "number changed"),
    ("The heart size is enlarged. There is no pleural effusion", "word changed"),
    ("The heart size is normal. ... Well-aerated lungs bilaterally.",
     "two non-adjacent sentences stitched behind an ellipsis"),
    ("The cardiac silhouette is markedly enlarged for technique",
     "wholly invented sentence"),
])
def test_rejected(corpus, quote, why):
    assert check_quote(quote, corpus).status == "unsupported", why


def test_nothing_is_grounded_against_an_empty_corpus():
    assert check_quote("The heart size is normal", Corpus([])).status == "unsupported"


# --- quote extraction -------------------------------------------------------

def test_extracts_across_quote_styles():
    for text in ['"The lungs are clear without focal consolidation."',
                 '“The lungs are clear without focal consolidation.”',
                 '`The lungs are clear without focal consolidation.`']:
        assert len(extract_quotes(text)) == 1


def test_short_quoted_spans_are_not_treated_as_citations():
    """A tool name in scare quotes is not a citation, and counting it as one
    buries the real signal in noise."""
    assert extract_quotes('I called "compute_ctr" and the result was "0.48".') == []


def test_duplicate_quotes_are_counted_once():
    answer = ('It says "There is no pleural effusion and no pneumothorax". '
              'Again: "There is no pleural effusion and no pneumothorax".')
    assert len(extract_quotes(answer)) == 1


# --- the answer-level report ------------------------------------------------

def test_answer_with_no_quotes_is_not_applicable():
    report = check_answer("The heart looks fine.", RESULTS)
    assert report["applicable"] is False
    # not a pass and not a fail: the metric did not apply
    assert report["quotes_found"] == 0


def test_report_separates_verbatim_from_repaired_from_fabricated():
    answer = ('One: "There is no pleural effusion and no pneumothorax". '
              'Two: "Follow-up imaging in 6 months is suggested". '
              'Three: "The cardiac silhouette is markedly enlarged for technique".')
    report = check_answer(answer, RESULTS)
    assert report["quotes_found"] == 3
    assert report["verbatim"] == 1
    assert len(report["repaired"]) == 1
    assert len(report["unsupported"]) == 1
    assert report["pass"] is False, "one unsupported quote fails the answer"


def test_repaired_quotes_alone_still_pass():
    report = check_answer('It says "Follow-up imaging in 6 months is suggested".', RESULTS)
    assert report["pass"] is True and report["verbatim"] == 0


def test_report_is_json_serialisable():
    """It is written straight into the eval result files. A dataclass in here
    turns a rescore into a TypeError three steps after the mistake."""
    import json
    json.dumps(check_answer('It says "Follow-up imaging in 6 months is suggested".', RESULTS))


# --- repairing an answer in place -------------------------------------------

def test_repair_answer_substitutes_the_real_span():
    answer = 'The report says "Follow-up imaging in 6 months is suggested".'
    fixed, repairs = repair_answer(answer, RESULTS)
    assert "Followup imaging in 6 months is suggested" in fixed
    assert repairs and repairs[0]["to"] == "Followup imaging in 6 months is suggested"


def test_repair_answer_leaves_a_suspected_fabrication_alone():
    """Deleting or rewriting an unsupported quote would hide the one failure
    this system most needs to surface."""
    answer = 'The report says "The cardiac silhouette is markedly enlarged for technique".'
    fixed, repairs = repair_answer(answer, RESULTS)
    assert fixed == answer and repairs == []


def test_repair_answer_is_idempotent():
    once, _ = repair_answer('It says "Follow-up imaging in 6 months is suggested".', RESULTS)
    twice, repairs = repair_answer(once, RESULTS)
    assert twice == once and repairs == []


# --- normalisation ----------------------------------------------------------

def test_normalise_folds_typography_but_not_words():
    assert normalise("Well‑aerated “lungs”") == 'well-aerated "lungs"'
    assert normalise("**Heart size**   normal") == "heart size normal"


def test_collect_strings_keeps_newlines_as_newlines():
    """str(dict) renders a real newline as backslash-n, which would make a quote
    spanning a line break look fabricated."""
    corpus = Corpus([{"report": "Line one.\nLine two continues here."}])
    assert check_quote("Line one. Line two continues here.", corpus).grounded


# --- delimiter pairing ------------------------------------------------------

def test_prose_between_two_code_spans_is_not_a_citation():
    """The bug that cost 19 points of groundedness on the 43-case run.

    With the length floor baked into the regex, a code span shorter than the
    floor was skipped and the scanner matched from its CLOSING backtick to the
    next OPENING one -- capturing the prose in between as a quotation. The model
    had cited nothing; the metric reported a possible fabrication.
    """
    answer = ('The tool returned `"ctr": 0.578` and indicated the measurement '
              'was plausible (`true`).')
    assert extract_quotes(answer) == []


def test_a_long_code_span_is_still_a_citation():
    answer = 'It reported `The heart size is normal and unchanged from prior`.'
    assert extract_quotes(answer) == ["The heart size is normal and unchanged from prior"]


def test_two_real_quotes_in_one_answer_are_both_found():
    answer = ('First "There is no pleural effusion and no pneumothorax" '
              'and second "Followup imaging in 6 months is suggested" done.')
    assert len(extract_quotes(answer)) == 2


def test_an_unclosed_delimiter_does_not_swallow_the_answer():
    """A stray quotation mark must not turn the rest of the answer into one
    enormous 'citation' that then fails and looks like a fabrication."""
    assert extract_quotes('The report says "There is no pleural effusion and more') == []


# --- fabricated identifiers -------------------------------------------------

LITERATURE = [{"ok": True, "hits": [
    {"pmid": "7541234", "title": "Cardiothoracic ratio on portable films",
     "journal": "Radiology", "year": 1995},
]}]


def test_a_pmid_that_came_from_the_tool_is_fine():
    assert check_answer("See PMID: 7541234 for the projection effect.", LITERATURE)["pass"]


def test_a_pmid_the_tool_never_returned_is_a_fabrication():
    """The failure this check was added for.

    The system prompt banned attributing quoted words to a paper. The model
    complied -- it opened by explaining it could not quote, because the tool
    returns no full text -- and then listed three papers with invented titles,
    journals, page ranges and PMIDs, having called no tool at all. The rule
    stopped one phrasing and the fabrication moved next door.
    """
    report = check_answer("See PMID: 12456789 and PMID 20876543.", LITERATURE)
    assert report["pass"] is False
    assert report["unsupported_pmids"] == ["12456789", "20876543"]


def test_pmids_are_matched_as_tokens_not_folded_text():
    """Whitespace folding would let a split identifier match a real one."""
    assert check_answer("PMID: 7541234", [{"hits": [{"pmid": "754 1234"}]}])["pass"] is False


def test_an_answer_citing_nothing_is_unaffected():
    report = check_answer("The heart looks fine.", LITERATURE)
    assert report["applicable"] is False and report["pmids_found"] == 0


# --- dict keys are tool output too ------------------------------------------

def test_a_classifier_label_quoted_from_a_dict_key_is_grounded():
    """classify_xray returns {"findings": {"Enlarged Cardiomediastinum": 0.43}},
    so every pathology label this system can name lives in a dict KEY.
    collect_strings walked values only, which made all eighteen invisible: an
    agent quoting a label straight out of the result it had just received was
    scored as having fabricated it. Caught on the final 43-case sweep.
    """
    results = [{"ok": True, "findings": {"Enlarged Cardiomediastinum": 0.43,
                                         "Cardiomegaly": 0.52}}]
    report = check_answer(
        'The classifier scored "Enlarged Cardiomediastinum" at 0.43.', results)
    assert report["pass"] and report["verbatim"] == 1


def test_collect_strings_returns_keys_and_values():
    from radreport.grounding import collect_strings
    got = collect_strings({"note": "heart is normal", "scores": {"Cardiomegaly": 0.5}})
    assert "Cardiomegaly" in got and "heart is normal" in got


def test_keys_do_not_let_a_fabrication_through():
    """Widening the corpus must not widen it to everything."""
    results = [{"findings": {"Cardiomegaly": 0.52}}]
    assert check_answer('It said "There is a large left pleural effusion here".',
                        results)["pass"] is False
