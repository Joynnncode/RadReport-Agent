"""Tests for BM25 retrieval and exact report lookup."""

from __future__ import annotations

import pytest

from radreport.tools.errors import ToolError
from radreport.tools.reports import (
    get_report_by_image, load_corpus, search_reports, tokenize,
)


def test_tokenize_is_lowercase_and_alphanumeric():
    assert tokenize("Cardiomegaly, present! (CXR-3821)") == [
        "cardiomegaly", "present", "cxr", "3821"
    ]


def test_empty_reports_are_skipped(report_csv):
    corpus = load_corpus(report_csv)
    assert len(corpus) == 4                      # row 5 had no text
    assert all(r.text for r in corpus)


def test_retrieves_the_obvious_match(report_csv):
    result = search_reports("cardiomegaly enlarged heart", k=2, csv_path=str(report_csv))
    assert result["hits"][0]["image_id"] == "CXR1"
    assert result["retriever"] == "bm25"


def test_k_is_respected(report_csv):
    assert len(search_reports("heart", k=1, csv_path=str(report_csv))["hits"]) <= 1


def test_no_lexical_overlap_returns_no_hits(report_csv):
    """This is BM25's defining weakness and we assert it explicitly, so that
    Weekend 3's embedding retriever has a documented gap to close."""
    result = search_reports("zebra xylophone", k=3, csv_path=str(report_csv))
    assert result["hits"] == []
    assert "No report contained any of the query terms" in result["note"]


def test_empty_query_raises(report_csv):
    with pytest.raises(ToolError, match="empty"):
        search_reports("   ", csv_path=str(report_csv))


def test_punctuation_only_query_raises(report_csv):
    with pytest.raises(ToolError, match="no indexable terms"):
        search_reports("???", csv_path=str(report_csv))


def test_missing_corpus_raises(tmp_path):
    with pytest.raises(ToolError, match="Report corpus not found"):
        load_corpus(tmp_path / "absent.csv")


def test_exact_lookup_found(report_csv):
    result = get_report_by_image("CXR2", csv_path=str(report_csv))
    assert result["found"] is True
    assert "pneumonia" in result["report"]["findings"].lower()


def test_exact_lookup_not_found_does_not_substitute(report_csv):
    """The safety-critical one. Asking for a case we do not have must return
    'not found', never the most similar case."""
    result = get_report_by_image("CXR9999", csv_path=str(report_csv))
    assert result["found"] is False
    assert "Do not substitute" in result["note"]
    assert "report" not in result


# --- id normalisation (regression, 2026-08-21) -----------------------------

def test_normalise_id_strips_stacked_suffixes():
    from radreport.tools.reports import normalise_id
    for variant in ("34_IM-1644-1001", "34_IM-1644-1001.dcm",
                    "34_IM-1644-1001.dcm.png", "  34_IM-1644-1001.DCM.PNG  "):
        assert normalise_id(variant) == "34_im-1644-1001"


def test_lookup_matches_regardless_of_suffix(report_csv):
    """Regression: the corpus stored ids as '<id>.dcm' while users and the agent
    say '<id>', so exact lookup missed cases that were genuinely present. A false
    negative in the one tool whose job is exact matching."""
    for variant in ("CXR2", "cxr2", "CXR2.dcm", "CXR2.dcm.png"):
        assert get_report_by_image(variant, csv_path=str(report_csv))["found"] is True


def test_normalisation_does_not_create_false_positives(report_csv):
    for variant in ("CXR", "CXR22", "CXR2X"):
        assert get_report_by_image(variant, csv_path=str(report_csv))["found"] is False
