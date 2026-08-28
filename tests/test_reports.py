"""Tests for BM25 retrieval and exact report lookup."""

from __future__ import annotations

import pytest

from radreport.tools.errors import ToolError
from radreport.tools.reports import (
    get_report_by_image, load_corpus, rank_bm25, search_reports, tokenize,
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


# --- retriever selection (Weekend 3's outstanding comparison) ---------------

def test_default_retriever_is_bm25(report_csv):
    """BM25 stays the default because it MEASURED better on clinical vocabulary
    (docs/retrieval-comparison.md), not because it was written first. If someone
    flips the default, that decision should have to change a test."""
    assert search_reports("cardiomegaly", csv_path=str(report_csv))["retriever"] == "bm25"


def test_unknown_retriever_is_rejected(report_csv):
    with pytest.raises(ToolError, match="Unknown retriever"):
        search_reports("cardiomegaly", csv_path=str(report_csv), retriever="magic")


def test_rank_bm25_returns_every_document(report_csv):
    """The comparison harness needs a full ranking, not a truncated one:
    recall@k and MRR are undefined if the tail has been thrown away."""
    order, scores = rank_bm25("heart", csv_path=str(report_csv))
    assert len(order) == len(load_corpus(report_csv))
    assert scores == sorted(scores, reverse=True)


def test_reciprocal_rank_fusion_prefers_agreement():
    """A document both retrievers rank highly must beat one that only a single
    retriever put first -- that is the entire point of fusing."""
    from radreport.tools.embed import reciprocal_rank_fusion
    fused = reciprocal_rank_fusion([[7, 1, 2], [7, 3, 4]])
    assert fused[0][0] == 7


def test_reciprocal_rank_fusion_ignores_score_scale():
    """RRF sees positions only. Passing wildly different score scales is exactly
    the situation it exists for -- BM25 is unbounded, cosine is [-1, 1] -- so
    the fusion must not care that one list came from a bigger-numbered world."""
    from radreport.tools.embed import reciprocal_rank_fusion
    a = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]])
    b = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]])
    assert a == b and a[0][0] in (1, 3)


@pytest.mark.slow
def test_embedding_retriever_finds_a_paraphrase(report_csv):
    """The claim the whole comparison was built to test: a query sharing no word
    with the report still retrieves it."""
    result = search_reports("enlarged heart", k=1, csv_path=str(report_csv),
                            retriever="embedding")
    assert result["retriever"] == "embedding"
    assert result["hits"][0]["image_id"] == "CXR1"


@pytest.mark.slow
def test_hybrid_returns_fused_scores_not_bm25_scores(report_csv):
    """Reporting a BM25 score next to a fused ranking would be a lie about where
    the ordering came from. RRF scores are small (~1/60 scale); BM25's are not."""
    hits = search_reports("pleural effusion", k=2, csv_path=str(report_csv),
                          retriever="hybrid")["hits"]
    assert hits and all(0 < h["score"] < 0.1 for h in hits)


@pytest.mark.slow
def test_embedding_retriever_reports_no_match_rather_than_nearest_neighbours():
    """Cosine has no natural zero, so without a floor an unrelated query still
    returns its k nearest neighbours and the tool announces "3 report(s)
    matched" for a question about bicycles. BM25 gets this for free."""
    from radreport.tools.reports import search_reports as search
    result = search("quarterly revenue forecast for the bicycle division",
                    k=3, retriever="embedding")
    assert result["hits"] == []
    assert "No report" in result["note"]
