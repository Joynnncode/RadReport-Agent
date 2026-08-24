"""Model-backed tests. Marked slow: they download and run real weights."""

from __future__ import annotations

import pytest

from radreport.tools.classify import classify_xray
from radreport.tools.errors import ToolError

pytestmark = pytest.mark.slow


def test_returns_all_pathologies(sample_image):
    result = classify_xray(sample_image)
    assert result["ok"] is True
    assert len(result["findings"]) >= 15
    assert all(0.0 <= v <= 1.0 for v in result["findings"].values())


def test_outputs_are_not_double_sigmoided(sample_image):
    """Regression test for a real bug found on 2026-08-18.

    torchxrayvision's DenseNet.forward already applies sigmoid + op_norm. An
    extra torch.sigmoid squashed everything into [0.5, 0.73], which looked
    plausible but meant every pathology read as mildly positive. The signature
    of that bug is a tiny spread with nothing below 0.5, so we assert a real
    spread instead of asserting specific values (which would be brittle).
    """
    probs = list(classify_xray(sample_image)["findings"].values())
    assert min(probs) < 0.35, "no low-probability findings; suspect double sigmoid"
    assert max(probs) - min(probs) > 0.3, "probability spread is implausibly narrow"


def test_known_cardiomegaly_case(sample_image):
    """The NIH sample 00000001_000.png is ground-truth labelled Cardiomegaly.
    One real case with a known answer is worth a lot of synthetic ones."""
    result = classify_xray(sample_image)
    ranked = list(result["findings"])
    assert "Cardiomegaly" in ranked[:3]


def test_threshold_filters(sample_image):
    strict = classify_xray(sample_image, threshold=0.99)
    assert strict["above_threshold"] == {}


def test_bad_path_propagates_tool_error(tmp_path):
    with pytest.raises(ToolError):
        classify_xray(tmp_path / "missing.png")
