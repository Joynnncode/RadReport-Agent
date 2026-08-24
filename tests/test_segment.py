"""Segmentation tests. Slow: real PSPNet weights."""

from __future__ import annotations

from pathlib import Path

import pytest

from radreport.tools.ctr import compute_ctr
from radreport.tools.segment import segment_lungs

pytestmark = pytest.mark.slow


def test_finds_lungs_and_heart(sample_image):
    result = segment_lungs(sample_image)
    assert result["missing_structures"] == []
    for name in ("Left Lung", "Right Lung", "Heart"):
        assert result["areas"][name]["found"]


def test_anatomical_plausibility(sample_image):
    """Lungs occupy more of a chest film than the heart does. If this fails the
    channel indices are wrong, which is otherwise invisible in the numbers."""
    areas = segment_lungs(sample_image)["areas"]
    lung_px = areas["Left Lung"]["pixels"] + areas["Right Lung"]["pixels"]
    assert lung_px > areas["Heart"]["pixels"]


def test_writes_readable_artifacts(sample_image):
    result = segment_lungs(sample_image)
    assert Path(result["mask_handle"]).exists()
    assert Path(result["overlay_png"]).exists()


def test_end_to_end_ctr_is_physiological(sample_image):
    """The full imaging chain. We do not assert an exact CTR, because that
    would pin the test to the current model weights. We assert it lands in a
    range that a chest X-ray can actually produce."""
    result = compute_ctr(segment_lungs(sample_image)["mask_handle"])
    assert 0.30 < result["ctr"] < 0.75
    assert result["plausible"] is True
