"""Tests for compute_ctr.

compute_ctr is the only tool with a provable right answer, so it gets the
strictest tests. Synthetic masks with rectangles of known width give an exact
expected ratio.
"""

from __future__ import annotations

import numpy as np
import pytest

from radreport.tools.ctr import compute_ctr, _max_horizontal_extent
from radreport.tools.errors import ToolError


def test_exact_ratio(synthetic_masks):
    result = compute_ctr(str(synthetic_masks(heart_width=100, lung_span=250)))
    assert result["ctr"] == pytest.approx(0.4, abs=1e-3)
    assert result["cardiac_diameter_px"] == 100
    assert result["thoracic_diameter_px"] == 250
    assert result["plausible"] is True


def test_above_threshold_is_flagged(synthetic_masks):
    result = compute_ctr(str(synthetic_masks(heart_width=150, lung_span=250)))
    assert result["ctr"] == pytest.approx(0.6, abs=1e-3)
    assert "exceeds the conventional 0.50 threshold" in result["interpretation"]
    # It must NOT assert cardiomegaly outright, because view is unknown.
    assert "IF this is" in result["interpretation"]


def test_implausible_ratio_is_refused(synthetic_masks):
    """Heart wider than the lungs means the segmentation failed, not that the
    patient is remarkable. The tool must say so rather than report a number."""
    result = compute_ctr(str(synthetic_masks(heart_width=240, lung_span=250)))
    assert result["plausible"] is False
    assert "Do not report this value" in result["interpretation"]


def test_missing_heart_raises(tmp_path):
    from tests.conftest import TARGETS
    masks = np.zeros((len(TARGETS), 512, 512), dtype=np.uint8)
    masks[TARGETS.index("Left Lung"), 100:400, 100:250] = 1
    path = tmp_path / "no_heart.npz"
    np.savez_compressed(path, masks=masks, targets=np.array(TARGETS))

    with pytest.raises(ToolError, match="No heart region"):
        compute_ctr(str(path))


def test_missing_mask_file_raises(tmp_path):
    with pytest.raises(ToolError, match="Call segment_lungs first"):
        compute_ctr(str(tmp_path / "never_made.npz"))


def test_extent_scans_rows_not_bounding_box():
    """A cross shape: widest row is 40 wide even though the bounding box is
    taller. Confirms we measure a real transverse diameter."""
    m = np.zeros((100, 100), dtype=bool)
    m[10:90, 45:55] = True     # vertical bar, 10 wide
    m[50:52, 30:70] = True     # horizontal bar, 40 wide
    width, row = _max_horizontal_extent(m)
    assert width == 40
    assert 50 <= row <= 51
