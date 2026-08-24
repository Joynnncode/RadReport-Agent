"""Tests for image loading. These are the 'known wrong input' tests.

Every one of these is a failure mode you WILL hit: a typo'd path, a JPEG that is
actually RGB, a zero-byte download. The reason to test them explicitly is that
the agent's recovery behaviour depends on getting a clean ToolError rather than
an obscure exception from deep inside PIL or numpy.
"""

from __future__ import annotations

import pytest

from radreport.imaging import load_xray
from radreport.tools.errors import ToolError


def test_missing_file_raises_tool_error(tmp_path):
    with pytest.raises(ToolError, match="not found"):
        load_xray(tmp_path / "does_not_exist.png")


def test_wrong_suffix_raises_tool_error(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("this is not an x-ray")
    with pytest.raises(ToolError, match="Unsupported image type"):
        load_xray(bad)


def test_corrupt_image_raises_tool_error(tmp_path):
    """A .png that is not actually a PNG. Happens with truncated downloads."""
    bad = tmp_path / "truncated.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage")
    with pytest.raises(ToolError, match="Could not decode"):
        load_xray(bad)


def test_tiny_image_rejected(tmp_path):
    from PIL import Image
    tiny = tmp_path / "tiny.png"
    Image.new("L", (8, 8)).save(tiny)
    with pytest.raises(ToolError, match="implausible shape"):
        load_xray(tiny)


def test_missing_file_is_not_recoverable(tmp_path):
    """The agent should not retry a bad path; recoverable=False tells it that."""
    with pytest.raises(ToolError) as exc:
        load_xray(tmp_path / "nope.png")
    assert exc.value.recoverable is False


def test_load_shape_and_range(sample_image):
    img = load_xray(sample_image, size=224)
    assert tuple(img.shape) == (1, 1, 224, 224)
    # The single most important assertion in this file: the model contract is
    # values in [-1024, 1024]. If this drifts to [0, 1] every downstream
    # probability silently becomes meaningless.
    assert -1100 <= float(img.min()) and float(img.max()) <= 1100
    assert float(img.max()) > 1.5, "values look like [0,1]; normalisation is wrong"


def test_resize_is_honoured(sample_image):
    assert tuple(load_xray(sample_image, size=512).shape) == (1, 1, 512, 512)
