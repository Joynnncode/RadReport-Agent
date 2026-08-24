"""Shared fixtures.

The important idea: MOST tests should not need a model or a network. Model-free
tests run in milliseconds, so you run them constantly. Reserve the slow ones for
the handful of cases that genuinely need real weights, and mark them so CI can
run the fast suite on every push and the slow suite on a schedule.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from radreport.config import IMAGE_DIR

TARGETS = [
    'Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula',
    'Left Lung', 'Right Lung', 'Left Hilus Pulmonis', 'Right Hilus Pulmonis',
    'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum', 'Weasand', 'Spine',
]


@pytest.fixture(scope="session")
def sample_image() -> Path:
    path = IMAGE_DIR / "sample_cxr.png"
    if not path.exists():
        pytest.skip("sample_cxr.png missing; run scripts/fetch_data.py")
    return path


@pytest.fixture
def synthetic_masks(tmp_path: Path):
    """Build a mask file with EXACTLY known geometry.

    This is the trick that makes compute_ctr properly testable. We do not test
    it against a real X-ray, because then we would have no ground truth and the
    test would just assert whatever the code currently does. Instead we paint
    rectangles of known width: heart 100px wide, lungs spanning 250px. The
    correct CTR is therefore 100/250 = 0.4, exactly, and any change to the
    geometry code that breaks that is caught immediately.
    """
    def _build(heart_width: int = 100, lung_span: int = 250, size: int = 512) -> Path:
        masks = np.zeros((len(TARGETS), size, size), dtype=np.uint8)
        mid = size // 2

        h0 = mid - heart_width // 2
        masks[TARGETS.index("Heart"), 200:320, h0:h0 + heart_width] = 1

        l0 = mid - lung_span // 2
        half = lung_span // 2
        masks[TARGETS.index("Left Lung"), 100:400, l0:l0 + half] = 1
        masks[TARGETS.index("Right Lung"), 100:400, l0 + half:l0 + lung_span] = 1

        out = tmp_path / f"masks_{heart_width}_{lung_span}.npz"
        np.savez_compressed(out, masks=masks, targets=np.array(TARGETS))
        return out
    return _build


@pytest.fixture
def report_csv(tmp_path: Path) -> Path:
    """A tiny corpus with known content, so retrieval assertions are exact."""
    rows = [
        {"uid": "1", "image_id": "CXR1", "findings": "The heart is enlarged. Cardiomegaly is present.",
         "impression": "Cardiomegaly."},
        {"uid": "2", "image_id": "CXR2", "findings": "Left lower lobe consolidation consistent with pneumonia.",
         "impression": "Pneumonia."},
        {"uid": "3", "image_id": "CXR3", "findings": "The lungs are clear. Heart size is normal.",
         "impression": "No acute cardiopulmonary abnormality."},
        {"uid": "4", "image_id": "CXR4", "findings": "Moderate right pleural effusion.",
         "impression": "Right pleural effusion."},
        {"uid": "5", "image_id": "CXR5", "findings": "", "impression": ""},   # must be skipped
    ]
    path = tmp_path / "reports.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["uid", "image_id", "findings", "impression"])
        w.writeheader()
        w.writerows(rows)
    return path
