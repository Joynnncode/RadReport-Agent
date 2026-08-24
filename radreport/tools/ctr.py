"""Tool 3: compute_ctr -> cardiothoracic ratio from segmentation masks.

Deliberately NOT a model. This is arithmetic on the masks that segment_lungs
produced. Keeping it deterministic means: identical input gives identical
output, it is unit-testable against hand-made synthetic masks, and when the
agent reports "CTR 0.56" you can point at the exact lines that produced 0.56.

CTR = maximum transverse cardiac diameter / maximum transverse thoracic diameter

CLINICAL CAVEATS, encoded here rather than left in a README:

  1. We approximate the thoracic diameter with the widest horizontal extent of
     the lung fields. The textbook measurement uses the INNER RIB MARGINS, which
     sit slightly outside the lung fields. Our denominator is therefore a little
     small, which biases our CTR a little HIGH. Direction of bias matters: this
     errs toward over-calling cardiomegaly, not under-calling it.

  2. CTR is only interpretable on a PA (back-to-front) erect film. On an AP or
     supine film the heart is further from the detector and is magnified, so a
     CTR above 0.5 is common in healthy people. We cannot tell PA from AP from
     pixels alone, so we never assert cardiomegaly, we report the number and
     say what it would mean IF the film is PA.

  3. A ratio is not a diagnosis. Pericardial effusion, chest wall deformity and
     poor inspiration all move this number without any cardiac enlargement.
"""

from __future__ import annotations

import numpy as np

from radreport.tools.errors import ToolError
from radreport.tools.segment import load_masks

PA_CARDIOMEGALY_THRESHOLD = 0.50


def _max_horizontal_extent(mask: np.ndarray) -> tuple[int, int]:
    """Width in pixels of the widest row of a binary mask, and which row.

    We scan row by row rather than taking the bounding box, because the heart's
    widest point and the lungs' widest point are at different heights, and a
    bounding box would silently take the max over the whole shape anyway. Doing
    it explicitly makes it easy to return WHERE the measurement was taken, which
    is what lets the Streamlit app draw the calliper line on the overlay.
    """
    widths = np.zeros(mask.shape[0], dtype=int)
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        if cols.size:
            widths[r] = cols[-1] - cols[0] + 1
    row = int(np.argmax(widths))
    return int(widths[row]), row


def compute_ctr(mask_handle: str) -> dict:
    """Compute the cardiothoracic ratio from a segment_lungs mask handle."""
    masks, targets = load_masks(mask_handle)

    def get(name: str) -> np.ndarray:
        if name not in targets:
            raise ToolError(f"Mask file has no '{name}' channel.", tool="compute_ctr")
        return masks[targets.index(name)].astype(bool)

    heart = get("Heart")
    lungs = get("Left Lung") | get("Right Lung")

    if not heart.any():
        raise ToolError(
            "No heart region was segmented, so CTR cannot be computed. "
            "The image may be a lateral view, poorly exposed, or not a chest X-ray.",
            tool="compute_ctr",
        )
    if not lungs.any():
        raise ToolError(
            "No lung region was segmented, so CTR cannot be computed.",
            tool="compute_ctr",
        )

    cardiac_px, cardiac_row = _max_horizontal_extent(heart)
    thoracic_px, thoracic_row = _max_horizontal_extent(lungs)

    if thoracic_px == 0:
        raise ToolError("Thoracic diameter measured as zero.", tool="compute_ctr")

    ctr = cardiac_px / thoracic_px

    # Guard against physically impossible values caused by segmentation failure
    # rather than by anatomy. A heart wider than the lung fields means the mask
    # is wrong, not that the patient has a remarkable heart.
    implausible = ctr > 0.85 or ctr < 0.20

    if implausible:
        interpretation = (
            f"CTR of {ctr:.2f} is outside the physiologically plausible range, "
            "which indicates a segmentation failure rather than a real finding. "
            "Do not report this value."
        )
    elif ctr > PA_CARDIOMEGALY_THRESHOLD:
        interpretation = (
            f"CTR {ctr:.2f} exceeds the conventional 0.50 threshold. IF this is "
            "an erect PA film, that would be consistent with cardiomegaly. On an "
            "AP or supine film this is not interpretable due to magnification."
        )
    else:
        interpretation = (
            f"CTR {ctr:.2f} is at or below the conventional 0.50 threshold, "
            "which does not suggest cardiac enlargement on a PA film."
        )

    return {
        "ok": True,
        "ctr": round(ctr, 3),
        "cardiac_diameter_px": cardiac_px,
        "thoracic_diameter_px": thoracic_px,
        "measured_at_rows": {"cardiac": cardiac_row, "thoracic": thoracic_row},
        "threshold_used": PA_CARDIOMEGALY_THRESHOLD,
        "plausible": not implausible,
        "interpretation": interpretation,
        "method": (
            "Thoracic diameter approximated by widest lung-field extent, not "
            "inner rib margins; this biases CTR slightly high. View (PA vs AP) "
            "is unknown and materially affects interpretation."
        ),
    }
