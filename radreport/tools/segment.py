"""Tool 2: segment_lungs -> anatomical masks, areas, and an overlay image.

Model: torchxrayvision's PSPNet from ChestX-Det, which segments 14 structures.
We only care about Left Lung, Right Lung and Heart, but the model produces all
14 in one pass so there is no saving in asking for fewer.

DESIGN POINT worth remembering for interviews: this tool cannot return the masks
themselves. Tool results are serialised to JSON and pasted into an LLM prompt,
and a 512x512x14 float array is both unserialisable and useless to a language
model. So the tool writes the masks to disk and returns a *handle* (a file path)
plus the small human-meaningful numbers. compute_ctr then takes that handle.
This "big data on disk, small handle in the conversation" pattern is how you
keep an agent's context window from exploding.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv

from radreport.config import ARTIFACT_DIR
from radreport.imaging import load_xray
from radreport.tools.errors import ToolError

STRUCTURES_OF_INTEREST = ("Left Lung", "Right Lung", "Heart")


@lru_cache(maxsize=1)
def _model() -> torch.nn.Module:
    model = xrv.baseline_models.chestx_det.PSPNet()
    model.eval()
    return model


def segment_lungs(image_path: str | Path, prob_threshold: float = 0.5) -> dict:
    """Segment lungs and heart. Returns areas plus a handle to the mask file."""
    model = _model()
    img = load_xray(image_path, size=512)   # this model wants 512, not 224

    with torch.no_grad():
        logits = model(img)
        probs = torch.sigmoid(logits)[0]    # (14, 512, 512), one map per target

    masks = (probs >= prob_threshold).numpy().astype(np.uint8)
    targets = list(model.targets)

    stem = Path(image_path).stem
    mask_path = ARTIFACT_DIR / f"{stem}_masks.npz"
    np.savez_compressed(mask_path, masks=masks, targets=np.array(targets))

    total_px = masks.shape[1] * masks.shape[2]
    areas = {}
    for name in STRUCTURES_OF_INTEREST:
        idx = targets.index(name)
        px = int(masks[idx].sum())
        areas[name] = {
            "pixels": px,
            "fraction_of_image": round(px / total_px, 4),
            "found": px > 0,
        }

    missing = [n for n, a in areas.items() if not a["found"]]

    overlay_path = _write_overlay(img, masks, targets, stem)

    return {
        "ok": True,
        "image": Path(image_path).name,
        "model": "chestx_det_pspnet",
        "mask_handle": str(mask_path),      # <- pass this to compute_ctr
        "overlay_png": str(overlay_path),
        "areas": areas,
        "missing_structures": missing,
        "note": (
            "Segmentation failed to find: " + ", ".join(missing)
            if missing else "All target structures segmented."
        ),
    }


def _write_overlay(img: torch.Tensor, masks: np.ndarray, targets: list[str], stem: str) -> Path:
    """Save a PNG with lungs in blue and heart in red over the X-ray.

    This is purely for humans: the Streamlit panel and your own sanity checking.
    A segmentation bug is almost invisible in a number and obvious in a picture.
    """
    from PIL import Image

    base = img[0, 0].numpy()
    base = (base - base.min()) / (np.ptp(base) + 1e-8)     # -> [0, 1] for display
    rgb = np.stack([base, base, base], axis=-1)

    colours = {"Left Lung": (0.2, 0.5, 1.0), "Right Lung": (0.2, 0.5, 1.0), "Heart": (1.0, 0.3, 0.3)}
    for name, colour in colours.items():
        m = masks[targets.index(name)].astype(bool)
        for c in range(3):
            rgb[..., c][m] = 0.55 * rgb[..., c][m] + 0.45 * colour[c]

    out = ARTIFACT_DIR / f"{stem}_overlay.png"
    Image.fromarray((rgb * 255).astype(np.uint8)).save(out)
    return out


def load_masks(mask_handle: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read back what segment_lungs wrote. Used by compute_ctr."""
    path = Path(mask_handle)
    if not path.exists():
        raise ToolError(
            f"Mask file {path.name} not found. Call segment_lungs first.",
            tool="load_masks",
        )
    data = np.load(path, allow_pickle=False)
    return data["masks"], [str(t) for t in data["targets"]]
