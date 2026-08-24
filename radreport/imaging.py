"""Image loading and preprocessing shared by every imaging tool.

The single most common source of silently wrong results in this project is
getting preprocessing wrong. torchxrayvision models do NOT take a normal [0,1]
image. They expect a single-channel float array scaled to [-1024, 1024]. If you
feed them [0,1] the model still runs and still returns confident-looking
probabilities, they are just meaningless. So preprocessing lives in one tested
function rather than being copy-pasted into each tool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

from radreport.tools.errors import ToolError

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".dcm", ".bmp", ".tif", ".tiff"}


def load_xray(image_path: str | Path, size: int = 224) -> torch.Tensor:
    """Load a chest X-ray and return a tensor of shape (1, 1, size, size).

    Steps, in order, and why each one is there:
      1. Existence / suffix check    -> fail loudly on a bad path, not silently
      2. Convert to 8-bit greyscale  -> X-rays are greyscale; RGB JPEGs happen
      3. normalize(img, 255)         -> rescales 0..255 into -1024..1024
      4. Add a channel axis          -> models want (C, H, W), not (H, W)
      5. CenterCrop then Resize      -> squares the image without stretching
                                        anatomy, which would change the CTR
    """
    path = Path(image_path)

    if not path.exists():
        raise ToolError(f"Image not found: {path}", tool="load_xray", recoverable=False)
    if path.suffix.lower() not in VALID_SUFFIXES:
        raise ToolError(
            f"Unsupported image type '{path.suffix}'. Expected one of "
            f"{sorted(VALID_SUFFIXES)}.",
            tool="load_xray",
            recoverable=False,
        )

    try:
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    except Exception as exc:  # truncated file, zero bytes, not an image at all
        raise ToolError(f"Could not decode image {path.name}: {exc}", tool="load_xray") from exc

    if img.ndim != 2 or min(img.shape) < 32:
        raise ToolError(
            f"Image {path.name} has implausible shape {img.shape} for a chest X-ray.",
            tool="load_xray",
        )

    img = xrv.datasets.normalize(img, 255)          # -> roughly [-1024, 1024]
    img = img[None, ...]                            # (H, W) -> (1, H, W)

    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(size)(img)

    return torch.from_numpy(img).float()[None, ...]  # (1, 1, size, size)
