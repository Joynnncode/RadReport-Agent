"""Tool 1: classify_xray -> pathology probabilities.

Model: torchxrayvision DenseNet-121, weights "densenet121-res224-all", trained
on the union of several public CXR datasets (NIH, CheXpert, MIMIC, PadChest and
others). It outputs 18 pathology labels. Some of those labels are not supported
by every training set, and torchxrayvision marks those as NaN rather than
guessing. We drop NaNs instead of reporting them as 0.0, because 0.0 means
"confidently absent" and NaN means "this model cannot speak to that", and
conflating the two is exactly the kind of error that matters clinically.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import torchxrayvision as xrv

from radreport.imaging import load_xray

WEIGHTS = "densenet121-res224-all"


@lru_cache(maxsize=1)
def _model() -> torch.nn.Module:
    """Load the model once per process.

    Loading DenseNet takes a couple of seconds and ~30 MB. The agent may call
    this tool several times in one run, and the eval harness calls it hundreds
    of times, so caching turns a 5 minute eval into a 30 second one.
    """
    model = xrv.models.DenseNet(weights=WEIGHTS)
    model.eval()   # disables dropout / batchnorm updates; inference only
    return model


def classify_xray(image_path: str | Path, threshold: float = 0.5) -> dict:
    """Return per-pathology probabilities for one chest X-ray.

    Returns a JSON-serialisable dict, because the result is fed back to an LLM
    as a tool message. Numpy floats are not JSON-serialisable, hence float().
    """
    model = _model()
    img = load_xray(image_path, size=224)

    # IMPORTANT: do NOT apply sigmoid here. torchxrayvision's DenseNet.forward
    # already applies sigmoid AND op_norm (operating-point normalisation) when
    # the weights ship with op_threshs, which these do. Applying sigmoid a
    # second time squashes every output into [0.5, 0.73] and produces the very
    # convincing failure mode of "every pathology is mildly positive".
    # The returned values are already calibrated so that 0.5 is the model's
    # operating threshold for each label. See DECISIONS.md, entry 2026-08-18.
    with torch.no_grad():                 # no gradients needed; saves memory
        probs = model(img)[0]

    findings: dict[str, float] = {}
    unsupported: list[str] = []

    for label, prob in zip(model.pathologies, probs.tolist()):
        if not label:
            continue
        if prob != prob:            # NaN check: NaN is the only value != itself
            unsupported.append(label)
        else:
            findings[label] = round(float(prob), 4)

    positives = {k: v for k, v in findings.items() if v >= threshold}

    return {
        "ok": True,
        "image": Path(image_path).name,
        "model": WEIGHTS,
        "threshold": threshold,
        "findings": dict(sorted(findings.items(), key=lambda kv: -kv[1])),
        "above_threshold": dict(sorted(positives.items(), key=lambda kv: -kv[1])),
        "unsupported_labels": unsupported,
    }
