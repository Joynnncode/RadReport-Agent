"""Precompute imaging results so the public demo needs no PyTorch.

WHY THIS EXISTS

Profiling the app before deploying: PSPNet segmentation peaks at ~1.8 GB RSS,
and torch plus the cached weights are ~820 MB on disk. Streamlit Community Cloud
gives roughly 1 GB of RAM. The app would install, start, and then be killed by
the OOM reaper the first time anyone pressed the overlay toggle.

Rather than drop the imaging story from the demo, run the models ONCE here and
commit the results. The deployed app serves those and imports no torch at all.
Identical UX, ~40 MB of dependencies, no OOM.

The honesty requirement: the deployed app must SAY it is serving precomputed
results, because "this model runs in 40 ms" and "I ran this model last Tuesday"
are different claims. radreport/tools/demo.py surfaces that in every result and
the UI shows a banner.

    python scripts/precompute_demo.py --n 40
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radreport.config import DATA_DIR, IMAGE_DIR, REPORT_CSV  # noqa: E402
from radreport.tools.classify import classify_xray            # noqa: E402
from radreport.tools.ctr import compute_ctr                   # noqa: E402
from radreport.tools.errors import ToolError                  # noqa: E402
from radreport.tools.segment import segment_lungs             # noqa: E402

DEST = DATA_DIR / "demo_cache.json"
MAX_PNG_BYTES = 400_000     # keep the committed file to a sane size


def _thumb(path: Path, max_px: int = 512) -> str | None:
    """Base64 PNG, so the deployed app needs no image files on disk either."""
    try:
        from PIL import Image
        import io
        img = Image.open(path).convert("L")
        img.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if len(data) > MAX_PNG_BYTES:
            img.thumbnail((320, 320))
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
        return base64.b64encode(data).decode()
    except Exception as exc:
        print(f"    ! thumbnail failed: {exc}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40, help="how many cases to bake in")
    args = ap.parse_args()

    rows = {r["image_id"]: r for r in csv.DictReader(REPORT_CSV.open(encoding="utf-8"))}
    cases = sorted(p.name.replace(".dcm.png", "") for p in IMAGE_DIR.glob("*.dcm.png"))

    # Prefer cases that have a report, so the demo can show the full chain.
    cases = [c for c in cases if c in rows][: args.n]

    cache: dict[str, dict] = {}
    for i, case_id in enumerate(cases, start=1):
        path = IMAGE_DIR / f"{case_id}.dcm.png"
        entry: dict = {"case_id": case_id}

        try:
            entry["classify"] = classify_xray(str(path))
        except ToolError as exc:
            entry["classify_error"] = str(exc)

        try:
            seg = segment_lungs(str(path))
            entry["segment"] = {k: v for k, v in seg.items() if k != "mask_handle"}
            entry["segment"]["mask_handle"] = f"demo://{case_id}"
            entry["ctr"] = compute_ctr(seg["mask_handle"])
            entry["overlay_b64"] = _thumb(Path(seg["overlay_png"]))
        except ToolError as exc:
            entry["segment_error"] = str(exc)

        entry["image_b64"] = _thumb(path)
        cache[case_id] = entry
        if i % 10 == 0:
            print(f"  {i}/{len(cases)} ...", flush=True)

    DEST.write_text(json.dumps(cache, separators=(",", ":")))
    size_mb = DEST.stat().st_size / 1e6
    print(f"\nwrote {len(cache)} cases to {DEST} ({size_mb:.1f} MB)")
    if size_mb > 45:
        print("  ! large for a git repo; lower --n or the thumbnail size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
