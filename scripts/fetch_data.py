"""Build the local working dataset from the Indiana University CXR collection.

Source: the Open-i / NLM Indiana University Chest X-ray Collection, mirrored
publicly on Hugging Face. Public domain, de-identified, no data use agreement.
We deliberately do NOT use MIMIC-CXR: it is credentialed, and a public repo that
touches it is a licensing problem.

    python scripts/fetch_data.py                # 200 images + full report corpus
    python scripts/fetch_data.py --n-images 500
    python scripts/fetch_data.py --reports-only

Two things this script does that matter downstream:

  1. It joins reports to projections and KEEPS THE PROJECTION LABEL. The CTR
     tool's biggest caveat is that the ratio is only interpretable on a frontal
     film. The dataset tells us which images are frontal, so we can enforce it
     rather than hedge about it.

  2. It records, but does not strip, the "XXXX" de-identification placeholder.
     The NLM replaced names, dates and some clinical terms with XXXX. Those
     tokens are pure noise for BM25 and will show up in retrieved quotes, so you
     need to know they are there. We strip them from the INDEXED text and keep
     the original text for quoting, so a quote stays faithful to the record.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radreport.config import DATA_DIR, IMAGE_DIR, REPORT_CSV  # noqa: E402

REPO = "sasi2004/chest-xrays-indiana-university"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
IMAGE_PREFIX = "images/images_normalized"
TIMEOUT = 60


def _fetch_csv(name: str) -> list[dict]:
    print(f"  downloading {name} ...", flush=True)
    resp = requests.get(f"{BASE}/{name}", timeout=TIMEOUT)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


def build_report_corpus() -> list[dict]:
    """Join reports to their frontal image and write data/reports.csv."""
    reports = _fetch_csv("indiana_reports.csv")
    projections = _fetch_csv("indiana_projections.csv")

    # One study (uid) can have several images. We want the frontal one, because
    # that is the only view CTR is defined on.
    frontal: dict[str, str] = {}
    for row in projections:
        if row.get("projection", "").strip().lower() == "frontal":
            frontal.setdefault(row["uid"].strip(), row["filename"].strip())

    rows = []
    for r in reports:
        uid = (r.get("uid") or "").strip()
        findings = (r.get("findings") or "").strip()
        impression = (r.get("impression") or "").strip()
        if not (findings or impression):
            continue
        filename = frontal.get(uid, "")
        rows.append({
            "uid": uid,
            # The clean case id: "1_IM-0001-4001.dcm.png" -> "1_IM-0001-4001".
            # Path.stem alone leaves the ".dcm" on, which then fails to match the
            # id a user actually types. See DECISIONS.md 2026-08-21.
            "image_id": filename.replace(".dcm.png", "").replace(".png", "")
                        if filename else f"uid{uid}",
            "filename": filename,
            "projection": "Frontal" if filename else "",
            "problems": (r.get("Problems") or "").strip(),
            "findings": findings,
            "impression": impression,
        })

    DATA_DIR.mkdir(exist_ok=True)
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with_image = sum(1 for r in rows if r["filename"])
    print(f"  wrote {len(rows)} reports to {REPORT_CSV} ({with_image} with a frontal image)")
    return rows


def fetch_images(rows: list[dict], n: int) -> int:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    targets = [r for r in rows if r["filename"]][:n]
    fetched = 0
    for i, row in enumerate(targets, start=1):
        dest = IMAGE_DIR / row["filename"]
        if dest.exists():
            continue
        try:
            resp = requests.get(f"{BASE}/{IMAGE_PREFIX}/{row['filename']}", timeout=TIMEOUT)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            fetched += 1
        except requests.RequestException as exc:
            print(f"  ! skipped {row['filename']}: {exc}")
        if i % 25 == 0:
            print(f"  {i}/{len(targets)} images ...", flush=True)
    print(f"  {fetched} new image(s) in {IMAGE_DIR}")
    return fetched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-images", type=int, default=200,
                    help="how many frontal images to keep locally (default 200)")
    ap.add_argument("--reports-only", action="store_true",
                    help="build the text corpus, skip image download")
    args = ap.parse_args()

    print("Building report corpus:")
    rows = build_report_corpus()

    if not args.reports_only:
        print(f"\nFetching up to {args.n_images} frontal images:")
        fetch_images(rows, args.n_images)

    print("\nDone. Images are gitignored; the corpus CSV is too. "
          "Anyone cloning the repo reruns this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
