"""Fetch a label-stratified image subset for the evaluation gold set.

The default fetch takes the first N frontal images, which follows the corpus's
own ordering and gave us 15 normals and 2 cardiomegalies out of 40. An eval set
built on that would measure almost nothing about the cardiomegaly path, which is
the project's headline capability.

So: sample per label from the dataset's own MeSH-derived `problems` column, so
every finding the agent is meant to handle appears enough times to say something
about. This is stratified sampling, and the reason to do it deliberately is that
an unbalanced eval set produces a confident number that hides the failure you
care about.

    python scripts/fetch_stratified.py
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radreport.config import IMAGE_DIR, REPORT_CSV  # noqa: E402

BASE = "https://huggingface.co/datasets/sasi2004/chest-xrays-indiana-university/resolve/main"
IMAGE_PREFIX = "images/images_normalized"

# label -> how many cases to pull. Weighted toward what the agent is asked about.
QUOTA = {
    "Cardiomegaly": 30,
    "Pleural Effusion": 20,
    "Pulmonary Atelectasis": 15,
    "Opacity": 15,
    "normal": 40,
}
SEED = 20260821     # fixed, so the eval set is reproducible by anyone

# Hugging Face rate-limits anonymous downloads and returned 429 for 18 of 120
# images on the first run. Self-throttle and back off rather than hammering
# someone else's free service until it says no -- the same rule the PubMed tool
# follows. Politeness here is also reliability: the retry loop is what turns a
# partial dataset into a complete one.
MIN_INTERVAL = 0.15
MAX_RETRIES = 4


def _get_with_backoff(url: str) -> bytes:
    delay = 1.0
    last = None
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, timeout=60)
        if resp.status_code == 429:
            last = "429 Too Many Requests"
            time.sleep(delay)
            delay *= 2          # exponential: 1s, 2s, 4s, 8s
            continue
        resp.raise_for_status()
        return resp.content
    raise requests.RequestException(f"gave up after {MAX_RETRIES} attempts ({last})")


def main() -> int:
    rows = list(csv.DictReader(REPORT_CSV.open(encoding="utf-8")))
    rows = [r for r in rows if r["filename"]]

    rng = random.Random(SEED)
    chosen: dict[str, dict] = {}

    for label, n in QUOTA.items():
        pool = [r for r in rows
                if label.lower() in [p.strip().lower() for p in r["problems"].split(";")]]
        rng.shuffle(pool)
        taken = 0
        for r in pool:
            if taken >= n:
                break
            if r["image_id"] not in chosen:
                chosen[r["image_id"]] = r
                taken += 1
        print(f"  {label:<24} pool={len(pool):>5}  selected={taken}")

    print(f"\n{len(chosen)} unique cases selected. Downloading...")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    fetched = skipped = failed = 0
    for i, r in enumerate(chosen.values(), start=1):
        dest = IMAGE_DIR / r["filename"]
        if dest.exists():
            skipped += 1
            continue
        try:
            time.sleep(MIN_INTERVAL)
            dest.write_bytes(_get_with_backoff(f"{BASE}/{IMAGE_PREFIX}/{r['filename']}"))
            fetched += 1
        except requests.RequestException as exc:
            print(f"  ! {r['filename']}: {exc}")
            failed += 1
        if i % 25 == 0:
            print(f"  {i}/{len(chosen)} ...", flush=True)

    print(f"\nfetched={fetched} already_present={skipped} failed={failed}")
    print(f"total images now: {len(list(IMAGE_DIR.glob('*.png')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
