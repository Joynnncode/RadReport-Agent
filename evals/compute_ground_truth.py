"""Precompute verified ground truth for gold-set cases.

The point: a gold set whose "expected answer" I made up is worthless. Every
numeric expectation here is produced by actually running the deterministic
tools, and every quoted expectation is copied from the real corpus. If a gold
answer and the system disagree, exactly one of them is wrong and I can tell
which by re-running this.

Writes evals/ground_truth.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radreport.config import IMAGE_DIR, REPORT_CSV        # noqa: E402
from radreport.tools import classify_xray, compute_ctr, segment_lungs  # noqa: E402
from radreport.tools.errors import ToolError              # noqa: E402


def main(limit: int = 40) -> int:
    rows = {r["image_id"]: r for r in csv.DictReader(REPORT_CSV.open(encoding="utf-8"))}
    local = sorted(p.name.replace(".dcm.png", "") for p in IMAGE_DIR.glob("*.dcm.png"))

    out = {}
    for i, case_id in enumerate(local[:limit], start=1):
        row = rows.get(case_id)
        if not row:
            continue
        path = IMAGE_DIR / row["filename"]
        record = {
            "case_id": case_id,
            "problems": row["problems"],
            "findings": row["findings"],
            "impression": row["impression"],
        }
        try:
            record["ctr"] = compute_ctr(segment_lungs(str(path))["mask_handle"])["ctr"]
        except ToolError as exc:
            record["ctr"] = None
            record["ctr_error"] = str(exc)

        probs = classify_xray(str(path))["findings"]
        record["cardiomegaly_prob"] = probs.get("Cardiomegaly")
        record["top_finding"] = next(iter(probs))
        out[case_id] = record

        if i % 10 == 0:
            print(f"  {i}/{min(limit, len(local))} ...", flush=True)

    dest = Path(__file__).parent / "ground_truth.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {len(out)} cases to {dest}")

    with_ctr = [r for r in out.values() if r["ctr"] is not None]
    print(f"  CTR computed for {len(with_ctr)}/{len(out)}")
    if with_ctr:
        ctrs = sorted(r["ctr"] for r in with_ctr)
        print(f"  CTR range {ctrs[0]:.3f} - {ctrs[-1]:.3f}, median {ctrs[len(ctrs)//2]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40))
