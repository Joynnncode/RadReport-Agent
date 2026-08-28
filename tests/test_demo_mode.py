"""Tests for the precomputed demo path used by the public deployment.

The property that matters: demo mode must produce the SAME numbers the real
models produced, and must never quietly invent a result for a case it does not
have. A demo that fabricates is worse than no demo.
"""

from __future__ import annotations

import json

import pytest

from radreport.config import DATA_DIR
from radreport.tools.errors import ToolError

CACHE = DATA_DIR / "demo_cache.json"
pytestmark = pytest.mark.skipif(not CACHE.exists(),
                                reason="run scripts/precompute_demo.py first")


@pytest.fixture(scope="module")
def demo():
    from radreport.tools import demo as d
    return d


def test_demo_tools_do_not_import_torch():
    """The whole reason this path exists. Checked in a subprocess because torch
    may already be imported by another test in the same session."""
    import subprocess
    import sys
    code = (
        "import sys; "
        "from radreport.tools import demo; "
        "demo.classify_xray(sorted(demo.available_cases())[0]); "
        "print('torch' in sys.modules)"
    )
    import os
    env = {**os.environ, "RADREPORT_DEMO": "1"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(DATA_DIR.parent), env=env)
    assert out.stdout.strip() == "False", out.stderr[-400:]


def test_cached_values_match_the_real_models(demo):
    """Byte-identical, because they ARE the real model outputs."""
    cache = json.loads(CACHE.read_text())
    case_id = sorted(cache)[0]
    entry = cache[case_id]

    result = demo.classify_xray(case_id)
    assert result["findings"] == entry["classify"]["findings"]
    assert result["precomputed"] is True


def test_every_result_declares_it_is_precomputed(demo):
    """'This model runs in 40ms' and 'I ran this last Tuesday' are different
    claims. A demo that blurs them misrepresents the system."""
    case_id = sorted(demo.available_cases())[0]
    seg = demo.segment_lungs(case_id)
    for result in (demo.classify_xray(case_id), seg, demo.compute_ctr(seg["mask_handle"])):
        assert result["precomputed"] is True
        assert "Precomputed" in result["note"]


def test_unknown_case_errors_rather_than_substituting(demo):
    with pytest.raises(ToolError, match="not in the precomputed demo set"):
        demo.classify_xray("a_case_that_does_not_exist")


def test_accepts_the_same_id_shapes_as_the_real_tools(demo):
    case_id = sorted(demo.available_cases())[0]
    for variant in (case_id, f"{case_id}.dcm.png", f"data/images/{case_id}.dcm.png"):
        assert demo.classify_xray(variant)["precomputed"] is True


def test_threshold_argument_still_applies(demo):
    """The cached findings are fixed, but the threshold must still filter, or the
    argument silently does nothing."""
    case_id = sorted(demo.available_cases())[0]
    assert demo.classify_xray(case_id, threshold=0.99)["above_threshold"] == {}
    assert len(demo.classify_xray(case_id, threshold=0.0)["above_threshold"]) > 0


def test_ctr_chains_from_the_demo_mask_handle(demo):
    case_id = sorted(demo.available_cases())[0]
    handle = demo.segment_lungs(case_id)["mask_handle"]
    assert handle.startswith("demo://")
    assert 0.2 <= demo.compute_ctr(handle)["ctr"] <= 0.85


def test_deploy_requirements_exclude_torch():
    from radreport.config import REPO_ROOT
    # Strip comments: the file explains WHY torch is absent, and naively
    # substring-matching the whole text flags that explanation.
    lines = [l.split("#")[0].strip()
             for l in (REPO_ROOT / "requirements-deploy.txt").read_text().lower().splitlines()]
    deps = " ".join(l for l in lines if l)
    # sentence-transformers is on this list because it depends on torch: adding
    # the dense retriever to the deploy set would reintroduce the 820 MB the
    # whole demo-mode design exists to avoid, and it would do it transitively,
    # where a reader scanning the file for "torch" would not see it.
    for heavy in ("torch", "torchvision", "torchxrayvision", "scikit-image",
                  "sentence-transformers"):
        assert heavy not in deps, f"{heavy} would blow the free-tier memory limit"


def test_missing_torch_without_demo_mode_gives_an_actionable_error():
    """A raw ImportError on a hosting dashboard is a bad afternoon."""
    import radreport.tools as t
    src = open(t.__file__, encoding="utf-8").read()
    assert "RADREPORT_DEMO=1" in src
