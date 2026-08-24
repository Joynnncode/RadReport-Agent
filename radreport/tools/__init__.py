"""The tool layer.

Everything here is plain Python: no LLM, no prompts, no API keys except
PubMed's optional one. That separation is deliberate. It means the whole tool
layer is unit-testable without spending a single token, and when the agent gives
a wrong answer you can immediately establish whether the tool was wrong or the
model's use of the tool was wrong. Those are very different bugs.

REGISTRY maps the name the model will use to the Python callable. Weekend 2's
agent.py imports this and nothing else from the tool layer.
"""

import os

from radreport.tools.errors import ToolError
from radreport.tools.literature import search_literature
from radreport.tools.reports import get_report_by_image, search_reports

# DEMO MODE. The retrieval and literature tools are pure Python and always load.
# The three imaging tools are the ones that need torch, and torch is why the
# public deployment cannot run them: PSPNet peaks at ~1.8 GB RSS against a ~1 GB
# free-tier limit, and torch plus weights are ~820 MB on disk.
#
# So RADREPORT_DEMO=1 swaps in precomputed results and torch is never imported.
# The import is conditional rather than a runtime branch precisely so that the
# deployed environment does not need the dependency installed at all.
DEMO_MODE = os.getenv("RADREPORT_DEMO", "0") == "1"

if DEMO_MODE:
    from radreport.tools.demo import classify_xray, compute_ctr, segment_lungs
else:
    try:
        from radreport.tools.classify import classify_xray
        from radreport.tools.ctr import compute_ctr
        from radreport.tools.segment import segment_lungs
    except ImportError as exc:      # pragma: no cover - deployment guard
        # The likeliest cause by far is a deployment installed from
        # requirements-deploy.txt (no torch) without RADREPORT_DEMO=1 set. A raw
        # "No module named 'torch'" on a hosting dashboard is a genuinely
        # confusing thing to debug, so say what actually went wrong.
        raise ImportError(
            f"Could not import the imaging tools ({exc}). If this is the public "
            "deployment, set RADREPORT_DEMO=1 so precomputed results are used "
            "instead; requirements-deploy.txt deliberately omits torch because "
            "PSPNet needs ~1.8 GB RAM. For live inference install "
            "requirements.txt."
        ) from exc

REGISTRY = {
    "classify_xray": classify_xray,
    "segment_lungs": segment_lungs,
    "compute_ctr": compute_ctr,
    "search_reports": search_reports,
    "get_report_by_image": get_report_by_image,
    "search_literature": search_literature,
}

__all__ = ["REGISTRY", "ToolError", *REGISTRY.keys()]
