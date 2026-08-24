"""Single place where paths and environment variables are resolved.

Why this file exists: every other module needs to know where the data lives and
where to write artifacts. If each module computes its own paths you end up with
subtle mismatches between the CLI, the tests and the Streamlit app. One module
owns it, everyone else imports from here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve the repo root from THIS file's location, not from the current working
# directory. This means `pytest` from anywhere and `streamlit run` from anywhere
# both find the same data.
REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
REPORT_DIR = DATA_DIR / "reports"
REPORT_CSV = DATA_DIR / "reports.csv"

ARTIFACT_DIR = REPO_ROOT / "artifacts"   # segmentation overlays, plots
TRACE_DIR = REPO_ROOT / "traces"         # JSONL agent traces
CACHE_DIR = REPO_ROOT / ".cache"         # LLM response cache (saves free-tier quota)

for _d in (ARTIFACT_DIR, TRACE_DIR, CACHE_DIR):
    _d.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# The single string that must appear on every clinical-looking output surface.
DISCLAIMER = (
    "Research prototype using public de-identified data. "
    "Not a medical device. Not for clinical use."
)
