"""Tool 5: search_literature -> cited abstracts from PubMed.

Uses NCBI E-utilities, which is free and needs no key. Two endpoints:
  esearch  - query string  -> list of PubMed IDs
  esummary - list of PMIDs -> title, journal, year, authors

We deliberately do NOT fetch full abstracts by default. The agent needs a
citation it can quote a title from, and pulling 3 full abstracts into context
costs a few thousand tokens per call for little benefit.

Two things this tool must get right, and both are about being a good citizen of
someone else's free service:
  - Rate limit. NCBI allows 3 requests/second without a key, 10 with one. We
    self-throttle rather than relying on them to reject us.
  - Timeouts. A tool that hangs forever hangs the whole agent loop. Every
    outbound request gets an explicit timeout.
"""

from __future__ import annotations

import threading
import time

import requests

from radreport.config import NCBI_API_KEY, NCBI_EMAIL
from radreport.tools.errors import ToolError

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TIMEOUT = 10           # seconds per request
MIN_INTERVAL = 0.34    # ~3 requests/second without an API key

_lock = threading.Lock()
_last_call = 0.0


def _throttle() -> None:
    """Block until at least MIN_INTERVAL has passed since the last request."""
    global _last_call
    with _lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _params(**kwargs) -> dict:
    p = {"db": "pubmed", "retmode": "json", **kwargs}
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    p["tool"] = "radreport-agent"
    return p


def _get(endpoint: str, **params) -> dict:
    _throttle()
    try:
        resp = requests.get(f"{BASE}/{endpoint}.fcgi", params=_params(**params), timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.Timeout as exc:
        raise ToolError(f"PubMed timed out after {TIMEOUT}s.", tool="search_literature") from exc
    except requests.RequestException as exc:
        raise ToolError(f"PubMed request failed: {exc}", tool="search_literature") from exc
    except ValueError as exc:
        raise ToolError("PubMed returned a non-JSON response.", tool="search_literature") from exc


def search_literature(query: str, k: int = 3) -> dict:
    """Search PubMed and return up to k citations."""
    if not query or not query.strip():
        raise ToolError("Query was empty.", tool="search_literature")

    k = max(1, min(int(k), 10))     # clamp: the model will sometimes ask for 100

    search = _get("esearch", term=query, retmax=k, sort="relevance")
    ids = search.get("esearchresult", {}).get("idlist", [])

    if not ids:
        return {"ok": True, "query": query, "citations": [],
                "note": "PubMed returned no results for this query."}

    summary = _get("esummary", id=",".join(ids))
    result = summary.get("result", {})

    citations = []
    for pmid in ids:
        rec = result.get(pmid)
        if not rec:
            continue
        authors = [a.get("name", "") for a in rec.get("authors", [])][:3]
        citations.append({
            "pmid": pmid,
            "title": rec.get("title", "").rstrip("."),
            "journal": rec.get("source", ""),
            "year": (rec.get("pubdate") or "")[:4],
            "authors": authors,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    return {"ok": True, "query": query, "citations": citations,
            "note": f"{len(citations)} citation(s) from PubMed."}
