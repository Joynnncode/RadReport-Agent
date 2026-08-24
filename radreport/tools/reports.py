"""Tool 4: search_reports -> retrieve radiologist reports by free-text query.

Retriever: BM25 (Okapi), via rank_bm25. BM25 is a lexical scorer: it rewards
documents that contain the query's rare words, discounts common ones, and
normalises for document length. It has no idea that "cardiomegaly" and "enlarged
heart" mean the same thing.

That limitation is the POINT of starting here. Weekend 3 adds an embedding
retriever which does understand that, and you measure whether it actually wins
on this corpus. A baseline you can beat is worth more than a clever system you
cannot evaluate. Radiology reports use a small, highly standardised vocabulary,
so BM25 is a genuinely strong baseline here, and it may not lose.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from radreport.config import REPORT_CSV
from radreport.tools.errors import ToolError

_TOKEN = re.compile(r"[a-z0-9]+")

# Case ids reach us in three shapes for the same study:
#   "34_IM-1644-1001"            what a user or the model says
#   "34_IM-1644-1001.dcm"        Path.stem of the filename, kept in the corpus
#   "34_IM-1644-1001.dcm.png"    the actual file on disk
# Comparing these raw makes get_report_by_image miss real cases, which is the
# worst possible false negative for the one tool whose job is exact matching.
# So every id passes through here before comparison.
_ID_SUFFIXES = (".png", ".jpg", ".jpeg", ".dcm")


def normalise_id(value: str) -> str:
    """Strip image suffixes and casing so the three shapes above compare equal."""
    out = (value or "").strip().lower()
    changed = True
    while changed:
        changed = False
        for suffix in _ID_SUFFIXES:
            if out.endswith(suffix):
                out = out[: -len(suffix)]
                changed = True
    return out


@dataclass(frozen=True)
class Report:
    uid: str
    image_id: str
    findings: str
    impression: str
    projection: str = ""    # "Frontal" if a frontal image exists for this study
    problems: str = ""      # dataset's own MeSH-derived labels; useful as weak
                            # ground truth when building the Weekend 4 gold set

    @property
    def text(self) -> str:
        return f"{self.findings} {self.impression}".strip()

    @property
    def index_text(self) -> str:
        """Text used for BM25 indexing, with de-identification noise removed.

        The NLM replaced identifiers and some clinical words with the literal
        token "XXXX". It appears thousands of times, so BM25 treats it as a
        common term and it contributes nothing but it does inflate document
        length, which the length normalisation then penalises. Stripping it from
        the INDEX while keeping it in `text` means retrieval improves and quoted
        evidence still matches the real record character for character.
        """
        return self.text.replace("XXXX", " ")


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumerics.

    Kept deliberately simple and shared between indexing and querying. If these
    two ever diverge, retrieval silently returns nothing and it is a miserable
    bug to find, so there is exactly one tokenizer.
    """
    return _TOKEN.findall(text.lower())


def load_corpus(csv_path: Path | None = None) -> list[Report]:
    path = Path(csv_path or REPORT_CSV)
    if not path.exists():
        raise ToolError(
            f"Report corpus not found at {path}. Run `python -m scripts.fetch_data` "
            "to build it from the Indiana University chest X-ray collection.",
            tool="search_reports",
            recoverable=False,
        )
    reports: list[Report] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            findings = (row.get("findings") or "").strip()
            impression = (row.get("impression") or "").strip()
            if not (findings or impression):
                continue        # a report with no text cannot be retrieved usefully
            reports.append(
                Report(
                    uid=(row.get("uid") or "").strip(),
                    image_id=(row.get("image_id") or "").strip(),
                    findings=findings,
                    impression=impression,
                    projection=(row.get("projection") or "").strip(),
                    problems=(row.get("problems") or "").strip(),
                )
            )
    if not reports:
        raise ToolError(f"Report corpus at {path} contained no usable rows.",
                        tool="search_reports", recoverable=False)
    return reports


@lru_cache(maxsize=4)
def _index(csv_path: str | None = None) -> tuple[BM25Okapi, tuple[Report, ...]]:
    """Build the BM25 index once and cache it.

    Indexing 3,900 short reports takes about a second. The eval harness runs
    hundreds of queries, so rebuilding per query would dominate the runtime.
    """
    corpus = load_corpus(Path(csv_path) if csv_path else None)
    bm25 = BM25Okapi([tokenize(r.index_text) for r in corpus])
    return bm25, tuple(corpus)


def search_reports(query: str, k: int = 3, csv_path: str | None = None) -> dict:
    """Return the k best-matching radiologist reports for a free-text query."""
    if not query or not query.strip():
        raise ToolError("Query was empty.", tool="search_reports")

    bm25, corpus = _index(csv_path)
    tokens = tokenize(query)
    if not tokens:
        raise ToolError(
            f"Query {query!r} contained no indexable terms.", tool="search_reports"
        )

    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(corpus)), key=lambda i: -scores[i])[: max(1, k)]

    hits = []
    for rank, i in enumerate(ranked, start=1):
        if scores[i] <= 0:
            continue        # BM25 gives 0 when no query term appears at all
        hits.append({"rank": rank, "score": round(float(scores[i]), 3), **asdict(corpus[i])})

    return {
        "ok": True,
        "query": query,
        "retriever": "bm25",
        "corpus_size": len(corpus),
        "hits": hits,
        "note": (
            "No report contained any of the query terms."
            if not hits else f"{len(hits)} report(s) matched."
        ),
    }


def get_report_by_image(image_id: str, csv_path: str | None = None) -> dict:
    """Exact lookup by image id. Not retrieval, a direct join.

    Separate from search_reports on purpose. "What does the report for CXR3821
    say" is a lookup and should never be answered by fuzzy scoring; if we do not
    have that exact case we must say so rather than return a similar one.
    """
    _, corpus = _index(csv_path)
    needle = normalise_id(image_id)
    for r in corpus:
        if normalise_id(r.image_id) == needle or r.uid.strip().lower() == needle:
            return {"ok": True, "found": True, "report": asdict(r)}
    return {
        "ok": True,
        "found": False,
        "image_id": image_id,
        "note": f"No report exists for {image_id!r} in this corpus. Do not substitute a similar case.",
    }
