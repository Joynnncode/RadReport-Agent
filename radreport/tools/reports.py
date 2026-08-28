"""Tool 4: search_reports -> retrieve radiologist reports by free-text query.

Retriever: BM25 (Okapi), via rank_bm25. BM25 is a lexical scorer: it rewards
documents that contain the query's rare words, discounts common ones, and
normalises for document length. It has no idea that "cardiomegaly" and "enlarged
heart" mean the same thing.

That limitation is the POINT of starting here. A baseline you can beat is worth
more than a clever system you cannot evaluate.

The embedding retriever now exists alongside it (`retriever="embedding"`), and
the guess above has been measured -- `evals/retrieval_compare.py` runs both over
the same queries and `docs/retrieval-comparison.md` reports what came out.
BM25 remains the DEFAULT, and the results are the reason: radiology reports use
a small, highly standardised vocabulary, so on queries that use that vocabulary
BM25 is not merely competitive, it wins. It loses exactly where predicted, on
paraphrase, which is why the fused retriever is the one to reach for when you do
not control how the query is phrased.
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


RETRIEVERS = ("bm25", "embedding", "hybrid")

# Cosine floor for the dense retriever, the counterpart to BM25's natural zero.
#
# BM25 scores exactly 0 when no query term appears, so "no match" is free. Cosine
# has no such floor: an unrelated query still returns its k nearest neighbours,
# and search_reports would report "3 report(s) matched" for a question about
# bicycles. Measured over this corpus, the two populations separate cleanly:
#
#     relevant   'pleural effusion' 0.683   'enlarged heart' 0.779   'broken rib' 0.601
#     nonsense   'zebra xylophone'  0.205   'quarterly revenue forecast' 0.117
#
# 0.35 sits in the gap, well above the best nonsense and well below the worst
# genuine hit. Reproduce the measurement before changing it.
EMBEDDING_FLOOR = 0.35


@lru_cache(maxsize=2)
def _embedding_index(csv_path: str | None = None):
    """Dense index over the same text BM25 sees. Built lazily and cached.

    Lazily because it costs a model load and, on a cold cache, a minute of
    encoding -- and the default retriever never touches it. A tool that pays for
    a capability nobody asked for on every import is a tool people stop importing.
    """
    from radreport.tools.embed import EmbeddingIndex
    _, corpus = _index(csv_path)
    return EmbeddingIndex([r.index_text for r in corpus])


def rank_bm25(query: str, csv_path: str | None = None) -> tuple[list[int], list[float]]:
    """Document ids best-first, with their scores."""
    bm25, corpus = _index(csv_path)
    tokens = tokenize(query)
    if not tokens:
        raise ToolError(
            f"Query {query!r} contained no indexable terms.", tool="search_reports"
        )
    scores = bm25.get_scores(tokens)
    order = sorted(range(len(corpus)), key=lambda i: -scores[i])
    return order, [float(scores[i]) for i in order]


def rank_embedding(query: str, csv_path: str | None = None) -> tuple[list[int], list[float]]:
    scores = _embedding_index(csv_path).scores(query)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return order, [float(scores[i]) for i in order]


def search_reports(query: str, k: int = 3, csv_path: str | None = None,
                   retriever: str = "bm25") -> dict:
    """Return the k best-matching radiologist reports for a free-text query.

    `retriever` selects lexical ("bm25", the default), dense ("embedding") or
    both fused by reciprocal rank ("hybrid"). The default is lexical because
    that is what the measurement supported, not because it was written first:
    see docs/retrieval-comparison.md.
    """
    if not query or not query.strip():
        raise ToolError("Query was empty.", tool="search_reports")
    if retriever not in RETRIEVERS:
        raise ToolError(
            f"Unknown retriever {retriever!r}. One of: {', '.join(RETRIEVERS)}.",
            tool="search_reports",
        )

    _, corpus = _index(csv_path)
    top = max(1, k)

    if retriever == "bm25":
        order, scores = rank_bm25(query, csv_path)
        # BM25 gives exactly 0 when no query term appears at all. That is a real
        # "no match" rather than a weak one, so those are dropped rather than
        # returned as the best of a bad set.
        pairs = [(i, sc) for i, sc in zip(order[:top], scores[:top]) if sc > 0]
    elif retriever == "embedding":
        order, scores = rank_embedding(query, csv_path)
        pairs = [(i, sc) for i, sc in zip(order[:top], scores[:top])
                 if sc >= EMBEDDING_FLOOR]
    else:
        bm_order, _ = rank_bm25(query, csv_path)
        em_order, _ = rank_embedding(query, csv_path)
        from radreport.tools.embed import reciprocal_rank_fusion
        # Fuse only the heads: the tail of a 3,826-document ranking is noise in
        # both retrievers, and including it lets a document neither ranked highly
        # accumulate its way up on two mediocre positions.
        fused = reciprocal_rank_fusion([list(bm_order[:50]), list(em_order[:50])])
        pairs = [(i, sc) for i, sc in fused[:top]]

    hits = [{"rank": rank, "score": round(float(score), 4), **asdict(corpus[i])}
            for rank, (i, score) in enumerate(pairs, start=1)]

    return {
        "ok": True,
        "query": query,
        "retriever": retriever,
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
