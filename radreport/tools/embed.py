"""The embedding retriever, and the machinery for comparing it against BM25.

Weekend 1 shipped BM25 and wrote down the reason to be suspicious of it: BM25
scores words, so it cannot know that "enlarged heart" and "cardiomegaly" are the
same finding. That was a hypothesis about a weakness, not a measurement of one,
and it sat in the README as a limitation for three weekends. This module exists
to settle it with numbers -- see `evals/retrieval_compare.py` for the harness
that produces them.

MODEL. all-MiniLM-L6-v2: 384 dimensions, 22M parameters, runs on CPU in about a
minute over the whole 3,826-report corpus. It is a general-purpose sentence
encoder with no medical pre-training, which is the honest baseline to start
from: if a generic encoder already beats BM25 on clinical paraphrase, that is
the interesting result, and if it does not, reaching for PubMedBERT next is a
motivated decision rather than a reflex.

CACHING. Encoding the corpus takes ~60s and the corpus does not change between
runs, so vectors are written to .cache keyed by a fingerprint of the corpus AND
the model name. Keying on the corpus alone would silently serve MiniLM vectors
after a model swap, which is the kind of bug that produces a plausible,
completely meaningless comparison table.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import numpy as np

from radreport.config import CACHE_DIR
from radreport.tools.errors import ToolError

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Reports are short and the queries are short. 256 tokens covers essentially
# every findings+impression pair in this corpus without truncating one.
MAX_SEQ_LENGTH = 256


def _fingerprint(texts: list[str], model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode())
    for text in texts:
        digest.update(text.encode("utf-8", "replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


@lru_cache(maxsize=2)
def _load_model(model_name: str = MODEL_NAME):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:      # pragma: no cover - depends on the install
        raise ToolError(
            "sentence-transformers is not installed. It is needed only for the "
            "embedding retriever; BM25 works without it. "
            "Install with: pip install sentence-transformers",
            tool="search_reports",
            recoverable=False,
        ) from exc

    # Defer to SENTENCE_TRANSFORMERS_HOME when the environment sets it. The
    # Docker image bakes the weights into /opt/sbert at build time so the first
    # request does not block on a download; hardcoding a cache folder here would
    # override that and re-fetch inside the container -- the exact failure the
    # build step exists to prevent, made invisible by working fine on a laptop.
    cache_folder = os.environ.get("SENTENCE_TRANSFORMERS_HOME") or str(CACHE_DIR / "sbert")
    model = SentenceTransformer(model_name, cache_folder=cache_folder)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def encode(texts: list[str], model_name: str = MODEL_NAME,
           batch_size: int = 64) -> np.ndarray:
    """Encode to L2-normalised vectors, so cosine similarity is a dot product."""
    model = _load_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,      # cosine == dot, and no per-query renorm
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


class EmbeddingIndex:
    """Dense vectors for a corpus, cached on disk.

    Exhaustive search, deliberately. 3,826 vectors x 384 dimensions is a 5.9 MB
    matrix and one numpy matmul answers a query in under a millisecond. An ANN
    index here would add a dependency, a build step and an approximation, in
    exchange for saving time that is not being spent.
    """

    def __init__(self, texts: list[str], model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.texts = texts
        cache_path = (CACHE_DIR / "embeddings" /
                      f"{_fingerprint(texts, model_name)}.npy")
        if cache_path.exists():
            self.vectors = np.load(cache_path)
        else:
            self.vectors = encode(texts, model_name)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.vectors)
        self.cache_path = cache_path

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity of the query against every document, in [-1, 1]."""
        q = encode([query], self.model_name)[0]
        return self.vectors @ q


def reciprocal_rank_fusion(rankings: list[list[int]],
                           k: int = 60) -> list[tuple[int, float]]:
    """Combine several rankings by rank position rather than by score.

    The reason not to add the scores directly: BM25 returns unbounded positive
    numbers whose scale depends on corpus statistics, and cosine similarity
    returns [-1, 1]. Summing those means the BM25 score is the answer and the
    embedding contributes rounding error. RRF only ever looks at *positions*, so
    the two retrievers get equal say without inventing a normalisation constant
    and pretending it was principled.

    k=60 is the value from the original RRF paper (Cormack et al., 2009); it
    damps the influence of the very top rank enough that one retriever's
    confident mistake does not decide the fused result on its own.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + position)
    return sorted(fused.items(), key=lambda kv: -kv[1])
