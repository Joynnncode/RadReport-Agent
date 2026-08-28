"""Does a dense retriever actually beat BM25 on this corpus?

    python -m evals.retrieval_compare                 # the full table
    python -m evals.retrieval_compare --show effusion # inspect one query

THE CLAIM BEING TESTED. reports.py has said since Weekend 1 that BM25 "has no
idea that 'cardiomegaly' and 'enlarged heart' mean the same thing". That is a
statement about a failure mode, and it had been sitting in the README as a
limitation without anyone checking whether it costs anything on real queries.
It might not: radiology reports use a small, rigidly conventional vocabulary,
and if every query uses that vocabulary too then lexical matching is not a
handicap, it is a precise instrument.

So the queries are split into two sets, and the split is the whole design:

  CLINICAL   the term as a radiologist writes it -- "pleural effusion",
             "cardiomegaly". BM25 should be excellent here and the dense model
             has nothing to add.
  LAY        the same finding as a patient, a clinician in a hurry, or a
             non-radiologist would type it -- "enlarged heart", "fluid around
             the lungs". This is where the stated weakness must show up if it
             is real.

Reporting one blended average over both sets would hide exactly the effect
under test: a retriever that is superb on half the queries and useless on the
other half averages out to "adequate".

RELEVANCE JUDGEMENTS. The corpus ships the NLM's own MeSH-derived `problems`
labels per study. A document is relevant to a query if its labels contain the
query's target label. These are weak labels -- assigned for indexing, not for
retrieval evaluation, and a report can discuss a finding the indexer did not
tag -- so the absolute numbers are worth less than the BM25-versus-dense gap
measured on identical judgements. Stated here rather than in a footnote,
because a reader who takes precision@5 = 0.72 as an absolute quality claim has
been misled by this harness.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radreport.tools.reports import (            # noqa: E402
    _index, rank_bm25, rank_embedding,
)

HERE = Path(__file__).parent

# (target label, clinical phrasing, lay phrasing)
#
# Chosen for labels with enough positives to measure (>=80 in 3,826 reports) and
# for a lay phrasing that shares as few content words as possible with the
# clinical one -- "enlarged heart" and "cardiomegaly" have no token in common,
# which is precisely the case BM25 cannot reach.
QUERIES = [
    ("Cardiomegaly",          "cardiomegaly",
                              "enlarged heart"),
    ("Pleural Effusion",      "pleural effusion",
                              "fluid around the lungs"),
    ("Pulmonary Atelectasis", "pulmonary atelectasis",
                              "collapsed lung tissue"),
    ("Calcified Granuloma",   "calcified granuloma",
                              "old healed infection spot"),
    ("Nodule",                "pulmonary nodule",
                              "small round spot in the lung"),
    ("Airspace Disease",      "airspace disease",
                              "hazy patch in the lung fields"),
    ("Fractures, Bone",       "bone fracture",
                              "broken rib"),
    ("Scoliosis",             "scoliosis",
                              "curved spine"),
    ("Emphysema",             "emphysema",
                              "overinflated damaged lungs"),
    ("Hernia, Hiatal",        "hiatal hernia",
                              "stomach pushing up into the chest"),
    ("Atherosclerosis",       "atherosclerosis",
                              "hardened calcified arteries"),
    ("Catheters, Indwelling", "indwelling catheter",
                              "tube line placed in the patient"),
]

K = 10


def relevant_ids(label: str, corpus) -> set[int]:
    needle = label.lower()
    return {i for i, r in enumerate(corpus)
            if needle in (r.problems or "").lower()}


def precision_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    top = ranked[:k]
    return sum(1 for i in top if i in gold) / k if top else 0.0


def recall_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    return sum(1 for i in ranked[:k] if i in gold) / len(gold) if gold else 0.0


def reciprocal_rank(ranked: list[int], gold: set[int]) -> float:
    """1/rank of the first relevant hit. Answers "how far do I scroll?"."""
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in gold:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """Binary-gain nDCG. Rewards putting relevant documents nearer the top,
    which precision@k -- indifferent to order within the cut -- does not."""
    import math
    dcg = sum(1 / math.log2(rank + 1)
              for rank, doc_id in enumerate(ranked[:k], start=1) if doc_id in gold)
    ideal = sum(1 / math.log2(rank + 1)
                for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def rank_hybrid(query: str) -> list[int]:
    from radreport.tools.embed import reciprocal_rank_fusion
    bm_order, _ = rank_bm25(query)
    em_order, _ = rank_embedding(query)
    return [i for i, _ in
            reciprocal_rank_fusion([list(bm_order[:50]), list(em_order[:50])])]


RETRIEVERS = {
    "bm25":      lambda q: rank_bm25(q)[0],
    "embedding": lambda q: rank_embedding(q)[0],
    "hybrid":    rank_hybrid,
}


def evaluate(phrasing: str) -> dict:
    """Score every retriever over one phrasing set. Returns per-query detail too."""
    _, corpus = _index()
    index = {"clinical": 1, "lay": 2}[phrasing]

    out: dict = {"phrasing": phrasing, "queries": [], "summary": {}}
    per_retriever: dict[str, dict[str, list[float]]] = {
        name: {"p": [], "r": [], "mrr": [], "ndcg": [], "latency_ms": []}
        for name in RETRIEVERS
    }

    for row in QUERIES:
        label, query = row[0], row[index]
        gold = relevant_ids(label, corpus)
        if not gold:
            print(f"  ! no documents labelled {label!r}; skipping", file=sys.stderr)
            continue

        record = {"label": label, "query": query, "n_relevant": len(gold),
                  "retrievers": {}}
        for name, rank in RETRIEVERS.items():
            started = time.perf_counter()
            ranked = rank(query)
            elapsed_ms = (time.perf_counter() - started) * 1000

            scores = {
                "p_at_10": round(precision_at_k(ranked, gold, K), 3),
                "recall_at_10": round(recall_at_k(ranked, gold, K), 3),
                "mrr": round(reciprocal_rank(ranked, gold), 3),
                "ndcg_at_10": round(ndcg_at_k(ranked, gold, K), 3),
                "latency_ms": round(elapsed_ms, 1),
            }
            record["retrievers"][name] = scores
            per_retriever[name]["p"].append(scores["p_at_10"])
            per_retriever[name]["r"].append(scores["recall_at_10"])
            per_retriever[name]["mrr"].append(scores["mrr"])
            per_retriever[name]["ndcg"].append(scores["ndcg_at_10"])
            per_retriever[name]["latency_ms"].append(elapsed_ms)
        out["queries"].append(record)

    for name, values in per_retriever.items():
        out["summary"][name] = {
            "p_at_10": round(statistics.mean(values["p"]), 3),
            "recall_at_10": round(statistics.mean(values["r"]), 3),
            "mrr": round(statistics.mean(values["mrr"]), 3),
            "ndcg_at_10": round(statistics.mean(values["ndcg"]), 3),
            "median_latency_ms": round(statistics.median(values["latency_ms"]), 1),
        }
    return out


def print_table(result: dict) -> None:
    print(f"\n{'=' * 74}")
    print(f"  {result['phrasing'].upper()} phrasing   "
          f"({len(result['queries'])} queries, k={K})")
    print("=" * 74)
    print(f"  {'retriever':<12} {'P@10':>7} {'R@10':>7} {'MRR':>7} "
          f"{'nDCG@10':>9} {'median ms':>11}")
    print(f"  {'-' * 12} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 11}")
    for name, s in result["summary"].items():
        print(f"  {name:<12} {s['p_at_10']:>7.3f} {s['recall_at_10']:>7.3f} "
              f"{s['mrr']:>7.3f} {s['ndcg_at_10']:>9.3f} "
              f"{s['median_latency_ms']:>11.1f}")


def print_per_query(clinical: dict, lay: dict) -> None:
    """The per-query view, because the average hides the finding.

    The claim is not "dense retrieval is better on average". It is that BM25
    fails on a SPECIFIC and predictable kind of query, and an average over
    twelve queries can be moved by one outlier in either direction.
    """
    print(f"\n{'=' * 74}")
    print("  Per query: nDCG@10, clinical phrasing -> lay phrasing")
    print("=" * 74)
    print(f"  {'label':<24} {'bm25':>15} {'embedding':>15} {'hybrid':>15}")
    print(f"  {'-' * 24} {'-' * 15} {'-' * 15} {'-' * 15}")
    lay_by_label = {q["label"]: q for q in lay["queries"]}
    for q in clinical["queries"]:
        other = lay_by_label.get(q["label"])
        if not other:
            continue
        cells = []
        for name in RETRIEVERS:
            a = q["retrievers"][name]["ndcg_at_10"]
            b = other["retrievers"][name]["ndcg_at_10"]
            cells.append(f"{a:.2f} -> {b:.2f}")
        print(f"  {q['label']:<24} {cells[0]:>15} {cells[1]:>15} {cells[2]:>15}")
        print(f"  {'':<24} {'':>15}   ({other['query']})")


def show_query(query: str) -> None:
    """Print what each retriever actually returns, for reading rather than scoring."""
    _, corpus = _index()
    for name, rank in RETRIEVERS.items():
        print(f"\n--- {name} : {query!r} ---")
        for position, doc_id in enumerate(rank(query)[:5], start=1):
            report = corpus[doc_id]
            print(f"  {position}. [{report.image_id}] problems={report.problems or '-'}")
            print(f"     {report.text[:150]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", help="print the top hits for one query and exit")
    ap.add_argument("--json", action="store_true",
                    help="also write docs/retrieval-comparison.json")
    ns = ap.parse_args()

    if ns.show:
        show_query(ns.show)
        return 0

    print("building indexes (the dense one encodes 3,826 reports on first run)...")
    clinical = evaluate("clinical")
    lay = evaluate("lay")

    print_table(clinical)
    print_table(lay)
    print_per_query(clinical, lay)

    print(f"\n{'=' * 74}")
    print("  The number this was built to produce")
    print("=" * 74)
    for name in RETRIEVERS:
        a = clinical["summary"][name]["ndcg_at_10"]
        b = lay["summary"][name]["ndcg_at_10"]
        drop = (a - b) / a * 100 if a else float("nan")
        print(f"  {name:<12} nDCG@10 falls {drop:5.1f}%  "
              f"({a:.3f} clinical -> {b:.3f} lay)")

    if ns.json:
        dest = HERE.parent / "docs" / "retrieval-comparison.json"
        dest.write_text(json.dumps({"k": K, "clinical": clinical, "lay": lay},
                                   indent=2))
        print(f"\n  written to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
