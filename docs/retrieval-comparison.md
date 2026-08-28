# Embeddings vs BM25 on 3,826 radiology reports

`reports.py` has carried this comment since Weekend 1:

> BM25 is a lexical scorer... It has no idea that "cardiomegaly" and "enlarged
> heart" mean the same thing.

That is a hypothesis about a weakness, and it sat in the README's limitations
section for three weekends without anyone checking what it costs. This is the
measurement. Reproduce with:

```bash
python -m evals.retrieval_compare --json
python -m evals.retrieval_compare --show "enlarged heart"    # read the hits
```

## The result

BM25 stays the default retriever, and the numbers are why.

**Clinical phrasing** — the term as a radiologist writes it (`pleural effusion`,
`cardiomegaly`). 12 queries, k=10:

| retriever | P@10 | MRR | nDCG@10 | median latency |
|---|---|---|---|---|
| **bm25** | **0.892** | **1.000** | **0.911** | 1.2 ms |
| embedding | 0.783 | 0.896 | 0.789 | 8.0 ms |
| hybrid (RRF) | 0.867 | 0.944 | 0.870 | 9.1 ms |

**Lay phrasing** — the same finding as a non-radiologist would type it
(`enlarged heart`, `fluid around the lungs`). Same 12 findings, same relevance
judgements, the query rewritten:

| retriever | P@10 | MRR | nDCG@10 | median latency |
|---|---|---|---|---|
| bm25 | 0.242 | 0.446 | 0.266 | 1.6 ms |
| **embedding** | **0.342** | **0.583** | **0.364** | 5.9 ms |
| hybrid (RRF) | 0.333 | 0.474 | 0.323 | 7.4 ms |

Three things came out of this, and only the first one was expected.

**1. The stated weakness is real, and larger than it looked.** BM25's nDCG@10
falls 70.8% between the two phrasings. This is not a rounding effect. Three
findings go to *exactly zero* on the lay phrasing: `fluid around the lungs`
returns nothing relevant to pleural effusion in the top ten, `overinflated
damaged lungs` nothing to emphysema, `old healed infection spot` nothing to
calcified granuloma — because no report contains those words, and words are all
BM25 has.

**2. Dense retrieval is not the fix.** The embedding model degrades less (53.9%
against BM25's 70.8%) and it wins the lay set on every metric — but it wins at
0.364 nDCG, which is not a working retriever, it is a less broken one. Anyone
reading "embeddings beat BM25 on paraphrase, +37% relative" as a reason to swap
the default would be shipping a retriever that is *worse on the queries the
system actually receives*: the agent constructs its own queries from clinical
vocabulary, so the clinical row is the operating condition, and BM25 wins it by
0.12 nDCG at a sixth of the latency.

**3. Neither retriever understands negation, and that is the more serious
finding.** The top hit for `enlarged heart` under the dense retriever is:

> The heart is not enlarged. Lungs are clear. No pleural effusion. No acute
> abnormality.

BM25 ranks the same report third. In a corpus where a large fraction of every
report is a list of things that are *absent*, a retriever that scores "no
pleural effusion" as a strong match for `pleural effusion` is a genuine
correctness problem, and it is invisible in the aggregate numbers above because
that report is labelled `normal`, so retrieving it costs both retrievers score
in exactly the same way an unrelated report would. The aggregate says they did
worse; it cannot say they returned the opposite of what was asked for. It is a
bigger threat to this system than the vocabulary gap, and it is what the next
retriever should be built to fix.

## Why the queries are split into two sets

Reporting one blended average over both phrasings would hide the entire effect.
A retriever that is superb on half the queries and useless on the other half
averages out to "adequate", and "adequate" is not a finding.

The split is also the honest way to state where each retriever belongs. BM25 is
not a weaker technique that a better one supersedes; it is a precise instrument
for a corpus with a small, rigidly conventional vocabulary, and this corpus is
exactly that. The dense model earns its place only when you do not control how
the query is phrased — which is a real situation (a user typing into the
Streamlit box) but not the one the agent is usually in.

## Method, and what these numbers are not worth

**Model.** `all-MiniLM-L6-v2`: 384 dimensions, 22M parameters, no medical
pre-training. That last part is deliberate. A general-purpose encoder is the
honest baseline: if it already closed the vocabulary gap, that would be the
interesting result, and since it does not, reaching for PubMedBERT next is a
motivated decision rather than a reflex.

**Search.** Exhaustive. 3,826 × 384 floats is a 5.9 MB matrix and one numpy
matmul answers a query in under a millisecond. An ANN index would add a
dependency, a build step and an approximation in exchange for saving time that
is not being spent.

**Fusion.** Reciprocal rank fusion, k=60, over the top 50 of each ranking.
Not score addition: BM25 returns unbounded positive numbers whose scale depends
on corpus statistics, cosine similarity returns [-1, 1], and summing them means
the BM25 score *is* the answer while the embedding contributes rounding error.
RRF only looks at rank positions, so neither retriever needs a normalisation
constant invented for it.

**Relevance judgements are weak, and the absolute numbers should not be quoted.**
A document counts as relevant if the NLM's MeSH-derived `problems` labels for
that study contain the query's target label. Those labels were assigned for
indexing, not for retrieval evaluation: a report can describe a finding the
indexer did not tag, and `P@10 = 0.892` should be read as "BM25 ranked labelled
documents highly", not as a quality score. What survives the weakness of the
labels is the *comparison* — every retriever is scored against identical
judgements, so the gap between them, and the collapse between the two phrasing
sets, are the numbers worth having.

**Twelve queries is small.** Findings were picked for having enough positives to
measure (≥80 of 3,826) and for a lay phrasing sharing as few content words as
possible with the clinical one. Per-query results are printed by the harness and
stored in `retrieval-comparison.json`, because with n=12 an average can be moved
by one outlier and the per-query table is where you check whether it was.
