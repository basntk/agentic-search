"""Retrievers: BM25 (pure python), dense (sentence-transformers, optional),
reciprocal rank fusion, and an optional cross-encoder reranker.

Every retriever exposes: search(query, k) -> list[(doc_id, score)].
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .corpus import Doc

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the of to in on for and or is are was were be by with as at from that this it its "
    "how what which who why when where do does did can could should would not no".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    name = "bm25"

    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf: list[Counter] = [Counter(tokenize(d.full_text)) for d in docs]
        self.dl = [sum(c.values()) for c in self.tf]
        self.avgdl = sum(self.dl) / max(len(docs), 1)
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.inv: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.inv[t].append(i)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.inv[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(self.docs[i].id, s) for i, s in top]


class Dense:
    """Bi-encoder retrieval. Requires `pip install sentence-transformers numpy`."""

    name = "dense"

    def __init__(self, docs: list[Doc], model: str = "BAAI/bge-small-en-v1.5"):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.docs = docs
        self.model = SentenceTransformer(model)
        self.emb = self.model.encode(
            [d.full_text for d in docs], normalize_embeddings=True, show_progress_bar=True
        )

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.emb @ q
        idx = self.np.argsort(-sims)[:k]
        return [(self.docs[i].id, float(sims[i])) for i in idx]


def rrf(rankings: list[list[tuple[str, float]]], k: int = 10, c: int = 60):
    """Reciprocal rank fusion. c=60 is the standard constant; try tuning it."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            fused[doc_id] += 1.0 / (c + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]


class Hybrid:
    name = "hybrid"

    def __init__(self, retrievers: list, pool: int = 50, c: int = 60):
        self.retrievers, self.pool, self.c = retrievers, pool, c

    def search(self, query: str, k: int = 10):
        return rrf([r.search(query, self.pool) for r in self.retrievers], k=k, c=self.c)


class Reranked:
    """Wraps any retriever with a cross-encoder rerank of the top `pool` candidates.
    Requires sentence-transformers."""

    def __init__(self, base, docs: list[Doc], model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", pool: int = 30):
        from sentence_transformers import CrossEncoder

        self.base, self.pool = base, pool
        self.by_id = {d.id: d for d in docs}
        self.ce = CrossEncoder(model)
        self.name = f"{base.name}+rerank"

    def search(self, query: str, k: int = 10):
        cands = self.base.search(query, self.pool)
        pairs = [(query, self.by_id[i].full_text) for i, _ in cands]
        scores = self.ce.predict(pairs)
        ranked = sorted(zip((i for i, _ in cands), scores), key=lambda x: -x[1])
        return [(i, float(s)) for i, s in ranked[:k]]


def build(kind: str, docs: list[Doc]):
    """kind: bm25 | dense | hybrid | bm25+rerank | dense+rerank | hybrid+rerank"""
    base_kind, _, rerank = kind.partition("+")
    if base_kind == "bm25":
        r = BM25(docs)
    elif base_kind == "dense":
        r = Dense(docs)
    elif base_kind == "hybrid":
        r = Hybrid([BM25(docs), Dense(docs)])
    else:
        raise ValueError(f"unknown retriever {kind}")
    return Reranked(r, docs) if rerank == "rerank" else r
