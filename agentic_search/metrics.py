"""Ranking metrics. `relevant` is a dict {doc_id: grade} (grade 1 for binary labels)."""
from __future__ import annotations

import math


def dcg(grades: list[float]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: list[str], relevant: dict[str, float], k: int = 10) -> float:
    gains = [relevant.get(d, 0) for d in ranked[:k]]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], relevant: dict[str, float], k: int = 10) -> float:
    rel = {d for d, g in relevant.items() if g > 0}
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def mrr(ranked: list[str], relevant: dict[str, float]) -> float:
    for i, d in enumerate(ranked):
        if relevant.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def precision(cited: list[str], relevant: dict[str, float]) -> float:
    return sum(1 for d in cited if relevant.get(d, 0) > 0) / len(cited) if cited else 0.0


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def bootstrap_ci(xs: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean. With 50-100 queries, differences smaller than this are noise."""
    import random

    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]
