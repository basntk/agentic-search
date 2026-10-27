"""Eval runners.

retrieval_eval: no LLM needed. Compares retriever configs on nDCG/recall/MRR + latency.
agent_eval:     runs the agent loop, reports evidence recall, citation precision,
                search count, latency, tokens, and (optionally) judged answer quality.

Queries file (JSONL): {"qid": "q1", "query": "...", "relevant": ["d1","d7"] or {"d1": 2, "d7": 1},
                       "reference_answer": "...", "type": "multi_hop"}
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from .agent import SearchAgent
from .corpus import Doc
from .judge import judge
from .metrics import bootstrap_ci, mrr, ndcg_at_k, percentile, precision, recall_at_k


def load_queries(path: str | Path) -> list[dict]:
    qs = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            q = json.loads(line)
            rel = q["relevant"]
            q["relevant"] = {str(d): 1 for d in rel} if isinstance(rel, list) else {str(d): g for d, g in rel.items()}
            qs.append(q)
    return qs


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def retrieval_eval(retrievers: dict, queries: list[dict], k: int = 10) -> dict:
    queries = [q for q in queries if q["relevant"]]  # unanswerable queries have no ranking to score
    report = {}
    for name, r in retrievers.items():
        rows, lat = [], []
        for q in queries:
            t = time.perf_counter()
            ranked = [d for d, _ in r.search(q["query"], k)]
            lat.append((time.perf_counter() - t) * 1000)
            rows.append(
                {
                    "qid": q["qid"],
                    "type": q.get("type", "all"),
                    f"ndcg@{k}": ndcg_at_k(ranked, q["relevant"], k),
                    f"recall@{k}": recall_at_k(ranked, q["relevant"], k),
                    "mrr": mrr(ranked, q["relevant"]),
                    "ranked": ranked,
                }
            )
        nd = [x[f"ndcg@{k}"] for x in rows]
        by_type: dict[str, list[float]] = defaultdict(list)
        for x in rows:
            by_type[x["type"]].append(x[f"ndcg@{k}"])
        report[name] = {
            f"ndcg@{k}": _mean(nd),
            f"ndcg@{k}_ci95": bootstrap_ci(nd),
            f"recall@{k}": _mean([x[f"recall@{k}"] for x in rows]),
            "mrr": _mean([x["mrr"] for x in rows]),
            "p50_ms": percentile(lat, 50),
            "p95_ms": percentile(lat, 95),
            "by_type_ndcg": {t: _mean(v) for t, v in by_type.items()},
            "failures": [x["qid"] for x in rows if x[f"recall@{k}"] == 0],  # read these by hand
            "rows": rows,
        }
    return report


def agent_eval(agent: SearchAgent, queries: list[dict], judge_llm=None, k: int = 10) -> dict:
    rows = []
    for q in queries:
        r = agent.run(q["query"])
        answerable = bool(q["relevant"])
        row = {
            "qid": q["qid"],
            "type": q.get("type", "all"),
            "answerable": answerable,
            "evidence_recall": recall_at_k(r.retrieved_ids, q["relevant"], len(r.retrieved_ids) or 1) if answerable else None,
            "cited_precision": precision(r.cited_ids, q["relevant"]) if answerable else None,
            "cited_recall": recall_at_k(r.cited_ids, q["relevant"], len(r.cited_ids) or 1) if answerable else None,
            "cited_ids_n": len(r.cited_ids),
            "n_searches": r.n_searches,
            "steps": r.steps,
            "latency_s": r.latency_s,
            "tokens": r.tokens_in + r.tokens_out,
            "finished": r.finished,
            "answer": r.answer,
            "queries": r.queries,
        }
        if judge_llm and q.get("reference_answer"):
            cited = [agent.by_id[d].full_text for d in r.cited_ids if d in agent.by_id]
            row["judge"] = judge(judge_llm, q["query"], q["reference_answer"], r.answer, cited)
        rows.append(row)

    ans = [x for x in rows if x["answerable"]]
    unans = [x for x in rows if not x["answerable"]]
    summary = {
        "n": len(rows),
        "evidence_recall": _mean([x["evidence_recall"] for x in ans]),
        "cited_precision": _mean([x["cited_precision"] for x in ans]),
        "cited_recall": _mean([x["cited_recall"] for x in ans]),
        # Unanswerable queries: a good agent cites nothing and says so. Fabrication shows up here.
        "unanswerable_cited_something_rate": _mean([1.0 if x["answer"] and x.get("cited_ids_n", 0) else 0.0 for x in unans]),
        "avg_searches": _mean([x["n_searches"] for x in rows]),
        "budget_exhausted_rate": _mean([0.0 if x["finished"] else 1.0 for x in rows]),
        "latency_p50_s": percentile([x["latency_s"] for x in rows], 50),
        "latency_p95_s": percentile([x["latency_s"] for x in rows], 95),
        "avg_tokens": _mean([x["tokens"] for x in rows]),
    }
    judged = [x["judge"] for x in rows if "judge" in x]
    if judged:
        summary["answer_score"] = _mean([j["correct"] / 2 for j in judged])
        summary["grounded_rate"] = _mean([1.0 if j.get("grounded") else 0.0 for j in judged])
    return {"summary": summary, "rows": rows}


def format_retrieval_table(report: dict, k: int = 10) -> str:
    head = f"| config | ndcg@{k} (95% CI) | recall@{k} | mrr | p50 ms | p95 ms | zero-recall queries |\n|---|---|---|---|---|---|---|\n"
    lines = []
    for name, m in report.items():
        lo, hi = m[f"ndcg@{k}_ci95"]
        lines.append(
            f"| {name} | {m[f'ndcg@{k}']:.3f} ({lo:.2f}-{hi:.2f}) | {m[f'recall@{k}']:.3f} | {m['mrr']:.3f} "
            f"| {m['p50_ms']:.1f} | {m['p95_ms']:.1f} | {len(m['failures'])} |"
        )
    return head + "\n".join(lines)
