"""Offline tests: no network, no API key, no model downloads.
Run: python -m pytest -q   (or python tests/test_offline.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_search.agent import SearchAgent
from agentic_search.corpus import load_jsonl
from agentic_search.evaluate import agent_eval, load_queries, retrieval_eval
from agentic_search.llm import ScriptedLLM
from agentic_search.metrics import mrr, ndcg_at_k, recall_at_k
from agentic_search.retrieval import BM25, Hybrid, rrf

DATA = Path(__file__).resolve().parents[1] / "data" / "sample"
docs = load_jsonl(DATA / "corpus.jsonl")
queries = load_queries(DATA / "queries.jsonl")


def test_metrics_hand_computed():
    rel = {"a": 1, "b": 1}
    assert recall_at_k(["a", "x", "b"], rel, 2) == 0.5
    assert mrr(["x", "a"], rel) == 0.5
    assert abs(ndcg_at_k(["a", "b"], rel, 10) - 1.0) < 1e-9
    assert ndcg_at_k(["x", "y"], rel, 10) == 0.0
    # one relevant at rank 2 of 1 relevant: dcg = 1/log2(3), idcg = 1
    assert abs(ndcg_at_k(["x", "a"], {"a": 1}, 10) - 0.6309297535714574) < 1e-9


def test_bm25_finds_obvious_doc():
    bm = BM25(docs)
    assert bm.search("term saturation length normalization ranking function", 3)[0][0] == "d01"


def test_rrf_prefers_docs_in_both_lists():
    fused = rrf([[("a", 0), ("b", 0), ("c", 0)], [("c", 0), ("b", 0), ("d", 0)]], k=4)
    assert fused[0][0] in {"b", "c"} and fused[-1][0] in {"a", "d"}


def test_hybrid_of_two_bm25_runs():
    h = Hybrid([BM25(docs), BM25(docs)])
    assert h.search("reciprocal rank fusion", 3)[0][0] == "d03"


def test_retrieval_eval_runs_and_skips_unanswerable():
    rep = retrieval_eval({"bm25": BM25(docs)}, queries, k=10)
    assert len(rep["bm25"]["rows"]) == sum(1 for q in queries if q["relevant"])
    assert 0.0 < rep["bm25"]["ndcg@10"] <= 1.0


def test_agent_loop_decompose_then_finish():
    script = [
        [("search", {"query": "BM25 rare entities"}), ("search", {"query": "dense retrieval paraphrase"})],
        [("read_document", {"doc_id": "d02"})],
        [("finish", {"answer": "Dense handles paraphrase, BM25 exact terms.", "cited_ids": ["d01", "d02"]})],
    ]
    agent = SearchAgent(ScriptedLLM(script), BM25(docs), docs, max_steps=5)
    res = agent.run("compare BM25 and dense retrieval on rare entities and paraphrases")
    assert res.finished and res.n_searches == 2 and res.steps == 3
    assert res.cited_ids == ["d01", "d02"] and "d02" in res.retrieved_ids


def test_agent_budget_exhaustion_is_flagged():
    loop = [[("search", {"query": "bm25"})]]  # never finishes
    res = SearchAgent(ScriptedLLM(loop), BM25(docs), docs, max_steps=3).run("x")
    assert not res.finished and res.steps == 3 and res.n_searches == 3


def test_agent_eval_end_to_end_with_scripted_llm():
    script = [[("search", {"query": "reciprocal rank fusion"})],
              [("finish", {"answer": "RRF.", "cited_ids": ["d03"]})]]
    agent = SearchAgent(ScriptedLLM(script), BM25(docs), docs)
    q2 = [q for q in queries if q["qid"] == "q02"]
    out = agent_eval(agent, q2)
    assert out["summary"]["cited_precision"] == 1.0 and out["summary"]["evidence_recall"] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
