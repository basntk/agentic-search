"""CLI.

  python -m agentic_search.cli retrieval --corpus data/corpus.jsonl --queries data/queries.jsonl --configs bm25,hybrid,hybrid+rerank
  python -m agentic_search.cli agent     --corpus data/corpus.jsonl --queries data/queries.jsonl --retriever hybrid --judge
  python -m agentic_search.cli ask       --corpus data/corpus.jsonl --retriever bm25 "your question"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import SearchAgent
from .corpus import load_jsonl
from .evaluate import agent_eval, format_retrieval_table, load_queries, retrieval_eval
from .retrieval import build


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("retrieval")
    r.add_argument("--corpus", required=True)
    r.add_argument("--queries", required=True)
    r.add_argument("--configs", default="bm25")
    r.add_argument("--k", type=int, default=10)
    r.add_argument("--out", default="results/retrieval.json")

    a = sub.add_parser("agent")
    a.add_argument("--corpus", required=True)
    a.add_argument("--queries", required=True)
    a.add_argument("--retriever", default="bm25")
    a.add_argument("--max-steps", type=int, default=8)
    a.add_argument("--judge", action="store_true")
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--out", default="results/agent.json")

    s = sub.add_parser("ask")
    s.add_argument("--corpus", required=True)
    s.add_argument("--retriever", default="bm25")
    s.add_argument("question")

    args = p.parse_args(argv)
    docs = load_jsonl(args.corpus)

    if args.cmd == "retrieval":
        qs = load_queries(args.queries)
        retrievers = {c: build(c, docs) for c in args.configs.split(",")}
        report = retrieval_eval(retrievers, qs, args.k)
        print(format_retrieval_table(report, args.k))
        for name, m in report.items():
            print(f"\n[{name}] by query type (ndcg@{args.k}):", {t: round(v, 3) for t, v in m["by_type_ndcg"].items()})
            print(f"[{name}] zero-recall queries:", m["failures"])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)

    elif args.cmd == "agent":
        from .llm import AnthropicLLM

        qs = load_queries(args.queries)
        if args.limit:
            qs = qs[: args.limit]
        agent = SearchAgent(AnthropicLLM(), build(args.retriever, docs), docs, max_steps=args.max_steps)
        out = agent_eval(agent, qs, judge_llm=AnthropicLLM() if args.judge else None)
        print(json.dumps(out["summary"], indent=1))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)

    elif args.cmd == "ask":
        from .llm import AnthropicLLM

        agent = SearchAgent(AnthropicLLM(), build(args.retriever, docs), docs)
        res = agent.run(args.question)
        print("queries:", res.queries, "\n")
        print(res.answer, "\ncited:", res.cited_ids)


if __name__ == "__main__":
    main()
