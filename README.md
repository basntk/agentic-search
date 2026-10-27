# Agentic Search over a Paper Corpus

A small, measured agentic search system: hybrid retrieval (BM25 + dense + RRF + cross-encoder rerank),
an LLM agent that decomposes questions, rewrites failing queries and decides when to stop, and an
eval harness that quantifies every change.

> **Status: scaffold.** Everything under "Results" and "What I learned" is blank until you run the experiments.
> Fill it with numbers you measured. Don't leave placeholders in the version you show anyone.

## Layout

```
agentic_search/
  corpus.py      JSONL loader
  retrieval.py   BM25 (pure python), Dense (sentence-transformers), RRF, Hybrid, Reranked
  agent.py       tool-use loop: search / read_document / finish, step budget, full trace
  llm.py         AnthropicLLM + ScriptedLLM (deterministic, for offline tests)
  evaluate.py    retrieval_eval (nDCG/recall/MRR/latency/CI) and agent_eval (evidence recall,
                 citation precision, #searches, latency, tokens, judged answer quality)
  judge.py       LLM-as-judge with a rubric
  cli.py         retrieval | agent | ask
scripts/
  fetch_arxiv.py draft corpus from the arXiv API
  gen_queries.py LLM-drafted eval set (you must hand-verify it)
tests/test_offline.py   runs with no network and no API key
data/sample/            18-doc toy corpus + 9 queries so everything runs immediately
```

## Quickstart

```bash
pip install -r requirements.txt
python tests/test_offline.py                      # sanity check, no API needed

# toy data
python -m agentic_search.cli retrieval --corpus data/sample/corpus.jsonl --queries data/sample/queries.jsonl --configs bm25,dense,hybrid,hybrid+rerank

# real data
python scripts/fetch_arxiv.py --query "cat:cs.IR" --n 3000 --out data/arxiv/corpus.jsonl
export ANTHROPIC_API_KEY=...
python scripts/gen_queries.py --corpus data/arxiv/corpus.jsonl --out data/arxiv/queries_draft.jsonl
#   -> hand-review into data/arxiv/queries.jsonl (see script docstring)
python -m agentic_search.cli retrieval --corpus data/arxiv/corpus.jsonl --queries data/arxiv/queries.jsonl --configs bm25,dense,hybrid,hybrid+rerank
python -m agentic_search.cli agent --corpus data/arxiv/corpus.jsonl --queries data/arxiv/queries.jsonl --retriever hybrid --judge
python -m agentic_search.cli ask --corpus data/arxiv/corpus.jsonl --retriever hybrid "how do papers handle position bias in LLM rerankers?"
```

## Eval design

- **Retrieval layer** is evaluated without any LLM: nDCG@10, recall@10, MRR, p50/p95 latency, bootstrap 95% CI.
  With ~60 queries, differences inside the CI are noise.
- **Agent layer** is evaluated separately: evidence recall (did anything it retrieved contain the gold docs),
  citation precision/recall, searches per query, budget-exhaustion rate, latency, tokens, judged answer score and groundedness.
- **Unanswerable queries** (~10%) check that the agent refuses instead of fabricating.
- Retrieval and generation are scored separately on purpose, so a bad answer can be traced to a miss vs. a bad read.

## Labeling process and its limits

_(Fill in: how many queries, how generated, how you verified, known label noise, e.g. unlabeled relevant docs
make precision-style metrics pessimistic.)_

## Results

_(Paste the tables from `results/`. Keep the ones where the result surprised you.)_

## Experiment log

| # | Change | Hypothesis | nDCG@10 before -> after | Latency before -> after | Verdict |
|---|--------|-----------|-------------------------|-------------------------|---------|
| 1 | Add dense to BM25 via RRF | Fixes paraphrase misses | | | |
| 2 | Cross-encoder rerank, pool=30 | Better top-3 ordering | | | |
| 3 | Agent: allow query rewriting | Recovers zero-recall queries | | | |
| 4 | Agent: step budget 4 vs 8 | Cost drops, multi-hop suffers | | | |
| 5 | Chunk size / RRF constant sweep | | | | |

## What I learned / what went wrong

_(Your war stories. Specific failure cases: query, what was retrieved, why, what fixed it, and what didn't.
Include one thing you tried that did not help.)_

## Known limitations

- Single-process, in-memory indexes (fine to ~100k docs). Next step would be a real ANN index (HNSW) and sharded inverted index.
- Static corpus; no freshness or incremental indexing.
- Offline-only relevance labels; no online metrics.
