"""Draft an eval set with Claude, then YOU verify it by hand. Needs ANTHROPIC_API_KEY.

  python scripts/gen_queries.py --corpus data/arxiv/corpus.jsonl --n-single 40 --n-multi 20 --out data/arxiv/queries_draft.jsonl

Generated labels are a starting point, not ground truth. Open the draft and:
  1. delete queries that are ambiguous or answerable by many docs you didn't label,
  2. for each query, search the corpus yourself (BM25 top 20) and add other relevant docs,
  3. keep ~10% unanswerable queries (relevant: []) to test refusal.
Document this process in the README; reviewers care that you know the labels are noisy.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_search.corpus import load_jsonl  # noqa: E402
from agentic_search.llm import AnthropicLLM  # noqa: E402

SINGLE = """Write ONE realistic search question a researcher might ask whose answer is in this paper abstract.
Do NOT copy distinctive phrases from the abstract; paraphrase so a keyword match is not trivial.
Also give a one-sentence reference answer. Return only JSON: {{"query": "...", "reference_answer": "..."}}

Title: {t1}
Abstract: {x1}"""

MULTI = """Write ONE question that REQUIRES information from BOTH abstracts below (a comparison or a two-hop question).
Return only JSON: {{"query": "...", "reference_answer": "..."}}

Paper A: {t1}\n{x1}\n\nPaper B: {t2}\n{x2}"""


def ask(llm, prompt):
    import re

    out = llm.step("You write evaluation data for search systems.", [{"role": "user", "content": prompt}], [])
    return json.loads(re.search(r"\{.*\}", out["text"], re.S).group(0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--n-single", type=int, default=40)
    p.add_argument("--n-multi", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    docs, rng, llm = load_jsonl(a.corpus), random.Random(a.seed), AnthropicLLM()
    rows = []
    for i, d in enumerate(rng.sample(docs, a.n_single)):
        q = ask(llm, SINGLE.format(t1=d.title, x1=d.text))
        rows.append({"qid": f"s{i:03d}", "type": "single", "relevant": [d.id], **q})
    for i in range(a.n_multi):
        d1, d2 = rng.sample(docs, 2)
        q = ask(llm, MULTI.format(t1=d1.title, x1=d1.text, t2=d2.title, x2=d2.text))
        rows.append({"qid": f"m{i:03d}", "type": "multi_hop", "relevant": [d1.id, d2.id], **q})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} draft queries to {a.out}. Now review them by hand.")


if __name__ == "__main__":
    main()
