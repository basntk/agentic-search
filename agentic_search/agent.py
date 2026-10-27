"""The agent loop: the model decides what to search, whether to read documents,
and when it has enough evidence to answer. Everything is traced for eval and debugging."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .corpus import Doc
from .llm import LLM

SYSTEM = """You answer questions using ONLY a private document collection, via tools.

Process:
1. Break the question into the sub-questions you need answered. Multi-part and comparison questions need several searches.
2. Call search with short, specific keyword queries. If results look off-target, REWRITE the query (synonyms, narrower or broader terms) instead of repeating it.
3. Use read_document when a snippet is promising but insufficient.
4. As soon as you have enough evidence, call finish. Do not keep searching to be thorough.
5. If the collection does not contain the answer, call finish and say so. Never answer from memory.

finish(answer, cited_ids): cite the doc ids that support each claim."""

TOOLS = [
    {
        "name": "search",
        "description": "Search the collection. Returns top results with id, title and a snippet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "description": "results to return (default 5, max 10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_document",
        "description": "Read the full text of one document by id.",
        "input_schema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
    },
    {
        "name": "finish",
        "description": "Give the final answer and end the search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "cited_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "cited_ids"],
        },
    },
]


@dataclass
class AgentResult:
    question: str
    answer: str = ""
    cited_ids: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)  # first-seen order across all searches
    queries: list[str] = field(default_factory=list)
    n_searches: int = 0
    steps: int = 0
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    finished: bool = False  # False means we hit the budget
    trace: list[dict] = field(default_factory=list)


class SearchAgent:
    def __init__(self, llm: LLM, retriever, docs: list[Doc], max_steps: int = 8, snippet_chars: int = 300):
        self.llm, self.retriever = llm, retriever
        self.by_id = {d.id: d for d in docs}
        self.max_steps, self.snippet_chars = max_steps, snippet_chars

    def _search(self, query: str, k: int) -> list[dict]:
        k = max(1, min(int(k or 5), 10))
        out = []
        for doc_id, score in self.retriever.search(query, k):
            d = self.by_id[doc_id]
            out.append({"id": doc_id, "title": d.title, "snippet": d.text[: self.snippet_chars], "score": round(score, 4)})
        return out

    def run(self, question: str) -> AgentResult:
        res = AgentResult(question=question)
        messages: list[dict] = [{"role": "user", "content": question}]
        seen: set[str] = set()
        t0 = time.perf_counter()

        for step in range(self.max_steps):
            out = self.llm.step(SYSTEM, messages, TOOLS)
            res.steps += 1
            res.tokens_in += out["usage"]["in"]
            res.tokens_out += out["usage"]["out"]
            messages.append({"role": "assistant", "content": out["assistant_content"]})

            if not out["tool_calls"]:
                # Model answered in plain text without calling finish; accept it but flag it.
                res.answer, res.finished = out["text"], True
                res.trace.append({"step": step, "event": "plain_text_answer"})
                break

            results, done = [], False
            for call in out["tool_calls"]:
                name, inp = call["name"], call["input"]
                ts = time.perf_counter()
                if name == "search":
                    hits = self._search(inp["query"], inp.get("k", 5))
                    res.n_searches += 1
                    res.queries.append(inp["query"])
                    for h in hits:
                        if h["id"] not in seen:
                            seen.add(h["id"])
                            res.retrieved_ids.append(h["id"])
                    payload = hits
                elif name == "read_document":
                    d = self.by_id.get(str(inp["doc_id"]))
                    payload = {"id": d.id, "title": d.title, "text": d.text} if d else {"error": "unknown doc_id"}
                    if d and d.id not in seen:
                        seen.add(d.id)
                        res.retrieved_ids.append(d.id)
                elif name == "finish":
                    res.answer, res.cited_ids = inp["answer"], [str(x) for x in inp.get("cited_ids", [])]
                    res.finished, done = True, True
                    payload = {"ok": True}
                else:
                    payload = {"error": f"unknown tool {name}"}
                res.trace.append(
                    {"step": step, "tool": name, "input": inp, "ms": round((time.perf_counter() - ts) * 1000, 1),
                     "n_results": len(payload) if isinstance(payload, list) else None}
                )
                results.append({"type": "tool_result", "tool_use_id": call["id"], "content": str(payload)})
            messages.append({"role": "user", "content": results})
            if done:
                break

        res.latency_s = round(time.perf_counter() - t0, 3)
        return res
