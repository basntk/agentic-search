"""LLM-as-judge for answer quality. Use a stronger or different model than the agent
when you can, and spot-check ~20 judgments by hand before trusting the numbers."""
from __future__ import annotations

import json
import re

from .llm import LLM

JUDGE_SYSTEM = """You grade an answer against a reference answer and the question.
Return ONLY JSON: {"correct": 0|1|2, "grounded": true|false, "reason": "<one sentence>"}
correct: 2 = fully matches the reference in substance, 1 = partially correct or missing a key part, 0 = wrong or no answer.
If the reference says the collection lacks the answer, a refusal to answer is correct (2).
grounded: false if the answer asserts specifics that are not supported by the cited document ids' content provided."""


def judge(llm: LLM, question: str, reference: str, answer: str, cited_texts: list[str]) -> dict:
    prompt = (
        f"Question: {question}\n\nReference answer: {reference}\n\nCandidate answer: {answer}\n\n"
        "Cited documents:\n" + "\n---\n".join(t[:1500] for t in cited_texts)
    )
    out = llm.step(JUDGE_SYSTEM, [{"role": "user", "content": prompt}], [])
    m = re.search(r"\{.*\}", out["text"], re.S)
    try:
        return json.loads(m.group(0)) if m else {"correct": 0, "grounded": False, "reason": "unparseable"}
    except json.JSONDecodeError:
        return {"correct": 0, "grounded": False, "reason": "unparseable"}
