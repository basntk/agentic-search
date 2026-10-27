"""LLM interface: a thin normalized layer so the agent loop is testable offline.

step() returns {"text": str, "tool_calls": [{"id","name","input"}], "assistant_content": list, "usage": {...}}
"""
from __future__ import annotations

import os
from typing import Protocol


class LLM(Protocol):
    def step(self, system: str, messages: list[dict], tools: list[dict]) -> dict: ...


class AnthropicLLM:
    """Requires `pip install anthropic` and ANTHROPIC_API_KEY. Set AGENT_MODEL to override."""

    def __init__(self, model: str | None = None, max_tokens: int = 1500):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        kwargs = {"tools": tools} if tools else {}
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            **kwargs,
        )
        text, calls, content = "", [], []
        for block in resp.content:
            content.append(block.model_dump(exclude_none=True))
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                calls.append({"id": block.id, "name": block.name, "input": block.input})
        return {
            "text": text,
            "tool_calls": calls,
            "assistant_content": content,
            "usage": {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens},
        }


class ScriptedLLM:
    """Deterministic stand-in for tests: replays a list of steps.
    Each scripted step is a list of (tool_name, input) tuples, or a string for plain text."""

    def __init__(self, script: list):
        self.script = list(script)
        self.i = 0

    def step(self, system, messages, tools):
        item = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        calls, content = [], []
        if isinstance(item, str):
            content.append({"type": "text", "text": item})
            return {"text": item, "tool_calls": [], "assistant_content": content, "usage": {"in": 0, "out": 0}}
        for j, (name, inp) in enumerate(item):
            cid = f"t{self.i}_{j}"
            calls.append({"id": cid, "name": name, "input": inp})
            content.append({"type": "tool_use", "id": cid, "name": name, "input": inp})
        return {"text": "", "tool_calls": calls, "assistant_content": content, "usage": {"in": 0, "out": 0}}
