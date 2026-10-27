"""Corpus loading. A corpus is a JSONL file: {"id": str, "title": str, "text": str}."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Doc:
    id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}"


def load_jsonl(path: str | Path) -> list[Doc]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                docs.append(Doc(id=str(d["id"]), title=d.get("title", ""), text=d["text"]))
    return docs
