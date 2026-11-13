"""Score a predictions file against the golden set."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .sqlexec import check, normalize_sql

CATEGORY_PATTERNS = {
    "aggregate": r"\b(count|sum|avg|min|max)\s*\(",
    "group_by": r"\bgroup\s+by\b",
    "order_by": r"\border\s+by\b",
    "join": r"\bjoin\b",
    "nested": r"\(\s*select\b",
}


def categories(gold_sql: str) -> list[str]:
    cats = [name for name, pat in CATEGORY_PATTERNS.items() if re.search(pat, gold_sql, re.IGNORECASE)]
    return cats or ["simple_filter"]


@dataclass
class ExampleScore:
    id: str
    correct: bool
    valid: bool
    exact: bool
    method: str
    categories: list[str]
    error: str | None = None


@dataclass
class Report:
    model: str
    n: int
    execution_accuracy: float
    exact_match: float
    valid_sql_rate: float
    exact_fallback_share: float
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    examples: list[ExampleScore] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Report:
        raw = json.loads(Path(path).read_text())
        raw["examples"] = [ExampleScore(**e) for e in raw.get("examples", [])]
        return cls(**raw)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score(golden: list[dict], predictions: dict[str, str], model: str) -> Report:
    examples = []
    for row in golden:
        pred = predictions.get(row["id"], "")
        res = check(row["context"], row["answer"], pred)
        examples.append(ExampleScore(
            id=row["id"], correct=res.correct, valid=res.valid,
            exact=normalize_sql(pred) == normalize_sql(row["answer"]),
            method=res.method, categories=categories(row["answer"]), error=res.error,
        ))

    n = len(examples)
    per_cat: dict[str, list[bool]] = defaultdict(list)
    for e in examples:
        for c in e.categories:
            per_cat[c].append(e.correct)
    return Report(
        model=model,
        n=n,
        execution_accuracy=round(sum(e.correct for e in examples) / n, 4),
        exact_match=round(sum(e.exact for e in examples) / n, 4),
        valid_sql_rate=round(sum(e.valid for e in examples) / n, 4),
        exact_fallback_share=round(sum(e.method == "exact_fallback" for e in examples) / n, 4),
        by_category={c: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for c, v in sorted(per_cat.items())},
        examples=examples,
    )
