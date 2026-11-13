"""Build fixed, non-overlapping train / dev / golden splits.

The golden set is the contract. It is chosen once, filtered to examples whose gold
query actually runs and returns rows on the synthetic databases, written to
data/golden.jsonl and committed. Training never sees it: the train split excludes
golden ids *and* any example with the same question and schema, so near-duplicates
in the source data can't leak answers into training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .sqlexec import gold_is_usable

DATASET = "b-mc2/sql-create-context"


def _key(row: dict) -> str:
    return (row["question"].strip().lower(), row["context"].strip().lower()).__repr__()


def prepare(out_dir: str | Path, golden_size: int = 400, dev_size: int = 200, train_size: int = 6000,
            seed: int = 42) -> dict[str, int]:
    from datasets import load_dataset  # heavy import; keep it off the scoring path

    rows = [dict(r, id=f"sc-{i}") for i, r in enumerate(load_dataset(DATASET, split="train"))]
    random.Random(seed).shuffle(rows)

    golden, dev, used_keys = [], [], set()
    i = 0
    while len(golden) < golden_size or len(dev) < dev_size:
        row = rows[i]
        i += 1
        if _key(row) in used_keys or not gold_is_usable(row["context"], row["answer"]):
            continue
        used_keys.add(_key(row))
        (golden if len(golden) < golden_size else dev).append(row)

    train = [r for r in rows[i:] if _key(r) not in used_keys][:train_size]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in [("golden", golden), ("dev", dev), ("train", train)]:
        with open(out / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps({k: r[k] for k in ("id", "question", "context", "answer")}) + "\n" for r in split)
    return {"golden": len(golden), "dev": len(dev), "train": len(train), "scanned": i}
