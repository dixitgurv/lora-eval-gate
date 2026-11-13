# lora-eval-gate

[![CI](https://github.com/dixitgurv/lora-eval-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/dixitgurv/lora-eval-gate/actions/workflows/ci.yml)

A LoRA fine-tune of a small language model for text-to-SQL, and the evaluation harness and promotion gate that decide whether it's allowed to replace the base model.

The fine-tune is the easy part. The point of this repo is the question that comes after it: **is the new model actually better, and is it safe to ship?** The answer comes from a fixed golden set, an execution-based scorer, and a gate that exits non-zero, so CI can block a merge.

## Result

Qwen2.5-0.5B-Instruct, before and after a 15-minute LoRA fine-tune on an Apple M1 laptop, scored on 400 held-out examples:

| Metric | Base | + LoRA | Change |
|---|---|---|---|
| Execution accuracy | 40.8% | **80.0%** | +39.3 pp |
| Exact match | 31.0% | 70.2% | +39.2 pp |
| Valid SQL | 87.5% | 97.5% | +10.0 pp |

| Query category | n | Base | + LoRA |
|---|---|---|---|
| simple filter | 289 | 42.2% | 85.5% |
| aggregate | 107 | 38.3% | 66.4% |
| order by | 4 | 25.0% | 75.0% |
| nested | 3 | 33.3% | 0.0% |
| group by | 2 | 50.0% | 100.0% |

**Gate: PASS.** On the same 400 questions, the candidate fixed 162 and broke 5 (exact McNemar p < 0.0001). No gated category regressed, and the valid-SQL rate cleared 95%. The full report is in [`runs/gate.md`](runs/gate.md).

### Reading the result honestly

- **About a third of the gain is learning the dataset's conventions.** In 54 of the 162 fixes, the base model wrote the right query but changed a value's form, for example `'Silverstone'` for the question's "silverstone", or `2011-12` for `2011–12`. Those return different rows, so they're genuinely wrong. But "copy the value exactly as the user wrote it" is a convention the fine-tune taught, not deeper SQL reasoning. Another 35 fixes turned unrunnable output into valid SQL. The other 73 are real logic fixes (right column, right aggregate, right condition).
- **Joins and nesting are barely tested.** The source data is mostly single-table WikiSQL, so the golden set has 3 nested queries and 0 joins. The fine-tuned model got all 3 nested ones wrong, and the gate correctly *doesn't* treat a 3-example category as evidence either way. A harder golden set is the first thing I'd add.
- **The 5 regressions are typical over-eagerness:** inventing a filter the question didn't ask for (`WHERE student = "All Students"`), wrapping a column in `COUNT`, or rewriting a correct `LIKE '%gmail.com%'` as `= "gmail.com"`.

## How it works

```
prepare ──► golden 400 (committed) · dev 200 · train 6,000   (no overlap, near-duplicates excluded)
train   ──► LoRA adapter (rank 16, all projections, 1.75% of parameters, 35 MB)
predict ──► runs/base.preds.jsonl, runs/lora.preds.jsonl
score   ──► execution accuracy on synthetic SQLite databases, per-category breakdown
gate    ──► PASS / FAIL, exit code 0 / 1, markdown summary
```

- **Scoring runs the SQL.** `count(*) where 5 < age` and `COUNT(*) WHERE age > 5` are the same answer, which string matching misses. Each schema is turned into small SQLite databases seeded with the gold query's own literal values, so `WHERE` clauses actually match rows. A prediction must return the same rows as the gold query on all three seeds.
- **The gate needs more than a higher average.** It requires a gain of at least 2 points, a significant paired test, no category with n ≥ 20 dropping more than 3 points, and a valid-SQL rate of at least 95%. A unit test covers a candidate that gains 30 points overall while `join` drops 25. The gate blocks it.
- **CI re-scores the committed predictions** on every push and runs the gate, so the numbers above are reproduced from raw model output by the current harness, not copied from a report. No GPU is needed.

Design notes for each decision, including a bug the harness found in itself and a training run that slowed down from 33 to 160 seconds per step, are in [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Run it

Scoring and the gate need only the core install:

```bash
uv sync
uv run evalgate score runs/lora.preds.jsonl --model "my run" --out my.report.json
uv run evalgate gate runs/base.report.json my.report.json
```

The full pipeline needs the `train` extra (PyTorch, transformers, PEFT). It runs on CUDA, Apple Silicon or CPU:

```bash
uv sync --extra train
uv run evalgate prepare                       # downloads b-mc2/sql-create-context, writes data/
uv run evalgate predict --out runs/base.preds.jsonl
uv run evalgate train --config config.yaml    # ~15 min on an M1
uv run evalgate predict --adapter adapters/sql-lora --out runs/lora.preds.jsonl
uv run evalgate score runs/lora.preds.jsonl --model "base + LoRA" --out runs/lora.report.json
uv run evalgate gate runs/base.report.json runs/lora.report.json
```

Gate thresholds are flags: `--min-gain`, `--max-p-value`, `--max-category-drop`, `--min-category-n`, `--min-valid-rate`.

## Layout

```
src/evalgate/
  sqlexec.py    schema parsing, synthetic databases, execution check
  metrics.py    scoring, categories, reports
  gate.py       promotion checks and exact McNemar test
  data.py       leak-free splits
  prompt.py     the one prompt format shared by training and inference
  modeling.py   LoRA training and batched generation
  cli.py        evalgate prepare | train | predict | score | gate
data/golden.jsonl   the 400-example contract
runs/               committed predictions, reports, gate output, training summary
```

## Data and licence

Training and evaluation data come from [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) (CC BY 4.0), which is derived from WikiSQL and Spider. The base model is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Apache 2.0). The code is MIT.
