# How it works

Design notes for every decision in this repo, written so that anyone can explain them.

## The question this repo answers

"Should this fine-tuned model replace the one in production?" Training loss can't answer that, and neither can a higher average score on its own. You need:

1. a fixed test set the model never trained on,
2. a scoring method that measures what you actually care about,
3. a decision rule that holds up against noise and catches regressions the average hides.

The repo builds all three for one concrete task, then runs a real fine-tune through them.

## Pipeline

```
prepare ──► data/golden.jsonl (committed, 400)   data/train.jsonl (6,000)
                    │                                     │
                    │                              train (LoRA) ──► adapters/sql-lora
                    ▼                                     │
predict (base) ──► runs/base.preds.jsonl      predict (base + adapter) ──► runs/lora.preds.jsonl
                    │                                     │
               score ──► base.report.json          score ──► lora.report.json
                                  └────────── gate ──────────┘
                                         exit 0 = promote, exit 1 = block
```

## The task and the model

**Text-to-SQL**: given a `CREATE TABLE` schema and a question, write a SQLite query. The data is [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) (CC BY 4.0, built from WikiSQL and Spider).

I chose text-to-SQL because it can be scored objectively. Whether an answer is "good" is a matter of opinion for most generation tasks. A SQL query either returns the right rows or it doesn't.

**Qwen2.5-0.5B-Instruct** is small enough to fine-tune on a laptop (Apple M1, 16 GB) in under an hour, and good enough that the baseline is a real model, not a strawman.

## Splits: no leakage

`data.py` shuffles once with a fixed seed and takes the golden set first, then dev, then train. Two details matter:

- **Near-duplicates.** The source data has repeated question/schema pairs. Removing golden *ids* from training isn't enough, so any training example with the same normalized question and schema as a golden example is excluded too.
- **Usable gold only.** An example enters the golden set only if its gold query runs on the synthetic databases *and returns rows*. If the gold query errors or returns nothing, there's no way to score a prediction against it. About 12% of candidates are skipped for this reason.

The golden set is committed. Anyone can re-score the committed predictions and get identical numbers, and CI does exactly that.

## Scoring: execution, not string match

`sqlexec.py` builds a small SQLite database from the schema, runs the gold query and the prediction, and compares the results. `count(*) ... where 5 < age` and `COUNT(*) ... WHERE age > 5` are both correct, and string matching would mark one of them wrong.

Synthetic data has a known weakness: two different queries can both return nothing and look "equal". Three defences:

1. **Data from the query's own literals.** If the gold query filters on `circuit = 'silverstone'` and `round > 10`, those values (and 9, 10, 11) are put into the data, so the filter actually selects rows.
2. **Three databases, three seeds.** A prediction must match on all of them.
3. **Empty-result fallback.** If the gold query returns nothing on all three, execution proves nothing, so that example falls back to exact match.

Results are compared as multisets, except when the gold query has `ORDER BY`, when order counts. Numbers are rounded to 4 places and numeric strings compare as numbers.

A progress handler aborts any query that runs too long. A bad prediction can be an accidental cross join.

### A bug the harness found in itself

The first baseline run reported **exact match 46.5% but execution accuracy 40.8%**. That is impossible if both are right: a query identical to the gold one must also return the same rows.

The cause was the exact-match normaliser. It lower-cased the whole query, string values included, so `'Silverstone'` matched `"silverstone"`. Those return different rows in SQLite, and the execution check was right. The normaliser now keeps the case of string literals, and exact match dropped to 31.0%, below execution accuracy where it belongs. There's a regression test for it.

The lesson: when two metrics should be ordered (exact ⊆ execution) and aren't, one of them is broken.

## Training: LoRA with a masked loss

`modeling.train()`:

- **LoRA** rank 16, alpha 32, on all attention and MLP projections (`q,k,v,o,gate,up,down`). That trains about 1.7% of the parameters, and the adapter file is a few tens of MB instead of a 2 GB checkpoint.
- **Prompt masking.** Labels are `-100` for every prompt token, so loss is computed only on the SQL answer. Without masking, much of the gradient goes into learning to reproduce schemas the model is given anyway.
- **One prompt format** (`prompt.py`) is shared by training and inference. A mismatch between the two is a classic silent failure.
- **fp32 on Apple Silicon.** fp16 training on MPS is numerically unreliable, and a 0.5B model fits in fp32 comfortably. On CUDA it uses bf16.
- Effective batch 16 (2 × 8 gradient accumulation), learning rate 2e-4, cosine decay, one epoch over 2,000 examples. Sequences are capped at 256 tokens; the longest training example is 229, so nothing is truncated.

### A training run that slowed itself down

The first run (4,000 examples, batch 4) started at 33 seconds per step and was at 160 seconds per step by step 100, with macOS swapping 8 GB. The cause is the MPS allocator: it caches freed GPU blocks for reuse, batches have different lengths, so the cache fragments and keeps growing until the OS swaps. Closing other apps didn't help.

The rerun used a Trainer callback that calls `torch.mps.empty_cache()` after every step, length-grouped batching (less padding, similar shapes), and a smaller per-device batch with more accumulation. It held a steady 8 seconds per step with swap under 2 GB and finished in 15 minutes. The point is that the step time stayed flat. A rising step time on Apple Silicon is a memory problem, not a compute one.

For inference the adapter is merged into the base weights (`merge_and_unload`): same outputs, faster generation. Decoding is greedy, so results are deterministic.

## The gate

`gate.py` passes the candidate only if **all** checks pass:

| Check | Why |
|---|---|
| Execution accuracy improves by ≥ 2 points | Retraining and redeploying has a cost; a 0.3-point gain isn't worth it |
| Exact McNemar test, p ≤ 0.05, more fixes than breaks | Separates real improvement from noise on a 400-example set |
| No category drops by more than 3 points (categories with n ≥ 20) | An average can rise while a whole query type gets worse |
| Valid-SQL rate ≥ 95% | A model that emits unrunnable SQL is a production incident, whatever its accuracy |

**Why McNemar.** Both models answer the same 400 questions, so the outcomes are paired. What matters is the discordant pairs: questions the candidate fixed versus questions it broke. Under "no real difference", those split 50/50, and the exact binomial test gives the probability of a split at least this lopsided. Comparing two accuracies as if they came from independent samples would ignore the pairing and be far less sensitive.

**Why per-category checks.** The tests include a case where the candidate gains 30 points overall while `join` accuracy drops 25 points. The average says ship it, and the gate says no. Categories are detected from the gold SQL (aggregate, group by, order by, join, nested, simple filter). Categories with fewer than 20 examples are reported but not gated, because a 3-point threshold on 4 examples means nothing.

The gate exits 1 on failure, so it can block a merge in CI. The markdown summary is written to the GitHub Actions job summary.

## What I'd do next

- A harder golden set: the source data is mostly single-table WikiSQL, so joins and nesting are rare (see the category counts in the README).
- Several training seeds, to separate the effect of the method from the luck of one run.
- Schema-grounding checks: flag predictions that use a column not in the schema before even running them.
