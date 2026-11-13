## Eval gate: PASS

Baseline `Qwen2.5-0.5B-Instruct (base)` vs candidate `Qwen2.5-0.5B-Instruct + LoRA` on 400 golden examples.

| Metric | Baseline | Candidate | Change |
|---|---|---|---|
| Execution accuracy | 40.8% | 80.0% | +39.3 pp |
| Exact match | 31.0% | 70.2% | +39.2 pp |
| Valid SQL | 87.5% | 97.5% | +10.0 pp |

| Category | n | Baseline | Candidate | Change |
|---|---|---|---|---|
| aggregate | 107 | 38.3% | 66.4% | +28.0 pp |
| group_by | 2 | 50.0% | 100.0% | +50.0 pp |
| nested | 3 | 33.3% | 0.0% | -33.3 pp |
| order_by | 4 | 25.0% | 75.0% | +50.0 pp |
| simple_filter | 289 | 42.2% | 85.5% | +43.3 pp |

Paired outcomes: candidate fixed 162, broke 5 (McNemar exact p < 0.0001).

| Check | Result | Detail |
|---|---|---|
| accuracy gain | pass | +39.3 pp (need at least +2.0 pp) |
| significance | pass | p < 0.0001 (need at most 0.05); fixed 162, broke 5 |
| valid SQL rate | pass | 97.5% (need at least 95%) |
| no regression: aggregate | pass | +28.0 pp on 107 examples (max drop 3 pp) |
| no regression: simple_filter | pass | +43.3 pp on 289 examples (max drop 3 pp) |
