"""The promotion gate: should the candidate model replace the baseline?

A higher average is not enough. The gate asks four questions, and all must pass:
  1. Did execution accuracy improve by at least `min_gain`?
  2. Is that improvement real, not noise? (exact McNemar test on paired outcomes)
  3. Did any query category get worse by more than `max_category_drop`?
  4. Is the candidate still producing runnable SQL at least `min_valid_rate` of the time?
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from .metrics import Report


@dataclass(frozen=True)
class GateConfig:
    min_gain: float = 0.02
    max_p_value: float = 0.05
    max_category_drop: float = 0.03
    min_category_n: int = 20  # categories smaller than this are reported, not gated
    min_valid_rate: float = 0.95


def _pp(delta: float) -> str:
    """A difference between two rates, in percentage points (not percent)."""
    return f"{delta * 100:+.1f} pp"


def _p(p: float) -> str:
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    passed: bool
    checks: list[Check]
    wins: int
    losses: int
    p_value: float

    def markdown(self, base: Report, cand: Report) -> str:
        lines = [
            f"## Eval gate: {'PASS' if self.passed else 'FAIL'}",
            "",
            f"Baseline `{base.model}` vs candidate `{cand.model}` on {cand.n} golden examples.",
            "",
            "| Metric | Baseline | Candidate | Change |",
            "|---|---|---|---|",
        ]
        for label, attr in [("Execution accuracy", "execution_accuracy"), ("Exact match", "exact_match"),
                            ("Valid SQL", "valid_sql_rate")]:
            b, c = getattr(base, attr), getattr(cand, attr)
            lines.append(f"| {label} | {b:.1%} | {c:.1%} | {_pp(c - b)} |")
        lines += ["", "| Category | n | Baseline | Candidate | Change |", "|---|---|---|---|---|"]
        for cat, stats in cand.by_category.items():
            b = base.by_category.get(cat, {}).get("accuracy", 0.0)
            c = stats["accuracy"]
            lines.append(f"| {cat} | {stats['n']} | {b:.1%} | {c:.1%} | {_pp(c - b)} |")
        lines += ["", (f"Paired outcomes: candidate fixed {self.wins}, broke {self.losses} "
                      f"(McNemar exact p {_p(self.p_value) if self.p_value < 0.0001 else "= " + _p(self.p_value)})."), "", "| Check | Result | Detail |", "|---|---|---|"]
        lines += [f"| {c.name} | {'pass' if c.passed else 'FAIL'} | {c.detail} |" for c in self.checks]
        return "\n".join(lines) + "\n"


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value: under H0, discordant pairs split 50/50."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def evaluate(base: Report, cand: Report, cfg: GateConfig = GateConfig()) -> GateResult:
    base_by_id = {e.id: e.correct for e in base.examples}
    cand_by_id = {e.id: e.correct for e in cand.examples}
    if base_by_id.keys() != cand_by_id.keys():
        raise ValueError("Baseline and candidate were scored on different golden sets.")

    wins = sum(cand_by_id[i] and not base_by_id[i] for i in base_by_id)
    losses = sum(base_by_id[i] and not cand_by_id[i] for i in base_by_id)
    p = mcnemar_exact(wins, losses)
    gain = cand.execution_accuracy - base.execution_accuracy

    checks = [
        Check("accuracy gain", gain >= cfg.min_gain,
              f"{_pp(gain)} (need at least {_pp(cfg.min_gain)})"),
        Check("significance", p <= cfg.max_p_value and wins > losses,
              f"p {'< 0.0001' if p < 0.0001 else '= ' + _p(p)} (need at most {cfg.max_p_value}); fixed {wins}, broke {losses}"),
        Check("valid SQL rate", cand.valid_sql_rate >= cfg.min_valid_rate,
              f"{cand.valid_sql_rate:.1%} (need at least {cfg.min_valid_rate:.0%})"),
    ]
    for cat, stats in cand.by_category.items():
        if stats["n"] < cfg.min_category_n or cat not in base.by_category:
            continue
        drop = base.by_category[cat]["accuracy"] - stats["accuracy"]
        checks.append(Check(f"no regression: {cat}", drop <= cfg.max_category_drop,
                            f"{_pp(-drop)} on {stats['n']} examples (max drop {cfg.max_category_drop * 100:.0f} pp)"))

    return GateResult(all(c.passed for c in checks), checks, wins, losses, round(p, 6))
