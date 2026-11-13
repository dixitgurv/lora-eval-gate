import pytest

from evalgate.gate import GateConfig, evaluate, mcnemar_exact
from evalgate.metrics import ExampleScore, Report


def _report(name, outcomes, cats=None, valid=1.0):
    cats = cats or ["aggregate"] * len(outcomes)
    ex = [ExampleScore(id=str(i), correct=o, valid=True, exact=o, method="execution", categories=[c])
          for i, (o, c) in enumerate(zip(outcomes, cats, strict=True))]
    by_cat = {}
    for c in set(cats):
        vals = [e.correct for e in ex if c in e.categories]
        by_cat[c] = {"n": len(vals), "accuracy": sum(vals) / len(vals)}
    acc = sum(outcomes) / len(outcomes)
    return Report(name, len(ex), acc, acc, valid, 0.0, by_cat, ex)


def test_mcnemar_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 10) == 1.0
    assert mcnemar_exact(15, 2) == pytest.approx(0.002349, abs=1e-5)


def test_clear_improvement_passes():
    base = _report("base", [False] * 60 + [True] * 40)
    cand = _report("lora", [True] * 30 + [False] * 30 + [True] * 40)
    result = evaluate(base, cand)
    assert result.passed, result.checks
    assert (result.wins, result.losses) == (30, 0)


def test_small_noisy_gain_fails_significance():
    base = _report("base", [True, False] * 50)
    # flips 12 wrong->right and 9 right->wrong: +3 points, but not distinguishable from noise
    outcomes = [True, False] * 50
    for i in range(1, 24, 2):
        outcomes[i] = True
    for i in range(0, 18, 2):
        outcomes[i] = False
    cand = _report("lora", outcomes)
    result = evaluate(base, cand)
    names = {c.name: c.passed for c in result.checks}
    assert names["accuracy gain"] and not names["significance"] and not result.passed


def test_category_regression_blocks_promotion():
    cats = ["join"] * 40 + ["aggregate"] * 60
    base = _report("base", [True] * 30 + [False] * 10 + [False] * 60, cats)
    cand = _report("lora", [True] * 20 + [False] * 20 + [True] * 60, cats)  # big win overall, join drops 25 pts
    result = evaluate(base, cand)
    failed = [c.name for c in result.checks if not c.passed]
    assert failed == ["no regression: join"]


def test_small_categories_are_not_gated():
    cats = ["nested"] * 5 + ["aggregate"] * 95
    base = _report("base", [True] * 5 + [False] * 95, cats)
    cand = _report("lora", [False] * 5 + [True] * 95, cats)
    assert evaluate(base, cand, GateConfig(min_category_n=20)).passed


def test_different_golden_sets_are_rejected():
    with pytest.raises(ValueError):
        evaluate(_report("a", [True] * 3), _report("b", [True] * 4))
