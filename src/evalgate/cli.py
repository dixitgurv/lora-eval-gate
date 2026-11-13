"""evalgate: prepare data, train a LoRA adapter, predict, score, and gate.

    evalgate prepare                          # build data/{golden,dev,train}.jsonl
    evalgate train   --config config.yaml     # LoRA fine-tune -> adapters/
    evalgate predict --adapter adapters/sql-lora --out runs/lora.preds.jsonl
    evalgate score   runs/lora.preds.jsonl --model lora --out runs/lora.report.json
    evalgate gate    runs/base.report.json runs/lora.report.json   # exit 1 if it fails
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .gate import GateConfig, evaluate
from .metrics import Report, read_jsonl, score


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evalgate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="build fixed golden/dev/train splits")
    sp.add_argument("--out", default="data")
    sp.add_argument("--golden-size", type=int, default=400)

    st = sub.add_parser("train", help="LoRA fine-tune")
    st.add_argument("--config", default="config.yaml")

    pr = sub.add_parser("predict", help="generate SQL for a split")
    pr.add_argument("--base-model", default=None, help="defaults to base_model in config.yaml")
    pr.add_argument("--adapter", default=None)
    pr.add_argument("--data", default="data/golden.jsonl")
    pr.add_argument("--out", required=True)
    pr.add_argument("--batch-size", type=int, default=16)
    pr.add_argument("--limit", type=int, default=None)

    sc = sub.add_parser("score", help="score predictions against the golden set")
    sc.add_argument("predictions")
    sc.add_argument("--golden", default="data/golden.jsonl")
    sc.add_argument("--model", required=True, help="label for the report")
    sc.add_argument("--out", required=True)

    gt = sub.add_parser("gate", help="compare two reports; exit 1 if the candidate fails")
    gt.add_argument("baseline")
    gt.add_argument("candidate")
    gt.add_argument("--summary", default=None, help="also write the markdown report here")
    for field, default in GateConfig().__dict__.items():
        gt.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)

    a = p.parse_args(argv)

    if a.cmd == "prepare":
        from .data import prepare

        print(json.dumps(prepare(a.out, golden_size=a.golden_size), indent=2))
        return 0

    if a.cmd == "train":
        from .modeling import train

        cfg = yaml.safe_load(Path(a.config).read_text())
        print(json.dumps(train(cfg), indent=2))
        return 0

    if a.cmd == "predict":
        from .modeling import predict

        base = a.base_model or yaml.safe_load(Path("config.yaml").read_text())["base_model"]
        print(json.dumps(predict(base, a.adapter, a.data, a.out, a.batch_size, limit=a.limit)))
        return 0

    if a.cmd == "score":
        golden = read_jsonl(a.golden)
        preds = {r["id"]: r["prediction"] for r in read_jsonl(a.predictions)}
        golden = [g for g in golden if g["id"] in preds] if len(preds) < len(golden) else golden
        report = score(golden, preds, a.model)
        Path(a.out).write_text(report.to_json())
        summary = {k: v for k, v in report.__dict__.items() if k != "examples"}
        print(json.dumps(summary, indent=2))
        return 0

    if a.cmd == "gate":
        cfg = GateConfig(**{f: getattr(a, f) for f in GateConfig().__dict__})
        base, cand = Report.load(a.baseline), Report.load(a.candidate)
        result = evaluate(base, cand, cfg)
        md = result.markdown(base, cand)
        print(md)
        if a.summary:
            Path(a.summary).write_text(md)
        return 0 if result.passed else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
