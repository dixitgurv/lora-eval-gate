"""Model loading, LoRA training and batched generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .metrics import read_jsonl
from .prompt import extract_sql, messages


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(base_model: str, adapter: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"  # decoder-only batched generation needs left padding
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # fp32 on Apple Silicon: fp16 training on MPS is numerically unreliable, and 0.5B fits easily.
    dtype = torch.bfloat16 if device() == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()  # plain weights: faster generation, same outputs
    return model.to(device()).eval(), tok


@torch.no_grad()
def predict(base_model: str, adapter: str | None, data_path: str, out_path: str, batch_size: int = 16,
            max_new_tokens: int = 128, limit: int | None = None) -> dict[str, Any]:
    model, tok = load(base_model, adapter)
    rows = read_jsonl(data_path)[:limit]
    started = time.time()
    with open(out_path, "w") as out:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            prompts = [tok.apply_chat_template(messages(r["question"], r["context"]), tokenize=False,
                                               add_generation_prompt=True) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out.writelines(json.dumps({"id": r["id"], "prediction": extract_sql(text), "raw": text}) + "\n" for r, text in zip(batch, texts, strict=True))
            print(f"  {min(i + batch_size, len(rows))}/{len(rows)}", flush=True)
    return {"n": len(rows), "seconds": round(time.time() - started, 1)}


def _tokenize(tok, row: dict, max_len: int) -> dict[str, list[int]]:
    """Tokens for prompt + answer, with the prompt masked out of the loss.

    Training on the prompt too would spend capacity learning to repeat schemas.
    Masking means the loss measures only what we want: writing the SQL.
    """
    prompt = tok.apply_chat_template(messages(row["question"], row["context"]), tokenize=False,
                                     add_generation_prompt=True)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tok(row["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def _release_mps_cache():
    """A Trainer callback that returns PyTorch's cached GPU memory to macOS after each step.

    On Apple Silicon the MPS allocator keeps freed blocks cached. Batches have different
    lengths, so the cache fragments and grows until the OS swaps. A first run without this
    went from 33 s to 160 s per step over 100 steps.
    """
    from transformers import TrainerCallback

    class ReleaseMpsCache(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            torch.mps.empty_cache()

    return ReleaseMpsCache()


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model, tok = load(cfg["base_model"])
    tok.padding_side = "right"  # training pads on the right
    model.train()
    model.enable_input_require_grads()
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=cfg["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()

    rows = read_jsonl(cfg["train_path"])[: cfg.get("train_examples")]
    ds = Dataset.from_list([_tokenize(tok, r, cfg["max_len"]) for r in rows]).shuffle(seed=cfg["seed"])
    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        lr_scheduler_type="cosine",
        warmup_steps=cfg["warmup_steps"],
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=cfg["seed"],
        dataloader_pin_memory=False,
        train_sampling_strategy="group_by_length",  # batch similar lengths together: less padding, steadier memory
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[_release_mps_cache()] if device() == "mps" else None)
    started = time.time()
    result = trainer.train()
    model.save_pretrained(cfg["output_dir"])
    summary = {
        "train_examples": len(rows),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "final_loss": round(result.training_loss, 4),
        "minutes": round((time.time() - started) / 60, 1),
        "device": device(),
        "loss_curve": [round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h],
    }
    Path(cfg["output_dir"], "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
