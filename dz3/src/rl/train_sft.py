"""Supervised fine-tuning on algorithmic gold trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.base.data import Data
from src.rl.config_utils import DEFAULT_TRAIN_CONFIG, load_config
from src.rl.reward import format_prompt


def _load_rows(train_path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    items = Data.from_jsonl_file(train_path)
    if limit is not None:
        items = items[:limit]

    rows: list[dict[str, str]] = []
    for item in items:
        if not item.gold_completion:
            continue
        prompt = format_prompt(item.question)
        text = f"{prompt}\n{item.gold_completion}"
        rows.append({"text": text})
    return rows


def _run_dry(config: dict[str, Any]) -> None:
    dry_cfg = config.get("dry_run", {})
    limit = int(dry_cfg.get("num_examples", 16))
    rows = _load_rows(config["data"]["train_path"], limit=limit)
    if not rows:
        raise RuntimeError("SFT dry run failed: no rows with gold_completion")
    avg_chars = sum(len(row["text"]) for row in rows) / len(rows)
    print(f"SFT dry run completed on {len(rows)} examples. mean_chars={avg_chars:.1f}")


def _train(config: dict[str, Any]) -> None:
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Missing dependencies for SFT (datasets/transformers/peft)") from exc

    model_cfg = config["model"]
    sft_cfg = config["sft"]

    rows = _load_rows(config["data"]["train_path"], limit=None)
    if not rows:
        raise RuntimeError("No SFT rows found. Run gold generation first.")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=True,
        device_map="auto",
    )

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(model_cfg["lora_r"]),
        lora_alpha=int(model_cfg["lora_alpha"]),
        lora_dropout=float(model_cfg["lora_dropout"]),
        target_modules=list(model_cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)

    hf_dataset = Dataset.from_list(rows)

    max_length = int(sft_cfg.get("max_length", 512))

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        encoded["labels"] = [ids[:] for ids in encoded["input_ids"]]
        return encoded

    tokenized = hf_dataset.map(_tokenize, batched=True, remove_columns=["text"])

    output_dir = Path(sft_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(sft_cfg["learning_rate"]),
        num_train_epochs=float(sft_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(sft_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(sft_cfg["gradient_accumulation_steps"]),
        logging_steps=int(sft_cfg["logging_steps"]),
        save_steps=int(sft_cfg["save_steps"]),
        report_to="none",
        remove_unused_columns=False,
        seed=int(sft_cfg["seed"]),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"SFT training completed. Saved model to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SFT model on gold trajectories")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, DEFAULT_TRAIN_CONFIG)
    if args.dry_run:
        _run_dry(config)
    else:
        _train(config)


if __name__ == "__main__":
    main()
