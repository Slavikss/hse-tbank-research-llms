"""GRPO training entrypoint for Qwen2.5-1.5B-Instruct with Unsloth."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from src.base.data import Data
from src.rl.config_utils import DEFAULT_TRAIN_CONFIG, load_config
from src.rl.datasets import to_training_rows
from src.rl.reward import build_reward_func


def _load_train_rows(train_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    data_items = Data.from_jsonl_file(train_path)
    if limit is not None:
        data_items = data_items[:limit]
    return to_training_rows(data_items)


def _run_dry(config: dict[str, Any]) -> None:
    dry_cfg = config.get("dry_run", {})
    limit = int(dry_cfg.get("num_examples", 16))
    train_path = config["data"]["train_path"]
    rows = _load_train_rows(train_path=train_path, limit=limit)
    if not rows:
        raise RuntimeError("Dry run failed: train dataset is empty")

    reward_fn = build_reward_func()
    fake = ["<answer>0</answer>" for _ in rows]
    rewards = reward_fn(
        completions=fake,
        answer=[str(item["answer"]) for item in rows],
        metadata=[item["metadata"] for item in rows],
    )
    avg_reward = sum(rewards) / len(rewards)
    print(f"Dry run completed on {len(rows)} samples. Mean dummy reward={avg_reward:.3f}")


def _load_training_modules() -> tuple[Any, Any, Any, Any]:
    """Load optional training modules with Unsloth imported first."""
    try:
        unsloth_mod = importlib.import_module("unsloth")
        datasets_mod = importlib.import_module("datasets")
        trl_mod = importlib.import_module("trl")
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional training dependencies. Install datasets, trl, and unsloth."
        ) from exc
    except NotImplementedError as exc:
        raise RuntimeError(
            "Unsloth runtime is unavailable on this machine. Use a GPU runtime (NVIDIA/AMD/Intel)."
        ) from exc

    return (
        unsloth_mod.FastLanguageModel,
        datasets_mod.Dataset,
        trl_mod.GRPOConfig,
        trl_mod.GRPOTrainer,
    )


def _train(config: dict[str, Any]) -> None:
    FastLanguageModel, HFDataset, GRPOConfig, GRPOTrainer = _load_training_modules()

    model_cfg = config["model"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    rows = _load_train_rows(train_path=config["data"]["train_path"], limit=None)
    if not rows:
        raise RuntimeError("Train dataset is empty. Generate data first.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=int(model_cfg["max_seq_length"]),
        dtype=None,
        load_in_4bit=bool(model_cfg["load_in_4bit"]),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(model_cfg["lora_r"]),
        lora_alpha=int(model_cfg["lora_alpha"]),
        lora_dropout=float(model_cfg["lora_dropout"]),
        target_modules=list(model_cfg["target_modules"]),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(train_cfg["seed"]),
    )

    train_dataset = HFDataset.from_list(rows)
    reward_fn = build_reward_func()

    training_args = GRPOConfig(
        output_dir=str(train_cfg["output_dir"]),
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        max_steps=int(train_cfg["max_steps"]),
        num_generations=int(train_cfg["num_generations"]),
        max_prompt_length=int(train_cfg["max_prompt_length"]),
        max_completion_length=int(train_cfg["max_completion_length"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        seed=int(train_cfg["seed"]),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    adapter_dir = Path(output_cfg["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if bool(output_cfg.get("save_merged", True)):
        merged_dir = Path(output_cfg["merged_dir"])
        merged_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained_merged"):
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        else:
            model.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))

    print("Training completed and checkpoints saved.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GRPO agent for modular arithmetic environment"
    )
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training config")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run smoke check without model training"
    )
    args = parser.parse_args()

    config = load_config(args.config, DEFAULT_TRAIN_CONFIG)
    if args.dry_run:
        _run_dry(config)
    else:
        _train(config)


if __name__ == "__main__":
    main()
