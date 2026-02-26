"""GRPO training entrypoint for Qwen2.5-1.5B-Instruct with Unsloth."""

from __future__ import annotations

import argparse
import importlib
import types
from pathlib import Path
from typing import Any

from src.base.data import Data
from src.rl.config_utils import DEFAULT_TRAIN_CONFIG, load_config
from src.rl.datasets import to_training_rows
from src.rl.reward import build_reward_func


def _align_completion_tensors(inputs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Truncate mismatched completion tensors to a shared token length."""
    try:
        import torch
    except Exception:
        return inputs, False

    token_keys = (
        "completion_ids",
        "completion_mask",
        "old_per_token_logps",
        "ref_per_token_logps",
        "sampling_per_token_logps",
        "importance_sampling_ratio",
    )
    lengths: list[int] = []
    for key in token_keys:
        value = inputs.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            lengths.append(int(value.size(1)))

    if not lengths:
        return inputs, False
    target = min(lengths)
    if all(length == target for length in lengths):
        return inputs, False

    aligned_inputs = dict(inputs)
    for key in token_keys:
        value = aligned_inputs.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2 and int(value.size(1)) != target:
            aligned_inputs[key] = value[:, :target]
    return aligned_inputs, True


def _patch_grpo_trainer_for_shape_guard(trainer: Any) -> None:
    """Guard against rare Unsloth/TRL completion length mismatches."""
    original_compute_loss = trainer.compute_loss
    warned = {"printed": False}

    def _safe_compute_loss(
        self: Any,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        aligned_inputs, changed = _align_completion_tensors(inputs)
        if changed and not warned["printed"]:
            print(
                "[train_grpo] Detected completion tensor length mismatch. "
                "Applying safe truncation to continue training."
            )
            warned["printed"] = True
        return original_compute_loss(model, aligned_inputs, return_outputs, num_items_in_batch)

    trainer.compute_loss = types.MethodType(_safe_compute_loss, trainer)


def _validate_sequence_lengths(
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> tuple[int, int]:
    max_seq_length = int(model_cfg["max_seq_length"])
    max_prompt_length = int(train_cfg["max_prompt_length"])
    max_completion_length = int(train_cfg["max_completion_length"])
    if max_prompt_length + max_completion_length > max_seq_length:
        raise ValueError(
            "Invalid lengths: max_prompt_length + max_completion_length "
            f"must be <= model.max_seq_length ({max_seq_length})."
        )
    return max_prompt_length, max_completion_length


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

    reward_fn = build_reward_func(log_samples=False)
    fake = ["<answer>0</answer>" for _ in rows]
    rewards = reward_fn(
        completions=fake,
        answer=[str(item["answer"]) for item in rows],
        metadata=[item["metadata"] for item in rows],
    )
    avg_reward = sum(rewards) / len(rewards)
    print(f"Dry run completed on {len(rows)} samples. Mean dummy reward={avg_reward:.3f}")


def _resolve_generation_kwargs(train_cfg: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    """Prepare generation kwargs and drop unsupported options for local generation."""
    use_vllm = bool(train_cfg.get("use_vllm", False))
    generation_kwargs_raw = train_cfg.get("generation_kwargs")

    if generation_kwargs_raw is None:
        generation_kwargs: dict[str, Any] | None = None
    elif isinstance(generation_kwargs_raw, dict):
        generation_kwargs = dict(generation_kwargs_raw)
    else:
        raise ValueError("training.generation_kwargs must be a dict when provided")

    if generation_kwargs and not use_vllm and "stop_strings" in generation_kwargs:
        generation_kwargs.pop("stop_strings", None)
        print(
            "[train_grpo] Dropped training.generation_kwargs.stop_strings for non-vLLM mode "
            "(transformers generate in GRPO does not receive tokenizer)."
        )

    if generation_kwargs == {}:
        generation_kwargs = None

    return generation_kwargs, use_vllm


def _load_training_modules() -> tuple[Any, Any, Any, Any]:
    """Load optional training modules with Unsloth imported first."""
    try:
        unsloth_mod = importlib.import_module("unsloth")
        datasets_mod = importlib.import_module("datasets")
        trl_mod = importlib.import_module("trl")
    except ImportError as exc:
        raise RuntimeError(
            "Missing or incompatible training dependencies. "
            "Install compatible unsloth/unsloth_zoo/vllm + datasets + trl."
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
    model_cfg = config["model"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    max_prompt_length, max_completion_length = _validate_sequence_lengths(model_cfg, train_cfg)

    FastLanguageModel, HFDataset, GRPOConfig, GRPOTrainer = _load_training_modules()

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

    # NOTE: trl==0.24.0 GRPOTrainer raises NotImplementedError for IterableDataset,
    # so we intentionally keep a map-style dataset here.
    train_dataset = HFDataset.from_list(rows)
    reward_fn = build_reward_func(
        log_samples=bool(train_cfg.get("log_samples", True)),
        log_every_calls=int(train_cfg.get("log_every_calls", 20)),
        log_max_items=int(train_cfg.get("log_max_items", 3)),
        log_max_chars=int(train_cfg.get("log_max_chars", 240)),
    )

    generation_kwargs, use_vllm = _resolve_generation_kwargs(train_cfg)

    training_args = GRPOConfig(
        output_dir=str(train_cfg["output_dir"]),
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        max_steps=int(train_cfg["max_steps"]),
        num_generations=int(train_cfg["num_generations"]),
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        temperature=float(train_cfg.get("temperature", 0.2)),
        top_p=float(train_cfg.get("top_p", 0.9)),
        use_vllm=use_vllm,
        generation_kwargs=generation_kwargs,
        mask_truncated_completions=bool(train_cfg.get("mask_truncated_completions", True)),
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
    _patch_grpo_trainer_for_shape_guard(trainer)
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
