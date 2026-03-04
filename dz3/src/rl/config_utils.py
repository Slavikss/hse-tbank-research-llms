"""Shared configuration loaders for DZ3 scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATA_CONFIG: dict[str, Any] = {
    "train": {
        "total_questions": 24000,
        "seed": 2026,
        "max_attempts": 100,
    },
    "val": {
        "difficulties": [2, 4, 6, 8, 10],
        "questions_per_difficulty": 300,
        "seed_base": 9000,
        "max_attempts": 100,
    },
    "output": {
        "train_path": "data/train/train.jsonl",
        "val_path": "data/val/val.jsonl",
        "val_dir": "data/val/by_difficulty",
    },
}

DEFAULT_HARD_CONFIG: dict[str, Any] = {
    "data": {
        "train_path": "data/train/train.jsonl",
        "val_path": "data/val/val.jsonl",
        "hard_train_path": "data/hard/hard_train.jsonl",
        "hard_val_path": "data/hard/hard_val.jsonl",
    },
    "model": {
        "baseline": "Qwen/Qwen2.5-0.5B-Instruct",
        "backend": "auto",
    },
    "sampling": {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 96,
    },
    "hard_mining": {
        "prefilter_n": 16,
        "certify_n": 128,
        "hard_train_target": 2048,
        "hard_val_target": 256,
        "seed": 2026,
    },
}

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "data": {
        "train_path": "data/train/train.jsonl",
        "val_path": "data/val/val.jsonl",
        "hard_train_path": "data/hard/hard_train.jsonl",
        "hard_val_path": "data/hard/hard_val.jsonl",
        "prediction_dir": "results/predictions",
        "curves_dir": "results/curves",
    },
    "models": {
        "baseline": "Qwen/Qwen2.5-0.5B-Instruct",
        "grpo": "outputs/grpo/merged_model",
        "sft": "outputs/sft",
        "sft_grpo": "outputs/sft_grpo/merged_model",
        "srft": "outputs/srft",
    },
    "model": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_seq_length": 512,
        "load_in_4bit": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "training": {
        "output_dir": "outputs/grpo/runs",
        "max_steps": 600,
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_generations": 4,
        "max_prompt_length": 416,
        "max_completion_length": 96,
        "mask_truncated_completions": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 96,
        "use_vllm": False,
        "log_samples": True,
        "log_every_calls": 20,
        "log_max_items": 3,
        "log_max_chars": 240,
        "logging_steps": 10,
        "save_steps": 100,
        "seed": 2026,
    },
    "sft": {
        "output_dir": "outputs/sft",
        "learning_rate": 2e-5,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "max_length": 512,
        "logging_steps": 10,
        "save_steps": 100,
        "seed": 2026,
    },
    "srft": {
        "output_dir": "outputs/srft",
        "learning_rate": 2e-5,
        "steps": 200,
        "batch_size": 2,
        "w_sft_coef": 0.5,
        "w_rl_coef": 0.1,
        "clip_eps": 0.2,
        "seed": 2026,
    },
    "inference": {
        "backend": "auto",
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 96,
    },
    "passk": {
        "n": 128,
        "k_values": [1, 4, 8, 16, 32, 64, 128],
    },
    "dry_run": {
        "num_examples": 16,
        "steps": 2,
    },
}


def load_config(path: str | Path, defaults: dict[str, Any]) -> dict[str, Any]:
    """Load YAML/JSON config and merge it with defaults."""
    config_path = Path(path)
    if not config_path.exists():
        return defaults

    raw = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        parsed = json.loads(raw)
    else:
        parsed = yaml.safe_load(raw) or {}

    return _deep_merge(defaults, parsed)


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(left)
    for key, value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
