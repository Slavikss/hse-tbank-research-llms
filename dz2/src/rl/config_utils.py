"""Shared configuration loaders for data/training/evaluation scripts."""

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
    "eval": {
        "difficulties": [2, 4, 6, 8, 10],
        "questions_per_difficulty": 300,
        "seed_base": 9000,
        "max_attempts": 100,
    },
    "output": {
        "train_path": "data/train/train.jsonl",
        "eval_dir": "data/eval",
    },
}

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "data": {
        "train_path": "data/train/train.jsonl",
        "eval_dir": "data/eval",
        "prediction_dir": "results/predictions",
    },
    "models": {
        "baseline": "Qwen/Qwen2.5-1.5B-Instruct",
        "trained": "outputs/merged_model",
    },
    "model": {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
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
        "output_dir": "outputs/grpo_runs",
        "max_steps": 800,
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_generations": 4,
        "max_prompt_length": 320,
        "max_completion_length": 128,
        "mask_truncated_completions": True,
        "logging_steps": 10,
        "save_steps": 100,
        "seed": 2026,
    },
    "output": {
        "adapter_dir": "outputs/adapter",
        "merged_dir": "outputs/merged_model",
        "save_merged": True,
    },
    "inference": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 96,
    },
    "dry_run": {
        "num_examples": 16,
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
