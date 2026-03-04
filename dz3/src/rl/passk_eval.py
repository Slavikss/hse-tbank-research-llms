"""Pass@k evaluation for baseline and trained models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from src.base.data import Data
from src.envs.arithmetic_mod.verifier import ArithmeticModVerifier
from src.rl.config_utils import DEFAULT_TRAIN_CONFIG, load_config
from src.rl.infer_backend import SamplingConfig, create_backend
from src.rl.reward import format_prompt


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimate from n samples and c correct."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if c < 0 or c > n:
        raise ValueError("c must be in [0, n]")
    if k <= 0 or k > n:
        raise ValueError("k must be in [1, n]")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _chunks[T](items: list[T], size: int) -> Iterable[list[T]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _approx_length(text: str) -> int:
    return len(text.split())


def evaluate_passk(
    model_path: str,
    items: list[Data],
    backend_name: str,
    n: int,
    k_values: list[int],
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    verifier = ArithmeticModVerifier()
    backend = create_backend(model_path=model_path, backend=backend_name)
    sampling = SamplingConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        seed=int(seed),
    )

    row_metrics: list[dict[str, Any]] = []
    all_lengths: list[int] = []

    for batch in _chunks(items, size=max(1, int(batch_size))):
        prompts = [format_prompt(item.question) for item in batch]
        generated = backend.generate(
            prompts=prompts,
            n=n,
            sampling=sampling,
            stop_strings=["</answer>"],
        )

        for item, completions in zip(batch, generated, strict=True):
            correctness = [verifier.verify(item, text) for text in completions]
            correct_count = sum(1 for ok in correctness if ok)
            lengths = [_approx_length(text) for text in completions]
            all_lengths.extend(lengths)

            row = {
                "question": item.question,
                "difficulty": item.difficulty,
                "n": n,
                "c": correct_count,
                "pass_at_k": {
                    str(k): compute_pass_at_k(n=n, c=correct_count, k=k) for k in k_values
                },
                "mean_gen_len": mean(lengths) if lengths else 0.0,
                "median_gen_len": median(lengths) if lengths else 0.0,
            }
            row_metrics.append(row)

    summary = {
        "total": len(row_metrics),
        "pass_at_k": {
            str(k): (
                mean(float(row["pass_at_k"][str(k)]) for row in row_metrics) if row_metrics else 0.0
            )
            for k in k_values
        },
        "mean_gen_len": mean(all_lengths) if all_lengths else 0.0,
        "median_gen_len": median(all_lengths) if all_lengths else 0.0,
    }

    return {
        "model_path": model_path,
        "backend": backend.backend_name,
        "n": n,
        "k_values": k_values,
        "summary": summary,
        "rows": row_metrics,
    }


def _resolve_model_path(model: str, config: dict[str, Any]) -> str:
    models = config.get("models", {})
    if model in models:
        return str(models[model])
    return model


def _resolve_split_path(split: str, config: dict[str, Any]) -> Path:
    data_cfg = config["data"]
    mapping = {
        "val": Path(data_cfg["val_path"]),
        "hard_val": Path(data_cfg["hard_val_path"]),
        "hard_train": Path(data_cfg["hard_train_path"]),
    }
    if split not in mapping:
        raise ValueError("split must be one of: val, hard_val, hard_train")
    return mapping[split]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pass@k with n samples per prompt")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", default="baseline")
    parser.add_argument("--split", required=True, choices=["val", "hard_val", "hard_train"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config, DEFAULT_TRAIN_CONFIG)
    model_path = _resolve_model_path(args.model, config)
    split_path = _resolve_split_path(args.split, config)
    if not split_path.exists():
        raise RuntimeError(f"Split path does not exist: {split_path}")

    items = Data.from_jsonl_file(split_path)
    infer_cfg = config["inference"]
    passk_cfg = config["passk"]

    result = evaluate_passk(
        model_path=model_path,
        items=items,
        backend_name=str(infer_cfg.get("backend", "auto")),
        n=int(passk_cfg["n"]),
        k_values=[int(k) for k in passk_cfg["k_values"]],
        temperature=float(infer_cfg["temperature"]),
        top_p=float(infer_cfg["top_p"]),
        max_tokens=int(infer_cfg["max_tokens"]),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )

    output_path = (
        Path(args.output)
        if args.output
        else Path("results/passk") / f"{args.model}_{args.split}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved pass@k metrics to {output_path}")
    print("Summary pass@k:", result["summary"]["pass_at_k"])


if __name__ == "__main__":
    main()
