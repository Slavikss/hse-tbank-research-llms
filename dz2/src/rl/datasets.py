"""Dataset generation utilities and CLI for train/eval JSONL files."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from src.base.data import Data
from src.envs.arithmetic_mod.env import ArithmeticModEnv
from src.rl.config_utils import DEFAULT_DATA_CONFIG, load_config
from src.rl.reward import format_prompt


def save_jsonl_dataset(items: list[Data], path: str | Path) -> None:
    Data.to_jsonl_file(items, path)


def load_jsonl_dataset(path: str | Path) -> list[Data]:
    return Data.from_jsonl_file(path)


def build_train_dataset(
    env: ArithmeticModEnv,
    total_questions: int,
    seed: int,
    max_attempts: int,
) -> list[Data]:
    if total_questions <= 0:
        raise ValueError("total_questions must be > 0")

    difficulties = list(range(1, 11))
    per_level = total_questions // len(difficulties)
    remainder = total_questions % len(difficulties)

    dataset: list[Data] = []
    for idx, difficulty in enumerate(difficulties):
        count = per_level + (1 if idx < remainder else 0)
        level_seed = seed * 100 + difficulty
        dataset.extend(
            env.generate(
                num_of_questions=count,
                max_attempts=max_attempts,
                difficulty=difficulty,
                seed=level_seed,
            )
        )
    return dataset


def build_eval_datasets(
    env: ArithmeticModEnv,
    difficulties: Iterable[int],
    questions_per_difficulty: int,
    seed_base: int,
    max_attempts: int,
) -> dict[int, list[Data]]:
    eval_sets: dict[int, list[Data]] = {}
    for difficulty in difficulties:
        eval_sets[int(difficulty)] = env.generate(
            num_of_questions=questions_per_difficulty,
            max_attempts=max_attempts,
            difficulty=int(difficulty),
            seed=seed_base + int(difficulty),
        )
    return eval_sets


def to_training_rows(items: list[Data]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in items:
        metadata = item.metadata or {}
        # Keep only fields required by reward verification to avoid pyarrow
        # overflow on huge intermediate integers (for example, raw expression value).
        row_metadata = {
            "modulus": int(metadata.get("modulus", 1)),
            "difficulty": item.difficulty,
        }
        rows.append(
            {
                "prompt": format_prompt(item.question),
                "question": item.question,
                "answer": item.answer,
                "difficulty": item.difficulty,
                "metadata": row_metadata,
            }
        )
    return rows


def summarize_dataset(items: list[Data]) -> dict[int, int]:
    counts: defaultdict[int, int] = defaultdict(int)
    for item in items:
        counts[item.difficulty] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate train/eval datasets for modular arithmetic RL task"
    )
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config YAML")
    parser.add_argument("--make-train", action="store_true", help="Generate train dataset")
    parser.add_argument("--make-eval", action="store_true", help="Generate eval datasets")
    parser.add_argument(
        "--print-summary", action="store_true", help="Print difficulty distributions"
    )
    args = parser.parse_args()

    config = load_config(args.config, DEFAULT_DATA_CONFIG)
    make_train = args.make_train or (not args.make_train and not args.make_eval)
    make_eval = args.make_eval or (not args.make_train and not args.make_eval)

    env = ArithmeticModEnv()

    if make_train:
        train_cfg = config["train"]
        train_items = build_train_dataset(
            env=env,
            total_questions=int(train_cfg["total_questions"]),
            seed=int(train_cfg["seed"]),
            max_attempts=int(train_cfg["max_attempts"]),
        )
        train_path = Path(config["output"]["train_path"])
        save_jsonl_dataset(train_items, train_path)
        print(f"Saved train dataset to {train_path} ({len(train_items)} rows)")
        if args.print_summary:
            print("Train difficulty distribution:", summarize_dataset(train_items))

    if make_eval:
        eval_cfg = config["eval"]
        eval_sets = build_eval_datasets(
            env=env,
            difficulties=eval_cfg["difficulties"],
            questions_per_difficulty=int(eval_cfg["questions_per_difficulty"]),
            seed_base=int(eval_cfg["seed_base"]),
            max_attempts=int(eval_cfg["max_attempts"]),
        )

        eval_dir = Path(config["output"]["eval_dir"])
        eval_dir.mkdir(parents=True, exist_ok=True)
        for difficulty, items in sorted(eval_sets.items()):
            path = eval_dir / f"difficulty_{difficulty}.jsonl"
            save_jsonl_dataset(items, path)
            print(f"Saved eval dataset to {path} ({len(items)} rows)")


if __name__ == "__main__":
    main()
