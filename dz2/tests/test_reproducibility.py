from __future__ import annotations

from pathlib import Path

from src.envs.arithmetic_mod.env import ArithmeticModEnv
from src.rl.datasets import build_eval_datasets, save_jsonl_dataset


def test_eval_dataset_reproducibility(tmp_path: Path) -> None:
    env = ArithmeticModEnv()
    first = build_eval_datasets(
        env=env,
        difficulties=[2],
        questions_per_difficulty=25,
        seed_base=9000,
        max_attempts=200,
    )[2]
    second = build_eval_datasets(
        env=env,
        difficulties=[2],
        questions_per_difficulty=25,
        seed_base=9000,
        max_attempts=200,
    )[2]

    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    save_jsonl_dataset(first, first_path)
    save_jsonl_dataset(second, second_path)

    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")
