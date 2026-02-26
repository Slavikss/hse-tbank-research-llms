from __future__ import annotations

import pytest

from src.rl.reward import SYSTEM_PROMPT, build_reward_func
from src.rl.train_grpo import _validate_sequence_lengths


def test_system_prompt_strict_pdf_format() -> None:
    assert "Respond in the following format:" in SYSTEM_PROMPT
    assert "<think>" in SYSTEM_PROMPT
    assert "</think>" in SYSTEM_PROMPT
    assert "<answer>" in SYSTEM_PROMPT
    assert "</answer>" in SYSTEM_PROMPT


class _SpyEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str]] = []

    def verify(self, data: object, test_solution: str) -> bool:
        self.calls.append((data, test_solution))
        return True


def test_reward_calls_env_verify() -> None:
    env = _SpyEnv()
    reward_fn = build_reward_func(env=env)

    rewards = reward_fn(
        completions=["<answer>4</answer>", "<answer>9</answer>"],
        answer=["4", "9"],
        metadata=[
            {"modulus": 11, "difficulty": 3},
            {"modulus": 19, "difficulty": 6},
        ],
    )

    assert rewards == [1.0, 1.0]
    assert len(env.calls) == 2

    first_data, first_solution = env.calls[0]
    assert first_data.answer == "4"
    assert first_data.metadata == {"modulus": 11, "difficulty": 3}
    assert first_solution == "<answer>4</answer>"


def test_training_lengths_guard() -> None:
    prompt_len, completion_len = _validate_sequence_lengths(
        model_cfg={"max_seq_length": 512},
        train_cfg={"max_prompt_length": 448, "max_completion_length": 64},
    )
    assert prompt_len == 448
    assert completion_len == 64

    with pytest.raises(ValueError):
        _validate_sequence_lengths(
            model_cfg={"max_seq_length": 512},
            train_cfg={"max_prompt_length": 490, "max_completion_length": 64},
        )
