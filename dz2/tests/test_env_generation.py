from __future__ import annotations

from src.envs.arithmetic_mod.env import ArithmeticModEnv


def test_generate_count() -> None:
    env = ArithmeticModEnv()
    items = env.generate(num_of_questions=25, max_attempts=200, difficulty=3, seed=42)
    assert len(items) == 25


def test_generate_difficulty_range() -> None:
    env = ArithmeticModEnv()
    for difficulty in range(1, 11):
        items = env.generate(
            num_of_questions=5,
            max_attempts=250,
            difficulty=difficulty,
            seed=100 + difficulty,
        )
        assert len(items) == 5
        for item in items:
            modulus = int(item.metadata["modulus"])
            answer = int(item.answer)
            assert item.difficulty == difficulty
            assert 0 <= answer < modulus


def test_generate_hyperparams_override() -> None:
    env = ArithmeticModEnv()
    items = env.generate(
        num_of_questions=8,
        max_attempts=300,
        difficulty=1,
        n_terms=5,
        modulus=19,
        operators=["*"],
        max_parentheses_depth=1,
        seed=777,
    )

    for item in items:
        cfg = item.metadata["config"]
        assert cfg["n_terms"] == 5
        assert cfg["modulus"] == 19
        assert tuple(cfg["operators"]) == ("*",)
        assert cfg["max_parentheses_depth"] == 1


def test_prompt_is_english_and_unambiguous() -> None:
    env = ArithmeticModEnv()
    sample = env.generate(num_of_questions=1, max_attempts=200, difficulty=4, seed=404)[0]
    q = sample.question

    assert "Rules:" in q
    assert "Expression:" in q
    assert "M:" in q
    assert "<answer>" in q
    assert "Output only one final integer" in q
    assert all(ord(ch) < 128 for ch in q)
