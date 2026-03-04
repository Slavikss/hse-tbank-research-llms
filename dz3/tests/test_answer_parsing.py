from __future__ import annotations

from src.envs.arithmetic_mod.verifier import ArithmeticModVerifier


def test_extract_answer_tags() -> None:
    verifier = ArithmeticModVerifier()
    text = """
<think>
step by step
</think>
<answer>
42
</answer>
"""
    assert verifier.extract_answer(text) == "42"


def test_extract_answer_fallback_last_int() -> None:
    verifier = ArithmeticModVerifier()
    text = "I tried 12 and then 27, final result is 33"
    assert verifier.extract_answer(text) == "33"
