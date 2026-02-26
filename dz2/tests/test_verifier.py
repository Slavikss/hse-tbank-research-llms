from __future__ import annotations

from src.base.data import Data
from src.envs.arithmetic_mod.verifier import ArithmeticModVerifier


def _make_data(answer: str, modulus: int) -> Data:
    return Data(question="", answer=answer, difficulty=1, metadata={"modulus": modulus})


def test_verify_correct() -> None:
    verifier = ArithmeticModVerifier()
    data = _make_data(answer="5", modulus=11)
    assert verifier.verify(data, "<answer>5</answer>")


def test_verify_wrong() -> None:
    verifier = ArithmeticModVerifier()
    data = _make_data(answer="5", modulus=11)
    assert not verifier.verify(data, "<answer>7</answer>")


def test_verify_parse_fail() -> None:
    verifier = ArithmeticModVerifier()
    data = _make_data(answer="5", modulus=11)
    assert not verifier.verify(data, "no integer answer here")


def test_modulo_normalization() -> None:
    verifier = ArithmeticModVerifier()
    data = _make_data(answer="8", modulus=11)
    assert verifier.verify(data, "<answer>-3</answer>")
