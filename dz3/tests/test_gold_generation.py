from __future__ import annotations

from src.base.data import Data
from src.rl.gold import attach_gold


def test_gold_completion_is_added_and_verifiable() -> None:
    item = Data(
        question="q",
        answer="3",
        difficulty=1,
        metadata={"modulus": 7, "expression": "10", "raw_value": 10},
    )
    out = attach_gold([item])[0]
    assert "<think>" in out.gold_completion
    assert "<answer>" in out.gold_completion
    assert "3" in out.gold_completion
