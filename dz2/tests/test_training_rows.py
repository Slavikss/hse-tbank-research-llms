from __future__ import annotations

from src.base.data import Data
from src.rl.datasets import to_training_rows


def test_to_training_rows_drops_large_raw_value_from_metadata() -> None:
    item = Data(
        question="Compute expression modulo M",
        answer="1",
        difficulty=9,
        metadata={
            "modulus": 37,
            "raw_value": 10**80,
            "expression": "500 * 500 * 500 * 500 * 500 * 500 * 500 * 500 * 500",
        },
    )

    rows = to_training_rows([item])
    assert len(rows) == 1
    metadata = rows[0]["metadata"]
    assert metadata == {"modulus": 37, "difficulty": 9}
