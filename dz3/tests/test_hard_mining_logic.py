from __future__ import annotations

from src.base.data import Data
from src.rl.hard_mining import (
    extract_zero_correct_indices,
    intersect_zero_index_sets,
    select_hard_items,
)


def test_extract_zero_correct_indices() -> None:
    rows = [{"c": 0}, {"c": 2}, {"c": 0}, {"c": 1}]
    assert extract_zero_correct_indices(rows) == [0, 2]


def test_select_hard_items_respects_target() -> None:
    items = [Data(question=str(i), answer="0") for i in range(10)]
    out = select_hard_items(items, zero_correct_indices=[0, 1, 2, 3, 4], target=3, seed=42)
    assert len(out) == 3
    questions = {item.question for item in out}
    assert questions.issubset({"0", "1", "2", "3", "4"})


def test_intersect_zero_index_sets() -> None:
    assert intersect_zero_index_sets([[0, 1, 2], [1, 2, 3], [2, 4]]) == [2]
    assert intersect_zero_index_sets([]) == []
