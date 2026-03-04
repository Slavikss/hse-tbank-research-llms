from __future__ import annotations

from src.rl.passk_eval import compute_pass_at_k


def test_passk_zero_when_no_correct() -> None:
    assert compute_pass_at_k(n=128, c=0, k=128) == 0.0


def test_passk_one_when_all_correct() -> None:
    assert compute_pass_at_k(n=128, c=128, k=64) == 1.0


def test_passk_monotonic() -> None:
    p1 = compute_pass_at_k(n=128, c=2, k=1)
    p8 = compute_pass_at_k(n=128, c=2, k=8)
    p64 = compute_pass_at_k(n=128, c=2, k=64)
    assert p1 <= p8 <= p64
