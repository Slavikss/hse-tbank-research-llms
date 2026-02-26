"""Prompt builder for modular arithmetic expression tasks."""

from __future__ import annotations


def build_question(expression: str, modulus: int) -> str:
    """Create an unambiguous English prompt for a single task."""
    return (
        "You are solving a one-step arithmetic task.\n"
        "Compute the value of the expression modulo M.\n"
        "Rules:\n"
        "1) Allowed operators are +, -, and *.\n"
        "2) Standard arithmetic precedence applies (* before + and -).\n"
        "3) Parentheses, if present, override precedence.\n"
        "4) Return one integer in the range [0, M-1].\n"
        "5) The final answer must be wrapped in <answer>...</answer>.\n"
        f"Expression: {expression}\n"
        f"M: {modulus}\n"
        "Output only one final integer in <answer> tags."
    )
