"""Verifier implementation for modular arithmetic expressions."""

from __future__ import annotations

import re

from src.base.data import Data
from src.base.verifier import Verifier

PARSE_FAIL = "__PARSE_FAIL__"

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
_INT_RE = re.compile(r"[-+]?\d+")


class ArithmeticModVerifier(Verifier):
    """Extract model answers and validate them against gold values."""

    def extract_answer(self, test_solution: str) -> str:
        if not isinstance(test_solution, str):
            return PARSE_FAIL

        tagged = _ANSWER_TAG_RE.search(test_solution)
        if tagged:
            inside = tagged.group(1)
            match = _INT_RE.search(inside)
            if match:
                return match.group(0)
            return PARSE_FAIL

        all_ints = _INT_RE.findall(test_solution)
        if all_ints:
            return all_ints[-1]
        return PARSE_FAIL

    def verify(self, data: Data, test_answer: str) -> bool:
        extracted = self.extract_answer(test_answer)
        if extracted == PARSE_FAIL:
            return False

        try:
            predicted = int(extracted)
            modulus = int((data.metadata or {}).get("modulus"))
            gold = int(str(data.answer).strip())
        except (TypeError, ValueError):
            return False

        normalized = predicted % modulus
        return normalized == gold
