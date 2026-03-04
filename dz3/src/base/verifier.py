"""Base verifier contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.base.data import Data


class Verifier(ABC):
    """Base class for verifier."""

    @abstractmethod
    def verify(self, data: Data, test_answer: str) -> bool:
        """Verify whether the test answer is consistent with the gold answer."""
        raise NotImplementedError("Verifier.verify() is not implemented")

    @abstractmethod
    def extract_answer(self, test_solution: str) -> str:
        """Extract the answer from a model response."""
        raise NotImplementedError("Verifier.extract_answer() is not implemented")
