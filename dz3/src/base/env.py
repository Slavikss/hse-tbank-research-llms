"""Base environment interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.base.data import Data
from src.base.verifier import Verifier


class Env(ABC):
    """Base class for one-step environments."""

    def __init__(self, name: str, verifier_cls: type[Verifier]):
        self.name = name
        self.verifier = verifier_cls()

    @abstractmethod
    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs: object,
    ) -> list[Data]:
        """Generate environment samples."""
        raise NotImplementedError("Env.generate() is not implemented")

    def verify(self, data: Data, test_solution: str) -> bool:
        """Proxy verification to verifier implementation."""
        return self.verifier.verify(data, test_solution)

    @abstractmethod
    def extract_answer(self, test_solution: str) -> str:
        """Extract answer from model completion."""
        raise NotImplementedError("Env.extract_answer() is not implemented")
