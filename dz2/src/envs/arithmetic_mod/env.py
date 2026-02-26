"""Environment implementation for modular arithmetic expressions."""

from __future__ import annotations

import random
from typing import Optional

from src.base.data import Data
from src.base.env import Env
from src.envs.arithmetic_mod.generator import generate_expression, resolve_config
from src.envs.arithmetic_mod.prompt import build_question
from src.envs.arithmetic_mod.verifier import ArithmeticModVerifier


class ArithmeticModEnv(Env):
    """One-step environment where models compute expression modulo M."""

    def __init__(self) -> None:
        super().__init__(name="arithmetic_mod", verifier_cls=ArithmeticModVerifier)

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs: object,
    ) -> list[Data]:
        if num_of_questions <= 0:
            raise ValueError("num_of_questions must be > 0")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")

        seed = kwargs.pop("seed", None)
        rng = random.Random(int(seed)) if seed is not None else random.Random()

        config = resolve_config(difficulty, **kwargs)

        results: list[Data] = []
        for _ in range(num_of_questions):
            sample = generate_expression(rng=rng, config=config, max_attempts=max_attempts)
            gold_answer = sample.raw_value % config.modulus
            question = build_question(sample.expression, config.modulus)
            metadata = {
                "modulus": config.modulus,
                "expression": sample.expression,
                "raw_value": sample.raw_value,
                "config": config.to_dict(),
            }
            results.append(
                Data(
                    question=question,
                    answer=str(gold_answer),
                    difficulty=int(difficulty or 1),
                    metadata=metadata,
                )
            )
        return results

    def extract_answer(self, test_solution: str) -> str:
        return self.verifier.extract_answer(test_solution)
