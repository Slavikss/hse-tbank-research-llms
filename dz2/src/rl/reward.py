"""Reward helpers for GRPO training."""

from __future__ import annotations

from collections.abc import Sequence

from src.base.data import Data
from src.envs.arithmetic_mod.verifier import ArithmeticModVerifier

SYSTEM_PROMPT = """
Respond in the following format:
<answer>
...
</answer>
Do not output anything outside <answer> tags.
""".strip()


def format_prompt(question: str) -> str:
    """Combine system prompt and task question into one model prompt."""
    return f"{SYSTEM_PROMPT}\n\n{question}"


def _completion_to_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "text" in completion:
            return str(completion["text"])
        if "content" in completion:
            return str(completion["content"])
    if isinstance(completion, Sequence):
        chunks: list[str] = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                chunks.append(str(item["content"]))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(completion)


def build_reward_func(verifier: ArithmeticModVerifier | None = None):
    """Create correctness reward callback for GRPOTrainer."""
    local_verifier = verifier or ArithmeticModVerifier()

    def correctness_reward_func(
        completions: Sequence[object],
        answer: Sequence[str] | None = None,
        metadata: Sequence[dict[str, object]] | None = None,
        **_: object,
    ) -> list[float]:
        if answer is None:
            return [0.0 for _ in completions]

        md = metadata or [{} for _ in answer]
        rewards: list[float] = []
        for completion, gold_answer, item_metadata in zip(completions, answer, md, strict=True):
            row = Data(
                question="",
                answer=str(gold_answer),
                difficulty=int(
                    item_metadata.get("difficulty", 1) if isinstance(item_metadata, dict) else 1
                ),
                metadata=item_metadata if isinstance(item_metadata, dict) else {},
            )
            ok = local_verifier.verify(row, _completion_to_text(completion))
            rewards.append(1.0 if ok else 0.0)
        return rewards

    return correctness_reward_func
