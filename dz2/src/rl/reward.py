"""Reward helpers for GRPO training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.base.data import Data
from src.envs.arithmetic_mod.env import ArithmeticModEnv
from src.envs.arithmetic_mod.verifier import PARSE_FAIL

SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
Keep <think> concise (one short line).
Stop immediately after </answer>.
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


def _shorten(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def build_reward_func(
    env: Any | None = None,
    log_samples: bool = False,
    log_every_calls: int = 20,
    log_max_items: int = 3,
    log_max_chars: int = 240,
):
    """Create correctness reward callback as an Env.verify wrapper."""
    local_env = env or ArithmeticModEnv()
    call_state = {"count": 0}

    def correctness_reward_func(
        completions: Sequence[object],
        answer: Sequence[str] | None = None,
        metadata: Sequence[dict[str, object]] | None = None,
        **extra: object,
    ) -> list[float]:
        if answer is None:
            return [0.0 for _ in completions]

        md = metadata or [{} for _ in answer]
        call_state["count"] += 1

        log_every = max(1, int(log_every_calls))
        should_log = bool(log_samples) and (call_state["count"] % log_every == 0)

        questions = extra.get("question")
        question_seq = None
        if isinstance(questions, Sequence) and not isinstance(questions, (str, bytes)):
            question_seq = questions

        rewards: list[float] = []
        for idx, (completion, gold_answer, item_metadata) in enumerate(
            zip(completions, answer, md, strict=True)
        ):
            row = Data(
                question="",
                answer=str(gold_answer),
                difficulty=int(
                    item_metadata.get("difficulty", 1) if isinstance(item_metadata, dict) else 1
                ),
                metadata=item_metadata if isinstance(item_metadata, dict) else {},
            )
            completion_text = _completion_to_text(completion)
            ok = local_env.verify(row, completion_text)
            rewards.append(1.0 if ok else 0.0)

            if should_log and idx < max(1, int(log_max_items)):
                extracted = local_env.extract_answer(completion_text)
                modulus = (row.metadata or {}).get("modulus")
                normalized = "n/a"
                if extracted != PARSE_FAIL:
                    try:
                        normalized = str(int(extracted) % int(modulus))
                    except Exception:
                        normalized = "n/a"

                question_text = ""
                if question_seq is not None and idx < len(question_seq):
                    question_text = str(question_seq[idx])

                print(
                    "[reward-log] "
                    f"call={call_state['count']} sample={idx} "
                    f"difficulty={row.difficulty} modulus={modulus} "
                    f"gold={row.answer} extracted={extracted} normalized={normalized} ok={ok}"
                )
                if question_text:
                    print(f"[reward-log] question={_shorten(question_text, int(log_max_chars))}")
                print(f"[reward-log] completion={_shorten(completion_text, int(log_max_chars))}")

        return rewards

    return correctness_reward_func
