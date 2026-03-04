"""Generate algorithmic gold trajectories for modular arithmetic tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.base.data import Data
from src.envs.arithmetic_mod.verifier import ArithmeticModVerifier


def build_gold_completion(item: Data) -> str:
    metadata = item.metadata or {}
    modulus = int(metadata.get("modulus", 1))
    expression = str(metadata.get("expression", "<unknown expression>"))

    raw_value = metadata.get("raw_value")
    if raw_value is None:
        raw_text = "unknown"
    else:
        raw_text = str(raw_value)

    answer = int(item.answer)
    return (
        "<think>\n"
        f"Evaluate {expression}. Raw value = {raw_text}. "
        f"Take modulo {modulus} to get {answer}.\n"
        "</think>\n"
        "<answer>\n"
        f"{answer}\n"
        "</answer>"
    )


def attach_gold(items: list[Data]) -> list[Data]:
    verifier = ArithmeticModVerifier()
    output: list[Data] = []
    for item in items:
        completion = build_gold_completion(item)
        if not verifier.verify(item, completion):
            raise RuntimeError("Generated gold completion failed verifier check")
        item.gold_completion = completion
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach algorithmic gold completions to JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    items = Data.from_jsonl_file(in_path)
    items = attach_gold(items)
    Data.to_jsonl_file(items, out_path)
    print(f"Saved gold trajectories to {out_path} ({len(items)} rows)")


if __name__ == "__main__":
    main()
