"""vLLM inference utility for evaluating baseline and trained models."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.base.data import Data
from src.rl.reward import format_prompt


def run_inference(
    model_path: str,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 96,
    limit: int | None = None,
) -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vllm is required for inference. Install vllm first.") from exc

    samples = Data.from_jsonl_file(input_jsonl)
    if limit is not None:
        samples = samples[:limit]

    prompts = [format_prompt(item.question) for item in samples]

    llm = LLM(model=model_path)
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts, sampling_params=params)

    for sample, result in zip(samples, outputs, strict=True):
        if result.outputs:
            sample.gpt_response = result.outputs[0].text
        else:
            sample.gpt_response = ""

    Data.to_jsonl_file(samples, output_jsonl)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM inference on JSONL dataset")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        limit=args.limit,
    )
    print(f"Predictions saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
