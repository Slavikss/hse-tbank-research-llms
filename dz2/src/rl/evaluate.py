"""Evaluate baseline and trained models on fixed difficulty datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.base.data import Data
from src.envs.arithmetic_mod.verifier import PARSE_FAIL, ArithmeticModVerifier
from src.rl.config_utils import DEFAULT_TRAIN_CONFIG, load_config
from src.rl.infer_vllm import run_inference


def _difficulty_from_path(path: Path) -> int:
    name = path.stem
    return int(name.split("_")[-1])


def _classify_error(data: Data, completion: str, verifier: ArithmeticModVerifier) -> str:
    extracted = verifier.extract_answer(completion)
    if extracted == PARSE_FAIL:
        return "parse"

    try:
        value = int(extracted)
        modulus = int((data.metadata or {}).get("modulus"))
        gold = int(data.answer)
    except (TypeError, ValueError):
        return "parse"

    if value % modulus == gold:
        return "correct"
    if value < 0 or value >= modulus:
        return "modulo_sign"
    return "arithmetic"


def _compute_metrics(items: list[Data]) -> dict[str, Any]:
    verifier = ArithmeticModVerifier()
    total = len(items)
    if total == 0:
        return {
            "total": 0,
            "accuracy": 0.0,
            "parse_success_rate": 0.0,
            "error_counts": {"parse": 0, "arithmetic": 0, "modulo_sign": 0},
        }

    correct = 0
    parse_fail = 0
    error_counts = {"parse": 0, "arithmetic": 0, "modulo_sign": 0}

    for item in items:
        completion = item.gpt_response or ""
        ok = verifier.verify(item, completion)
        if ok:
            correct += 1
            continue
        err = _classify_error(item, completion, verifier)
        if err in error_counts:
            error_counts[err] += 1
        if err == "parse":
            parse_fail += 1

    return {
        "total": total,
        "accuracy": correct / total,
        "parse_success_rate": (total - parse_fail) / total,
        "error_counts": error_counts,
    }


def evaluate_model(
    model_alias: str,
    config: dict[str, Any],
    skip_inference: bool,
) -> dict[str, Any]:
    if model_alias not in {"baseline", "trained"}:
        raise ValueError("--model must be baseline or trained")

    model_path = str(config["models"][model_alias])
    eval_dir = Path(config["data"]["eval_dir"])
    prediction_root = Path(config["data"]["prediction_dir"]) / model_alias
    prediction_root.mkdir(parents=True, exist_ok=True)

    infer_cfg = config.get("inference", {})
    metrics_by_difficulty: dict[int, dict[str, Any]] = {}

    eval_files = sorted(eval_dir.glob("difficulty_*.jsonl"), key=_difficulty_from_path)
    if not eval_files:
        raise RuntimeError(f"No eval files found in {eval_dir}")

    for eval_file in eval_files:
        difficulty = _difficulty_from_path(eval_file)
        pred_file = prediction_root / eval_file.name

        if not skip_inference:
            run_inference(
                model_path=model_path,
                input_jsonl=eval_file,
                output_jsonl=pred_file,
                temperature=float(infer_cfg.get("temperature", 0.0)),
                top_p=float(infer_cfg.get("top_p", 1.0)),
                max_tokens=int(infer_cfg.get("max_tokens", 96)),
            )

        if not pred_file.exists():
            raise RuntimeError(
                f"Prediction file {pred_file} does not exist. Run without --skip-inference first."
            )

        predicted = Data.from_jsonl_file(pred_file)
        metrics_by_difficulty[difficulty] = _compute_metrics(predicted)

    macro_accuracy = sum(m["accuracy"] for m in metrics_by_difficulty.values()) / len(
        metrics_by_difficulty
    )

    result = {
        "model_alias": model_alias,
        "model_path": model_path,
        "macro_accuracy": macro_accuracy,
        "per_difficulty": metrics_by_difficulty,
    }
    return result


def save_metrics(result: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    alias = result["model_alias"]
    json_path = out / f"metrics_{alias}.json"
    csv_path = out / f"metrics_{alias}.csv"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "difficulty",
                "accuracy",
                "parse_success_rate",
                "parse_errors",
                "arithmetic_errors",
                "modulo_sign_errors",
                "total",
            ]
        )
        for difficulty, metrics in sorted(result["per_difficulty"].items()):
            errors = metrics["error_counts"]
            writer.writerow(
                [
                    difficulty,
                    f"{metrics['accuracy']:.6f}",
                    f"{metrics['parse_success_rate']:.6f}",
                    errors.get("parse", 0),
                    errors.get("arithmetic", 0),
                    errors.get("modulo_sign", 0),
                    metrics["total"],
                ]
            )

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model quality across fixed difficulty datasets"
    )
    parser.add_argument(
        "--config", default="configs/train.yaml", help="Path to training/evaluation config"
    )
    parser.add_argument("--model", required=True, choices=["baseline", "trained"])
    parser.add_argument(
        "--skip-inference", action="store_true", help="Use already generated predictions"
    )
    args = parser.parse_args()

    config = load_config(args.config, DEFAULT_TRAIN_CONFIG)
    result = evaluate_model(
        model_alias=args.model, config=config, skip_inference=args.skip_inference
    )
    json_path, csv_path = save_metrics(result, out_dir="results/metrics")
    print(f"Saved metrics to {json_path} and {csv_path}")
    print(f"Macro accuracy: {result['macro_accuracy']:.4f}")


if __name__ == "__main__":
    main()
