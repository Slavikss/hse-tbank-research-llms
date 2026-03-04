"""Hard subset mining where baseline pass@n equals zero."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.base.data import Data
from src.rl.config_utils import DEFAULT_HARD_CONFIG, load_config
from src.rl.passk_eval import evaluate_passk


def extract_zero_correct_indices(rows: list[dict[str, Any]]) -> list[int]:
    return [idx for idx, row in enumerate(rows) if int(row["c"]) == 0]


def select_hard_items(
    items: list[Data],
    zero_correct_indices: list[int],
    target: int,
    seed: int,
) -> list[Data]:
    selected = [items[idx] for idx in zero_correct_indices]
    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected[:target] if target > 0 else selected


def _mine_one_split(
    model_path: str,
    items: list[Data],
    backend_name: str,
    prefilter_n: int,
    certify_n: int,
    target: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> tuple[list[Data], dict[str, Any]]:
    prefilter_result = evaluate_passk(
        model_path=model_path,
        items=items,
        backend_name=backend_name,
        n=prefilter_n,
        k_values=[1, prefilter_n],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        batch_size=8,
        seed=seed,
    )
    prefilter_indices = extract_zero_correct_indices(prefilter_result["rows"])
    prefilter_candidates = [items[idx] for idx in prefilter_indices]

    certify_result = evaluate_passk(
        model_path=model_path,
        items=prefilter_candidates,
        backend_name=backend_name,
        n=certify_n,
        k_values=[1, certify_n],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        batch_size=4,
        seed=seed + 1,
    )
    certify_indices = extract_zero_correct_indices(certify_result["rows"])
    hard_items = select_hard_items(
        items=prefilter_candidates,
        zero_correct_indices=certify_indices,
        target=target,
        seed=seed,
    )

    stats = {
        "total": len(items),
        "prefilter_n": prefilter_n,
        "prefilter_zero": len(prefilter_indices),
        "certify_n": certify_n,
        "certify_zero": len(certify_indices),
        "selected": len(hard_items),
        "target": target,
    }
    return hard_items, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard subsets with baseline pass@n=0")
    parser.add_argument("--config", default="configs/hard.yaml")
    args = parser.parse_args()

    config = load_config(args.config, DEFAULT_HARD_CONFIG)
    data_cfg = config["data"]
    model_cfg = config["model"]
    sampling = config["sampling"]
    mining = config["hard_mining"]

    train_items = Data.from_jsonl_file(data_cfg["train_path"])
    val_items = Data.from_jsonl_file(data_cfg["val_path"])

    hard_train, train_stats = _mine_one_split(
        model_path=str(model_cfg["baseline"]),
        items=train_items,
        backend_name=str(model_cfg.get("backend", "auto")),
        prefilter_n=int(mining["prefilter_n"]),
        certify_n=int(mining["certify_n"]),
        target=int(mining["hard_train_target"]),
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        max_tokens=int(sampling["max_tokens"]),
        seed=int(mining["seed"]),
    )
    hard_val, val_stats = _mine_one_split(
        model_path=str(model_cfg["baseline"]),
        items=val_items,
        backend_name=str(model_cfg.get("backend", "auto")),
        prefilter_n=int(mining["prefilter_n"]),
        certify_n=int(mining["certify_n"]),
        target=int(mining["hard_val_target"]),
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        max_tokens=int(sampling["max_tokens"]),
        seed=int(mining["seed"] + 17),
    )

    hard_train_path = Path(data_cfg["hard_train_path"])
    hard_val_path = Path(data_cfg["hard_val_path"])
    Data.to_jsonl_file(hard_train, hard_train_path)
    Data.to_jsonl_file(hard_val, hard_val_path)

    report = {
        "hard_train": train_stats,
        "hard_val": val_stats,
        "paths": {
            "hard_train": str(hard_train_path),
            "hard_val": str(hard_val_path),
        },
    }
    report_path = Path("results/passk/hard_mining_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved hard-train: {hard_train_path} ({len(hard_train)} rows)")
    print(f"Saved hard-val: {hard_val_path} ({len(hard_val)} rows)")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
