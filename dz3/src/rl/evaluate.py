"""Convenience wrapper over pass@k evaluation for standard splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.base.data import Data
from src.rl.config_utils import DEFAULT_TRAIN_CONFIG, load_config
from src.rl.passk_eval import evaluate_passk


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one model on val and hard-val splits")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=["val", "hard_val", "both"], default="both")
    args = parser.parse_args()

    config = load_config(args.config, DEFAULT_TRAIN_CONFIG)
    model_path = str(config.get("models", {}).get(args.model, args.model))

    split_paths = {
        "val": Path(config["data"]["val_path"]),
        "hard_val": Path(config["data"]["hard_val_path"]),
    }
    infer_cfg = config["inference"]
    passk_cfg = config["passk"]

    run_splits = [args.split] if args.split != "both" else ["val", "hard_val"]

    for split in run_splits:
        items = Data.from_jsonl_file(split_paths[split])
        result = evaluate_passk(
            model_path=model_path,
            items=items,
            backend_name=str(infer_cfg["backend"]),
            n=int(passk_cfg["n"]),
            k_values=[int(k) for k in passk_cfg["k_values"]],
            temperature=float(infer_cfg["temperature"]),
            top_p=float(infer_cfg["top_p"]),
            max_tokens=int(infer_cfg["max_tokens"]),
            batch_size=8,
            seed=2026,
        )

        out = Path("results/passk") / f"{args.model}_{split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved {split} metrics to {out}")


if __name__ == "__main__":
    main()
