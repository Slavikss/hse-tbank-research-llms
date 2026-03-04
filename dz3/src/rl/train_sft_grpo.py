"""Two-stage training: SFT first, then GRPO initialized from SFT."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml

from src.rl.config_utils import DEFAULT_TRAIN_CONFIG, load_config
from src.rl.train_grpo import _run_dry as grpo_dry
from src.rl.train_grpo import _train as grpo_train
from src.rl.train_sft import _run_dry as sft_dry
from src.rl.train_sft import _train as sft_train


def _build_stage2_config(config: dict) -> dict:
    updated = yaml.safe_load(yaml.safe_dump(config))

    sft_output = str(updated["sft"]["output_dir"])
    updated["training"]["init_model_path"] = sft_output
    updated["training"]["output_dir"] = "outputs/sft_grpo/runs"
    updated["models"]["grpo"] = str(updated["models"]["sft_grpo"])
    return updated


def _run_dry(config: dict) -> None:
    sft_dry(config)
    stage2 = _build_stage2_config(config)
    grpo_dry(stage2)
    print("SFT->GRPO dry run completed")


def _train(config: dict) -> None:
    sft_train(config)
    stage2 = _build_stage2_config(config)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(stage2, tmp)
        temp_path = Path(tmp.name)

    loaded = load_config(temp_path, DEFAULT_TRAIN_CONFIG)
    grpo_train(loaded)
    print("SFT->GRPO training completed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT followed by GRPO")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, DEFAULT_TRAIN_CONFIG)
    if args.dry_run:
        _run_dry(config)
    else:
        _train(config)


if __name__ == "__main__":
    main()
