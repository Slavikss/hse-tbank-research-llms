from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.base.data import Data
from src.envs.arithmetic_mod.env import ArithmeticModEnv
from src.rl.gold import attach_gold


def _make_dataset(path: Path) -> None:
    env = ArithmeticModEnv()
    items = env.generate(num_of_questions=20, max_attempts=200, difficulty=3, seed=2026)
    items = attach_gold(items)
    Data.to_jsonl_file(items, path)


def _write_config(path: Path, train_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "data:",
                f"  train_path: {train_path}",
                "  val_path: data/val/val.jsonl",
                "  hard_train_path: data/hard/hard_train.jsonl",
                "  hard_val_path: data/hard/hard_val.jsonl",
                "  prediction_dir: results/predictions",
                "  curves_dir: results/curves",
                "dry_run:",
                "  num_examples: 8",
            ]
        ),
        encoding="utf-8",
    )


def _run(module: str, config_path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", module, "--config", str(config_path), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{module} failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc.stdout


def test_train_scripts_dry_run(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    _make_dataset(train_path)

    config_path = tmp_path / "train.yaml"
    _write_config(config_path, train_path)

    out_sft = _run("src.rl.train_sft", config_path)
    out_grpo = _run("src.rl.train_grpo", config_path)
    out_sft_grpo = _run("src.rl.train_sft_grpo", config_path)
    out_srft = _run("src.rl.train_srft", config_path)

    assert "dry run" in out_sft.lower()
    assert "dry run" in out_grpo.lower()
    assert "dry run" in out_sft_grpo.lower()
    assert "dry run" in out_srft.lower()
