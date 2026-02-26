from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.base.data import Data
from src.envs.arithmetic_mod.env import ArithmeticModEnv


def test_train_script_smoke(tmp_path: Path) -> None:
    env = ArithmeticModEnv()
    items = env.generate(num_of_questions=20, max_attempts=200, difficulty=3, seed=2026)

    train_path = tmp_path / "train.jsonl"
    Data.to_jsonl_file(items, train_path)

    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                f"  train_path: {train_path}",
                "  eval_dir: data/eval",
                "  prediction_dir: results/predictions",
                "dry_run:",
                "  num_examples: 8",
            ]
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.rl.train_grpo",
            "--config",
            str(config_path),
            "--dry-run",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "Dry run completed" in proc.stdout
