"""Plot reward/entropy/length curves from training logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _plot_metric(metric: str, title: str, output: Path, files: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, path in files.items():
        rows = _load_jsonl(path)
        if not rows:
            continue
        x = [int(row.get("step", idx + 1) or (idx + 1)) for idx, row in enumerate(rows)]
        y: list[float] = []
        for row in rows:
            value = row.get(metric)
            if value is None:
                continue
            y.append(float(value))
        if not y:
            continue
        x = x[: len(y)]
        ax.plot(x, y, label=label)

    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.parse_args()

    files = {
        "GRPO": Path("results/curves/grpo_train.jsonl"),
        "SRFT": Path("results/curves/srft_train.jsonl"),
    }

    _plot_metric("reward_mean", "Reward Curve", Path("results/figures/reward_curve.png"), files)
    _plot_metric(
        "gen_len_mean", "Generation Length Curve", Path("results/figures/gen_len_curve.png"), files
    )
    _plot_metric("entropy", "Entropy Curve", Path("results/figures/entropy_curve.png"), files)
    print("Saved training curves to results/figures")


if __name__ == "__main__":
    main()
