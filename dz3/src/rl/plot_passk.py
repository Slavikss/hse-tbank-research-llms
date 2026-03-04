"""Plot pass@k curves for multiple models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

MODEL_ORDER = ["baseline", "grpo", "sft", "sft_grpo", "srft"]
MODEL_LABEL = {
    "baseline": "Baseline",
    "grpo": "GRPO-only",
    "sft": "SFT-only",
    "sft_grpo": "SFT->GRPO",
    "srft": "SRFT",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_split(split: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in MODEL_ORDER:
        path = Path("results/passk") / f"{model}_{split}.json"
        if not path.exists():
            continue
        data = _load(path)
        passk = data["summary"]["pass_at_k"]
        x = [int(k) for k in sorted(passk, key=lambda v: int(v))]
        y = [float(passk[str(k)]) for k in x]
        ax.plot(x, y, marker="o", label=MODEL_LABEL.get(model, model))

    ax.set_xscale("log", base=2)
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title(f"pass@k on {split}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pass@k curves")
    parser.add_argument("--split", choices=["val", "hard_val"], default=None)
    args = parser.parse_args()

    if args.split:
        out = plot_split(args.split, Path(f"results/figures/passk_{args.split}.png"))
        print(f"Saved chart to {out}")
    else:
        out1 = plot_split("val", Path("results/figures/passk_val.png"))
        out2 = plot_split("hard_val", Path("results/figures/passk_hard_val.png"))
        print(f"Saved charts to {out1} and {out2}")


if __name__ == "__main__":
    main()
