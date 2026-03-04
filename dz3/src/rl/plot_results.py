"""Plot grouped bar chart for baseline vs trained metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_metrics(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_accuracy_by_difficulty(metrics: dict) -> tuple[list[int], list[float]]:
    per_diff_raw = metrics["per_difficulty"]
    normalized = {int(k): v for k, v in per_diff_raw.items()}
    difficulties = sorted(normalized.keys())
    values = [float(normalized[d]["accuracy"]) for d in difficulties]
    return difficulties, values


def plot_grouped_bars(
    baseline_metrics_path: str | Path,
    trained_metrics_path: str | Path,
    output_path: str | Path,
) -> Path:
    baseline = _load_metrics(baseline_metrics_path)
    trained = _load_metrics(trained_metrics_path)

    diffs_b, acc_b = _extract_accuracy_by_difficulty(baseline)
    diffs_t, acc_t = _extract_accuracy_by_difficulty(trained)

    if diffs_b != diffs_t:
        raise ValueError("Difficulty sets do not match between baseline and trained metrics")

    x = list(range(len(diffs_b)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([v - width / 2 for v in x], acc_b, width, label="Baseline", color="#7aa6c2")
    ax.bar([v + width / 2 for v in x], acc_t, width, label="Trained", color="#cc7a5c")

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Difficulty")
    ax.set_title("Baseline vs Trained Accuracy by Difficulty")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in diffs_b])
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline vs trained accuracy grouped bars")
    parser.add_argument("--baseline-metrics", default="results/metrics/metrics_baseline.json")
    parser.add_argument("--trained-metrics", default="results/metrics/metrics_trained.json")
    parser.add_argument("--output", default="results/figures/baseline_vs_trained.png")
    args = parser.parse_args()

    out = plot_grouped_bars(
        baseline_metrics_path=args.baseline_metrics,
        trained_metrics_path=args.trained_metrics,
        output_path=args.output,
    )
    print(f"Saved chart to {out}")


if __name__ == "__main__":
    main()
