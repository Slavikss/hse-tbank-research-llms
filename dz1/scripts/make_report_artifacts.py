from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.logging import ensure_dir
from src.utils.plotting import plot_coverage_curve, plot_distribution_shift, plot_rl_learning_curves


def _safe_copy_csv(source: Path, target: Path) -> None:
    if source.exists():
        df = pd.read_csv(source)
        ensure_dir(target.parent)
        df.to_csv(target, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report-ready tables and figures.")
    parser.add_argument("--input", type=str, default="artifacts", help="Artifacts root")
    parser.add_argument("--output", type=str, default="report/assets", help="Report assets dir")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    ensure_dir(output_root)

    rl_dir = input_root / "rl"
    bc_dir = input_root / "bc"

    rl_summary_src = rl_dir / "aggregate_results.csv"
    rl_experiment_summary_src = rl_dir / "experiment_summary.csv"
    rl_metrics_src = rl_dir / "combined_metrics.csv"

    _safe_copy_csv(rl_summary_src, output_root / "rl_final_table.csv")
    _safe_copy_csv(rl_experiment_summary_src, output_root / "rl_experiment_summary.csv")

    if rl_metrics_src.exists():
        metrics_df = pd.read_csv(rl_metrics_src)
        plot_rl_learning_curves(metrics_df, output_root / "rl_learning_curves.png")

    coverage_src = bc_dir / "failure_experiments" / "coverage_vs_performance.csv"
    shift_src = bc_dir / "failure_experiments" / "distribution_shift.csv"

    _safe_copy_csv(coverage_src, output_root / "bc_coverage_table.csv")
    _safe_copy_csv(shift_src, output_root / "bc_shift_table.csv")

    if coverage_src.exists():
        plot_coverage_curve(pd.read_csv(coverage_src), output_root / "bc_coverage_curve.png")
    if shift_src.exists():
        plot_distribution_shift(pd.read_csv(shift_src), output_root / "bc_distribution_shift.png")

    bc_main_summary = bc_dir / "bc_main" / "summary.json"
    if bc_main_summary.exists():
        summary_df = pd.read_json(bc_main_summary, typ="series").to_frame(name="value")
        summary_df.to_csv(output_root / "bc_main_summary.csv")

    print(f"Saved report artifacts to: {output_root}")


if __name__ == "__main__":
    main()
