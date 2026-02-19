from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.bc.dataset import generate_expert_dataset
from src.bc.experiments import run_bc_failure_experiments
from src.bc.train_bc import train_bc_from_npz
from src.config import load_bc_config
from src.utils.logging import ensure_dir, write_json
from src.utils.plotting import plot_coverage_curve, plot_distribution_shift


def main() -> None:
    parser = argparse.ArgumentParser(description="Run behaviour cloning pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--expert", type=str, required=True, help="Checkpoint of expert RL policy")
    args = parser.parse_args()

    config = load_bc_config(args.config)
    out_dir = Path(config.output_dir)
    ensure_dir(out_dir)

    dataset_npz = out_dir / "expert_dataset.npz"
    dataset_meta = out_dir / "expert_dataset.json"
    dataset_stats = generate_expert_dataset(
        expert_checkpoint=args.expert,
        output_npz=dataset_npz,
        env_id=config.env_id,
        episodes=config.expert_dataset_episodes,
        min_reward=config.expert_min_reward,
        seed=config.seed,
        metadata_path=dataset_meta,
    )

    bc_result = train_bc_from_npz(dataset_npz=dataset_npz, config=config, run_name="bc_main")

    failure_summary = run_bc_failure_experiments(
        config=config,
        dataset_npz=dataset_npz,
        expert_checkpoint=args.expert,
        bc_checkpoint=bc_result.checkpoint_path,
    )

    coverage_path = Path(failure_summary["coverage_table"])
    shift_path = Path(failure_summary["distribution_shift_table"])
    coverage_df = pd.read_csv(coverage_path)
    shift_df = pd.read_csv(shift_path)

    plot_coverage_curve(coverage_df, out_dir / "coverage_vs_performance.png")
    plot_distribution_shift(shift_df, out_dir / "distribution_shift.png")

    summary_payload = {
        "dataset": dataset_stats,
        "bc_main_summary": bc_result.summary_json,
        "bc_main_checkpoint": bc_result.checkpoint_path,
        "failure_experiments": failure_summary,
    }
    write_json(out_dir / "suite_summary.json", summary_payload)

    overview = pd.DataFrame(
        [
            {
                "artifact": "expert_dataset",
                "path": str(dataset_npz),
            },
            {
                "artifact": "bc_main_checkpoint",
                "path": str(bc_result.checkpoint_path),
            },
            {
                "artifact": "coverage_table",
                "path": str(coverage_path),
            },
            {
                "artifact": "distribution_shift_table",
                "path": str(shift_path),
            },
        ]
    )
    overview_path = out_dir / "artifacts_overview.csv"
    overview.to_csv(overview_path, index=False)

    print(f"Saved BC suite summary to: {out_dir / 'suite_summary.json'}")


if __name__ == "__main__":
    main()
