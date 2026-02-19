from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.config import build_pg_train_config, load_rl_suite_config
from src.rl.train_pg import train_policy_gradient
from src.utils.logging import ensure_dir
from src.utils.plotting import plot_rl_learning_curves


def _collect_steps_to_thresholds(df: pd.DataFrame, threshold: float) -> float | None:
    eval_df = df.dropna(subset=["eval_return_mean"]).sort_values("env_steps")
    passed = eval_df[eval_df["eval_return_mean"] >= threshold]
    if passed.empty:
        return None
    return float(passed.iloc[0]["env_steps"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL experiment suite for policy gradients.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    suite = load_rl_suite_config(args.config)
    out_root = Path(suite.common.output_dir)
    ensure_dir(out_root)

    summaries: list[dict] = []
    combined_metrics_frames: list[pd.DataFrame] = []
    best_reward = float("-inf")
    best_checkpoint = None

    for experiment in suite.experiments:
        for seed in suite.seeds:
            run_cfg = build_pg_train_config(suite.common, experiment=experiment, seed=seed)
            result = train_policy_gradient(run_cfg)
            summary = pd.read_json(result.summary_json, typ="series").to_dict()
            summary["experiment"] = experiment.name
            summary["seed"] = seed

            metrics_df = pd.read_csv(result.metrics_csv)
            metrics_df["experiment"] = experiment.name
            metrics_df["seed"] = seed
            steps_to_475 = _collect_steps_to_thresholds(metrics_df, threshold=475.0)
            steps_to_495 = _collect_steps_to_thresholds(metrics_df, threshold=495.0)
            metrics_df["steps_to_475"] = steps_to_475
            metrics_df["steps_to_495"] = steps_to_495
            combined_metrics_frames.append(metrics_df)
            summary["steps_to_475"] = steps_to_475
            summary["steps_to_495"] = steps_to_495

            summaries.append(summary)
            if float(summary["best_eval_reward"]) > best_reward:
                best_reward = float(summary["best_eval_reward"])
                best_checkpoint = summary["best_checkpoint"]

    summary_df = pd.DataFrame(summaries)
    summary_path = out_root / "aggregate_results.csv"
    summary_df.to_csv(summary_path, index=False)
    experiment_summary = (
        summary_df.groupby("experiment", as_index=False)
        .agg(
            best_eval_reward_mean=("best_eval_reward", "mean"),
            best_eval_reward_std=("best_eval_reward", "std"),
            total_env_steps_mean=("total_env_steps", "mean"),
            total_env_steps_std=("total_env_steps", "std"),
            steps_to_475_mean=("steps_to_475", "mean"),
            steps_to_475_std=("steps_to_475", "std"),
            steps_to_495_mean=("steps_to_495", "mean"),
            steps_to_495_std=("steps_to_495", "std"),
        )
        .sort_values("best_eval_reward_mean", ascending=False)
    )
    experiment_summary_path = out_root / "experiment_summary.csv"
    experiment_summary.to_csv(experiment_summary_path, index=False)

    combined_metrics = pd.concat(combined_metrics_frames, ignore_index=True)
    combined_metrics_path = out_root / "combined_metrics.csv"
    combined_metrics.to_csv(combined_metrics_path, index=False)

    if best_checkpoint is not None:
        best_target = out_root / "best_expert.pt"
        shutil.copy2(best_checkpoint, best_target)

    plot_rl_learning_curves(combined_metrics, out_root / "learning_curves.png")

    ranking = summary_df.sort_values("best_eval_reward", ascending=False).head(10)
    ranking_path = out_root / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    print(f"Saved RL summaries to: {summary_path}")
    print(f"Saved RL experiment summary to: {experiment_summary_path}")
    print(f"Saved RL combined metrics to: {combined_metrics_path}")
    if best_checkpoint is not None:
        print(f"Best expert checkpoint copied to: {out_root / 'best_expert.pt'}")


if __name__ == "__main__":
    main()
