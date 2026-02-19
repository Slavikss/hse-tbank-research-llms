from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.logging import ensure_dir

sns.set_theme(style="whitegrid")


def plot_rl_learning_curves(combined_metrics: pd.DataFrame, output_png: str | Path) -> None:
    ensure_dir(Path(output_png).parent)
    eval_df = combined_metrics.dropna(subset=["eval_return_mean"]).copy()
    if eval_df.empty:
        return

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=eval_df,
        x="env_steps",
        y="eval_return_mean",
        hue="experiment",
        estimator="mean",
        errorbar=("sd"),
    )
    plt.title("RL: learning curves (eval mean +/- sd)")
    plt.ylabel("Eval return")
    plt.xlabel("Environment steps")
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def plot_coverage_curve(coverage_df: pd.DataFrame, output_png: str | Path) -> None:
    ensure_dir(Path(output_png).parent)
    if coverage_df.empty:
        return

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=coverage_df, x="fraction", y="eval_reward_mean", marker="o")
    plt.title("BC: dataset coverage vs performance")
    plt.xlabel("Dataset fraction")
    plt.ylabel("Eval reward")
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def plot_distribution_shift(shift_df: pd.DataFrame, output_png: str | Path) -> None:
    ensure_dir(Path(output_png).parent)
    if shift_df.empty:
        return

    plot_df = shift_df.melt(
        id_vars=["obs_perturb_std"],
        value_vars=["bc_reward_mean", "expert_reward_mean"],
        var_name="policy",
        value_name="reward",
    )
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=plot_df, x="obs_perturb_std", y="reward", hue="policy", marker="o")
    plt.title("BC under observation perturbations")
    plt.xlabel("Observation noise std")
    plt.ylabel("Eval reward")
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def plot_error_compounding(compounding_df: pd.DataFrame, output_png: str | Path) -> None:
    ensure_dir(Path(output_png).parent)
    filtered = compounding_df[compounding_df["count"] > 0]
    if filtered.empty:
        return

    plt.figure(figsize=(8, 4))
    sns.lineplot(data=filtered, x="timestep", y="mismatch_rate")
    plt.title("BC error compounding")
    plt.xlabel("Timestep")
    plt.ylabel("Mismatch rate vs expert")
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()
