from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.bc.dataset import load_dataset_npz, narrow_state_subset, save_dataset_npz, subset_dataset
from src.bc.train_bc import train_bc_from_arrays, train_bc_from_npz
from src.config import BCConfig
from src.envs import make_env
from src.models import load_policy_checkpoint
from src.rl.evaluate import evaluate_policy
from src.utils.logging import ensure_dir
from src.utils.plotting import plot_error_compounding


def _measure_error_compounding(
    env_id: str,
    bc_policy: torch.nn.Module,
    expert_policy: torch.nn.Module,
    episodes: int,
    max_steps: int,
    seed: int,
) -> pd.DataFrame:
    env = make_env(env_id=env_id, seed=seed, max_steps=max_steps)
    mismatches = np.zeros(max_steps, dtype=np.float64)
    counts = np.zeros(max_steps, dtype=np.int64)

    for ep_idx in range(episodes):
        obs, _ = env.reset(seed=seed + ep_idx)
        for t in range(max_steps):
            obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), dtype=torch.float32).unsqueeze(
                0
            )
            bc_action = int(torch.argmax(bc_policy(obs_t), dim=-1).item())
            expert_action = int(torch.argmax(expert_policy(obs_t), dim=-1).item())
            mismatches[t] += float(bc_action != expert_action)
            counts[t] += 1

            obs, _, terminated, truncated, _ = env.step(bc_action)
            if terminated or truncated:
                break

    env.close()
    rates = np.divide(mismatches, counts, out=np.zeros_like(mismatches), where=counts > 0)
    return pd.DataFrame(
        {
            "timestep": np.arange(max_steps, dtype=np.int64),
            "mismatch_rate": rates,
            "count": counts,
        }
    )


def run_bc_failure_experiments(
    config: BCConfig,
    dataset_npz: str | Path,
    expert_checkpoint: str | Path,
    bc_checkpoint: str | Path,
    max_steps: int = 500,
) -> dict[str, Any]:
    base_dir = Path(config.output_dir) / "failure_experiments"
    ensure_dir(base_dir)

    states, actions = load_dataset_npz(dataset_npz)

    coverage_rows: list[dict[str, Any]] = []
    for fraction in config.coverage_fractions:
        subset_npz = base_dir / f"coverage_{int(fraction * 100)}.npz"
        subset_dataset(dataset_npz, subset_npz, fraction=fraction, seed=config.seed)
        local_cfg = replace(config, epochs=max(8, config.epochs // 2), early_stop_patience=4)
        result = train_bc_from_npz(
            dataset_npz=subset_npz,
            config=local_cfg,
            run_name=f"coverage_{int(fraction * 100)}",
            max_steps=max_steps,
        )
        coverage_rows.append(
            {
                "fraction": float(fraction),
                "samples": int(max(1, int(len(actions) * fraction))),
                "eval_reward_mean": float(result.eval_reward_mean),
            }
        )

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = base_dir / "coverage_vs_performance.csv"
    coverage_df.to_csv(coverage_path, index=False)

    expert_policy = load_policy_checkpoint(expert_checkpoint, device="cpu")
    bc_policy = load_policy_checkpoint(bc_checkpoint, device="cpu")

    shift_rows: list[dict[str, Any]] = []
    for std in config.shift_perturbation_stds:
        bc_eval = evaluate_policy(
            env_id=config.env_id,
            policy=bc_policy,
            episodes=config.bc_eval_episodes,
            seed=config.seed + 900_000,
            max_steps=max_steps,
            deterministic=True,
            obs_perturb_std=std,
        )
        expert_eval = evaluate_policy(
            env_id=config.env_id,
            policy=expert_policy,
            episodes=config.bc_eval_episodes,
            seed=config.seed + 910_000,
            max_steps=max_steps,
            deterministic=True,
            obs_perturb_std=std,
        )
        shift_rows.append(
            {
                "obs_perturb_std": float(std),
                "bc_reward_mean": float(bc_eval["mean"]),
                "expert_reward_mean": float(expert_eval["mean"]),
            }
        )

    shift_df = pd.DataFrame(shift_rows)
    shift_path = base_dir / "distribution_shift.csv"
    shift_df.to_csv(shift_path, index=False)

    narrow_states, narrow_actions = narrow_state_subset(states, actions)
    narrow_npz = base_dir / "narrow_dataset.npz"
    save_dataset_npz(narrow_npz, narrow_states, narrow_actions)
    narrow_cfg = replace(config, epochs=max(8, config.epochs // 2), early_stop_patience=4)
    narrow_result = train_bc_from_arrays(
        narrow_states,
        narrow_actions,
        config=narrow_cfg,
        run_name="narrow_state_train",
        max_steps=max_steps,
    )

    narrow_policy = load_policy_checkpoint(narrow_result.checkpoint_path, device="cpu")
    narrow_eval = evaluate_policy(
        env_id=config.env_id,
        policy=narrow_policy,
        episodes=config.bc_eval_episodes,
        seed=config.seed + 920_000,
        max_steps=max_steps,
        deterministic=True,
        obs_perturb_std=0.0,
    )

    compounding_df = _measure_error_compounding(
        env_id=config.env_id,
        bc_policy=bc_policy,
        expert_policy=expert_policy,
        episodes=config.error_compounding_episodes,
        max_steps=config.error_compounding_max_steps,
        seed=config.seed + 930_000,
    )
    compounding_path = base_dir / "error_compounding.csv"
    compounding_df.to_csv(compounding_path, index=False)
    plot_error_compounding(compounding_df, base_dir / "error_compounding.png")

    summary = {
        "coverage_table": str(coverage_path),
        "distribution_shift_table": str(shift_path),
        "error_compounding_table": str(compounding_path),
        "narrow_dataset_samples": int(len(narrow_actions)),
        "narrow_bc_eval_reward": float(narrow_eval["mean"]),
    }
    summary_path = base_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
