from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.envs import make_env
from src.models import load_policy_checkpoint
from src.utils.logging import ensure_dir


def save_dataset_npz(path: str | Path, states: np.ndarray, actions: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez(
        path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
    )


def load_dataset_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(path))
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    return states, actions


def generate_expert_dataset(
    expert_checkpoint: str | Path,
    output_npz: str | Path,
    env_id: str,
    episodes: int,
    min_reward: float,
    seed: int,
    max_steps: int = 500,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    policy = load_policy_checkpoint(expert_checkpoint, device="cpu")
    env = make_env(env_id=env_id, seed=seed, max_steps=max_steps)

    all_states: list[np.ndarray] = []
    all_actions: list[int] = []
    kept_episode_returns: list[float] = []
    raw_episode_returns: list[float] = []

    for ep_idx in range(episodes):
        obs, _ = env.reset(seed=seed + ep_idx)
        ep_states: list[np.ndarray] = []
        ep_actions: list[int] = []
        ep_return = 0.0

        for _ in range(max_steps):
            obs_arr = np.asarray(obs, dtype=np.float32)
            obs_tensor = torch.tensor(obs_arr, dtype=torch.float32).unsqueeze(0)
            action = policy.act(obs_tensor, deterministic=True)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_states.append(obs_arr)
            ep_actions.append(action)
            ep_return += float(reward)
            obs = next_obs
            if terminated or truncated:
                break

        raw_episode_returns.append(ep_return)
        if ep_return >= min_reward:
            kept_episode_returns.append(ep_return)
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)

    env.close()

    if not all_states:
        msg = (
            "No trajectories passed min_reward threshold. "
            f"min_reward={min_reward}, max_collected={max(raw_episode_returns, default=0.0):.2f}"
        )
        raise RuntimeError(msg)

    states = np.asarray(all_states, dtype=np.float32)
    actions = np.asarray(all_actions, dtype=np.int64)
    save_dataset_npz(output_npz, states=states, actions=actions)

    stats = {
        "expert_checkpoint": str(expert_checkpoint),
        "output_npz": str(output_npz),
        "episodes_requested": int(episodes),
        "episodes_kept": int(len(kept_episode_returns)),
        "samples": int(len(actions)),
        "kept_return_mean": float(np.mean(kept_episode_returns)),
        "raw_return_mean": float(np.mean(raw_episode_returns)),
        "min_reward_threshold": float(min_reward),
    }

    if metadata_path is not None:
        meta_path = Path(metadata_path)
        ensure_dir(meta_path.parent)
        meta_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return stats


def subset_dataset(
    source_npz: str | Path,
    output_npz: str | Path,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < fraction <= 1.0):
        msg = f"Fraction must be in (0, 1], got {fraction}"
        raise ValueError(msg)

    states, actions = load_dataset_npz(source_npz)
    n = len(actions)
    keep = max(1, int(n * fraction))

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)[:keep]

    subset_states = states[indices]
    subset_actions = actions[indices]
    save_dataset_npz(output_npz, subset_states, subset_actions)
    return subset_states, subset_actions


def narrow_state_subset(
    states: np.ndarray,
    actions: np.ndarray,
    position_threshold: float = 0.05,
    angle_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    # CartPole state: [cart_pos, cart_vel, pole_angle, pole_vel].
    mask = (np.abs(states[:, 0]) <= position_threshold) & (np.abs(states[:, 2]) <= angle_threshold)
    if not np.any(mask):
        return states, actions
    return states[mask], actions[mask]
