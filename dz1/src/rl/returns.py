from __future__ import annotations

import numpy as np


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    running = 0.0
    out: list[float] = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        out.append(float(running))
    out.reverse()
    return out


def flatten_discounted_returns(trajectories: list[list[float]], gamma: float) -> np.ndarray:
    returns: list[float] = []
    for rewards in trajectories:
        returns.extend(discounted_returns(rewards, gamma))
    return np.asarray(returns, dtype=np.float32)


def rloo_baselines(trajectory_total_returns: list[float]) -> list[float]:
    n = len(trajectory_total_returns)
    if n <= 1:
        return [0.0 for _ in trajectory_total_returns]
    total_sum = float(np.sum(trajectory_total_returns))
    return [float((total_sum - ret) / (n - 1)) for ret in trajectory_total_returns]


def normalize_advantages(
    advantages: np.ndarray,
    eps: float = 1e-8,
    center: bool = True,
) -> np.ndarray:
    base = advantages - float(np.mean(advantages)) if center else advantages
    std = float(np.std(base))
    return base / (std + eps)
