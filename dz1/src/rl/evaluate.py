from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.envs import make_env


@torch.no_grad()
def evaluate_policy(
    env_id: str,
    policy: torch.nn.Module,
    episodes: int,
    seed: int,
    max_steps: int,
    device: str = "cpu",
    deterministic: bool = True,
    obs_perturb_std: float = 0.0,
    reset_noise_std: float = 0.0,
) -> dict[str, Any]:
    env = make_env(env_id, seed=seed, max_steps=max_steps)
    returns: list[float] = []

    for ep_idx in range(episodes):
        obs, _ = env.reset(seed=seed + ep_idx)
        if reset_noise_std > 0.0 and hasattr(env.unwrapped, "state"):
            state = np.asarray(env.unwrapped.state, dtype=np.float32)
            state = state + np.random.normal(0.0, reset_noise_std, size=state.shape)
            env.unwrapped.state = state
            obs = state.copy()

        ep_return = 0.0
        for _ in range(max_steps):
            obs_for_policy = np.asarray(obs, dtype=np.float32)
            if obs_perturb_std > 0.0:
                obs_for_policy = obs_for_policy + np.random.normal(
                    0.0,
                    obs_perturb_std,
                    size=obs_for_policy.shape,
                )

            obs_tensor = torch.tensor(obs_for_policy, dtype=torch.float32, device=device).unsqueeze(
                0
            )
            logits = policy(obs_tensor)
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            if terminated or truncated:
                break

        returns.append(ep_return)

    env.close()
    return {
        "episode_returns": returns,
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
    }
