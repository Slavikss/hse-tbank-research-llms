from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str, seed: int, max_steps: int | None = None) -> gym.Env:
    env = gym.make(env_id)
    if max_steps is not None and hasattr(env, "_max_episode_steps"):
        env._max_episode_steps = int(max_steps)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
